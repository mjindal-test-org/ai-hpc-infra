"""
dataset.py — ultrachat_200k dataset for Mistral-7B-v0.3.

ultrachat_200k message format:
  {"messages": [
    {"role": "user",      "content": "..."},
    {"role": "assistant", "content": "..."},
    ...
  ]}

Uses tokenizer.apply_chat_template() for correct Mistral formatting.
Labels are masked to -100 for all non-assistant tokens.
"""

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
from typing import Optional


def make_tokenizer(model_name: str, hf_token: Optional[str] = None):
    """Load Mistral tokeniser. No token needed."""
    tok = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    if tok.pad_token is None:
        tok.pad_token    = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "right"
    return tok


def _build_labels(input_ids: list, messages: list, tokenizer) -> list:
    """
    Return labels array: input_ids for assistant tokens, -100 for everything else.
    -100 is ignored by PyTorch CrossEntropyLoss.
    """
    labels   = [-100] * len(input_ids)
    prev_len = 0
    cumulative = []

    for msg in messages:
        cumulative.append(msg)
        partial     = tokenizer.apply_chat_template(
            cumulative, tokenize=False, add_generation_prompt=False
        )
        partial_ids = tokenizer(
            partial, truncation=False, return_tensors=None
        )["input_ids"]
        current_len = len(partial_ids)

        if msg["role"] == "assistant":
            for i in range(prev_len, min(current_len, len(input_ids))):
                labels[i] = input_ids[i]

        prev_len = current_len

    return labels


def format_and_tokenize(example: dict, tokenizer, max_seq_len: int) -> dict:
    """Convert one ultrachat example to model inputs with masked labels."""
    messages  = example["messages"]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    tokenised = tokenizer(
        formatted, truncation=True, max_length=max_seq_len,
        padding=False, return_tensors=None,
    )
    input_ids      = tokenised["input_ids"]
    attention_mask = tokenised["attention_mask"]
    labels         = _build_labels(input_ids, messages, tokenizer)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


class UltraChatDataset(Dataset):
    def __init__(self, dataset_name: str, split: str, tokenizer,
                 max_seq_len: int, max_samples: int = -1,
                 hf_token: Optional[str] = None):
        print(f"Loading {dataset_name} ({split})...")
        ds = load_dataset(dataset_name, split=split, token=hf_token)
        if max_samples > 0:
            ds = ds.select(range(min(max_samples, len(ds))))
        ds = ds.filter(lambda x: len(x["messages"]) >= 2)
        self.data        = ds
        self.tokenizer   = tokenizer
        self.max_seq_len = max_seq_len
        print(f"  {len(self.data):,} examples ready.")

    def __len__(self):  return len(self.data)
    def __getitem__(self, idx):
        return format_and_tokenize(self.data[idx], self.tokenizer, self.max_seq_len)


def make_dataloader_gpu(dataset: UltraChatDataset, batch_size: int,
                        shuffle: bool = True,
                        sampler=None) -> DataLoader:
    """
    Dynamic padding per batch — efficient for GPU.

    sampler: pass a DistributedSampler for multi-GPU training.
             When a sampler is provided, shuffle must be False
             (the sampler handles shuffling internally).
    """
    from transformers import DataCollatorForSeq2Seq
    collator = DataCollatorForSeq2Seq(
        dataset.tokenizer, padding=True, pad_to_multiple_of=8,
        return_tensors="pt", label_pad_token_id=-100,
    )
    # If a sampler is provided, pass it directly into the constructor.
    # Never set loader.sampler after init — raises ValueError in PyTorch 2.x.
    return DataLoader(dataset, batch_size=batch_size,
                      shuffle=(shuffle if sampler is None else False),
                      sampler=sampler,          # passed here, not set after
                      collate_fn=collator, num_workers=4,
                      pin_memory=True, drop_last=True)


def make_dataloader_tpu(dataset: UltraChatDataset, batch_size: int,
                        max_seq_len: int, device, shuffle: bool = True):
    """Fixed max_length padding — required for XLA shape consistency."""
    import torch_xla.distributed.parallel_loader as pl
    from transformers import DataCollatorForSeq2Seq
    collator = DataCollatorForSeq2Seq(
        dataset.tokenizer, padding="max_length", max_length=max_seq_len,
        return_tensors="pt", label_pad_token_id=-100,
    )
    base = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collator, num_workers=4, drop_last=True)
    return pl.MpDeviceLoader(base, device)