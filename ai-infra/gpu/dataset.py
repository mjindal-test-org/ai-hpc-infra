"""
dataset.py — Dataset loading for HuggingFaceH4/ultrachat_200k with Mistral-7B-v0.3.

ultrachat_200k format:
    {
        "prompt_id": "abc123",
        "messages": [
            {"role": "user",      "content": "What is ML?"},
            {"role": "assistant", "content": "ML is ..."},
            {"role": "user",      "content": "Give an example."},
            {"role": "assistant", "content": "Sure! ..."}
        ]
    }

We use tokenizer.apply_chat_template() to format conversations correctly for
Mistral's [INST]...[/INST] format. Labels are masked to -100 for all
non-assistant tokens so the model only learns to generate assistant responses.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
from typing import Optional


def make_tokenizer(model_name: str, hf_token: Optional[str] = None):
    """Load Mistral tokeniser. No token needed for Mistral."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=hf_token,
    )
    # Mistral has no pad token by default — use EOS as pad
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.padding_side = "right"
    return tokenizer


def _build_labels(input_ids: list, messages: list, tokenizer) -> list:
    """
    Mask all tokens except assistant responses with -100.
    -100 = ignored by PyTorch CrossEntropyLoss.
    Model only learns to predict assistant turns.
    """
    labels   = [-100] * len(input_ids)
    prev_len = 0

    cumulative = []
    for msg in messages:
        cumulative.append(msg)

        partial     = tokenizer.apply_chat_template(
            cumulative, tokenize=False, add_generation_prompt=False
        )
        partial_ids = tokenizer(partial, truncation=False,
                                return_tensors=None)["input_ids"]
        current_len = len(partial_ids)

        if msg["role"] == "assistant":
            start = prev_len
            end   = min(current_len, len(input_ids))
            for i in range(start, end):
                labels[i] = input_ids[i]

        prev_len = current_len

    return labels


def format_and_tokenize(example: dict, tokenizer, max_seq_len: int) -> dict:
    """Convert one ultrachat example into model inputs + masked labels."""
    messages = example["messages"]

    # apply_chat_template produces Mistral format:
    #   <s>[INST] user [/INST] assistant </s>[INST] user [/INST] ...
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    tokenised = tokenizer(
        formatted,
        truncation=True,
        max_length=max_seq_len,
        padding=False,
        return_tensors=None,
    )

    input_ids      = tokenised["input_ids"]
    attention_mask = tokenised["attention_mask"]
    labels         = _build_labels(input_ids, messages, tokenizer)

    return {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "labels":         labels,
    }


class UltraChatDataset(Dataset):
    """
    PyTorch Dataset for HuggingFaceH4/ultrachat_200k.
    Downloads and caches to ~/.cache/huggingface/datasets/ on first run (~1 GB).
    Applies formatting lazily (at item fetch time) to keep memory usage low.
    """

    def __init__(self, dataset_name: str, split: str, tokenizer,
                 max_seq_len: int, max_samples: int = -1,
                 hf_token: Optional[str] = None):

        print(f"Loading {dataset_name} ({split})...")
        ds = load_dataset(dataset_name, split=split, token=hf_token)

        # Optionally cap dataset size for quick testing
        if max_samples > 0:
            ds = ds.select(range(min(max_samples, len(ds))))

        # Drop examples with fewer than 2 messages
        ds = ds.filter(lambda x: len(x["messages"]) >= 2)

        self.data        = ds
        self.tokenizer   = tokenizer
        self.max_seq_len = max_seq_len
        print(f"  {len(self.data):,} examples ready.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return format_and_tokenize(
            self.data[idx], self.tokenizer, self.max_seq_len
        )


def make_dataloader_gpu(dataset: UltraChatDataset, batch_size: int,
                        shuffle: bool = True) -> DataLoader:
    """
    GPU DataLoader — dynamic padding (each batch padded to its longest sequence).
    GPU handles variable shapes natively; no XLA recompilation risk.
    """
    from transformers import DataCollatorForSeq2Seq

    collator = DataCollatorForSeq2Seq(
        dataset.tokenizer,
        padding=True,
        pad_to_multiple_of=8,   # Tensor Core alignment
        return_tensors="pt",
        label_pad_token_id=-100,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )


def make_dataloader_tpu(dataset: UltraChatDataset, batch_size: int,
                        max_seq_len: int, device,
                        shuffle: bool = True):
    """
    TPU DataLoader — FIXED padding to max_seq_len.
    XLA compiles once on the first batch shape; fixed shape = no recompilation.
    """
    import torch_xla.distributed.parallel_loader as pl
    from transformers import DataCollatorForSeq2Seq

    collator = DataCollatorForSeq2Seq(
        dataset.tokenizer,
        padding="max_length",
        max_length=max_seq_len,
        return_tensors="pt",
        label_pad_token_id=-100,
    )
    base = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        num_workers=4,
        drop_last=True,
    )
    return pl.MpDeviceLoader(base, device)