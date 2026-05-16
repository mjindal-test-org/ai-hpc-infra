"""
dataset.py — Dataset loading for ultrachat_200k with Mistral-7B-v0.3.

Key changes from the Llama / support-conversations version:

1. Dataset format:
   ultrachat_200k has "messages" column — a list of role/content dicts:
   [
     {"role": "user",      "content": "Explain quantum computing..."},
     {"role": "assistant", "content": "Quantum computing uses..."},
     {"role": "user",      "content": "Can you give an example?"},
     {"role": "assistant", "content": "Sure! Imagine..."}
   ]
   The old dataset had flat "user" and "assistant" string columns.

2. Chat template:
   We use tokenizer.apply_chat_template() instead of manual string formatting.
   This is cleaner and automatically handles the correct format for each model:
     Llama 3:  <|begin_of_text|><|start_header_id|>...<|end_header_id|>...
     Mistral:  <s>[INST] ... [/INST] ... </s>[INST] ... [/INST]
   Switching models only requires changing config.model_name — no template edits.

3. Label masking:
   We mask all tokens except the assistant's responses so the model only
   learns to generate assistant turns, not repeat user messages.
   Works identically for multi-turn conversations.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
from typing import Optional


def make_tokenizer(model_name: str, hf_token: Optional[str] = None):
    """
    Load Mistral tokeniser.

    Mistral-7B-v0.3 includes a chat template in the tokeniser config,
    so apply_chat_template() works out of the box — no manual formatting.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=hf_token,   # None for Mistral (not gated)
    )

    # Mistral's tokeniser does not have a pad token by default.
    # Standard practice: use EOS token as pad token for decoder-only models.
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Pad on the right side for causal language modelling.
    # Left padding is used for batch inference but not training.
    tokenizer.padding_side = "right"

    return tokenizer


def format_and_tokenize(example: dict, tokenizer, max_seq_len: int) -> dict:
    """
    Convert one ultrachat_200k example into model inputs.

    ultrachat_200k example structure:
    {
        "prompt_id": "abc123",
        "messages": [
            {"role": "user",      "content": "What is machine learning?"},
            {"role": "assistant", "content": "Machine learning is..."},
            {"role": "user",      "content": "Give me an example."},
            {"role": "assistant", "content": "Sure! Consider..."}
        ]
    }

    Steps:
    1. Apply Mistral's chat template to get the correctly formatted string
    2. Tokenise with truncation to max_seq_len
    3. Build labels: -100 for all non-assistant tokens (masked from loss)
    """
    messages = example["messages"]

    # apply_chat_template handles all the Mistral-specific formatting:
    #   <s>[INST] user message [/INST] assistant response </s>
    #   [INST] user message [/INST] assistant response </s>
    # tokenize=False returns the string so we can tokenise ourselves
    # with full control over padding and truncation
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,  # False during training (labels include response)
    )

    # Tokenise the full conversation
    tokenised = tokenizer(
        formatted,
        truncation=True,
        max_length=max_seq_len,
        padding=False,        # padding happens in the DataLoader collator
        return_tensors=None,  # return plain lists — collator handles tensors
    )

    input_ids      = tokenised["input_ids"]
    attention_mask = tokenised["attention_mask"]

    # Build labels — start as a copy of input_ids then mask non-assistant tokens
    labels = _build_labels(input_ids, messages, tokenizer)

    return {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "labels":         labels,
    }


def _build_labels(input_ids: list, messages: list, tokenizer) -> list:
    """
    Mask all tokens except assistant responses with -100.

    -100 is PyTorch's "ignore this token" convention for CrossEntropyLoss.
    The model is trained ONLY on predicting assistant tokens.
    User turns and system prompts are masked so the model doesn't
    waste capacity learning to reproduce them.

    Strategy:
    - Rebuild each turn individually and find its token boundaries
    - Mask everything that isn't an assistant turn
    """
    labels = [-100] * len(input_ids)  # start: mask everything

    # We find assistant turn boundaries by tokenising turn by turn.
    # For each assistant message, find its tokens in input_ids and unmask them.

    # Build cumulative conversation to find turn positions
    cumulative_messages = []
    prev_len = 0

    for msg in messages:
        cumulative_messages.append(msg)

        # Tokenise conversation up to and including this message
        partial = tokenizer.apply_chat_template(
            cumulative_messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        partial_ids = tokenizer(
            partial, truncation=False, return_tensors=None
        )["input_ids"]
        current_len = len(partial_ids)

        # If this is an assistant message, unmask the tokens it added
        if msg["role"] == "assistant":
            start = prev_len
            end   = min(current_len, len(input_ids))
            for i in range(start, end):
                labels[i] = input_ids[i]   # unmask — model should predict this

        prev_len = current_len

    return labels


class UltraChatDataset(Dataset):
    """
    PyTorch Dataset wrapping ultrachat_200k.

    Downloads from HuggingFace Hub on first use (~1GB).
    Cached locally in ~/.cache/huggingface/datasets/ for subsequent runs.

    Applies formatting and tokenisation at item-fetch time (lazy).
    This keeps memory usage low — we don't tokenise all 200K examples upfront.
    """

    def __init__(self, dataset_name: str, split: str, tokenizer,
                 max_seq_len: int, max_samples: int = -1,
                 hf_token: Optional[str] = None):

        print(f"Loading {dataset_name} ({split} split)...")

        ds = load_dataset(
            dataset_name,
            split=split,
            token=hf_token,   # None for ultrachat (public dataset)
        )

        # Optionally limit dataset size for quick testing
        if max_samples > 0:
            ds = ds.select(range(min(max_samples, len(ds))))

        # Filter out examples with no messages or only one turn
        ds = ds.filter(lambda x: len(x["messages"]) >= 2)

        self.data        = ds
        self.tokenizer   = tokenizer
        self.max_seq_len = max_seq_len

        print(f"  Loaded {len(self.data):,} examples from {split}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        return format_and_tokenize(example, self.tokenizer, self.max_seq_len)


def make_dataloader_gpu(dataset: UltraChatDataset, batch_size: int,
                        shuffle: bool = True) -> DataLoader:
    """
    GPU DataLoader with dynamic padding.

    Each batch is padded to the length of its longest sequence.
    This is memory-efficient: short batches don't waste compute on padding.
    GPU handles variable tensor shapes natively — no recompilation.
    """
    from transformers import DataCollatorForSeq2Seq

    collator = DataCollatorForSeq2Seq(
        dataset.tokenizer,
        padding=True,            # "longest" — pad to longest in batch
        pad_to_multiple_of=8,    # align to 8 for Tensor Core efficiency
        return_tensors="pt",
        label_pad_token_id=-100,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        num_workers=4,
        pin_memory=True,   # faster CPU→GPU transfer
        drop_last=True,    # drop last incomplete batch
    )


def make_dataloader_tpu(dataset: UltraChatDataset, batch_size: int,
                        max_seq_len: int, device,
                        shuffle: bool = True):
    """
    TPU DataLoader with fixed-length padding.

    MUST pad to max_seq_len (fixed shape) — XLA recompiles on shape changes.
    Every batch has identical shape → compiled once → all subsequent steps fast.
    """
    import torch_xla.distributed.parallel_loader as pl
    from transformers import DataCollatorForSeq2Seq

    collator = DataCollatorForSeq2Seq(
        dataset.tokenizer,
        padding="max_length",    # FIXED — every sequence padded to max_seq_len
        max_length=max_seq_len,
        return_tensors="pt",
        label_pad_token_id=-100,
    )

    base_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        num_workers=4,
        drop_last=True,    # essential — last batch must match fixed shape
    )

    # Wraps PyTorch DataLoader to pre-load batches to TPU device
    return pl.MpDeviceLoader(base_loader, device)