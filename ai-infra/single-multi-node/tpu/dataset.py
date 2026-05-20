"""
dataset.py — Dataset loading and preprocessing.
Shared between GPU and TPU training scripts.

The key difference: on GPU, padding="longest" (dynamic shapes) is fine.
On TPU, we MUST use padding="max_length" (fixed shapes) or XLA will
recompile the entire graph every time the batch shape changes.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
import json
import os


SYSTEM_PROMPT = (
    "You are a helpful customer support agent. "
    "Be concise, empathetic, and resolve the user's issue clearly."
)


def format_conversation(example: dict) -> str:
    """
    Convert a raw support conversation dict into the chat template format
    that Llama 3 expects.

    Input example format:
        {"user": "My order hasn't arrived", "assistant": "I'm sorry to hear..."}

    Output (Llama 3 chat format):
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are a helpful customer support agent...
        <|start_header_id|>user<|end_header_id|>
        My order hasn't arrived
        <|start_header_id|>assistant<|end_header_id|>
        I'm sorry to hear...
    """
    return (
        f"<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n{SYSTEM_PROMPT}\n"
        f"<|start_header_id|>user<|end_header_id|>\n{example['user']}\n"
        f"<|start_header_id|>assistant<|end_header_id|>\n{example['assistant']}"
        f"<|eot_id|>"
    )


class SupportDataset(Dataset):
    """
    Custom PyTorch Dataset for customer support conversations.
    Works identically on GPU and TPU — tokenisation happens here,
    padding strategy is controlled by the DataLoader collator.
    """

    def __init__(self, data_path: str, tokenizer, max_seq_len: int):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        # Load data — supports JSON lines or HuggingFace dataset
        if os.path.exists(data_path):
            with open(data_path) as f:
                self.examples = [json.loads(line) for line in f]
        else:
            # Load from HuggingFace hub
            ds = load_dataset(data_path, split="train")
            self.examples = list(ds)

        print(f"Loaded {len(self.examples)} examples from {data_path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        text = format_conversation(self.examples[idx])

        # Tokenise — no padding here, that happens in the collator
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors=None,     # return lists, not tensors — collator handles that
        )

        # For causal language modelling, labels = input_ids shifted by 1.
        # HuggingFace handles the shift internally when labels == input_ids.
        # We mask the system+user tokens so loss is only computed on assistant output.
        input_ids = encoding["input_ids"]
        labels = self._mask_user_tokens(input_ids, self.tokenizer)

        return {
            "input_ids": input_ids,
            "attention_mask": encoding["attention_mask"],
            "labels": labels,
        }

    def _mask_user_tokens(self, input_ids: list, tokenizer) -> list:
        """
        Set label = -100 for all tokens before the assistant's response.
        This means the model only learns to generate the assistant turn,
        not to regurgitate the system prompt or user message.
        -100 is PyTorch's convention for "ignore this token in loss".
        """
        labels = list(input_ids)  # copy

        # Find the position of the assistant header token sequence
        # "<|start_header_id|>assistant<|end_header_id|>"
        assistant_header = tokenizer.encode(
            "<|start_header_id|>assistant<|end_header_id|>",
            add_special_tokens=False
        )

        # Search for the assistant header in input_ids
        for i in range(len(input_ids) - len(assistant_header)):
            if input_ids[i:i+len(assistant_header)] == assistant_header:
                # Mask everything up to and including the header
                for j in range(i + len(assistant_header)):
                    labels[j] = -100
                break

        return labels


def make_tokenizer(model_name: str):
    """Load tokeniser with correct padding settings for Llama 3."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Llama 3 has no pad token by default — use EOS as pad.
    # This is standard practice for decoder-only models.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Pad on the right for causal LM training
    tokenizer.padding_side = "right"
    return tokenizer


def make_dataloader_gpu(dataset: SupportDataset, batch_size: int,
                        shuffle: bool = True) -> DataLoader:
    """
    GPU DataLoader — uses dynamic padding (pad each batch to its longest
    sequence). This is more memory-efficient than fixed-length padding
    because short sequences don't waste compute on padding tokens.

    GPU handles variable tensor shapes natively — no recompilation occurs.
    """
    from transformers import DataCollatorForSeq2Seq

    collator = DataCollatorForSeq2Seq(
        dataset.tokenizer,
        padding=True,            # "longest" — dynamic, batch-by-batch
        pad_to_multiple_of=8,    # align to 8 for Tensor Core efficiency
        return_tensors="pt",
        label_pad_token_id=-100,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        num_workers=4,           # parallel data loading workers
        pin_memory=True,         # speeds up CPU→GPU transfer
        drop_last=True,          # drop incomplete final batch for consistency
    )


def make_dataloader_tpu(dataset: SupportDataset, batch_size: int,
                        max_seq_len: int, device, shuffle: bool = True) -> DataLoader:
    """
    TPU DataLoader — uses FIXED padding to max_seq_len.

    CRITICAL: XLA (TPU's compiler) traces the computation graph based on
    tensor shapes at the first step. If shapes change in subsequent steps,
    XLA must retrace and recompile — a 30–120 second penalty per recompile.

    By padding everything to the same fixed length, all batches have
    identical shapes → XLA compiles once → all subsequent steps are fast.

    Trade-off: short sequences waste compute on padding tokens.
    This is acceptable because TPU's throughput is so high that the
    wasted computation is less costly than recompilation.
    """
    import torch_xla.distributed.parallel_loader as pl
    from transformers import DataCollatorForSeq2Seq

    collator = DataCollatorForSeq2Seq(
        dataset.tokenizer,
        padding="max_length",    # FIXED — pads every sequence to max_seq_len
        max_length=max_seq_len,  # must match ModelConfig.max_seq_len exactly
        return_tensors="pt",
        label_pad_token_id=-100,
    )

    base_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        num_workers=4,
        drop_last=True,          # essential on TPU — last batch size must match
    )

    # MpDeviceLoader wraps the PyTorch DataLoader to:
    # 1. Pre-load the next batch to TPU memory while current batch is processing
    # 2. Handle the XLA device placement automatically
    # This overlaps data transfer with computation, hiding latency.
    return pl.MpDeviceLoader(base_loader, device)