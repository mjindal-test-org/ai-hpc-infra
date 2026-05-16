"""
config.py — Configuration for Mistral-7B-v0.3 fine-tuning on ultrachat_200k.

Changes from Llama version:
  - model_name       → mistralai/Mistral-7B-v0.3  (no token needed)
  - dataset_name     → HuggingFaceH4/ultrachat_200k
  - max_seq_len      → 2048 (ultrachat has longer multi-turn conversations)
  - lora_target_modules → same (Mistral uses identical layer names to Llama)
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    # ── Model ─────────────────────────────────────────────────────────────────
    # Mistral-7B-v0.3 — fully open, no HuggingFace token required.
    # v0.3 adds vocabulary extension + improved tokeniser over v0.1.
    model_name: str = "mistralai/Mistral-7B-v0.3"

    # No token needed — Mistral is not gated.
    # Only set this if you switch back to a gated model like Llama.
    hf_token: Optional[str] = None

    # ultrachat conversations are multi-turn and longer than simple Q&A.
    # 2048 covers ~95% of ultrachat examples without truncation.
    # On TPU this MUST be fixed — never changes between batches.
    max_seq_len: int = 2048

    # GPU QLoRA: quantise base model to 4-bit — saves ~75% VRAM.
    # TPU: ignored — bitsandbytes not supported on TPU.
    use_4bit_quantisation: bool = True

    # ── LoRA ──────────────────────────────────────────────────────────────────
    # Mistral and Llama use identical projection layer names — no change needed.
    lora_r: int          = 16     # rank — higher = more params, better quality
    lora_alpha: int      = 32     # scaling = 2 × rank is standard
    lora_dropout: float  = 0.05

    lora_target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",   # attention
        "gate_proj", "up_proj", "down_proj",        # feed-forward (SwiGLU)
    ])


@dataclass
class TrainingConfig:
    # ── Dataset ───────────────────────────────────────────────────────────────
    # ultrachat_200k: 200K curated multi-turn conversations, fully public.
    # Downloaded automatically from HuggingFace Hub on first run.
    dataset_name: str = "HuggingFaceH4/ultrachat_200k"
    train_split: str  = "train_sft"   # SFT = Supervised Fine-Tuning split
    eval_split: str   = "test_sft"

    # Limit samples for quick testing. Set to -1 to use everything.
    # Start with 5000 to verify the pipeline works end-to-end (~20 min on 2×A100).
    # Then set to -1 for a full run (several hours).
    max_train_samples: int = 5000
    max_eval_samples: int  = 500

    output_dir: str     = "./output"
    checkpoint_dir: str = "./checkpoints"

    # ── Batch & steps ─────────────────────────────────────────────────────────
    # Smaller batch than support-chat example because ultrachat sequences are longer.
    # Effective batch = per_device × num_GPUs × gradient_accumulation_steps
    # Example: 2 × 2 GPUs × 8 = 32 — good for instruction tuning
    per_device_batch_size: int       = 2
    gradient_accumulation_steps: int = 8
    num_epochs: int                  = 1   # 1 epoch over 200K is plenty
    max_steps: int                   = -1  # -1 = full epoch

    # ── Optimiser ─────────────────────────────────────────────────────────────
    learning_rate: float = 2e-4
    weight_decay: float  = 0.01
    warmup_ratio: float  = 0.03
    lr_scheduler: str    = "cosine"

    # ── Precision ─────────────────────────────────────────────────────────────
    bf16: bool = True    # A100/H100/TPU native — use this
    fp16: bool = False   # only for V100 or older GPUs without BF16

    # ── Memory ────────────────────────────────────────────────────────────────
    gradient_checkpointing: bool = True

    # ── Logging & saving ──────────────────────────────────────────────────────
    logging_steps: int    = 10
    save_steps: int       = 500
    eval_steps: int       = 500
    save_total_limit: int = 3
    seed: int             = 42


@dataclass
class InferenceConfig:
    model_path: str     = "./output"
    max_new_tokens: int = 512
    temperature: float  = 0.7
    top_p: float        = 0.9
    do_sample: bool     = True
    host: str           = "0.0.0.0"
    port: int           = 8000