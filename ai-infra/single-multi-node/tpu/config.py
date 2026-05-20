"""
config.py — Shared configuration for GPU and TPU fine-tuning scripts.
Identical for both; hardware-specific settings are in each training script.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    # Model to fine-tune. Any HuggingFace causal LM works.
    model_name: str = "meta-llama/Llama-3.1-8B"

    # Maximum sequence length. On TPU this MUST be a fixed power-of-2-friendly
    # number because XLA recompiles on shape changes. On GPU this can be dynamic.
    max_seq_len: int = 512

    # For GPU QLoRA: quantise base model to 4-bit before adding LoRA adapters.
    # Saves ~75% VRAM on GPU. NOT supported on TPU — TPU uses BF16 full weights.
    use_4bit_quantisation: bool = True   # GPU only — ignored on TPU

    # LoRA configuration
    lora_r: int = 16          # LoRA rank — higher = more parameters updated
    lora_alpha: int = 32      # LoRA scaling factor (usually 2 × rank)
    lora_dropout: float = 0.05
    # Which weight matrices to add LoRA adapters to
    lora_target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])


@dataclass
class TrainingConfig:
    # --- Data ---
    dataset_name: str = "support_conversations"   # local path or HF dataset id
    output_dir: str = "./output"
    checkpoint_dir: str = "./checkpoints"

    # --- Batch & steps ---
    # per_device_batch_size × num_devices = total batch size per step
    # GPU: num_devices = number of GPUs (e.g. 2 for 2×A100)
    # TPU: num_devices = number of TPU chips (e.g. 8 for v4-8)
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 4   # effective batch = per_device × devices × accum
    num_epochs: int = 3
    max_steps: int = -1                    # -1 = run all epochs; set >0 to override

    # --- Optimiser ---
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03            # 3% of total steps used for LR warmup
    lr_scheduler: str = "cosine"

    # --- Precision ---
    # GPU: use bf16 if A100/H100, fp16 for older GPUs (V100)
    # TPU: always bf16 — it is natively supported and fastest on TPU hardware
    bf16: bool = True
    fp16: bool = False

    # --- Memory optimisation ---
    # Activation checkpointing: recompute activations in backward pass
    # instead of storing them. Saves ~60% memory at ~30% speed cost.
    gradient_checkpointing: bool = True

    # --- Logging & saving ---
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3             # keep only the last N checkpoints

    # --- Reproducibility ---
    seed: int = 42


@dataclass
class InferenceConfig:
    model_path: str = "./output"          # path to fine-tuned model
    max_new_tokens: int = 200
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True

    # For production serving
    host: str = "0.0.0.0"
    port: int = 8000
    max_concurrent_requests: int = 256    # GPU vLLM handles this automatically