"""
config.py — Configuration for Mistral-7B-v0.3 + ultrachat_200k.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    model_name: str = "mistralai/Mistral-7B-v0.3"

    # Mistral is not gated — no token needed
    hf_token: Optional[str] = None

    # Fixed sequence length.
    # On TPU: never change this between runs — XLA recompiles on shape changes.
    # On GPU: can be dynamic but fixed is also fine.
    max_seq_len: int = 2048

    # GPU only: quantise base model to 4-bit (QLoRA).
    # TPU: ignored — bitsandbytes not supported on TPU.
    use_4bit_quantisation: bool = True

    # LoRA — same layer names for Mistral and Llama
    lora_r: int         = 16
    lora_alpha: int     = 32
    lora_dropout: float = 0.05
    lora_target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])


@dataclass
class TrainingConfig:
    dataset_name: str = "HuggingFaceH4/ultrachat_200k"
    train_split: str  = "train_sft"
    eval_split: str   = "test_sft"

    # Set to -1 to use full 200K examples.
    # Use 5000 first to verify the pipeline works end-to-end.
    max_train_samples: int = 5000
    max_eval_samples: int  = 500

    output_dir: str     = "./output"
    checkpoint_dir: str = "./checkpoints"

    per_device_batch_size: int       = 2
    gradient_accumulation_steps: int = 8
    num_epochs: int                  = 1
    max_steps: int                   = -1   # -1 = full epoch

    learning_rate: float = 2e-4
    weight_decay: float  = 0.01
    warmup_ratio: float  = 0.03
    lr_scheduler: str    = "cosine"

    # BF16 for A100/H100/TPU. Set fp16=True only for older GPUs (V100).
    bf16: bool = True
    fp16: bool = False

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