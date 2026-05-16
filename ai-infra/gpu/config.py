"""
config.py — Mistral-7B-v0.3 + ultrachat_200k configuration.
No HuggingFace token required — Mistral is fully open.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    model_name: str         = "mistralai/Mistral-7B-v0.3"
    hf_token: Optional[str] = None      # not needed for Mistral
    max_seq_len: int        = 2048      # fixed on TPU, flexible on GPU
    use_4bit_quantisation: bool = True  # GPU only — ignored on TPU

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
    dataset_name: str  = "HuggingFaceH4/ultrachat_200k"
    train_split: str   = "train_sft"
    eval_split: str    = "test_sft"
    max_train_samples: int = 5000   # -1 = full 200K
    max_eval_samples: int  = 500

    output_dir: str     = "./output"
    checkpoint_dir: str = "./checkpoints"

    per_device_batch_size: int       = 2
    gradient_accumulation_steps: int = 8
    num_epochs: int                  = 1
    max_steps: int                   = -1

    learning_rate: float = 2e-4
    weight_decay: float  = 0.01
    warmup_ratio: float  = 0.03
    lr_scheduler: str    = "cosine"

    bf16: bool = True   # A100/H100/TPU
    fp16: bool = False  # V100 only

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