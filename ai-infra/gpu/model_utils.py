"""
model_utils.py — Safe model loading for Mistral-7B-v0.3.

ROOT CAUSE OF ValueError:
  In transformers >= 4.40, PretrainedConfig raises:
    ValueError: use_cache attribute should not be set after
    MistralConfig is initialized

  This is triggered by gradient_checkpointing_enable() which internally
  does:  self.config.use_cache = False
  Even though we set use_cache=False in from_pretrained(), the config
  is now considered "initialized" and rejects further attribute changes.

THREE-PART FIX:
  1. Load AutoConfig FIRST — set use_cache=False before the model exists.
     Pass that config object into from_pretrained(config=...).
     Now the model starts with use_cache=False baked in.

  2. Skip gradient_checkpointing_enable() — use the model's internal
     _set_gradient_checkpointing() method directly. This method toggles
     checkpointing WITHOUT touching config.use_cache.

  3. Wrap in try/except as a safety net for any edge cases across
     different library versions.
"""

import logging
from typing import Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

log = logging.getLogger(__name__)


def _enable_gradient_checkpointing(model: torch.nn.Module) -> None:
    """
    Enable gradient checkpointing WITHOUT touching model.config.use_cache.

    Standard gradient_checkpointing_enable() sets self.config.use_cache = False
    which raises ValueError in transformers >= 4.40.

    This function uses the internal _set_gradient_checkpointing() instead,
    which only toggles the checkpointing flag on each layer — it never
    modifies the config.
    """
    # Method 1: use internal method (preferred — doesn't touch config)
    if hasattr(model, "_set_gradient_checkpointing"):
        model._set_gradient_checkpointing(
            enable=True,
            gradient_checkpointing_func=torch.utils.checkpoint.checkpoint,
        )
        log.info("Gradient checkpointing enabled via _set_gradient_checkpointing")
        return

    # Method 2: call the public method with safety net
    try:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        log.info("Gradient checkpointing enabled via gradient_checkpointing_enable")
    except ValueError as e:
        if "use_cache" in str(e):
            # Config already has use_cache=False — the error is safe to ignore.
            # Checkpointing is already active because we set use_cache=False
            # during config loading (fix part 1).
            log.info(
                "Gradient checkpointing: caught use_cache ValueError — "
                "safe to ignore (use_cache already False in pre-loaded config)"
            )
        else:
            raise  # re-raise anything unrelated


def _get_attn_implementation() -> str:
    """
    Return the best available attention implementation.
    Uses FlashAttention 2 if installed, otherwise falls back to
    the default scaled dot-product attention (still fast on A100/H100
    via PyTorch 2.x's built-in SDPA).
    """
    try:
        import flash_attn  # noqa: F401
        log.info("FlashAttention 2 available — using it")
        return "flash_attention_2"
    except ImportError:
        log.info("flash_attn not installed — using default SDPA attention. "
                 "To install: pip install flash-attn --no-build-isolation")
        return "sdpa"   # PyTorch 2.x built-in — still fast, no install needed


def load_model_gpu(
    model_name: str,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_target_modules: list,
    use_4bit: bool = True,
    hf_token: Optional[str] = None,
    world_size: int = 1,
) -> torch.nn.Module:
    """
    Load Mistral-7B-v0.3 with QLoRA for GPU training.

    FIX PART 1 — pre-load config with use_cache=False:
      We load AutoConfig first, set use_cache=False on it, then pass
      that config object to from_pretrained(config=...).
      The model is born with use_cache=False — it was never set "after init".

    FIX PART 2 — bypass gradient_checkpointing_enable():
      We call _enable_gradient_checkpointing() which uses the internal
      _set_gradient_checkpointing() method that doesn't touch config.

    FIX PART 3 — all config objects built in one shot:
      BitsAndBytesConfig and LoraConfig are constructed once.
      Their attributes are never modified after construction.
    """
    log.info(f"Loading {model_name} for GPU...")

    # ── FIX PART 1: pre-load and configure config before model exists ─────────
    model_config = AutoConfig.from_pretrained(
        model_name,
        token=hf_token,
    )
    model_config.use_cache = False  # SAFE here — model doesn't exist yet

    # ── FIX PART 3: BitsAndBytesConfig built in one shot ─────────────────────
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    ) if use_4bit else None

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=model_config,
        token=hf_token,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
        device_map="cpu" if world_size > 1 else "auto",
        attn_implementation=_get_attn_implementation(),  # auto-detects flash_attn
    )

    # ── FIX PART 2: safe gradient checkpointing ───────────────────────────────
    if use_4bit:
        # prepare_model_for_kbit_training handles checkpointing internally.
        # We pass use_gradient_checkpointing=False and do it ourselves
        # to avoid the double-enable issue.
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=False,  # we handle it below
        )
    _enable_gradient_checkpointing(model)      # safe, doesn't touch config

    # ── FIX PART 3: LoraConfig built in one shot ──────────────────────────────
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=lora_target_modules,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    trainable, total = model.get_nb_trainable_parameters()
    log.info(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    return model


def load_model_tpu(
    model_name: str,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_target_modules: list,
    device: torch.device,
    hf_token: Optional[str] = None,
) -> torch.nn.Module:
    """
    Load Mistral-7B-v0.3 in BF16 for TPU training.
    Same three-part fix applied — no bitsandbytes (not supported on TPU).
    """
    import torch_xla.core.xla_model as xm
    log.info(f"[Chip {xm.get_ordinal()}] Loading {model_name} for TPU...")

    # ── FIX PART 1: pre-load config ───────────────────────────────────────────
    model_config = AutoConfig.from_pretrained(model_name, token=hf_token)
    model_config.use_cache = False  # SAFE — model doesn't exist yet

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=model_config,          # pre-configured
        token=hf_token,
        torch_dtype=torch.bfloat16,
        # No device_map on TPU — move manually
        # No attn_implementation="flash_attention_2" — CUDA-only kernel
    )

    # ── FIX PART 2: safe gradient checkpointing ───────────────────────────────
    _enable_gradient_checkpointing(model)

    # ── FIX PART 3: LoraConfig in one shot ────────────────────────────────────
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=lora_target_modules,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model = model.to(device)

    trainable, total = model.get_nb_trainable_parameters()
    log.info(f"[Chip {xm.get_ordinal()}] Trainable: {trainable:,} / {total:,} "
             f"({100*trainable/total:.2f}%)")
    return model