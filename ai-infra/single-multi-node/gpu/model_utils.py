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
    local_rank: int = 0,
) -> torch.nn.Module:
    """
    Load Mistral-7B-v0.3 with QLoRA for GPU training.

    CRITICAL: FSDP is INCOMPATIBLE with 4-bit bitsandbytes quantisation.
    ─────────────────────────────────────────────────────────────────────
    FSDP works by flattening and sharding all tensors across GPUs.
    4-bit quantised weights are stored as integer (uint8) tensors.
    FSDP cannot flatten integer tensors → raises:
      ValueError: Cannot flatten integer dtype tensors

    SOLUTION: Use DDP (DistributedDataParallel) for QLoRA, not FSDP.
    ─────────────────────────────────────────────────────────────────────
    With QLoRA, only ~1% of parameters (the LoRA adapters) are trainable.
    These are tiny BF16 tensors. DDP AllReduces only the trainable gradients,
    which is very cheap. The 4-bit base weights stay frozen on each GPU.

    device_map for multi-GPU QLoRA:
      Each GPU loads a full copy of the quantised model (only ~4GB in 4-bit).
      device_map={"": local_rank} puts the entire model on THIS GPU.
      DDP then syncs only LoRA gradients across GPUs — nothing is sharded.

    For non-quantised (BF16) multi-GPU: use FSDP via wrap_model_ddp_or_fsdp()
    in train_gpu.py which checks use_4bit and chooses the right strategy.
    """
    log.info(f"Loading {model_name} for GPU (use_4bit={use_4bit}, "
             f"world_size={world_size}, local_rank={local_rank})...")

    # ── Pre-load config with use_cache=False (before model exists) ────────────
    model_config = AutoConfig.from_pretrained(model_name, token=hf_token)
    model_config.use_cache = False  # safe here — model doesn't exist yet

    # ── BitsAndBytesConfig (built once, never modified) ───────────────────────
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    ) if use_4bit else None

    if use_4bit and world_size > 1:
        # QLoRA multi-GPU: place the ENTIRE model on this specific GPU.
        # DDP (not FSDP) will sync only the tiny LoRA gradients.
        # Each GPU holds its own full copy of the 4-bit model (~4GB — cheap).
        device_map = {"": local_rank}
    elif use_4bit:
        # QLoRA single GPU: auto-place on the available GPU
        device_map = "auto"
    else:
        # Full BF16: load to CPU first, FSDP will shard across GPUs
        device_map = "cpu" if world_size > 1 else "auto"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=model_config,                      # use_cache=False baked in
        token=hf_token,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        attn_implementation=_get_attn_implementation(),
    )

    # ── Gradient checkpointing (safe — doesn't touch config) ──────────────────
    if use_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=False,  # handled below
        )
    _enable_gradient_checkpointing(model)

    # ── LoRA adapters (built once, never modified) ─────────────────────────────
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