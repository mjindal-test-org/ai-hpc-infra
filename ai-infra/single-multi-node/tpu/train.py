"""
train_tpu.py — Full fine-tuning script for TPU.

Supports:
  - Single TPU chip:      python train_tpu.py  (rare — dev only)
  - TPU v4-8 (1 host):   python train_tpu.py  (8 chips on 1 machine)
  - TPU v4-32 (4 hosts):  Launch via GCP AI Platform or xm.spawn across hosts
  - TPU v4-pod (512+):    Managed via Vertex AI Training or TPU Pod orchestration

IMPORTANT: A TPU "pod slice" like v4-8 means:
  - 8 TPU chips
  - All on a single host machine (1 physical server)
  - Connected via ICI (Inter-Chip Interconnect) — not InfiniBand
  - Run ONE Python process, which xmp.spawn fans out to 8 chip processes

A TPU v4-32 means:
  - 32 chips across 4 host machines
  - Each machine has 8 chips
  - Run the script on all 4 hosts simultaneously (GCP handles this)
  - Cross-host communication via ICI fabric — still much faster than InfiniBand

Install dependencies:
  pip install torch~=2.3.0 torch_xla[tpu]~=2.3.0 -f https://storage.googleapis.com/libtpu-releases/index.html
  pip install transformers peft datasets wandb accelerate
  (Note: bitsandbytes NOT supported on TPU — no INT4 quantisation)
"""

import os
import math
import time
import logging
from pathlib import Path

import torch
import torch_xla
import torch_xla.core.q as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.xla_backend   # registers 'xla' as a dist backend

import torch.distributed as dist
from transformers import (
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType
import wandb

from config import ModelConfig, TrainingConfig
from dataset import SupportDataset, make_tokenizer, make_dataloader_tpu

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ── SINGLE vs MULTI-CHIP EXPLANATION FOR TPU ─────────────────────────────────
#
# TPU v4-8 (the most common starting point):
#   - 8 TPU chips on a SINGLE host machine
#   - Run: python train_tpu.py
#   - xmp.spawn() launches 8 processes, one per chip, all on the same machine
#   - Chips communicate via ICI (hardware interconnect) — transparent to your code
#   - This is "single machine, multi-chip" — analogous to single-machine multi-GPU
#
# TPU v4-32 (4 host machines, 8 chips each = 32 chips total):
#   - Run the SAME script on all 4 host machines simultaneously
#   - GCP TPU VM handles cross-host ICI communication automatically
#   - Each host runs xmp.spawn(8 chips) → 32 processes total
#   - Gradient AllReduce happens across all 32 chips via ICI
#   - This IS multi-machine — analogous to multi-node GPU with InfiniBand
#
# TPU Pod (512 chips = 64 hosts):
#   - Launch via Vertex AI Training or TPU Queued Resource API
#   - GCP manages all 64 host processes, ICI routing, and fault tolerance
#   - Your code is IDENTICAL — xm handles the scale transparently
#
# KEY DIFFERENCE from GPU multi-machine:
#   GPU multi-machine REQUIRES you to manage:
#     - torchrun coordinator process
#     - InfiniBand network configuration
#     - NCCL environment variables
#   TPU multi-machine: GCP + ICI handles everything. Your code doesn't change.
#
# ─────────────────────────────────────────────────────────────────────────────


def load_model_tpu(cfg: ModelConfig, device):
    """
    Load Llama 3 for TPU training in BF16.

    KEY DIFFERENCE from GPU:
    - No 4-bit quantisation (bitsandbytes not supported on TPU)
    - Load in BF16 directly (TPU's native high-performance format)
    - Must move to TPU device explicitly — no device_map="auto"
    - XLA will compile the model graph on first forward pass

    Memory on TPU v4-8 (32GB HBM per chip, 256GB total across 8 chips):
      - 7B model in BF16: 14GB total → ~1.75GB per chip (sharded via FSDP equivalent)
      - LoRA adapters (BF16): ~0.5GB per chip
      - Activations + KV cache: ~8GB per chip
      - Optimizer states (AdamW, FP32): ~14GB total → ~1.75GB per chip
      - Total per chip: ~12GB of 32GB available → comfortable headroom
    """
    log.info(f"Loading {cfg.model_name} for TPU (BF16)...")

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.bfloat16,   # TPU's native precision — no INT4 support
        # Do NOT use device_map="auto" on TPU — it tries to use CUDA
        # Do NOT use attn_implementation="flash_attention_2" — CUDA-only kernel
        use_cache=False,              # Required for gradient checkpointing
    )

    # Add LoRA adapters (same config as GPU — architecture is hardware-agnostic)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=cfg.lora_target_modules,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    # Move entire model to the TPU device
    # On TPU v4-8 with xmp.spawn: each of the 8 processes gets its own chip
    # device = xm.xla_device() returns the chip assigned to THIS process
    model = model.to(device)

    # Enable gradient checkpointing on TPU
    # Works the same as GPU — recomputes activations in backward pass
    model.gradient_checkpointing_enable()

    trainable, total = model.get_nb_trainable_parameters()
    log.info(f"[Chip {xm.get_ordinal()}] Trainable params: {trainable:,} / {total:,} "
             f"({100*trainable/total:.2f}%)")

    return model


def train_one_chip(index: int, model_cfg: ModelConfig, train_cfg: TrainingConfig):
    """
    Training function that runs on ONE TPU chip.

    xmp.spawn() calls this function for each chip simultaneously.
    index = chip index (0–7 for v4-8).

    Each chip:
    1. Loads its own copy of the model
    2. Processes a different subset of the data (different batch each step)
    3. Computes gradients locally
    4. AllReduces gradients across all chips via ICI
    5. Updates its local copy of the model weights (now identical on all chips)

    This is Data Parallelism — same model, different data per chip.
    """

    # ── 1. Device setup ───────────────────────────────────────────────────────
    # xm.xla_device() returns the TPU chip assigned to this process.
    # Each of the 8 processes in a v4-8 gets a different chip.
    device = xm.xla_device()

    # Initialise distributed process group using the 'xla' backend.
    # This registers XLA's AllReduce for gradient synchronisation across chips.
    # Replaces NCCL on GPU — uses ICI instead of NVLink/InfiniBand.
    dist.init_process_group(
        backend="xla",
        init_method="xla://",       # XLA discovers all chips automatically
    )

    is_main = xm.is_master_ordinal()   # True only for chip 0

    if is_main:
        wandb.init(project="llama-support-finetune-tpu", config={
            **vars(model_cfg), **vars(train_cfg),
            "num_chips": xm.xrt_world_size(),
            "hardware": "TPU",
            "tpu_type": os.environ.get("TPU_ACCELERATOR_TYPE", "unknown"),
        })

    # ── 2. Tokeniser ──────────────────────────────────────────────────────────
    tokenizer = make_tokenizer(model_cfg.model_name)

    # ── 3. Dataset ────────────────────────────────────────────────────────────
    dataset = SupportDataset(train_cfg.dataset_name, tokenizer, model_cfg.max_seq_len)

    # Each chip sees a different partition of the dataset.
    # DistributedSampler splits the dataset across all chips by index.
    from torch.utils.data.distributed import DistributedSampler
    sampler = DistributedSampler(
        dataset,
        num_replicas=xm.xrt_world_size(),   # total chips (e.g. 8)
        rank=xm.get_ordinal(),              # this chip's index (0–7)
        shuffle=True,
    )

    # CRITICAL: must use max_length padding for TPU (fixed shapes for XLA)
    loader = make_dataloader_tpu(
        dataset,
        batch_size=train_cfg.per_device_batch_size,
        max_seq_len=model_cfg.max_seq_len,
        device=device,
        shuffle=False,   # sampler handles shuffling
    )

    # ── 4. Model ──────────────────────────────────────────────────────────────
    model = load_model_tpu(model_cfg, device)

    # ── 5. Optimiser ──────────────────────────────────────────────────────────
    # Standard AdamW — no 8-bit version needed on TPU (memory is less of a concern)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    # ── 6. LR schedule ────────────────────────────────────────────────────────
    num_chips = xm.xrt_world_size()
    steps_per_epoch = math.ceil(
        len(dataset) / (train_cfg.per_device_batch_size
                        * num_chips
                        * train_cfg.gradient_accumulation_steps)
    )
    total_steps = steps_per_epoch * train_cfg.num_epochs
    warmup_steps = int(total_steps * train_cfg.warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    if is_main:
        log.info(f"TPU training: {num_chips} chips, {total_steps} steps, "
                 f"effective batch={train_cfg.per_device_batch_size * num_chips}")

    # ── 7. Training loop ──────────────────────────────────────────────────────
    global_step = 0
    epoch_loss = 0.0
    optimizer.zero_grad()

    for epoch in range(train_cfg.num_epochs):
        model.train()
        sampler.set_epoch(epoch)
        t0 = time.time()

        for step, batch in enumerate(loader):
            # batch is already on TPU device (MpDeviceLoader handled it)
            # No .to(device) needed here

            # Forward pass — all computation is lazy on TPU.
            # Nothing actually executes until xm.mark_step() is called.
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = outputs.loss / train_cfg.gradient_accumulation_steps

            # Backward pass — adds gradient computation to the XLA graph.
            # Still lazy — no actual computation yet.
            loss.backward()

            if (step + 1) % train_cfg.gradient_accumulation_steps == 0:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # ── MARK STEP: the most important TPU-specific call ───────────
                # This triggers XLA to:
                #   1. Compile the traced computation graph (first call only, ~60s)
                #   2. Execute ALL queued operations on the TPU hardware
                #   3. AllReduce gradients across all chips via ICI
                # Without this, no computation ever happens.
                # After the first call, compilation is cached — subsequent calls
                # execute the compiled graph directly (fast).
                xm.mark_step()

                global_step += 1

                # ── Logging ───────────────────────────────────────────────────
                if global_step % train_cfg.logging_steps == 0 and is_main:
                    # .item() forces synchronisation — reads the loss value from TPU.
                    # This is a sync point: Python waits for TPU to finish.
                    # Use sparingly — too many .item() calls fragment the XLA graph.
                    loss_val = loss.item() * train_cfg.gradient_accumulation_steps
                    epoch_loss += loss_val

                    elapsed = time.time() - t0
                    tokens_per_sec = (
                        train_cfg.logging_steps
                        * train_cfg.per_device_batch_size
                        * num_chips
                        * train_cfg.gradient_accumulation_steps
                        * model_cfg.max_seq_len
                    ) / elapsed

                    log.info(
                        f"[Chip {xm.get_ordinal()}] "
                        f"Epoch {epoch+1} | Step {global_step}/{total_steps} | "
                        f"Loss: {loss_val:.4f} | "
                        f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                        f"{tokens_per_sec:,.0f} tok/s"
                    )

                    if is_main:
                        wandb.log({
                            "train/loss": loss_val,
                            "train/learning_rate": scheduler.get_last_lr()[0],
                            "train/tokens_per_sec": tokens_per_sec,
                            "train/global_step": global_step,
                            "train/epoch": epoch + 1,
                        })
                    t0 = time.time()

                # ── Checkpoint ────────────────────────────────────────────────
                if global_step % train_cfg.save_steps == 0:
                    if is_main:
                        save_path = Path(train_cfg.checkpoint_dir) / f"step-{global_step}"
                        save_path.mkdir(parents=True, exist_ok=True)

                        # xm.save(): saves checkpoint only from chip 0.
                        # torch.save() would save 8 identical copies (one per chip).
                        # Always use xm.save() for checkpointing on TPU.
                        xm.save(model.state_dict(), str(save_path / "model.pt"))
                        tokenizer.save_pretrained(str(save_path))
                        log.info(f"Checkpoint saved to {save_path}")

                    # All chips must reach this barrier before continuing.
                    # Prevents chip 0 from racing ahead while others are still computing.
                    xm.rendezvous("checkpoint")

        log.info(f"[Chip {xm.get_ordinal()}] Epoch {epoch+1} complete")

    # ── 8. Save final model ───────────────────────────────────────────────────
    if is_main:
        output_path = Path(train_cfg.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        xm.save(model.state_dict(), str(output_path / "model.pt"))
        tokenizer.save_pretrained(str(output_path))
        log.info(f"Final model saved to {output_path}")
        wandb.finish()

    dist.destroy_process_group()


def train(model_cfg: ModelConfig, train_cfg: TrainingConfig):
    """
    Entry point. xmp.spawn() fans this out to one process per TPU chip.

    For TPU v4-8: nprocs=8 → 8 processes × 1 chip each
    For single chip testing: nprocs=1

    This is the equivalent of torchrun --nproc_per_node=N for GPU,
    except xmp.spawn() manages the process lifecycle automatically.
    """
    num_chips = 8   # v4-8 has 8 chips; set to 1 for dev/debug

    xmp.spawn(
        train_one_chip,
        args=(model_cfg, train_cfg),
        nprocs=num_chips,
        start_method="fork",   # faster than "spawn" for TPU
    )


if __name__ == "__main__":
    model_cfg = ModelConfig(use_4bit_quantisation=False)  # no INT4 on TPU
    train_cfg = TrainingConfig()
    train(model_cfg, train_cfg)