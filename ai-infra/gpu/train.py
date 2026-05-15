"""
train_gpu.py — Full fine-tuning script for GPU.

Supports:
  - Single GPU:       python train_gpu.py
  - Multi-GPU (1 machine, N GPUs):
                      torchrun --nproc_per_node=2 train_gpu.py
  - Multi-node (M machines × N GPUs):
                      torchrun --nnodes=2 --nproc_per_node=8 \
                               --node_rank=0 \
                               --master_addr=192.168.1.1 \
                               --master_port=29500 \
                               train_gpu.py

Hardware used in this example:
  Single machine, 2× NVIDIA A100 80GB (NVLink)
  Total VRAM: 160GB
  Interconnect: NVLink (within machine) — no InfiniBand needed for 1 node

Install dependencies:
  pip install torch transformers peft bitsandbytes accelerate datasets wandb
  pip install flash-attn --no-build-isolation
"""

import os
import math
import time
import logging
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import bitsandbytes as bnb
import wandb
import functools

from config import ModelConfig, TrainingConfig
from dataset import SupportDataset, make_tokenizer, make_dataloader_gpu

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ── SINGLE vs MULTI-MACHINE EXPLANATION ──────────────────────────────────────
#
# Single machine, 1 GPU:
#   - Run: python train_gpu.py
#   - LOCAL_RANK=0, WORLD_SIZE=1
#   - No distributed setup needed
#   - Batch size = per_device_batch_size × 1
#
# Single machine, 2+ GPUs (most common setup):
#   - Run: torchrun --nproc_per_node=2 train_gpu.py
#   - torchrun launches 2 Python processes, one per GPU
#   - Each process gets LOCAL_RANK=0 or 1
#   - Processes communicate via NVLink (within machine) — very fast
#   - FSDP shards model parameters across both GPUs
#   - Effective batch size = per_device_batch_size × 2
#
# Multiple machines, each with multiple GPUs:
#   - Run on each machine with different --node_rank
#   - Processes communicate via InfiniBand (cross-machine) — slower
#   - Master node coordinates all processes
#   - Effective batch = per_device_batch_size × machines × GPUs_per_machine
#
# ─────────────────────────────────────────────────────────────────────────────


def setup_distributed():
    """
    Initialise process group for multi-GPU training.
    Called automatically by torchrun — env vars (LOCAL_RANK, RANK, WORLD_SIZE)
    are injected by torchrun. For single-GPU, these default to 0/0/1.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        # NCCL backend: uses GPU-direct RDMA for inter-GPU communication.
        # Within a single machine: uses NVLink (fast, ~900 GB/s).
        # Across machines: uses InfiniBand or Ethernet RDMA.
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        log.info(f"Distributed: rank={rank}/{world_size}, local_rank={local_rank}")
    else:
        log.info("Single GPU training (no distributed setup)")

    return local_rank, rank, world_size


def is_main_process(rank: int) -> bool:
    """Only rank 0 should log, save checkpoints, and report to W&B."""
    return rank == 0


def load_model_gpu(cfg: ModelConfig, local_rank: int):
    """
    Load Llama 3 with 4-bit quantisation (QLoRA) for GPU.

    Memory layout for QLoRA on 2× A100 80GB:
      - Base model (4-bit quantised): ~4 GB per GPU (total 4GB — shared via FSDP)
      - LoRA adapters (BF16):         ~0.5 GB per GPU
      - Activations (during training): ~8–15 GB per GPU (depends on seq len)
      - Gradient + optimizer states:  ~4 GB per GPU (only LoRA params)
      - Total per GPU:                ~17–24 GB of 80GB available → very comfortable

    Without 4-bit (full BF16):
      - Base model: 14 GB per GPU → 28 GB total across 2 GPUs via FSDP
      - Still fits in 80GB per GPU but much less headroom for large batches
    """
    log.info(f"Loading {cfg.model_name} on GPU...")

    # 4-bit quantisation configuration (bitsandbytes)
    bnb_config = bnb.nn.modules.Params4bit if cfg.use_4bit_quantisation else None

    from transformers import BitsAndBytesConfig
    quantisation_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,  # compute in BF16, store in 4-bit
        bnb_4bit_use_double_quant=True,          # nested quantisation for extra savings
        bnb_4bit_quant_type="nf4",              # NormalFloat4 — best quality 4-bit type
    ) if cfg.use_4bit_quantisation else None

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        quantization_config=quantisation_config,
        torch_dtype=torch.bfloat16,
        # device_map: "auto" places layers across available GPUs automatically.
        # For FSDP (multi-GPU), we use "cpu" and let FSDP handle placement.
        device_map="cpu" if int(os.environ.get("WORLD_SIZE", 1)) > 1 else "auto",
        attn_implementation="flash_attention_2",   # 3x faster, 10x less memory
        use_cache=False,   # must disable for gradient checkpointing
    )

    # prepare_model_for_kbit_training: enables gradient checkpointing on
    # quantised model and casts layer norms to BF16 for stability
    if cfg.use_4bit_quantisation:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
        )

    # Add LoRA adapters on top of the quantised base model
    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    trainable, total = model.get_nb_trainable_parameters()
    log.info(f"Trainable parameters: {trainable:,} / {total:,} "
             f"({100*trainable/total:.2f}%) — LoRA efficiency")

    return model


def wrap_model_fsdp(model, local_rank: int, world_size: int):
    """
    Wrap model in FSDP for multi-GPU training on a single machine or cluster.

    FSDP (Fully Sharded Data Parallel) shards model parameters across all GPUs:
    - Each GPU holds 1/N of the parameters
    - Before a forward pass, FSDP all-gathers the full parameters temporarily
    - After the forward pass, it shards them again
    - Gradients are also sharded and reduced across GPUs

    For 2× A100 with FSDP:
    - Each GPU holds ~7GB of model parameters (14GB total / 2)
    - All-gather temporarily brings full layer to each GPU during compute
    - Much more memory efficient than DDP (which replicates full model)

    We only need FSDP for multi-GPU. Single GPU uses the model directly.
    """
    if world_size <= 1:
        return model.to(f"cuda:{local_rank}")

    # Auto-wrap policy: wrap each LlamaDecoderLayer as its own FSDP unit.
    # Each layer is gathered, computed, then sharded independently.
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={LlamaDecoderLayer},
    )

    # Mixed precision: parameters and grads in BF16, but keep a master FP32
    # copy of parameters for the optimizer. Standard "mixed precision training".
    mixed_precision_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,   # gradient AllReduce in BF16
        buffer_dtype=torch.bfloat16,
    )

    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,  # shard params + grads + optimizer
        device_id=torch.device(f"cuda:{local_rank}"),
        use_orig_params=True,   # required for LoRA compatibility
    )

    log.info(f"Model wrapped in FSDP across {world_size} GPUs")
    return model


def train(model_cfg: ModelConfig, train_cfg: TrainingConfig):
    # ── 1. Distributed setup ──────────────────────────────────────────────────
    local_rank, rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    torch.manual_seed(train_cfg.seed + rank)  # different seed per process

    if is_main_process(rank):
        wandb.init(project="llama-support-finetune-gpu", config={
            **vars(model_cfg), **vars(train_cfg),
            "num_gpus": world_size,
            "hardware": "GPU",
        })

    # ── 2. Tokeniser ──────────────────────────────────────────────────────────
    tokenizer = make_tokenizer(model_cfg.model_name)

    # ── 3. Dataset & DataLoader ───────────────────────────────────────────────
    dataset = SupportDataset(train_cfg.dataset_name, tokenizer, model_cfg.max_seq_len)

    # For multi-GPU: each GPU sees a different subset of the data each epoch.
    # DistributedSampler handles the splitting automatically.
    if world_size > 1:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(dataset, num_replicas=world_size,
                                     rank=rank, shuffle=True)
        loader = make_dataloader_gpu(dataset, train_cfg.per_device_batch_size,
                                     shuffle=False)  # sampler handles shuffling
        # Override the sampler in the loader
        loader.sampler = sampler
    else:
        loader = make_dataloader_gpu(dataset, train_cfg.per_device_batch_size)

    # ── 4. Model ──────────────────────────────────────────────────────────────
    model = load_model_gpu(model_cfg, local_rank)
    if world_size > 1:
        model = wrap_model_fsdp(model, local_rank, world_size)

    if train_cfg.gradient_checkpointing and world_size <= 1:
        # For single GPU, enable gradient checkpointing directly.
        # For FSDP, it was handled in prepare_model_for_kbit_training.
        model.gradient_checkpointing_enable()

    # ── 5. Optimiser (only LoRA parameters) ──────────────────────────────────
    # AdamW with 8-bit quantisation from bitsandbytes — saves memory on optimizer states.
    # Standard FP32 Adam would use 8 bytes per param (m + v vectors).
    # 8-bit Adam uses 2 bytes per param — 4x savings on optimizer memory.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    log.info(f"Training {sum(p.numel() for p in trainable_params):,} parameters")

    optimizer = bnb.optim.AdamW8bit(
        trainable_params,
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
        betas=(0.9, 0.999),
    )

    # ── 6. Learning rate schedule ─────────────────────────────────────────────
    steps_per_epoch = math.ceil(len(dataset) / (train_cfg.per_device_batch_size
                                                 * world_size
                                                 * train_cfg.gradient_accumulation_steps))
    total_steps = steps_per_epoch * train_cfg.num_epochs
    warmup_steps = int(total_steps * train_cfg.warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    log.info(f"Training: {total_steps} steps, {warmup_steps} warmup, "
             f"batch={train_cfg.per_device_batch_size * world_size * train_cfg.gradient_accumulation_steps}")

    # ── 7. Training loop ──────────────────────────────────────────────────────
    global_step = 0
    best_loss = float("inf")

    for epoch in range(train_cfg.num_epochs):
        model.train()
        if world_size > 1:
            loader.sampler.set_epoch(epoch)  # shuffle differently each epoch

        epoch_loss = 0.0
        optimizer.zero_grad()
        t0 = time.time()

        for step, batch in enumerate(loader):
            # Move batch to GPU — each process moves its own batch to its GPU
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass — autocast to BF16 for Tensor Core acceleration
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                # Divide loss by accumulation steps so gradients are averaged
                # correctly across accumulation steps (not summed)
                loss = outputs.loss / train_cfg.gradient_accumulation_steps

            # Backward pass — computes gradients for all LoRA parameters
            loss.backward()

            # Only update weights every gradient_accumulation_steps
            if (step + 1) % train_cfg.gradient_accumulation_steps == 0:
                # Clip gradients to prevent training instability (exploding gradients)
                if world_size > 1:
                    model.clip_grad_norm_(1.0)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                epoch_loss += loss.item() * train_cfg.gradient_accumulation_steps

                # ── Logging ───────────────────────────────────────────────────
                if global_step % train_cfg.logging_steps == 0 and is_main_process(rank):
                    elapsed = time.time() - t0
                    tokens_per_sec = (
                        train_cfg.logging_steps
                        * train_cfg.per_device_batch_size
                        * world_size
                        * train_cfg.gradient_accumulation_steps
                        * model_cfg.max_seq_len
                    ) / elapsed

                    log.info(f"Epoch {epoch+1} | Step {global_step}/{total_steps} | "
                             f"Loss: {epoch_loss/global_step:.4f} | "
                             f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                             f"{tokens_per_sec:,.0f} tok/s")

                    wandb.log({
                        "train/loss": epoch_loss / global_step,
                        "train/learning_rate": scheduler.get_last_lr()[0],
                        "train/tokens_per_sec": tokens_per_sec,
                        "train/global_step": global_step,
                        "train/epoch": epoch + 1,
                    })
                    t0 = time.time()

                # ── Checkpoint ────────────────────────────────────────────────
                if global_step % train_cfg.save_steps == 0 and is_main_process(rank):
                    save_path = Path(train_cfg.checkpoint_dir) / f"step-{global_step}"
                    save_path.mkdir(parents=True, exist_ok=True)

                    # For FSDP, must use FSDP's state_dict method to gather
                    # sharded parameters before saving
                    if world_size > 1:
                        from torch.distributed.fsdp import FullStateDictConfig, StateDictType
                        with FSDP.state_dict_type(
                            model,
                            StateDictType.FULL_STATE_DICT,
                            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
                        ):
                            state_dict = model.state_dict()
                            torch.save(state_dict, save_path / "model.pt")
                    else:
                        model.save_pretrained(str(save_path))

                    tokenizer.save_pretrained(str(save_path))
                    log.info(f"Checkpoint saved to {save_path}")

        log.info(f"Epoch {epoch+1} complete. Avg loss: {epoch_loss/global_step:.4f}")

    # ── 8. Save final model ───────────────────────────────────────────────────
    if is_main_process(rank):
        output_path = Path(train_cfg.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(output_path))
        tokenizer.save_pretrained(str(output_path))
        log.info(f"Final model saved to {output_path}")
        wandb.finish()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    model_cfg = ModelConfig()
    train_cfg = TrainingConfig()
    train(model_cfg, train_cfg)