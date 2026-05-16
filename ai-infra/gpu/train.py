"""
train_gpu.py — Fine-tune Mistral-7B-v0.3 on ultrachat_200k (GPU).

All ValueError fixes are in model_utils.py:load_model_gpu().
See model_utils.py for full explanation of the three-part fix.

Run:
  Single GPU:         python train_gpu.py
  2 GPUs, 1 machine:  torchrun --nproc_per_node=2 train_gpu.py
  8 GPUs, 1 machine:  torchrun --nproc_per_node=8 train_gpu.py
  Multi-node:
    Machine 1: torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0
                        --master_addr=<IP> --master_port=29500 train_gpu.py
    Machine 2: torchrun --nnodes=2 --nproc_per_node=8 --node_rank=1
                        --master_addr=<IP> --master_port=29500 train_gpu.py
"""

import os
import math
import time
import logging
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import get_cosine_schedule_with_warmup
import bitsandbytes as bnb
import wandb

from config import ModelConfig, TrainingConfig
from dataset import UltraChatDataset, make_tokenizer, make_dataloader_gpu
from model_utils import load_model_gpu

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def setup_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank       = int(os.environ.get("RANK",       0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        log.info(f"Distributed: rank={rank}/{world_size}, local={local_rank}")
    else:
        log.info("Single GPU")
    return local_rank, rank, world_size


def is_main(rank): return rank == 0


def wrap_model(model, local_rank: int, world_size: int, use_4bit: bool):
    """
    Choose the correct multi-GPU strategy based on quantisation:

    QLoRA (use_4bit=True)  → DDP
    ─────────────────────────────────────────────────────────────────
    FSDP cannot shard 4-bit integer tensors → ValueError: Cannot
    flatten integer dtype tensors.

    With QLoRA, the base model weights are FROZEN 4-bit integers.
    Only LoRA adapters (~1% of params, BF16) are trainable.
    DDP AllReduces only the tiny LoRA gradients → very cheap.
    Each GPU holds a full copy of the 4-bit model (~4 GB) — fine.

    Full BF16 (use_4bit=False) → FSDP
    ─────────────────────────────────────────────────────────────────
    Full BF16 model = 14 GB. FSDP shards this across GPUs so each
    GPU holds only 14/N GB. Necessary for large models or small GPUs.
    """
    if world_size <= 1:
        return model  # already on correct device via device_map in load_model_gpu

    if use_4bit:
        # DDP: sync only LoRA gradients — base model stays frozen on each GPU
        # find_unused_parameters=False: all LoRA params are used, no overhead
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )
        log.info(f"QLoRA + DDP across {world_size} GPUs "
                 f"(FSDP skipped — incompatible with 4-bit tensors)")
    else:
        # FSDP: shard full BF16 model across GPUs
        import functools
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            MixedPrecision, ShardingStrategy,
        )
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
        from transformers.models.mistral.modeling_mistral import MistralDecoderLayer

        auto_wrap = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={MistralDecoderLayer},
        )
        mp = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
        model = FSDP(
            model,
            auto_wrap_policy=auto_wrap,
            mixed_precision=mp,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            device_id=torch.device(f"cuda:{local_rank}"),
            use_orig_params=True,
        )
        log.info(f"Full BF16 + FSDP across {world_size} GPUs")

    return model


def evaluate(model, loader, device, world_size):
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                total += model(**batch).loss.item()
            n += 1
    avg = total / max(n, 1)
    if world_size > 1:
        t = torch.tensor(avg, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.AVG)
        avg = t.item()
    model.train()
    return avg


def train(model_cfg: ModelConfig, train_cfg: TrainingConfig):

    # 1. Distributed setup
    local_rank, rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    torch.manual_seed(train_cfg.seed + rank)

    if is_main(rank):
        wandb.init(project="mistral-ultrachat-gpu",
                   config={**vars(model_cfg), **vars(train_cfg),
                           "num_gpus": world_size})

    # 2. Tokeniser
    tokenizer = make_tokenizer(model_cfg.model_name, model_cfg.hf_token)

    # 3. Datasets
    train_ds = UltraChatDataset(
        train_cfg.dataset_name, train_cfg.train_split,
        tokenizer, model_cfg.max_seq_len, train_cfg.max_train_samples)
    eval_ds = UltraChatDataset(
        train_cfg.dataset_name, train_cfg.eval_split,
        tokenizer, model_cfg.max_seq_len, train_cfg.max_eval_samples)

    if world_size > 1:
        from torch.utils.data.distributed import DistributedSampler
        tr_sampler = DistributedSampler(train_ds, world_size, rank, shuffle=True)
        ev_sampler = DistributedSampler(eval_ds,  world_size, rank, shuffle=False)
        # Pass sampler directly into constructor — never set loader.sampler after init
        # (raises ValueError: sampler attribute should not be set after
        #  DataLoader is initialized — PyTorch 2.x breaking change)
        train_loader = make_dataloader_gpu(train_ds, train_cfg.per_device_batch_size,
                                           sampler=tr_sampler)
        eval_loader  = make_dataloader_gpu(eval_ds,  train_cfg.per_device_batch_size,
                                           sampler=ev_sampler)
    else:
        tr_sampler   = None
        train_loader = make_dataloader_gpu(train_ds, train_cfg.per_device_batch_size)
        eval_loader  = make_dataloader_gpu(eval_ds,  train_cfg.per_device_batch_size,
                                           shuffle=False)

    # 4. Model
    model = load_model_gpu(
        model_name=model_cfg.model_name,
        lora_r=model_cfg.lora_r,
        lora_alpha=model_cfg.lora_alpha,
        lora_dropout=model_cfg.lora_dropout,
        lora_target_modules=model_cfg.lora_target_modules,
        use_4bit=model_cfg.use_4bit_quantisation,
        hf_token=model_cfg.hf_token,
        world_size=world_size,
        local_rank=local_rank,       # needed for device_map={"": local_rank}
    )
    model = wrap_model(model, local_rank, world_size,
                       use_4bit=model_cfg.use_4bit_quantisation)

    # 5. Optimiser
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = bnb.optim.AdamW8bit(trainable,
                                     lr=train_cfg.learning_rate,
                                     weight_decay=train_cfg.weight_decay)

    # 6. Schedule
    steps_per_epoch = math.ceil(
        len(train_ds) /
        (train_cfg.per_device_batch_size * world_size
         * train_cfg.gradient_accumulation_steps))
    total_steps  = steps_per_epoch * train_cfg.num_epochs
    warmup_steps = int(total_steps * train_cfg.warmup_ratio)
    scheduler    = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    log.info(f"Steps={total_steps}  warmup={warmup_steps}  "
             f"eff_batch={train_cfg.per_device_batch_size * world_size * train_cfg.gradient_accumulation_steps}")

    # 7. Training loop
    global_step, running_loss = 0, 0.0
    optimizer.zero_grad()

    for epoch in range(train_cfg.num_epochs):
        model.train()
        if world_size > 1 and tr_sampler is not None:
            tr_sampler.set_epoch(epoch)   # reshuffle differently each epoch
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(**batch).loss / train_cfg.gradient_accumulation_steps
            loss.backward()
            running_loss += loss.item() * train_cfg.gradient_accumulation_steps

            if (step + 1) % train_cfg.gradient_accumulation_steps == 0:
                # Gradient clipping — same API for DDP and single GPU
                # (FSDP has its own clip_grad_norm_ but DDP/single use torch's)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % train_cfg.logging_steps == 0 and is_main(rank):
                    elapsed  = time.time() - t0
                    avg_loss = running_loss / train_cfg.logging_steps
                    tok_s    = (train_cfg.logging_steps
                                * train_cfg.per_device_batch_size
                                * world_size
                                * train_cfg.gradient_accumulation_steps
                                * model_cfg.max_seq_len) / elapsed
                    log.info(f"Epoch {epoch+1} | Step {global_step}/{total_steps} | "
                             f"Loss {avg_loss:.4f} | LR {scheduler.get_last_lr()[0]:.2e} | "
                             f"{tok_s:,.0f} tok/s")
                    wandb.log({"train/loss": avg_loss,
                               "train/lr": scheduler.get_last_lr()[0],
                               "train/tok_s": tok_s, "step": global_step})
                    running_loss, t0 = 0.0, time.time()

                if global_step % train_cfg.eval_steps == 0:
                    ev = evaluate(model, eval_loader, device, world_size)
                    if is_main(rank):
                        log.info(f"Eval loss {ev:.4f}")
                        wandb.log({"eval/loss": ev, "step": global_step})

                if global_step % train_cfg.save_steps == 0 and is_main(rank):
                    p = Path(train_cfg.checkpoint_dir) / f"step-{global_step}"
                    p.mkdir(parents=True, exist_ok=True)
                    # DDP wraps model in .module — unwrap for saving
                    # FSDP has its own state_dict mechanism (handled in else)
                    save_model = model.module if hasattr(model, "module") else model
                    save_model.save_pretrained(str(p))
                    tokenizer.save_pretrained(str(p))
                    log.info(f"Checkpoint → {p}")

        log.info(f"Epoch {epoch+1} complete")

    # 8. Final save
    if is_main(rank):
        out = Path(train_cfg.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        # Unwrap DDP .module wrapper before saving
        save_model = model.module if hasattr(model, "module") else model
        save_model.save_pretrained(str(out))
        tokenizer.save_pretrained(str(out))
        log.info(f"Model saved → {out}")
        wandb.finish()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    train(ModelConfig(), TrainingConfig())