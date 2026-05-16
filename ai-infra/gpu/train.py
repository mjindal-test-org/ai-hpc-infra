"""
train_gpu.py — Fine-tune Mistral-7B-v0.3 on ultrachat_200k using GPU.

Changes from Llama version:
  - No HuggingFace token needed (Mistral is not gated)
  - Uses MistralDecoderLayer in FSDP wrap policy (not LlamaDecoderLayer)
  - Dataset is UltraChatDataset (not SupportDataset)
  - Eval loop added (ultrachat has a test_sft split)
  - attn_implementation changed: flash_attention_2 works with Mistral too

Run commands:
  Single GPU:            python train_gpu.py
  2 GPUs, 1 machine:     torchrun --nproc_per_node=2 train_gpu.py
  8 GPUs, 1 machine:     torchrun --nproc_per_node=8 train_gpu.py
  2 machines × 8 GPUs:
    Machine 1: torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 \
                        --master_addr=192.168.1.10 --master_port=29500 train_gpu.py
    Machine 2: torchrun --nnodes=2 --nproc_per_node=8 --node_rank=1 \
                        --master_addr=192.168.1.10 --master_port=29500 train_gpu.py
"""

import os
import math
import time
import logging
import functools
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
    get_cosine_schedule_with_warmup,
)

# Mistral uses MistralDecoderLayer — different from LlamaDecoderLayer
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer

from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import bitsandbytes as bnb
import wandb

from config import ModelConfig, TrainingConfig
from dataset import UltraChatDataset, make_tokenizer, make_dataloader_gpu

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def setup_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank       = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        log.info(f"Distributed: rank={rank}/{world_size}, local_rank={local_rank}")
    else:
        log.info("Single GPU training")

    return local_rank, rank, world_size


def is_main(rank): return rank == 0


def load_model(cfg: ModelConfig):
    """
    Load Mistral-7B-v0.3 with QLoRA (4-bit base + BF16 LoRA adapters).

    No token= needed — Mistral is fully open.

    Memory on 2× A100 80GB with QLoRA:
      Base model (4-bit):    ~4 GB total (sharded across 2 GPUs via FSDP → ~2 GB each)
      LoRA adapters (BF16):  ~0.5 GB each GPU
      Activations:           ~10–15 GB each GPU (2048 seq len is heavier than 512)
      Optimizer (8-bit Adam):~2 GB each GPU
      Total per GPU:         ~15–20 GB of 80 GB → comfortable
    """
    log.info(f"Loading {cfg.model_name}...")

    if cfg.use_4bit_quantisation:
        from transformers import BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        quant_config = None

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        # No token= needed for Mistral
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
        # Use "cpu" for multi-GPU (FSDP places layers); "auto" for single GPU
        device_map="cpu" if int(os.environ.get("WORLD_SIZE", 1)) > 1 else "auto",
        attn_implementation="flash_attention_2",  # 3x faster, 10x less memory
        use_cache=False,   # required for gradient checkpointing
    )

    if cfg.use_4bit_quantisation:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True
        )

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
    log.info(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    return model


def wrap_fsdp(model, local_rank: int, world_size: int):
    """Wrap model in FSDP for multi-GPU sharding."""
    if world_size <= 1:
        return model.to(f"cuda:{local_rank}")

    # Use MistralDecoderLayer — the FSDP wrap unit for Mistral
    auto_wrap = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={MistralDecoderLayer},
    )
    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )
    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap,
        mixed_precision=mp_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=torch.device(f"cuda:{local_rank}"),
        use_orig_params=True,
    )
    log.info(f"FSDP across {world_size} GPUs")
    return model


def evaluate(model, eval_loader, device, world_size: int) -> float:
    """Run one pass over the eval set and return average loss."""
    model.eval()
    total_loss, steps = 0.0, 0
    with torch.no_grad():
        for batch in eval_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(**batch)
            total_loss += outputs.loss.item()
            steps += 1
    avg = total_loss / max(steps, 1)
    # Average across all GPUs
    if world_size > 1:
        t = torch.tensor(avg, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.AVG)
        avg = t.item()
    model.train()
    return avg


def train(model_cfg: ModelConfig, train_cfg: TrainingConfig):

    # ── 1. Distributed setup ──────────────────────────────────────────────────
    local_rank, rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    torch.manual_seed(train_cfg.seed + rank)

    if is_main(rank):
        wandb.init(project="mistral-ultrachat-gpu", config={
            **vars(model_cfg), **vars(train_cfg),
            "num_gpus": world_size,
        })

    # ── 2. Tokeniser ──────────────────────────────────────────────────────────
    # No token needed for Mistral
    tokenizer = make_tokenizer(model_cfg.model_name, hf_token=model_cfg.hf_token)

    # ── 3. Datasets & DataLoaders ─────────────────────────────────────────────
    train_dataset = UltraChatDataset(
        train_cfg.dataset_name,
        split=train_cfg.train_split,
        tokenizer=tokenizer,
        max_seq_len=model_cfg.max_seq_len,
        max_samples=train_cfg.max_train_samples,
    )
    eval_dataset = UltraChatDataset(
        train_cfg.dataset_name,
        split=train_cfg.eval_split,
        tokenizer=tokenizer,
        max_seq_len=model_cfg.max_seq_len,
        max_samples=train_cfg.max_eval_samples,
    )

    if world_size > 1:
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size,
                                           rank=rank, shuffle=True)
        eval_sampler  = DistributedSampler(eval_dataset,  num_replicas=world_size,
                                           rank=rank, shuffle=False)
        train_loader = make_dataloader_gpu(train_dataset, train_cfg.per_device_batch_size,
                                           shuffle=False)
        eval_loader  = make_dataloader_gpu(eval_dataset,  train_cfg.per_device_batch_size,
                                           shuffle=False)
        train_loader.sampler = train_sampler
        eval_loader.sampler  = eval_sampler
    else:
        train_loader = make_dataloader_gpu(train_dataset, train_cfg.per_device_batch_size)
        eval_loader  = make_dataloader_gpu(eval_dataset,  train_cfg.per_device_batch_size,
                                           shuffle=False)

    # ── 4. Model ──────────────────────────────────────────────────────────────
    model = load_model(model_cfg)
    if world_size > 1:
        model = wrap_fsdp(model, local_rank, world_size)
    elif train_cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # ── 5. Optimiser ──────────────────────────────────────────────────────────
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = bnb.optim.AdamW8bit(
        trainable_params,
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
    )

    # ── 6. LR Schedule ────────────────────────────────────────────────────────
    steps_per_epoch = math.ceil(
        len(train_dataset) /
        (train_cfg.per_device_batch_size * world_size * train_cfg.gradient_accumulation_steps)
    )
    total_steps  = steps_per_epoch * train_cfg.num_epochs
    warmup_steps = int(total_steps * train_cfg.warmup_ratio)
    scheduler    = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    log.info(f"Steps: {total_steps} total, {warmup_steps} warmup | "
             f"Effective batch: {train_cfg.per_device_batch_size * world_size * train_cfg.gradient_accumulation_steps}")

    # ── 7. Training loop ──────────────────────────────────────────────────────
    global_step = 0
    running_loss = 0.0
    optimizer.zero_grad()

    for epoch in range(train_cfg.num_epochs):
        model.train()
        if world_size > 1:
            train_loader.sampler.set_epoch(epoch)

        t0 = time.time()

        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(**batch)
                loss    = outputs.loss / train_cfg.gradient_accumulation_steps

            loss.backward()
            running_loss += loss.item() * train_cfg.gradient_accumulation_steps

            if (step + 1) % train_cfg.gradient_accumulation_steps == 0:
                # Gradient clipping
                if world_size > 1:
                    model.clip_grad_norm_(1.0)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # ── Logging ───────────────────────────────────────────────────
                if global_step % train_cfg.logging_steps == 0 and is_main(rank):
                    elapsed = time.time() - t0
                    avg_loss = running_loss / train_cfg.logging_steps
                    tokens_sec = (
                        train_cfg.logging_steps
                        * train_cfg.per_device_batch_size
                        * world_size
                        * train_cfg.gradient_accumulation_steps
                        * model_cfg.max_seq_len
                    ) / elapsed

                    log.info(
                        f"Epoch {epoch+1} | Step {global_step}/{total_steps} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                        f"{tokens_sec:,.0f} tok/s"
                    )
                    wandb.log({
                        "train/loss":          avg_loss,
                        "train/learning_rate": scheduler.get_last_lr()[0],
                        "train/tokens_per_sec":tokens_sec,
                        "train/step":          global_step,
                        "train/epoch":         epoch + 1,
                    })
                    running_loss = 0.0
                    t0 = time.time()

                # ── Evaluation ────────────────────────────────────────────────
                if global_step % train_cfg.eval_steps == 0:
                    eval_loss = evaluate(model, eval_loader, device, world_size)
                    if is_main(rank):
                        log.info(f"Eval loss: {eval_loss:.4f}")
                        wandb.log({"eval/loss": eval_loss, "train/step": global_step})

                # ── Checkpoint ────────────────────────────────────────────────
                if global_step % train_cfg.save_steps == 0 and is_main(rank):
                    ckpt_path = Path(train_cfg.checkpoint_dir) / f"step-{global_step}"
                    ckpt_path.mkdir(parents=True, exist_ok=True)

                    if world_size > 1:
                        from torch.distributed.fsdp import FullStateDictConfig, StateDictType
                        with FSDP.state_dict_type(
                            model, StateDictType.FULL_STATE_DICT,
                            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
                        ):
                            torch.save(model.state_dict(), ckpt_path / "model.pt")
                    else:
                        model.save_pretrained(str(ckpt_path))

                    tokenizer.save_pretrained(str(ckpt_path))
                    log.info(f"Checkpoint → {ckpt_path}")

        log.info(f"Epoch {epoch+1} complete")

    # ── 8. Final save ─────────────────────────────────────────────────────────
    if is_main(rank):
        out = Path(train_cfg.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(out))
        tokenizer.save_pretrained(str(out))
        log.info(f"Model saved → {out}")
        wandb.finish()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    train(ModelConfig(), TrainingConfig())