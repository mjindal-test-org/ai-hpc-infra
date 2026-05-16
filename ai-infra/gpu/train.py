"""
train_gpu.py — Fine-tune Mistral-7B-v0.3 on ultrachat_200k (GPU).

Fixes applied vs previous version:
  [1] use_cache=False passed inside from_pretrained() — not set after init
  [2] gradient_checkpointing_kwargs={"use_reentrant": False} prevents conflict
      with prepare_model_for_kbit_training
  [3] gradient_checkpointing_enable() only called when NOT using kbit training
  [4] All config objects (BitsAndBytesConfig, LoraConfig) built in one shot
  [5] MistralDecoderLayer used in FSDP wrap policy (not LlamaDecoderLayer)
  [6] No HuggingFace token needed (Mistral is not gated)

Run:
  Single GPU:         python train_gpu.py
  2 GPUs, 1 machine:  torchrun --nproc_per_node=2 train_gpu.py
  8 GPUs, 1 machine:  torchrun --nproc_per_node=8 train_gpu.py
  Multi-node (2 machines × 8 GPUs each):
    Machine 1: torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0
                        --master_addr=<IP> --master_port=29500 train_gpu.py
    Machine 2: torchrun --nnodes=2 --nproc_per_node=8 --node_rank=1
                        --master_addr=<IP> --master_port=29500 train_gpu.py
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

# FIX [5]: Mistral uses MistralDecoderLayer, not LlamaDecoderLayer
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)
from peft import (
    LoraConfig, get_peft_model, TaskType,
    prepare_model_for_kbit_training,
)
import bitsandbytes as bnb
import wandb

from config import ModelConfig, TrainingConfig
from dataset import UltraChatDataset, make_tokenizer, make_dataloader_gpu

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Distributed helpers
# ─────────────────────────────────────────────────────────────────────────────

def setup_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank       = int(os.environ.get("RANK",       0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        log.info(f"Distributed: rank={rank}/{world_size}, local_rank={local_rank}")
    else:
        log.info("Single GPU — no distributed setup needed")
    return local_rank, rank, world_size


def is_main(rank: int) -> bool:
    return rank == 0


# ─────────────────────────────────────────────────────────────────────────────
# Model loading  — all fixes applied here
# ─────────────────────────────────────────────────────────────────────────────

def load_model(cfg: ModelConfig) -> torch.nn.Module:
    """
    Load Mistral-7B-v0.3 with QLoRA (4-bit base + BF16 LoRA adapters).

    FIX [1]: use_cache=False set inside from_pretrained(), never after.
             Setting model.config.use_cache = False after init raises:
             ValueError: use_cache attribute should not be set after
             MistralConfig is initialized

    FIX [2]: gradient_checkpointing_kwargs={"use_reentrant": False}
             Required by newer versions of PEFT / transformers to avoid
             conflicts between prepare_model_for_kbit_training and
             torch's gradient checkpointing implementation.

    FIX [3]: gradient_checkpointing_enable() only called for the non-kbit path.
             Calling it after prepare_model_for_kbit_training(use_gradient_
             checkpointing=True) double-enables it and raises an error.

    FIX [4]: BitsAndBytesConfig and LoraConfig built in a single constructor
             call. Never modify their attributes after construction.
    """
    log.info(f"Loading {cfg.model_name}...")

    # ── Quantisation config (built once, never modified) ──────────────────────
    # FIX [4]: all fields set at construction time
    if cfg.use_4bit_quantisation:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        quant_config = None

    # ── Load base model ───────────────────────────────────────────────────────
    # FIX [1]: use_cache=False is passed HERE — not set on model.config afterward
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        token=cfg.hf_token,                 # None for Mistral (not gated)
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
        device_map=(
            "cpu" if int(os.environ.get("WORLD_SIZE", 1)) > 1
            else "auto"
        ),
        attn_implementation="flash_attention_2",
        use_cache=False,                    # FIX [1]: set here, not after
    )

    # ── Gradient checkpointing ────────────────────────────────────────────────
    if cfg.use_4bit_quantisation:
        # FIX [2] + FIX [3]: prepare_model_for_kbit_training handles gradient
        # checkpointing internally. Do NOT call gradient_checkpointing_enable()
        # separately — that causes a double-enable conflict.
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},  # FIX [2]
        )
    else:
        # Non-kbit path: enable gradient checkpointing directly
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False},  # FIX [2]
        )

    # ── LoRA adapters (built once, never modified) ────────────────────────────
    # FIX [4]: all LoraConfig fields set at construction time
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
    log.info(f"Trainable params: {trainable:,} / {total:,} "
             f"({100 * trainable / total:.2f}%) — LoRA efficiency")
    return model


def wrap_fsdp(model, local_rank: int, world_size: int):
    """Shard model across GPUs with FSDP. Skipped for single-GPU."""
    if world_size <= 1:
        return model.to(f"cuda:{local_rank}")

    # FIX [5]: MistralDecoderLayer is the correct wrap unit for Mistral
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
    log.info(f"FSDP enabled across {world_size} GPUs")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model, loader, device, world_size: int) -> float:
    model.eval()
    total, steps = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                total += model(**batch).loss.item()
            steps += 1
    avg = total / max(steps, 1)
    if world_size > 1:
        t = torch.tensor(avg, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.AVG)
        avg = t.item()
    model.train()
    return avg


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

def train(model_cfg: ModelConfig, train_cfg: TrainingConfig):

    # ── 1. Distributed ────────────────────────────────────────────────────────
    local_rank, rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    torch.manual_seed(train_cfg.seed + rank)

    if is_main(rank):
        wandb.init(
            project="mistral-ultrachat-gpu",
            config={**vars(model_cfg), **vars(train_cfg), "num_gpus": world_size},
        )

    # ── 2. Tokeniser ──────────────────────────────────────────────────────────
    tokenizer = make_tokenizer(model_cfg.model_name, model_cfg.hf_token)

    # ── 3. Datasets & loaders ─────────────────────────────────────────────────
    train_ds = UltraChatDataset(
        train_cfg.dataset_name, train_cfg.train_split,
        tokenizer, model_cfg.max_seq_len, train_cfg.max_train_samples,
    )
    eval_ds = UltraChatDataset(
        train_cfg.dataset_name, train_cfg.eval_split,
        tokenizer, model_cfg.max_seq_len, train_cfg.max_eval_samples,
    )

    if world_size > 1:
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(train_ds, world_size, rank, shuffle=True)
        eval_sampler  = DistributedSampler(eval_ds,  world_size, rank, shuffle=False)
        train_loader  = make_dataloader_gpu(train_ds, train_cfg.per_device_batch_size, shuffle=False)
        eval_loader   = make_dataloader_gpu(eval_ds,  train_cfg.per_device_batch_size, shuffle=False)
        train_loader.sampler = train_sampler
        eval_loader.sampler  = eval_sampler
    else:
        train_loader = make_dataloader_gpu(train_ds, train_cfg.per_device_batch_size)
        eval_loader  = make_dataloader_gpu(eval_ds,  train_cfg.per_device_batch_size, shuffle=False)

    # ── 4. Model ──────────────────────────────────────────────────────────────
    model = load_model(model_cfg)
    model = wrap_fsdp(model, local_rank, world_size)

    # ── 5. Optimiser ──────────────────────────────────────────────────────────
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = bnb.optim.AdamW8bit(
        trainable,
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
    )

    # ── 6. LR schedule ────────────────────────────────────────────────────────
    steps_per_epoch = math.ceil(
        len(train_ds) /
        (train_cfg.per_device_batch_size * world_size
         * train_cfg.gradient_accumulation_steps)
    )
    total_steps  = steps_per_epoch * train_cfg.num_epochs
    warmup_steps = int(total_steps * train_cfg.warmup_ratio)
    scheduler    = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps, total_steps
    )
    log.info(f"Total steps: {total_steps}  |  Warmup: {warmup_steps}  |  "
             f"Effective batch: {train_cfg.per_device_batch_size * world_size * train_cfg.gradient_accumulation_steps}")

    # ── 7. Training loop ──────────────────────────────────────────────────────
    global_step  = 0
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
                loss = model(**batch).loss / train_cfg.gradient_accumulation_steps

            loss.backward()
            running_loss += loss.item() * train_cfg.gradient_accumulation_steps

            if (step + 1) % train_cfg.gradient_accumulation_steps == 0:
                if world_size > 1:
                    model.clip_grad_norm_(1.0)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Logging
                if global_step % train_cfg.logging_steps == 0 and is_main(rank):
                    elapsed  = time.time() - t0
                    avg_loss = running_loss / train_cfg.logging_steps
                    tok_sec  = (
                        train_cfg.logging_steps
                        * train_cfg.per_device_batch_size
                        * world_size
                        * train_cfg.gradient_accumulation_steps
                        * model_cfg.max_seq_len
                    ) / elapsed
                    log.info(f"Epoch {epoch+1} | Step {global_step}/{total_steps} | "
                             f"Loss {avg_loss:.4f} | LR {scheduler.get_last_lr()[0]:.2e} | "
                             f"{tok_sec:,.0f} tok/s")
                    wandb.log({"train/loss": avg_loss,
                               "train/lr": scheduler.get_last_lr()[0],
                               "train/tok_per_sec": tok_sec,
                               "step": global_step})
                    running_loss = 0.0
                    t0 = time.time()

                # Evaluation
                if global_step % train_cfg.eval_steps == 0:
                    eval_loss = evaluate(model, eval_loader, device, world_size)
                    if is_main(rank):
                        log.info(f"Eval loss: {eval_loss:.4f}")
                        wandb.log({"eval/loss": eval_loss, "step": global_step})

                # Checkpoint
                if global_step % train_cfg.save_steps == 0 and is_main(rank):
                    path = Path(train_cfg.checkpoint_dir) / f"step-{global_step}"
                    path.mkdir(parents=True, exist_ok=True)
                    if world_size > 1:
                        from torch.distributed.fsdp import (
                            FullStateDictConfig, StateDictType
                        )
                        with FSDP.state_dict_type(
                            model, StateDictType.FULL_STATE_DICT,
                            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
                        ):
                            torch.save(model.state_dict(), path / "model.pt")
                    else:
                        model.save_pretrained(str(path))
                    tokenizer.save_pretrained(str(path))
                    log.info(f"Checkpoint saved → {path}")

        log.info(f"Epoch {epoch+1} done.")

    # ── 8. Final save ─────────────────────────────────────────────────────────
    if is_main(rank):
        out = Path(train_cfg.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(out))
        tokenizer.save_pretrained(str(out))
        log.info(f"Final model saved → {out}")
        wandb.finish()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    train(ModelConfig(), TrainingConfig())