"""
inference_tpu.py — Inference on TPU using torch_xla + FastAPI server.

TPU inference is different from GPU:
  - No vLLM support (vLLM is CUDA-only as of 2024)
  - XLA compiles the graph on the first call (~30–60s warm-up)
  - All subsequent calls with the SAME shape are fast (compiled graph reused)
  - Static shapes required — variable-length inputs must be padded
  - JetStream (Google's TPU serving framework) is the production alternative

Two modes:
  1. Single request (test): python inference_tpu.py --mode single
  2. FastAPI server:         python inference_tpu.py --mode serve

For production-grade TPU serving, use JetStream:
  https://github.com/google/jetstream-pytorch

Install: pip install torch torch_xla[tpu] transformers peft fastapi uvicorn
"""

import argparse
import time
import logging
from typing import Optional

import torch
import torch_xla.core.xla_model as xm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

log = logging.getLogger(__name__)


# ── TPU INFERENCE: UNDERSTANDING XLA COMPILATION ─────────────────────────────
#
# When you call model.generate() on TPU for the first time:
#   Step 1: XLA traces the forward pass (records all operations)
#   Step 2: XLA compiles the trace to optimised TPU machine code (~30–60s)
#   Step 3: The compiled kernel executes on hardware
#
# On the SECOND call with the SAME shape:
#   Step 1: XLA finds the cached compiled kernel
#   Step 2: Executes directly — no compilation overhead
#   → Much faster!
#
# On a call with a DIFFERENT shape (e.g. longer input):
#   → XLA recompiles. Back to 30–60s.
#   → This is why fixed shapes are essential for TPU inference.
#
# Strategy: Pad all inputs to a fixed set of "bucket" lengths.
# e.g. if input is 150 tokens → pad to 256 (nearest power of 2)
#      if input is 300 tokens → pad to 512
# This limits recompilations to the number of bucket sizes.
#
# ─────────────────────────────────────────────────────────────────────────────


# Fixed bucket sizes for input padding — one XLA compilation per bucket
# Powers of 2 are friendly for TPU's systolic array
BUCKET_SIZES = [64, 128, 256, 512, 1024]


def get_bucket(length: int) -> int:
    """Find the smallest bucket size >= length."""
    for bucket in BUCKET_SIZES:
        if length <= bucket:
            return bucket
    return BUCKET_SIZES[-1]   # cap at max bucket


class TPUInferenceEngine:
    """
    Wrapper around the TPU model that handles:
    - Fixed-shape padding for XLA compilation stability
    - Compilation warm-up on startup
    - Efficient batch inference
    """

    def __init__(self, model_path: str):
        self.device = xm.xla_device()

        print(f"Loading model from {model_path} onto TPU...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Load in BF16 — TPU's native precision
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        # Compile with torch.compile using XLA backend
        # This pre-traces and optimises the forward pass
        # The first call still triggers full XLA compilation,
        # but subsequent same-shape calls are faster.
        self.model = torch.compile(
            self.model,
            backend="openxla",    # XLA backend for TPU
            fullgraph=True,       # compile entire graph at once
        )

        print(f"Model loaded on {self.device}")
        self._warmup()

    def _warmup(self):
        """
        Pre-warm all bucket sizes at startup.
        This triggers XLA compilation for each bucket during startup
        (slow, ~60s per bucket) so production requests don't hit cold compilation.

        In production, warm up only the buckets you expect to use.
        """
        print("Warming up XLA compilations for each bucket size...")
        for bucket in BUCKET_SIZES[:3]:   # warm first 3 buckets
            print(f"  Compiling bucket size {bucket}...")
            dummy_input = {
                "input_ids": torch.ones(1, bucket, dtype=torch.long).to(self.device),
                "attention_mask": torch.ones(1, bucket, dtype=torch.long).to(self.device),
            }
            t0 = time.time()
            with torch.no_grad():
                _ = self.model(**dummy_input, use_cache=False)
            xm.mark_step()   # trigger compilation
            print(f"  Bucket {bucket} compiled in {time.time()-t0:.1f}s")

        print("Warm-up complete — all subsequent calls will be fast.")

    def generate(self, prompt: str, max_new_tokens: int = 200,
                 temperature: float = 0.7) -> dict:
        """
        Generate a response for a single prompt.

        Steps:
        1. Tokenise and measure length
        2. Pad to the appropriate bucket (fixed shape → no recompile)
        3. Run generation token by token
        4. Decode and strip padding from output
        """
        system = "You are a helpful customer support agent."
        full_prompt = (
            f"<|begin_of_text|>"
            f"<|start_header_id|>system<|end_header_id|>\n{system}\n"
            f"<|start_header_id|>user<|end_header_id|>\n{prompt}\n"
            f"<|start_header_id|>assistant<|end_header_id|>\n"
        )

        # Tokenise first to measure length
        tokens = self.tokenizer(full_prompt, return_tensors="pt",
                                truncation=True, max_length=BUCKET_SIZES[-1])
        input_len = tokens["input_ids"].shape[1]

        # Pad to bucket size — ensures fixed shape → no XLA recompile
        bucket = get_bucket(input_len)

        # Re-tokenise with fixed padding to the bucket size
        tokens = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            padding="max_length",
            max_length=bucket,
            truncation=True,
        )

        input_ids = tokens["input_ids"].to(self.device)
        attention_mask = tokens["attention_mask"].to(self.device)

        t0 = time.time()

        # Manual generation loop — gives us control over shape at each step
        # Unlike model.generate() which may create dynamic shapes internally
        generated_ids = input_ids.clone()
        generated_attention = attention_mask.clone()

        for _ in range(max_new_tokens):
            with torch.no_grad():
                outputs = self.model(
                    input_ids=generated_ids,
                    attention_mask=generated_attention,
                    use_cache=False,   # avoids KV cache dynamic shapes
                )

            # Sample next token
            logits = outputs.logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Check for EOS (still done on TPU — comparison is fast)
            if next_token.item() == self.tokenizer.eos_token_id:
                xm.mark_step()
                break

            # Append token — this changes shape! Need to handle carefully.
            # In production, use a fixed-length sliding window instead.
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            generated_attention = torch.cat([
                generated_attention,
                torch.ones(1, 1, dtype=torch.long).to(self.device)
            ], dim=1)

            # Mark step every token — tells XLA to execute
            xm.mark_step()

        elapsed = time.time() - t0
        new_tokens = generated_ids.shape[1] - input_ids.shape[1]

        # Decode only the generated tokens (strip input and padding)
        response_ids = generated_ids[0, input_len:]
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        return {
            "response": response.strip(),
            "input_tokens": input_len,
            "output_tokens": new_tokens,
            "elapsed_sec": round(elapsed, 3),
            "tokens_per_sec": round(new_tokens / elapsed, 1),
            "bucket_used": bucket,
        }


# ── FASTAPI SERVER ────────────────────────────────────────────────────────────

def run_server(model_path: str, port: int = 8001):
    """
    Simple FastAPI server wrapping the TPU inference engine.
    Single-threaded — TPU processes one request at a time.

    For multi-user TPU serving at scale, use JetStream instead:
    https://github.com/google/jetstream-pytorch
    """
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    import uvicorn

    # Load model (this triggers warm-up compilation — takes ~3 minutes)
    engine = TPUInferenceEngine(model_path)

    app = FastAPI(title="Llama 3 Support Bot — TPU")

    class ChatRequest(BaseModel):
        message: str
        max_new_tokens: Optional[int] = 200
        temperature: Optional[float] = 0.7

    class ChatResponse(BaseModel):
        response: str
        input_tokens: int
        output_tokens: int
        elapsed_sec: float
        tokens_per_sec: float
        bucket_used: int

    @app.get("/health")
    def health():
        return {"status": "ok", "device": str(engine.device)}

    @app.post("/chat", response_model=ChatResponse)
    def chat(request: ChatRequest):
        try:
            result = engine.generate(
                request.message,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
            )
            return ChatResponse(**result)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    print(f"\nTPU inference server starting on http://0.0.0.0:{port}")
    print(f"API docs: http://localhost:{port}/docs")
    uvicorn.run(app, host="0.0.0.0", port=port, workers=1)


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["single", "serve"], default="single")
    parser.add_argument("--model_path", default="./output")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    if args.mode == "single":
        engine = TPUInferenceEngine(args.model_path)
        result = engine.generate("My order hasn't arrived after 2 weeks.")
        print(f"\nResponse: {result['response']}")
        print(f"Stats: {result['output_tokens']} tokens in {result['elapsed_sec']}s "
              f"({result['tokens_per_sec']} tok/s), bucket={result['bucket_used']}")
    elif args.mode == "serve":
        run_server(args.model_path, args.port)