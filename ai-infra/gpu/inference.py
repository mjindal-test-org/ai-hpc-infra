"""
inference_gpu.py — Inference for Mistral-7B-v0.3 fine-tuned model.

Fixes:
  - Generation params passed directly to generate() — never set on
    model.generation_config after init (that raises ValueError)
  - apply_chat_template() used for correct Mistral [INST] format
  - No token needed (Mistral not gated)

Run:
  python inference_gpu.py --mode single
  python inference_gpu.py --mode serve
  python inference_gpu.py --mode benchmark
"""

import argparse
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def _get_attn_implementation() -> str:
    """Auto-detect best attention: FlashAttention 2 if installed, else SDPA."""
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except ImportError:
        return "sdpa"   # PyTorch 2.x built-in — still fast, no extra install


def run_single(model_path: str, message: str):
    print(f"Loading {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation=_get_attn_implementation(),
    )
    model.eval()

    # Correct Mistral chat format via apply_chat_template
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": message}],
        tokenize=False,
        add_generation_prompt=True,  # adds [/INST] — model continues from here
    )
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    t0 = time.time()
    with torch.no_grad():
        # ALL params here — never on model.generation_config
        output = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - t0

    n_in  = inputs["input_ids"].shape[1]
    n_out = output.shape[1] - n_in
    resp  = tokenizer.decode(output[0][n_in:], skip_special_tokens=True)

    print(f"\nUser:      {message}")
    print(f"Assistant: {resp}")
    print(f"\n({n_out} tokens / {elapsed:.2f}s = {n_out/elapsed:.1f} tok/s)")
    return resp


def run_server(model_path: str, tensor_parallel_size: int = 1, port: int = 8000):
    from vllm import LLM, SamplingParams
    print(f"Starting vLLM — {tensor_parallel_size} GPU(s), port {port}...")
    llm = LLM(model=model_path, dtype="bfloat16",
               tensor_parallel_size=tensor_parallel_size,
               max_model_len=8192, gpu_memory_utilization=0.90)

    tokenizer   = llm.get_tokenizer()
    test_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is machine learning?"}],
        tokenize=False, add_generation_prompt=True)

    out = llm.generate([test_prompt],
                        SamplingParams(temperature=0.7, max_tokens=100))
    print(f"Test OK: {out[0].outputs[0].text[:80]}...")

    print(f"\nvllm serve {model_path} --port {port} "
          f"--tensor-parallel-size {tensor_parallel_size} "
          f"--max-model-len 8192 --dtype bfloat16")


def run_benchmark(model_path: str, num_requests: int = 100):
    from vllm import LLM, SamplingParams
    llm       = LLM(model=model_path, dtype="bfloat16", max_model_len=4096)
    tokenizer = llm.get_tokenizer()
    sp        = SamplingParams(temperature=0.7, max_tokens=200)
    prompts   = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": f"Query {i}: Explain overfitting."}],
            tokenize=False, add_generation_prompt=True)
        for i in range(num_requests)
    ]
    t0      = time.time()
    outputs = llm.generate(prompts, sp)
    elapsed = time.time() - t0
    tokens  = sum(len(o.outputs[0].token_ids) for o in outputs)
    print(f"Requests: {num_requests} | Tokens: {tokens:,} | "
          f"Time: {elapsed:.2f}s | {tokens/elapsed:,.0f} tok/s | "
          f"{elapsed/num_requests*1000:.1f}ms/req")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["single","serve","benchmark"], default="single")
    p.add_argument("--model_path",           default="./output")
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--port",                 type=int, default=8000)
    p.add_argument("--num_requests",         type=int, default=100)
    p.add_argument("--message",
                   default="Explain the difference between supervised and unsupervised ML.")
    a = p.parse_args()

    if a.mode == "single":     run_single(a.model_path, a.message)
    elif a.mode == "serve":    run_server(a.model_path, a.tensor_parallel_size, a.port)
    elif a.mode == "benchmark":run_benchmark(a.model_path, a.num_requests)