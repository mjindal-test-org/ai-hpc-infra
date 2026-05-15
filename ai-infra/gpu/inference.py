"""
inference_gpu.py — Production inference on GPU using vLLM.

Two modes:
  1. Single request (test):  python inference_gpu.py --mode single
  2. Production server:      python inference_gpu.py --mode serve

vLLM handles:
  - Continuous batching (serving multiple users simultaneously)
  - PagedAttention (efficient KV cache management)
  - Tensor parallelism (split model across multiple GPUs for large models)
  - OpenAI-compatible REST API

Single machine examples:
  1 GPU:  vllm serve ./output --port 8000
  2 GPUs: vllm serve ./output --port 8000 --tensor-parallel-size 2
  4 GPUs: vllm serve ./output --port 8000 --tensor-parallel-size 4

Install: pip install vllm transformers peft
"""

import argparse
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


# ── MODE 1: Direct inference (single request, no server) ─────────────────────
# Use this for: testing your fine-tuned model, evaluation, batch processing
# NOT suitable for: serving multiple concurrent users

def run_single_inference(model_path: str, prompt: str):
    """
    Load model and run a single inference pass.
    No server, no batching — straightforward generate() call.
    """
    print(f"Loading model from {model_path}...")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load in BF16 — fastest on A100/H100 via Tensor Cores
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",          # auto-places across available GPUs
        attn_implementation="flash_attention_2",  # faster attention
    )
    model.eval()

    # Format prompt in Llama 3 chat format
    system = "You are a helpful customer support agent."
    full_prompt = (
        f"<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n{system}\n"
        f"<|start_header_id|>user<|end_header_id|>\n{prompt}\n"
        f"<|start_header_id|>assistant<|end_header_id|>\n"
    )

    inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")

    t0 = time.time()
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - t0

    new_tokens = output.shape[1] - inputs["input_ids"].shape[1]
    response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:],
                                skip_special_tokens=True)

    print(f"\nResponse ({new_tokens} tokens in {elapsed:.2f}s = "
          f"{new_tokens/elapsed:.1f} tok/s):")
    print(response)
    return response


# ── MODE 2: vLLM production server ───────────────────────────────────────────
# Use this for: serving multiple concurrent users, production deployment
# vLLM handles batching, scheduling, and memory management automatically

def run_vllm_server(model_path: str, tensor_parallel_size: int = 1, port: int = 8000):
    """
    Start a vLLM server with OpenAI-compatible API.

    tensor_parallel_size:
      1 = single GPU (7B model in BF16 needs ~14GB — fits on 1×A100 80GB)
      2 = 2 GPUs (70B model needs ~140GB — needs 2×A100 80GB with TP=2)
      4 = 4 GPUs (for very large models or very high throughput)

    The server runs until killed (Ctrl+C).
    Clients send requests to http://localhost:8000/v1/chat/completions
    """
    from vllm import LLM, SamplingParams

    print(f"Starting vLLM server on port {port}...")
    print(f"Tensor parallel size: {tensor_parallel_size}")
    print(f"This enables {tensor_parallel_size} GPU(s) on 1 machine.")

    llm = LLM(
        model=model_path,
        tokenizer=model_path,
        dtype="bfloat16",
        tensor_parallel_size=tensor_parallel_size,  # GPUs on THIS machine
        # pipeline_parallel_size=1  # for multi-machine pipeline parallelism
        max_model_len=4096,         # max context window for inference
        gpu_memory_utilization=0.90,  # use 90% of GPU memory for KV cache
        # Continuous batching settings:
        max_num_seqs=256,           # max concurrent requests
        max_num_batched_tokens=8192,  # max tokens processed per step
    )

    # Test the server with a sample request
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=200,
    )

    test_prompt = (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n"
        "You are a helpful customer support agent.\n"
        "<|start_header_id|>user<|end_header_id|>\n"
        "My order hasn't arrived yet.\n"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )

    print("\nTest request:")
    outputs = llm.generate([test_prompt], sampling_params)
    for output in outputs:
        print(f"Response: {output.outputs[0].text[:200]}...")

    print(f"\nvLLM is ready. For production, run:")
    print(f"  vllm serve {model_path} \\")
    print(f"    --port {port} \\")
    print(f"    --tensor-parallel-size {tensor_parallel_size} \\")
    print(f"    --max-model-len 4096 \\")
    print(f"    --dtype bfloat16")
    print(f"\nOpenAI-compatible API available at:")
    print(f"  http://localhost:{port}/v1/chat/completions")


# ── MODE 3: vLLM client (how users call the server) ──────────────────────────

def call_vllm_api(user_message: str, server_url: str = "http://localhost:8000"):
    """
    Example client that calls the vLLM server.
    Uses the OpenAI Python SDK — drop-in replacement for GPT-4 calls.
    """
    from openai import OpenAI

    client = OpenAI(
        base_url=f"{server_url}/v1",
        api_key="not-needed",   # vLLM doesn't require auth (add nginx for that)
    )

    t0 = time.time()
    response = client.chat.completions.create(
        model="llama-support",   # name you give the model in vLLM
        messages=[
            {"role": "system", "content": "You are a helpful customer support agent."},
            {"role": "user", "content": user_message},
        ],
        max_tokens=200,
        temperature=0.7,
        stream=False,
    )
    elapsed = time.time() - t0

    content = response.choices[0].message.content
    usage = response.usage
    print(f"Response ({usage.completion_tokens} tokens in {elapsed:.2f}s):")
    print(content)
    return content


# ── MODE 4: Benchmark (throughput test) ──────────────────────────────────────

def benchmark_gpu(model_path: str, num_requests: int = 100):
    """
    Measure inference throughput: tokens per second across concurrent requests.
    This simulates production load and shows the value of continuous batching.
    """
    from vllm import LLM, SamplingParams

    llm = LLM(model=model_path, dtype="bfloat16", max_model_len=2048)
    sampling_params = SamplingParams(temperature=0.7, max_tokens=100)

    # Simulate concurrent user requests
    prompts = [
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n"
        f"Support query #{i}: My order is delayed.\n"
        "<|start_header_id|>assistant<|end_header_id|>\n"
        for i in range(num_requests)
    ]

    print(f"Benchmarking {num_requests} concurrent requests...")
    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - t0

    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    print(f"\nResults:")
    print(f"  Requests:     {num_requests}")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Wall time:    {elapsed:.2f}s")
    print(f"  Throughput:   {total_tokens/elapsed:,.0f} tokens/sec")
    print(f"  Latency avg:  {elapsed/num_requests*1000:.1f}ms per request")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["single", "serve", "benchmark"],
                        default="single")
    parser.add_argument("--model_path", default="./output")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--num_requests", type=int, default=100)
    args = parser.parse_args()

    if args.mode == "single":
        run_single_inference(args.model_path, "My order hasn't arrived after 2 weeks.")
    elif args.mode == "serve":
        run_vllm_server(args.model_path, args.tensor_parallel_size, args.port)
    elif args.mode == "benchmark":
        benchmark_gpu(args.model_path, args.num_requests)