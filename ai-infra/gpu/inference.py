"""
inference_gpu.py — Inference for Mistral-7B-v0.3 fine-tuned on ultrachat_200k.

Changes from Llama version:
  - No token needed for loading base model
  - Uses tokenizer.apply_chat_template() for correct Mistral format
  - max_new_tokens increased to 512 (ultrachat model produces longer responses)

Run:
  Single request:    python inference_gpu.py --mode single
  Production server: python inference_gpu.py --mode serve --tensor_parallel_size 1
  Benchmark:         python inference_gpu.py --mode benchmark
"""

import argparse
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def run_single(model_path: str, user_message: str):
    """Load model and generate one response — no server, no batching."""
    print(f"Loading {model_path}...")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    model.eval()

    # Build prompt using Mistral's chat template
    # Mistral format: <s>[INST] user [/INST] assistant </s>
    messages = [{"role": "user", "content": user_message}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,   # True at inference — adds [/INST] ready for response
    )

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    t0 = time.time()
    with torch.no_grad():
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

    # Decode only the new tokens (strip the input prompt)
    new_tokens = output.shape[1] - inputs["input_ids"].shape[1]
    response   = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:],
                                  skip_special_tokens=True)

    print(f"\nUser: {user_message}")
    print(f"\nAssistant: {response}")
    print(f"\n({new_tokens} tokens in {elapsed:.2f}s = {new_tokens/elapsed:.1f} tok/s)")
    return response


def run_vllm_server(model_path: str, tensor_parallel_size: int = 1, port: int = 8000):
    """
    Start vLLM server with OpenAI-compatible API.

    tensor_parallel_size controls how many GPUs on THIS machine serve the model:
      1 GPU:  7B model in BF16 = ~14 GB → fits on 1×A100 80 GB easily
      2 GPUs: needed for 70B models or for higher inference throughput
    """
    from vllm import LLM, SamplingParams

    print(f"Starting vLLM server — {tensor_parallel_size} GPU(s)...")

    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=8192,             # Mistral supports up to 32K context
        gpu_memory_utilization=0.90,
        max_num_seqs=256,
    )

    # Quick test
    sampling_params = SamplingParams(temperature=0.7, max_tokens=200)
    messages = [{"role": "user", "content": "What is the capital of France?"}]

    # Apply chat template for vLLM
    tokenizer = llm.get_tokenizer()
    test_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    outputs = llm.generate([test_prompt], sampling_params)
    print(f"Test response: {outputs[0].outputs[0].text[:100]}...")

    print(f"\nFor production, run directly:")
    print(f"  vllm serve {model_path} \\")
    print(f"    --port {port} \\")
    print(f"    --tensor-parallel-size {tensor_parallel_size} \\")
    print(f"    --max-model-len 8192 \\")
    print(f"    --dtype bfloat16")
    print(f"\nOpenAI API at: http://localhost:{port}/v1/chat/completions")


def run_benchmark(model_path: str, num_requests: int = 100):
    """Measure throughput across concurrent requests."""
    from vllm import LLM, SamplingParams

    llm = LLM(model=model_path, dtype="bfloat16", max_model_len=4096)
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(temperature=0.7, max_tokens=200)

    # Build prompts using Mistral chat template
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": f"Query {i}: Explain machine learning briefly."}],
            tokenize=False, add_generation_prompt=True
        )
        for i in range(num_requests)
    ]

    print(f"Benchmarking {num_requests} requests...")
    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - t0

    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    print(f"Requests:     {num_requests}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Wall time:    {elapsed:.2f}s")
    print(f"Throughput:   {total_tokens/elapsed:,.0f} tokens/sec")
    print(f"Avg latency:  {elapsed/num_requests*1000:.1f}ms per request")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["single", "serve", "benchmark"],
                        default="single")
    parser.add_argument("--model_path",           default="./output")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--port",                 type=int, default=8000)
    parser.add_argument("--num_requests",         type=int, default=100)
    parser.add_argument("--message",
                        default="Explain the difference between supervised and unsupervised learning.")
    args = parser.parse_args()

    if args.mode == "single":
        run_single(args.model_path, args.message)
    elif args.mode == "serve":
        run_vllm_server(args.model_path, args.tensor_parallel_size, args.port)
    elif args.mode == "benchmark":
        run_benchmark(args.model_path, args.num_requests)