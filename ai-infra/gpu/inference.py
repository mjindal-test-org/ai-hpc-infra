"""
inference_gpu.py — Inference for Mistral-7B-v0.3 fine-tuned on ultrachat_200k.

Fixes applied vs previous version:
  [1] Generation params passed directly to generate() — never set on
      model.generation_config after init (raises ValueError)
  [2] tokenizer.apply_chat_template() used for correct Mistral format
  [3] No HuggingFace token needed (Mistral is not gated)
  [4] use_cache not modified after model init

Modes:
  python inference_gpu.py --mode single      # one request, no server
  python inference_gpu.py --mode serve       # vLLM OpenAI-compatible server
  python inference_gpu.py --mode benchmark   # throughput test
"""

import argparse
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ─────────────────────────────────────────────────────────────────────────────
# Mode 1 — Single request (no server)
# ─────────────────────────────────────────────────────────────────────────────

def run_single(model_path: str, user_message: str):
    """Load model and generate one response."""
    print(f"Loading {model_path}...")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # FIX [4]: do NOT set use_cache or any config after init
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
        # use_cache defaults to True for inference — leave it alone
    )
    model.eval()

    # FIX [2]: apply_chat_template produces correct Mistral format:
    # <s>[INST] user message [/INST]
    messages = [{"role": "user", "content": user_message}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,   # adds [/INST] ready for model to continue
    )

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    t0 = time.time()
    with torch.no_grad():
        # FIX [1]: ALL generation params passed directly to generate().
        # Never do model.generation_config.max_new_tokens = 512 — that
        # raises: ValueError: max_new_tokens attribute should not be set
        # after GenerationConfig is initialized
        output = model.generate(
            **inputs,
            max_new_tokens=512,       # FIX [1]: here, not on generation_config
            temperature=0.7,          # FIX [1]
            top_p=0.9,                # FIX [1]
            do_sample=True,           # FIX [1]
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - t0

    # Decode only the new tokens (strip the input prompt)
    n_input  = inputs["input_ids"].shape[1]
    n_output = output.shape[1] - n_input
    response = tokenizer.decode(output[0][n_input:], skip_special_tokens=True)

    print(f"\nUser:      {user_message}")
    print(f"Assistant: {response}")
    print(f"\n({n_output} tokens in {elapsed:.2f}s = {n_output/elapsed:.1f} tok/s)")
    return response


# ─────────────────────────────────────────────────────────────────────────────
# Mode 2 — vLLM production server
# ─────────────────────────────────────────────────────────────────────────────

def run_server(model_path: str, tensor_parallel_size: int = 1, port: int = 8000):
    """
    Start vLLM server with OpenAI-compatible API.
    vLLM handles continuous batching and PagedAttention automatically.

    tensor_parallel_size = number of GPUs on THIS machine to use:
      1 GPU:  Mistral 7B BF16 = ~14 GB → fits on 1×A100 80 GB
      2 GPUs: for higher throughput or larger models

    FIX [1]: vLLM accepts generation params at request time via SamplingParams.
    vLLM never modifies generation_config — no issue here.
    """
    from vllm import LLM, SamplingParams

    print(f"Starting vLLM — {tensor_parallel_size} GPU(s), port {port}...")

    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=8192,
        gpu_memory_utilization=0.90,
        max_num_seqs=256,
    )

    # Test request
    tokenizer      = llm.get_tokenizer()
    sampling_params = SamplingParams(
        temperature=0.7, top_p=0.9, max_tokens=200   # FIX [1]: here in SamplingParams
    )
    test_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is machine learning?"}],
        tokenize=False, add_generation_prompt=True,
    )
    outputs = llm.generate([test_prompt], sampling_params)
    print(f"Test OK: {outputs[0].outputs[0].text[:80]}...")

    print(f"\nFor production use, run directly:")
    print(f"  vllm serve {model_path} \\")
    print(f"    --port {port} \\")
    print(f"    --tensor-parallel-size {tensor_parallel_size} \\")
    print(f"    --max-model-len 8192 \\")
    print(f"    --dtype bfloat16")
    print(f"\nOpenAI-compatible API: http://localhost:{port}/v1/chat/completions")


# ─────────────────────────────────────────────────────────────────────────────
# Mode 3 — Benchmark
# ─────────────────────────────────────────────────────────────────────────────

def run_benchmark(model_path: str, num_requests: int = 100):
    """Measure throughput across concurrent requests via vLLM."""
    from vllm import LLM, SamplingParams

    llm       = LLM(model=model_path, dtype="bfloat16", max_model_len=4096)
    tokenizer = llm.get_tokenizer()

    # FIX [1]: generation params go into SamplingParams, not generation_config
    sampling_params = SamplingParams(temperature=0.7, max_tokens=200)

    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": f"Query {i}: Explain overfitting briefly."}],
            tokenize=False, add_generation_prompt=True,
        )
        for i in range(num_requests)
    ]

    print(f"Benchmarking {num_requests} concurrent requests...")
    t0      = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - t0

    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    print(f"\nRequests:     {num_requests}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Wall time:    {elapsed:.2f}s")
    print(f"Throughput:   {total_tokens / elapsed:,.0f} tokens/sec")
    print(f"Avg latency:  {elapsed / num_requests * 1000:.1f}ms / request")


# ─────────────────────────────────────────────────────────────────────────────
# Mode 4 — Client example (calling the vLLM server)
# ─────────────────────────────────────────────────────────────────────────────

def call_server(user_message: str, server_url: str = "http://localhost:8000"):
    """Call the running vLLM server using the OpenAI Python SDK."""
    from openai import OpenAI

    client = OpenAI(base_url=f"{server_url}/v1", api_key="not-needed")
    t0     = time.time()
    resp   = client.chat.completions.create(
        model="mistral",
        messages=[{"role": "user", "content": user_message}],
        # FIX [1]: generation params set here in the API call — correct place
        max_tokens=512,
        temperature=0.7,
        top_p=0.9,
        stream=False,
    )
    elapsed = time.time() - t0
    content = resp.choices[0].message.content
    print(f"Response ({resp.usage.completion_tokens} tokens in {elapsed:.2f}s):")
    print(content)
    return content


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",
                        choices=["single", "serve", "benchmark", "client"],
                        default="single")
    parser.add_argument("--model_path",           default="./output")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--port",                 type=int, default=8000)
    parser.add_argument("--num_requests",         type=int, default=100)
    parser.add_argument("--message",
                        default="Explain the difference between supervised and "
                                "unsupervised machine learning.")
    args = parser.parse_args()

    if args.mode == "single":
        run_single(args.model_path, args.message)
    elif args.mode == "serve":
        run_server(args.model_path, args.tensor_parallel_size, args.port)
    elif args.mode == "benchmark":
        run_benchmark(args.model_path, args.num_requests)
    elif args.mode == "client":
        call_server(args.message)