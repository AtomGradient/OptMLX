"""
Benchmark: --pre-allocate-kv impact on memory and latency.

Compares three modes:
  1. Default (dynamic KV cache, grows in 256-token steps)
  2. --pre-allocate-kv 2048
  3. --pre-allocate-kv 4096

Measures:
  - Memory after model load (before generation)
  - Memory after KV cache creation (before generation)
  - First token latency (time to first token)
  - Prompt processing tok/s
  - Generation tok/s
  - Peak memory

Usage:
    python3.11 benchmark_preallocate_kv.py
"""

import gc
import json
import sys
import time
from pathlib import Path

_MLX_BUILD = Path(__file__).parent / "mlx" / "build" / "lib.macosx-15.0-arm64-cpython-311"
if _MLX_BUILD.exists():
    sys.path.insert(0, str(_MLX_BUILD))

import mlx.core as mx

_MLX_LM = Path(__file__).parent / "mlx-lm"
sys.path.insert(0, str(_MLX_LM))

from mlx_lm import load, stream_generate
from mlx_lm.models.cache import make_prompt_cache, KVCache

# ---------------------------------------------------------------------------
MODEL_BASE = Path("/Users/alex/Documents/Codes/Narrus/LLModels")
MODELS = [
    "Qwen3-4B-4bit",
    "Qwen3-8B-4bit",
    "Qwen3-14B-4bit",
]
PRE_ALLOCATE_SIZES = [None, 2048, 4096]
PROMPT = "Write a detailed introduction to the theory of general relativity, covering its historical context, key principles, and major experimental confirmations."
MAX_TOKENS = 200


def get_memory_gb():
    """Get current MLX memory usage in GB."""
    mx.synchronize()
    return mx.metal.get_active_memory() / (1024**3)


def get_peak_memory_gb():
    mx.synchronize()
    return mx.metal.get_peak_memory() / (1024**3)


def calc_kv_cache_size(model, pre_alloc):
    """Calculate theoretical KV cache size in MB for a given pre-allocation."""
    args = model.args
    n_kv_heads = getattr(args, "num_key_value_heads", None) or getattr(args, "num_attention_heads", None)
    head_dim = getattr(args, "head_dim", None)
    if head_dim is None:
        head_dim = args.hidden_size // args.num_attention_heads
    num_layers = len(model.layers)
    # Align to 256 step boundary
    aligned = ((pre_alloc + 255) // 256) * 256
    # keys + values, float16 = 2 bytes
    bytes_per_layer = 2 * 1 * n_kv_heads * aligned * head_dim * 2  # 2 for k+v, 2 for fp16
    total = bytes_per_layer * num_layers
    return total / (1024**2)


def run_benchmark(model, tokenizer, prompt_tokens, pre_alloc_size):
    """Run one generation and collect detailed timing."""
    gc.collect()
    mx.synchronize()

    # Reset peak memory tracking
    mx.metal.reset_peak_memory()
    mem_before_cache = get_memory_gb()

    # Build kwargs
    kwargs = {"max_tokens": MAX_TOKENS}
    if pre_alloc_size is not None:
        kwargs["max_context_length"] = pre_alloc_size

    # Measure time to first token and overall stats
    first_token_time = None
    tic = time.perf_counter()
    response = None
    for i, response in enumerate(stream_generate(model, tokenizer, prompt=prompt_tokens, **kwargs)):
        if i == 0:
            first_token_time = time.perf_counter() - tic

    mem_after = get_memory_gb()
    peak_mem = get_peak_memory_gb()

    return {
        "mem_before_cache": round(mem_before_cache, 3),
        "first_token_latency": round(first_token_time, 4) if first_token_time else None,
        "prompt_tps": round(response.prompt_tps, 1),
        "generation_tps": round(response.generation_tps, 1),
        "prompt_tokens": response.prompt_tokens,
        "generation_tokens": response.generation_tokens,
        "peak_memory": round(peak_mem, 3),
        "mem_after_gen": round(mem_after, 3),
    }


def benchmark_model(model_name):
    model_path = MODEL_BASE / model_name
    print(f"\n{'='*70}")
    print(f"Model: {model_name}")
    print(f"{'='*70}")

    model, tokenizer = load(str(model_path))
    mx.eval(model.parameters())
    mem_model = get_memory_gb()
    print(f"  Model memory: {mem_model:.3f} GB")

    # Prepare prompt
    messages = [{"role": "user", "content": PROMPT}]
    if tokenizer.has_chat_template:
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
    else:
        prompt_tokens = tokenizer.encode(PROMPT)

    # Show theoretical KV cache sizes
    for sz in [2048, 4096]:
        theory_mb = calc_kv_cache_size(model, sz)
        print(f"  Theoretical KV cache ({sz} tokens): {theory_mb:.1f} MB")

    results = {}
    for pre_alloc in PRE_ALLOCATE_SIZES:
        label = f"pre-alloc={pre_alloc}" if pre_alloc else "dynamic"
        print(f"\n  [{label}]")

        # Warm-up
        print(f"    warming up ...", end="", flush=True)
        _ = run_benchmark(model, tokenizer, prompt_tokens, pre_alloc)
        print(" done")

        # Timed run
        r = run_benchmark(model, tokenizer, prompt_tokens, pre_alloc)
        print(f"    First token latency: {r['first_token_latency']*1000:.1f} ms")
        print(f"    Prompt: {r['prompt_tps']:.1f} tok/s ({r['prompt_tokens']} tokens)")
        print(f"    Generation: {r['generation_tps']:.1f} tok/s ({r['generation_tokens']} tokens)")
        print(f"    Peak memory: {r['peak_memory']:.3f} GB")

        results[label] = r

    del model, tokenizer
    gc.collect()

    return {"model": model_name, "model_memory_gb": round(mem_model, 3), "results": results}


def main():
    print("MLX-LM --pre-allocate-kv Benchmark")
    print(f"MLX: {getattr(mx, '__version__', 'unknown')}")

    all_results = []
    for model_name in MODELS:
        if not (MODEL_BASE / model_name).exists():
            print(f"\nSkipping {model_name}")
            continue
        all_results.append(benchmark_model(model_name))

    # Summary table
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    print(f"{'Model':<18} {'Mode':<16} {'1st tok (ms)':>12} {'Prompt t/s':>11} {'Gen t/s':>9} {'Peak (GB)':>10}")
    print("-" * 76)
    for r in all_results:
        for mode, data in r["results"].items():
            print(
                f"{r['model']:<18} {mode:<16} "
                f"{data['first_token_latency']*1000:>11.1f} "
                f"{data['prompt_tps']:>11.1f} "
                f"{data['generation_tps']:>9.1f} "
                f"{data['peak_memory']:>10.3f}"
            )
        print()

    out_path = Path(__file__).parent / "benchmark_preallocate_kv_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
