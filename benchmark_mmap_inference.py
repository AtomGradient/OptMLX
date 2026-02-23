"""
Benchmark: mmap vs standard loading — inference speed impact.

Measures prompt processing tps and generation tps for both loading modes.
Uses the mlx_lm Python API directly.

Usage:
    python3.11 benchmark_mmap_inference.py
"""

import gc
import json
import sys
import time
from pathlib import Path

# Use locally-built MLX with use_mmap support
_MLX_BUILD = Path(__file__).parent / "mlx" / "build" / "lib.macosx-15.0-arm64-cpython-311"
if _MLX_BUILD.exists():
    sys.path.insert(0, str(_MLX_BUILD))

import mlx.core as mx

if "use_mmap" not in (mx.load.__doc__ or ""):
    print("ERROR: installed MLX does not support use_mmap.")
    sys.exit(1)

# Add mlx-lm to path
_MLX_LM = Path(__file__).parent / "mlx-lm"
sys.path.insert(0, str(_MLX_LM))

from mlx_lm import load, stream_generate

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_BASE = Path("/Users/alex/Documents/Codes/Narrus/LLModels")
MODELS = [
    "Qwen3-4B-4bit",
    "Qwen3-4B-8bit",
    "Qwen3-8B-3bit",
    "Qwen3-8B-4bit",
    "Qwen3-8B-6bit",
    "Qwen3-8B-8bit",
    "Qwen3-14B-4bit",
    "Qwen3-14B-6bit",
]
PROMPT = "Write a detailed introduction to the theory of general relativity, covering its historical context, key principles, and major experimental confirmations."
MAX_TOKENS = 200
RUNS = 2  # timed runs per config (inference is slower, keep it manageable)


def run_inference(model, tokenizer, prompt_tokens):
    """Run one generation pass and return stats from the last stream_generate response."""
    response = None
    for response in stream_generate(model, tokenizer, prompt=prompt_tokens, max_tokens=MAX_TOKENS):
        pass
    return (
        response.prompt_tps,
        response.generation_tps,
        response.prompt_tokens,
        response.generation_tokens,
        response.peak_memory,
    )


def benchmark_model(model_name: str):
    model_path = MODEL_BASE / model_name
    print(f"\n{'='*70}")
    print(f"Model: {model_name}")
    print(f"{'='*70}")

    results = {}

    for use_mmap in [False, True]:
        label = "mmap" if use_mmap else "standard"
        print(f"\n  [{label}] loading model ...", end="", flush=True)

        tic = time.perf_counter()
        model, tokenizer = load(str(model_path), use_mmap=use_mmap)
        load_time = time.perf_counter() - tic
        print(f" done ({load_time:.2f}s)")

        # Prepare prompt tokens
        messages = [{"role": "user", "content": PROMPT}]
        if tokenizer.has_chat_template:
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
        else:
            prompt_tokens = tokenizer.encode(PROMPT)

        # Warm-up run
        print(f"  [{label}] warming up ...", end="", flush=True)
        _ = run_inference(model, tokenizer, prompt_tokens)
        print(" done")

        runs = []
        for i in range(RUNS):
            p_tps, g_tps, p_tok, g_tok, peak_mem = run_inference(
                model, tokenizer, prompt_tokens
            )
            runs.append({
                "prompt_tps": p_tps,
                "generation_tps": g_tps,
                "prompt_tokens": p_tok,
                "generation_tokens": g_tok,
                "peak_memory": peak_mem,
            })
            print(
                f"  [{label}] run {i+1}/{RUNS}: "
                f"prompt {p_tps:.1f} tok/s, "
                f"gen {g_tps:.1f} tok/s, "
                f"peak {peak_mem:.2f} GB"
            )

        avg_p = sum(r["prompt_tps"] for r in runs) / len(runs)
        avg_g = sum(r["generation_tps"] for r in runs) / len(runs)
        avg_mem = sum(r["peak_memory"] for r in runs) / len(runs)

        results[label] = {
            "load_time": round(load_time, 3),
            "avg_prompt_tps": round(avg_p, 1),
            "avg_gen_tps": round(avg_g, 1),
            "avg_peak_memory": round(avg_mem, 2),
            "runs": runs,
        }

        # Release model
        del model, tokenizer
        gc.collect()

    std = results["standard"]
    mm = results["mmap"]
    print(f"\n  Summary:")
    print(f"    {'':20s} {'Standard':>12s} {'Mmap':>12s} {'Ratio':>8s}")
    print(f"    {'Load time':20s} {std['load_time']:>11.3f}s {mm['load_time']:>11.3f}s {std['load_time']/mm['load_time'] if mm['load_time'] else 0:>7.2f}x")
    print(f"    {'Prompt tok/s':20s} {std['avg_prompt_tps']:>12.1f} {mm['avg_prompt_tps']:>12.1f} {mm['avg_prompt_tps']/std['avg_prompt_tps'] if std['avg_prompt_tps'] else 0:>7.2f}x")
    print(f"    {'Generation tok/s':20s} {std['avg_gen_tps']:>12.1f} {mm['avg_gen_tps']:>12.1f} {mm['avg_gen_tps']/std['avg_gen_tps'] if std['avg_gen_tps'] else 0:>7.2f}x")
    print(f"    {'Peak memory (GB)':20s} {std['avg_peak_memory']:>12.2f} {mm['avg_peak_memory']:>12.2f}")

    return {
        "model": model_name,
        "standard": results["standard"],
        "mmap": results["mmap"],
    }


def main():
    print("MLX-LM mmap Inference Benchmark")
    print(f"Prompt: {len(PROMPT)} chars, Max tokens: {MAX_TOKENS}, Runs: {RUNS}")
    print(f"MLX: {getattr(mx, '__version__', 'unknown')}")

    all_results = []
    for model_name in MODELS:
        model_path = MODEL_BASE / model_name
        if not model_path.exists():
            print(f"\nSkipping {model_name} (not found)")
            continue
        result = benchmark_model(model_name)
        all_results.append(result)

    # Final summary table
    print(f"\n{'='*70}")
    print("Final Summary")
    print(f"{'='*70}")
    print(f"{'Model':<20} {'Prompt tok/s':>14} {'':>14} {'Gen tok/s':>12} {'':>12} {'Peak Mem':>10}")
    print(f"{'':20} {'Std':>7} {'Mmap':>7} {'Ratio':>7} {'Std':>6} {'Mmap':>6} {'Ratio':>7} {'Std':>5} {'Mmap':>5}")
    print("-" * 95)
    for r in all_results:
        s, m = r["standard"], r["mmap"]
        p_ratio = m["avg_prompt_tps"] / s["avg_prompt_tps"] if s["avg_prompt_tps"] else 0
        g_ratio = m["avg_gen_tps"] / s["avg_gen_tps"] if s["avg_gen_tps"] else 0
        print(
            f"{r['model']:<20} "
            f"{s['avg_prompt_tps']:>7.1f} {m['avg_prompt_tps']:>7.1f} {p_ratio:>6.2f}x "
            f"{s['avg_gen_tps']:>6.1f} {m['avg_gen_tps']:>6.1f} {g_ratio:>6.2f}x "
            f"{s['avg_peak_memory']:>5.1f} {m['avg_peak_memory']:>5.1f}"
        )

    # Save
    out_path = Path(__file__).parent / "benchmark_mmap_inference_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
