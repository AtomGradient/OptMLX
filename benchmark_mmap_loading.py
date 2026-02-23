"""
Benchmark: mmap vs standard model loading in mlx-lm.

Measures wall-clock time for model weight loading with and without use_mmap.
Each configuration is tested multiple times; the first run warms the OS page cache,
subsequent runs measure steady-state performance.

Usage:
    python3.11 benchmark_mmap_loading.py
    (requires the locally-built MLX with use_mmap support)
"""

import gc
import json
import subprocess
import sys
import time
from pathlib import Path

# Use the locally-built MLX which supports use_mmap
_MLX_BUILD = Path(__file__).parent / "mlx" / "build" / "lib.macosx-15.0-arm64-cpython-311"
if _MLX_BUILD.exists():
    sys.path.insert(0, str(_MLX_BUILD))

import mlx.core as mx

# Verify use_mmap support
if "use_mmap" not in (mx.load.__doc__ or ""):
    print("ERROR: installed MLX does not support use_mmap.")
    print(f"  MLX loaded from: {mx.__file__}")
    print("  Run with: python3.11 benchmark_mmap_loading.py")
    sys.exit(1)

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
RUNS = 3  # number of timed runs per configuration


def get_model_size_gb(model_path: Path) -> float:
    """Sum up all safetensors file sizes."""
    total = sum(f.stat().st_size for f in model_path.glob("model*.safetensors"))
    return total / (1024**3)


def purge_disk_cache():
    """Attempt to purge the macOS disk cache so loads hit disk."""
    try:
        subprocess.run(["sudo", "purge"], check=True, capture_output=True, timeout=10)
    except Exception:
        pass  # non-fatal; results will still be relative


def load_weights(model_path: Path, use_mmap: bool) -> float:
    """
    Load all safetensors weights and return elapsed seconds.
    We call mx.eval() on every array to force actual materialization,
    ensuring mmap pages are faulted in and the comparison is fair.
    """
    import glob as globmod

    weight_files = sorted(globmod.glob(str(model_path / "model*.safetensors")))
    if not weight_files:
        raise FileNotFoundError(f"No safetensors in {model_path}")

    gc.collect()

    tic = time.perf_counter()
    weights = {}
    for wf in weight_files:
        weights.update(mx.load(wf, use_mmap=use_mmap))
    # Force evaluation so that mmap pages are actually read
    mx.eval(*weights.values())
    toc = time.perf_counter()

    # Release memory
    del weights
    gc.collect()

    return toc - tic


def benchmark_model(model_name: str):
    model_path = MODEL_BASE / model_name
    size_gb = get_model_size_gb(model_path)

    print(f"\n{'='*60}")
    print(f"Model: {model_name}  ({size_gb:.2f} GB on disk)")
    print(f"{'='*60}")

    results = {"standard": [], "mmap": []}

    for use_mmap in [False, True]:
        label = "mmap" if use_mmap else "standard"
        print(f"\n  [{label}] warming up ...", end="", flush=True)
        # Warm-up run (populates OS page cache)
        _ = load_weights(model_path, use_mmap=use_mmap)
        print(" done")

        for i in range(RUNS):
            t = load_weights(model_path, use_mmap=use_mmap)
            results[label].append(t)
            throughput = size_gb / t if t > 0 else 0
            print(f"  [{label}] run {i+1}/{RUNS}: {t:.3f}s  ({throughput:.2f} GB/s)")

    avg_std = sum(results["standard"]) / len(results["standard"])
    avg_mmap = sum(results["mmap"]) / len(results["mmap"])
    speedup = avg_std / avg_mmap if avg_mmap > 0 else 0

    print(f"\n  Summary:")
    print(f"    standard avg: {avg_std:.3f}s")
    print(f"    mmap     avg: {avg_mmap:.3f}s")
    print(f"    speedup:      {speedup:.2f}x")

    return {
        "model": model_name,
        "size_gb": round(size_gb, 2),
        "standard_avg": round(avg_std, 3),
        "mmap_avg": round(avg_mmap, 3),
        "speedup": round(speedup, 2),
        "standard_runs": [round(t, 3) for t in results["standard"]],
        "mmap_runs": [round(t, 3) for t in results["mmap"]],
    }


def main():
    print("MLX-LM mmap Loading Benchmark")
    print(f"Runs per config: {RUNS}")
    print(f"MLX version: {mx.__version__ if hasattr(mx, '__version__') else 'unknown'}")

    all_results = []
    for model_name in MODELS:
        model_path = MODEL_BASE / model_name
        if not model_path.exists():
            print(f"\nSkipping {model_name} (not found)")
            continue
        result = benchmark_model(model_name)
        all_results.append(result)

    # Final summary table
    print(f"\n{'='*60}")
    print("Final Summary")
    print(f"{'='*60}")
    print(f"{'Model':<20} {'Size':>6} {'Std(s)':>8} {'Mmap(s)':>8} {'Speedup':>8}")
    print(f"{'-'*20} {'-'*6} {'-'*8} {'-'*8} {'-'*8}")
    for r in all_results:
        print(
            f"{r['model']:<20} {r['size_gb']:>5.1f}G "
            f"{r['standard_avg']:>7.3f}s {r['mmap_avg']:>7.3f}s "
            f"{r['speedup']:>7.2f}x"
        )

    # Save JSON results
    out_path = Path(__file__).parent / "benchmark_mmap_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
