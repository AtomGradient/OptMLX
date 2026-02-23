# OptMLX

Exploratory research on zero-copy mmap model loading and KV cache pre-allocation for [MLX](https://github.com/ml-explore/mlx) on Apple Silicon.

This project investigates whether two memory optimization techniques commonly used in [llama.cpp](https://github.com/ggerganov/llama.cpp) — `mmap()` zero-copy loading and KV cache pre-allocation — can improve MLX's inference performance. Our benchmarks across 8 Qwen3 quantized models on an M1 Max (32GB) reveal that **MLX's existing implementation is already well-designed for Apple Silicon's unified memory architecture**, and these techniques yield mixed results depending on model size and quantization.

**[Project Page](https://atomgradient.github.io/OptMLX/)** · **[Paper (PDF)](docs/paper.pdf)**

## Key Findings

| Technique | Result |
|-----------|--------|
| mmap Zero-Copy Loading | Up to 20x faster for large models (8B-8bit, 14B-4bit), but slower for small models due to lazy evaluation overhead |
| KV Cache Pre-Allocation | No meaningful throughput gain; increases initial memory with no benefit for short conversations |
| Quantized dtype Bug | Discovered and fixed a bug where `QuantizedLinear.weight.dtype` returns `uint32` instead of the actual compute dtype |

## Installation

### Prerequisites

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.11+
- Xcode Command Line Tools (`xcode-select --install`)

### Setup

Clone this repository (with submodules already included):

```bash
git clone https://github.com/AtomGradient/OptMLX.git
cd OptMLX
```

Create and activate a virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Install the modified MLX framework (C++ build with Python bindings):

```bash
pip install ./mlx --no-build-isolation
```

> This compiles MLX from source with mmap support. The build may take a few minutes.

Install the modified mlx-lm inference engine:

```bash
pip install -e ./mlx-lm
```

Verify installation:

```bash
python -c "import mlx.core as mx; print('MLX version:', mx.__version__)"
python -c "import mlx_lm; print('mlx-lm imported successfully')"
```

## Usage

### CLI

```bash
# Standard loading (baseline)
python -m mlx_lm generate \
  --model mlx-community/Qwen3-8B-4bit \
  --prompt "Hello" \
  --max-tokens 200

# mmap zero-copy loading
python -m mlx_lm generate \
  --model mlx-community/Qwen3-8B-4bit \
  --prompt "Hello" \
  --max-tokens 200 \
  --use-mmap

# KV cache pre-allocation (4096-token context window)
python -m mlx_lm generate \
  --model mlx-community/Qwen3-8B-4bit \
  --prompt "Hello" \
  --max-tokens 200 \
  --pre-allocate-kv 4096

# Both techniques combined
python -m mlx_lm generate \
  --model mlx-community/Qwen3-8B-4bit \
  --prompt "Hello" \
  --max-tokens 200 \
  --use-mmap \
  --pre-allocate-kv 4096
```

### Python API

```python
import mlx.core as mx
from mlx_lm import load, generate

# Load model with mmap
model, tokenizer = load("mlx-community/Qwen3-8B-4bit", use_mmap=True)

# Generate text
response = generate(model, tokenizer, prompt="Hello", max_tokens=200)
print(response)

# Low-level mmap loading
data = mx.load("model.safetensors", use_mmap=True)
```

## Running Benchmarks

Three benchmark scripts are included to reproduce our results.

### 1. mmap Loading Speed

Measures model loading time with standard `pread()` vs `mmap()`:

```bash
python benchmark_mmap_loading.py
```

Results are saved to `benchmark_mmap_results.json`.

### 2. mmap Inference Impact

Measures whether mmap loading affects inference throughput (prompt processing and token generation):

```bash
python benchmark_mmap_inference.py
```

Results are saved to `benchmark_mmap_inference_results.json`.

### 3. KV Cache Pre-Allocation

Compares dynamic KV cache growth vs pre-allocated cache at different context sizes:

```bash
python benchmark_preallocate_kv.py
```

Results are saved to `benchmark_preallocate_kv_results.json`.

> **Note:** Benchmarks download Qwen3 models from Hugging Face on first run. Ensure you have sufficient disk space and network access.

## What We Modified

### MLX Framework (C++)

| File | Change |
|------|--------|
| `mlx/mlx/io/load.h` | Added `MmapReader` class with `mmap()`, `madvise()`, and Metal buffer sharing |
| `mlx/mlx/io/load.cpp` | Full `MmapReader` implementation with lifecycle management |
| `mlx/mlx/backend/common/load.cpp` | Zero-copy path in `Load::eval_cpu` with alignment check and fallback |
| `mlx/mlx/backend/metal/primitives.cpp` | `Load::eval_gpu` delegates to `eval_cpu` (unified memory) |
| `mlx/mlx/io/safetensors.cpp` | Added `load_safetensors_mmap()` function |
| `mlx/mlx/io.h` | Exported `load_safetensors_mmap()` via `MLX_API` |
| `mlx/python/src/load.h` | Added `use_mmap` parameter to function signatures |
| `mlx/python/src/load.cpp` | Python binding calls `load_safetensors_mmap` |
| `mlx/python/src/ops.cpp` | Added `use_mmap` keyword argument to `mx.load()` |

### mlx-lm (Python)

| File | Change |
|------|--------|
| `mlx-lm/mlx_lm/models/cache.py` | `KVCache` supports pre-allocation; `make_prompt_cache` accepts `max_context_length` |
| `mlx-lm/mlx_lm/generate.py` | Added `--use-mmap` and `--pre-allocate-kv` CLI arguments |
| `mlx-lm/mlx_lm/utils.py` | `load_model()` / `load()` pass through `use_mmap` |

## Project Structure

```
OptMLX/
├── mlx/                        # Modified MLX framework (C++ core + Python bindings)
├── mlx-lm/                     # Modified mlx-lm inference engine (Python)
├── paper/
│   └── paper.tex               # LaTeX source
├── docs/
│   ├── index.html              # GitHub Pages project page
│   └── paper.pdf               # Compiled paper
├── benchmark_mmap_loading.py   # Loading speed benchmark
├── benchmark_mmap_inference.py # Inference impact benchmark
├── benchmark_preallocate_kv.py # KV cache pre-allocation benchmark
├── benchmark_*_results.json    # Benchmark results data
└── README.md
```

## Citation

```bibtex
@misc{optmlx2026,
  title   = {Exploring Zero-Copy mmap Loading and KV Cache Pre-Allocation for MLX on Apple Silicon},
  author  = {AtomGradient},
  year    = {2026},
  url     = {https://github.com/AtomGradient/OptMLX},
}
```

## License

This project is for research purposes. The modified MLX and mlx-lm code is based on Apple's original repositories under the MIT License.
