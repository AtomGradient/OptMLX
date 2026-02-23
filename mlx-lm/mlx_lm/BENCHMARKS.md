# Benchmarks

## Commands 

The command for evaluating on MMLU Pro:

```
mlx_lm.evaluate --model model/repo --task mmlu_pro
```
 
The command for efficiency benchmarks:

```
mlx_lm.benchmark --model model/repo -p 2048 -g 128
```

To get the package versions run:

```
python -m mlx --version && python -m mlx_lm --version
```

## Models

<details>

 <summary> Qwen/Qwen3-4B-Instruct-2507 </summary>

Precision | MMLU Pro | Prompt (2048) tok/sec | Generation (128) tok/sec | Memory GB | Repo
--------- | -------- | ------------------- | ------------------------ | --------- | ----
bf16      | 64.05    | 1780.63             | 52.47                    | 9.02    | Qwen/Qwen3-4B-Instruct-2507
q8 | 63.85 | 1606.573| 86.907 | 5.254 | mlx-community/Qwen3-4B-Instruct-2507-8bit
q6 | 63.53 | 1576.73 | 104.68 | 4.25 | mlx-community/Qwen3-4B-Instruct-2507-6bit
q5 g32 | 63.16 | 1570.80 | 110.29 | 4.00 | mlx-community/Qwen3-4B-Instruct-2507-5bit-g32
q5 | 62.38 | 1584.33 | 116.39 | 3.86 | mlx-community/Qwen3-4B-Instruct-2507-5bit
q4 g32 | 61.46 | 1610.03 | 126.00 | 3.603 | mlx-community/Qwen3-4B-Instruct-2507-4bit-g32
q4 | 60.72 | 1622.27 | 134.52 | 3.35 | mlx-community/Qwen3-4B-Instruct-2507-4bit

- Performance benchmark on 64GB M4 Max
- mlx 0.29.2.dev20251008+85a8824a8
- mlx-lm 0.28.2
- macOS 26.1
 
</details>

<details>
<summary> Qwen/Qwen3-30B-A3B-Instruct-2507 </summary>

Precision | MMLU Pro | Prompt (2048) tok/sec | Generation (128) tok/sec | Memory GB | Repo
--------- | -------- | ------------------- | ------------------------ | --------- | ----
bf16 | 72.62 | :skull: | :skull: | :skull: | Qwen/Qwen3-30B-A3B-Instruct-2507
q8 | 72.46 | 1719.47 | 83.16 | 33.46 | mlx-community/Qwen3-30B-A3B-Instruct-2507-8bit 
q6 | 72.41 | 1667.45 | 94.14 | 25.82 | mlx-community/Qwen3-30B-A3B-Instruct-2507-6bit
q5 | 71.97 | 1664.24 | 101.00 |22.01 | mlx-community/Qwen3-30B-A3B-Instruct-2507-5bit
q4 | 70.71 | 1753.90 | 113.33 |18.20 | mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit

 
- Performance benchmarks on 64GB M4 Max
- mlx 0.29.2.dev20251008+85a8824a8
- mlx-lm 0.28.2
- macOS 26.1

</details>

## Memory-Mapped Loading (`--use-mmap`)

Comparison of model weight loading time with standard I/O vs memory-mapped I/O (`--use-mmap`).
Each configuration runs 3 times after a warm-up run (OS page cache populated).

Model | Size (GB) | Standard (s) | Mmap (s) | Speedup
----- | --------- | ------------ | -------- | -------
Qwen3-4B-4bit | 2.1 | 0.101 | 0.131 | 0.77x
Qwen3-4B-8bit | 4.0 | 0.184 | 0.246 | 0.75x
Qwen3-8B-3bit | 3.3 | 0.146 | 0.075 | 1.95x
Qwen3-8B-4bit | 4.3 | 0.352 | 0.328 | 1.07x
Qwen3-8B-6bit | 6.2 | 0.637 | 0.435 | 1.46x
Qwen3-8B-8bit | 8.1 | 2.572 | 0.125 | **20.65x**
Qwen3-14B-4bit | 7.7 | 2.323 | 0.535 | **4.34x**
Qwen3-14B-6bit | 11.2 | 3.701 | 5.388 | 0.69x

- Benchmark on M4 Max (16C CPU / 40C GPU / 48GB)
- mlx 0.30.7.dev20260223+d4c81062
- macOS 26.1

### Inference Speed Impact

Does `--use-mmap` affect prompt processing or token generation speed?
Prompt ~50 tokens, generation 200 tokens, 2 timed runs after warm-up.

Model | Prompt tok/s (Std) | Prompt tok/s (Mmap) | Ratio | Gen tok/s (Std) | Gen tok/s (Mmap) | Ratio | Peak Mem Std (GB) | Peak Mem Mmap (GB)
----- | ------------------ | ------------------- | ----- | --------------- | ---------------- | ----- | ----------------- | ------------------
Qwen3-4B-4bit | 189.1 | 188.6 | 1.00x | 56.9 | 58.5 | 1.03x | 2.38 | 2.38
Qwen3-4B-8bit | 165.8 | 159.7 | 0.96x | 41.0 | 42.0 | 1.02x | 4.37 | 4.37
Qwen3-8B-3bit | 100.8 | 92.3 | 0.92x | 29.9 | 29.4 | 0.98x | 4.37 | 4.37
Qwen3-8B-4bit | 89.8 | 100.7 | 1.12x | 30.1 | 30.1 | 1.00x | 4.72 | 8.82
Qwen3-8B-6bit | 86.7 | 75.7 | 0.87x | 20.7 | 20.7 | 1.00x | 8.82 | 8.82
Qwen3-8B-8bit | 64.4 | 80.6 | 1.25x | 21.6 | 21.3 | 0.99x | 8.82 | 8.84
Qwen3-14B-4bit | 43.4 | 43.2 | 1.00x | 16.2 | 16.5 | 1.02x | 8.84 | 11.09
Qwen3-14B-6bit | 39.7 | 37.0 | 0.93x | 12.1 | 12.0 | 0.99x | 12.08 | 12.16

- Benchmark on M4 Max (16C CPU / 40C GPU / 48GB)
- mlx 0.30.7.dev20260223+d4c81062
- macOS 26.1
