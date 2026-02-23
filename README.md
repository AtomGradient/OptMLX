# OptMLX: mmap 零拷贝加载 + KV Cache 预分配

在 macOS / Apple Silicon 上，为 [MLX](https://github.com/ml-explore/mlx) 框架和 [mlx-lm](https://github.com/ml-explore/mlx-lm) 推理引擎实现两项关键内存优化，使其在大模型推理场景下接近 llama.cpp 的内存效率。

## 背景与动机

llama.cpp 在 macOS 上能做到"大模型、小内存、极低内存涨幅"，其核心依赖两项技术：

| 技术 | llama.cpp | mlx (原始) |
|------|-----------|-----------|
| 模型加载 | `mmap()` 内存映射，操作系统按需换页，支持超物理内存的模型 | `pread()` 全量拷贝到 Metal 缓冲区，内存占用 = 模型大小 |
| KV Cache | 启动时一次性预分配 | 动态增长（每 256 token 扩展），内存阶梯式上升 |

**实现后的效果：** 模型权重零拷贝 + KV Cache 一次性分配 → 推理期间内存占用几乎不变。

## 核心架构

### Feature 1: mmap 零拷贝模型加载

```
┌─────────────────────────────────────────────────────────┐
│                   safetensors 文件                        │
│  [8B header_len] [JSON header ...] [tensor data ...]     │
└────────────────────────────┬────────────────────────────┘
                             │ mmap()
                             ▼
┌─────────────────────────────────────────────────────────┐
│              虚拟内存映射 (MAP_PRIVATE)                    │
│         mmap_ptr ──────────────────► file_size            │
└────────────────────────────┬────────────────────────────┘
                             │ Metal newBuffer(ptr, size, Shared)
                             ▼
┌─────────────────────────────────────────────────────────┐
│              MTLBuffer (共享内存模式)                      │
│   整个文件映射为一个 Metal buffer                          │
└──────┬──────────┬──────────┬───────────────────────────┘
       │          │          │  copy_shared_buffer(offset)
       ▼          ▼          ▼
   tensor_A    tensor_B    tensor_C
   (偏移视图)  (偏移视图)  (偏移视图)
```

**关键思想：** Apple Silicon 的统一内存架构意味着 CPU 和 GPU 共享同一片物理内存。`mmap()` 映射的地址可以直接包装为 Metal `MTLBuffer`（`StorageModeShared`），每个张量通过 `array::copy_shared_buffer()` 创建偏移视图——全程零拷贝，无需任何 `memcpy`。

#### 数据流

```
Python: mx.load("model.safetensors", use_mmap=True)
  │
  ▼
C++ Python 绑定: mlx_load_safetensor_helper(file, s, use_mmap=true)
  │
  ▼
C++: load_safetensors_mmap(file, s)
  │  创建 MmapReader (mmap + madvise)
  │  解析 JSON header
  │  为每个张量创建 Load primitive (携带 reader + byte_offset)
  ▼
懒求值: Load::eval_cpu()
  │
  ├─ [对齐检查通过] → 零拷贝路径
  │   MmapReader::get_metal_buffer()  // 懒初始化，整个文件一个 MTLBuffer
  │   创建 parent array (uint8, 持有 buffer 引用)
  │   out.copy_shared_buffer(parent, offset)  // 偏移视图，零拷贝
  │
  └─ [偏移未对齐] → 回退路径
      allocator::malloc() + memcpy from mmap  // 仍比 pread 快
```

#### 偏移对齐问题

safetensors 文件中张量数据的绝对偏移 = `8 + header_length + data_offset`。`copy_shared_buffer` 的 offset 参数以**元素**为单位，内部乘以 `itemsize()` 转为字节。当字节偏移不能被 `itemsize()` 整除时，整数除法会截断：

```
例: offset_ = 83 bytes, itemsize = 4 (float32)
    element_offset = 83 / 4 = 20 (截断)
    实际字节偏移 = 20 * 4 = 80 ≠ 83  ← 错误！
```

**解决方案：** 添加对齐检查 `offset_ % out.itemsize() == 0`。HuggingFace 的 safetensors 库会填充 JSON header 保证对齐，因此主流模型文件都走零拷贝路径。MLX 自身 `save_safetensors` 生成的文件可能不对齐，此时优雅回退到 memcpy。

#### 生命周期管理

```
tensor array ─── Data.deleter lambda ─── shared_ptr<Reader> ─── MmapReader
                                                                     │
                                                                metal_buffer_
                                                                mmap_ptr_
```

- deleter lambda 捕获 `shared_ptr<Reader>`，延长 `MmapReader` 生命周期
- `MmapReader` 持有 `metal_buffer_` 和 `mmap_ptr_`，析构时释放
- `MmapReader` 不反向引用任何 tensor array → **无循环引用**
- 即使删除所有上层引用，只要 tensor array 存活，mmap 映射就存活

### Feature 2: KV Cache 预分配

```
原始行为 (动态增长):
  内存 ▲
       │          ┌──┐
       │       ┌──┘  │
       │    ┌──┘     │    每 256 token 阶梯式增长
       │ ┌──┘        │
       │─┘           │
       └─────────────┴──► token

预分配行为:
  内存 ▲
       │──────────────     启动时一次性分配
       │              │
       │              │    推理期间内存平坦
       │              │
       │              │
       └──────────────┴──► token
```

**关键思想：** 在推理开始前，根据模型维度（`n_kv_heads`, `head_dim`）和目标上下文长度，一次性分配全部 KV Cache 空间。推理过程中只做 slice 写入，不再触发任何内存分配。

```python
# KVCache 预分配构造
cache = KVCache(
    n_kv_heads=8,       # 从 model.args 提取
    head_dim=128,        # 从 model.args 提取
    max_context_length=4096,
    dtype=mx.float16     # 从模型权重推断
)
# → keys shape: (1, 8, 4096, 128)
# → values shape: (1, 8, 4096, 128)
# → mx.eval() 立即分配物理内存
```

`update_and_fetch` 方法**无需修改**——现有逻辑天然兼容：
- 预分配后 `self.keys is not None` 且 `(prev + keys.shape[2]) <= self.keys.shape[2]`
- 直接执行 slice 写入 `self.keys[..., prev:offset, :] = keys`
- 若超出预分配长度，自动回退到动态增长（优雅降级）

## 修改的文件

### mlx 框架层 (C++)

| 文件 | 修改内容 |
|------|---------|
| `mlx/mlx/io/load.h` | `Reader` 基类添加 `is_mmap()` / `data_at()` 虚方法；新增 `MmapReader` 类 |
| `mlx/mlx/io/load.cpp` | `MmapReader` 完整实现：mmap/munmap/madvise + Metal buffer 懒初始化 |
| `mlx/mlx/backend/common/load.cpp` | `Load::eval_cpu` 添加 mmap 零拷贝快速路径（含对齐检查） |
| `mlx/mlx/backend/metal/primitives.cpp` | `Load::eval_gpu` 委托给 `eval_cpu`（统一内存，无需区分） |
| `mlx/mlx/io/safetensors.cpp` | 新增 `load_safetensors_mmap()` 函数 |
| `mlx/mlx/io.h` | 导出 `load_safetensors_mmap()` 声明（`MLX_API`） |
| `mlx/python/src/load.h` | 函数签名添加 `use_mmap` 参数 |
| `mlx/python/src/load.cpp` | Python 绑定调用 `load_safetensors_mmap` |
| `mlx/python/src/ops.cpp` | `mx.load()` 添加 `use_mmap` 关键字参数 |

### mlx-lm 应用层 (Python)

| 文件 | 修改内容 |
|------|---------|
| `mlx-lm/mlx_lm/models/cache.py` | `KVCache.__init__` 支持预分配；`make_prompt_cache` 支持 `max_context_length` |
| `mlx-lm/mlx_lm/generate.py` | `generate_step()` 传递 `max_context_length`；添加 CLI 参数 |
| `mlx-lm/mlx_lm/utils.py` | `load_model()` / `load()` 传递 `use_mmap` |

## 使用方法

```bash
# mmap 零拷贝加载
python -m mlx_lm generate \
  --model mlx-community/Llama-3.2-1B-Instruct-4bit \
  --prompt "Hello" \
  --use-mmap

# KV Cache 预分配 (4096 token 上下文)
python -m mlx_lm generate \
  --model mlx-community/Llama-3.2-1B-Instruct-4bit \
  --prompt "Hello" \
  --pre-allocate-kv 4096

# 两者同时使用
python -m mlx_lm generate \
  --model mlx-community/Llama-3.2-1B-Instruct-4bit \
  --prompt "Hello" \
  --use-mmap \
  --pre-allocate-kv 4096
```

Python API:

```python
import mlx.core as mx
from mlx_lm import load, generate

# mmap 加载
model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit", use_mmap=True)

# 底层 API
data = mx.load("model.safetensors", use_mmap=True)
```

## 实现过程中解决的关键问题

### 1. 共享库符号导出

**问题：** `MmapReader` 在 `pip install` 构建时（`-DBUILD_SHARED_LIBS=ON`）链接失败，因为缺少 `MLX_API` 导出注解。

**解决：** 不直接在 Python 绑定中构造 `MmapReader`，而是创建导出函数 `load_safetensors_mmap()`（带 `MLX_API`），由它在内部构造 `MmapReader`。Python 绑定只调用这个导出函数。

### 2. 字节偏移截断

**问题：** `copy_shared_buffer(parent, strides, flags, size, offset)` 的 offset 以元素为单位。当文件偏移（字节）不能被 `itemsize()` 整除时，整数除法截断导致读取位置偏移。

**解决：** 在零拷贝路径入口添加 `offset_ % out.itemsize() == 0` 检查。不对齐时回退到 memcpy（仍从 mmap 读取，速度快于 pread）。

### 3. Metal buffer 生命周期

**问题：** 多个 tensor array 共享同一个 Metal buffer（覆盖整个 mmap 区域）。需要确保只要任意 tensor 存活，底层 mmap 映射和 Metal buffer 就不被释放。

**解决：** 每个 tensor 的 deleter lambda 捕获 `shared_ptr<Reader>`（即 `MmapReader`）的引用计数。`MmapReader` 持有 `metal_buffer_` 和 `mmap_ptr_`，析构时依次释放。引用链单向流动，无循环。

### 4. KV Cache 向后兼容

**问题：** `KVCache` 的构造函数签名改变后，需要确保所有现有模型代码（不传参）仍然正常工作。

**解决：** 所有新参数都有默认值（`n_kv_heads=0, head_dim=0, max_context_length=0`）。只有三个参数都 > 0 时才预分配。默认构造 `KVCache()` 行为完全不变。

## 设计决策与权衡

### 为什么用 MAP_PRIVATE 而不是 MAP_SHARED？

`MAP_PRIVATE` 提供 copy-on-write 语义。虽然模型权重是只读的，但某些操作（如量化）可能原地修改张量数据。`MAP_PRIVATE` 确保写入时自动创建私有副本，不会破坏原始文件。

### 为什么 eval_gpu 直接委托 eval_cpu？

Apple Silicon 统一内存架构下，CPU 分配的 `MTLBuffer`（`StorageModeShared`）可以直接被 GPU 核函数访问。不需要单独的 GPU 拷贝路径。

### 为什么不强制要求对齐？

safetensors 规范没有强制数据对齐要求。虽然 HuggingFace 库填充 header 保证对齐，但我们不能假设所有文件都对齐。对齐检查 + 优雅回退保证了正确性，同时不牺牲主流场景的性能。

### 为什么 KV Cache 预分配在 Python 层而非 C++ 层？

KV Cache 的生命周期管理、模型维度提取、dtype 推断都在 Python 层完成。在 C++ 层实现需要大量跨层传递，复杂度远高于收益。Python 层的 `mx.zeros()` + `mx.eval()` 已经能高效完成物理内存分配。

## 构建与测试

```bash
# 创建虚拟环境
python3.11 -m venv .venv
source .venv/bin/activate

# 构建安装 mlx
cd mlx && python3 -m pip install . --no-build-isolation

# 安装 mlx-lm (开发模式)
python3 -m pip install -e ./mlx-lm
```

## 项目结构

```
OptMLX/
├── README.md              ← 本文件
├── mlx/                   ← MLX 框架 (修改了 C++ 核心和 Python 绑定)
│   ├── mlx/
│   │   ├── io/
│   │   │   ├── load.h     ← MmapReader 声明
│   │   │   ├── load.cpp   ← MmapReader 实现
│   │   │   └── safetensors.cpp ← load_safetensors_mmap()
│   │   ├── io.h           ← 导出声明
│   │   └── backend/
│   │       ├── common/load.cpp   ← eval_cpu 零拷贝路径
│   │       └── metal/primitives.cpp ← eval_gpu 委托
│   └── python/src/
│       ├── load.h         ← use_mmap 参数
│       ├── load.cpp       ← Python 绑定实现
│       └── ops.cpp        ← mx.load() 绑定
├── mlx-lm/                ← mlx-lm 推理引擎 (修改了 Python 层)
│   └── mlx_lm/
│       ├── models/cache.py ← KVCache 预分配
│       ├── generate.py     ← CLI 参数 + 管线集成
│       └── utils.py        ← use_mmap 传递
├── llama.cpp/             ← 参考实现
└── .venv/                 ← Python 虚拟环境
```
