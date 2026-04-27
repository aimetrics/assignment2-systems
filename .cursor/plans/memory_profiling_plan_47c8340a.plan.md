---
name: Memory Profiling Plan
overview: 在 benchmarking.py 中新增 --memory_profile flag，调用 PyTorch memory profiler API 对 2.7B 模型的 forward-only 和完整 training step（含 optimizer step）进行内存 snapshot，输出 .pickle 文件供 pytorch.org/memory_viz 可视化。不修改 model.py。
todos:
  - id: add-imports-mem
    content: 新增 import os 和 torch.nn.functional as F
    status: completed
  - id: add-run-memory-profile
    content: 实现 run_memory_profile() 函数，含 warmup、开启记录、单次 step、dump snapshot 、关闭记录
    status: completed
  - id: add-memory-profile-arg
    content: 在 parse_args() 中新增 --memory_profile 和 --output_dir 参数
    status: completed
  - id: update-main-memory
    content: 在 main() 中加分支：--memory_profile 时调用 run_memory_profile()，否则调用 benchmark()
    status: completed
isProject: false
---

# Memory Profiling 实现方案

## 目标文件
- 只修改 [`benchmarking.py`](benchmarking.py)

## 改动清单

### 1. 新增 `run_memory_profile()` 函数

接收 `model, x, y, mode, dtype, output_path`，内部逻辑：

```python
def run_memory_profile(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    mode: str,
    dtype: torch.dtype | None,
    output_path: str,
    warmup_steps: int = 2,
):
    assert mode in ["forward", "train"]
    autocast_ctx = (
        torch.autocast(device=DEVICE, dtype=dtype) if dtype is not None else nullcontext()
    )

    # 初始化 optimizer（仅 train mode 需要）
    optimizer = torch.optim.AdamW(model.parameters()) if mode == "train" else None

    # warmup（不记录，让 CUDA allocator 稳定）
    for _ in range(warmup_steps):
        model.zero_grad(set_to_none=True)
        if mode == "forward":
            with autocast_ctx, torch.no_grad():
                _ = model(x)
        else:
            with autocast_ctx:
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss.backward()
            optimizer.step()
        synchronize()

    # 开启记录
    torch.cuda.memory._record_memory_history(max_entries=1_000_000)

    # 单次 step
    model.zero_grad(set_to_none=True)
    if mode == "forward":
        with autocast_ctx, torch.no_grad():
            _ = model(x)
    else:
        with autocast_ctx:
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()
        optimizer.step()
    synchronize()

    # dump & 关闭
    torch.cuda.memory._dump_snapshot(output_path)
    torch.cuda.memory._record_memory_history(enabled=None)
    print(f"Memory snapshot saved to: {output_path}", flush=True)
```

### 2. `parse_args()` 新增参数

```python
parser.add_argument("--memory_profile", action="store_true", default=False,
    help="Run memory profiler instead of timing benchmark")
parser.add_argument("--output_dir", type=str, default=".",
    help="Directory to write memory snapshot pickle files")
```

### 3. `main()` 分支逻辑

在 `benchmark()` 调用前加分支：

```python
if args.memory_profile:
    fname = f"memory_{args.model_size}_{args.mode}_seq{args.seq_len}"
    if args.bf16:
        fname += "_bf16"
    fname += ".pickle"
    output_path = os.path.join(args.output_dir, fname)
    run_memory_profile(
        model=model, x=x, y=y,
        mode=args.mode, dtype=torch.bfloat16 if args.bf16 else None,
        output_path=output_path,
    )
else:
    timings, mean_time, std_time = benchmark(...)
```

### 4. import 补充

```python
import os
import torch.nn.functional as F
```

## 运行命令与子问题对应

### 问题 (a)：Forward-only vs Full training step 的 Active Memory Timeline

各跑一次，得到两个 pickle，分别拖入 [pytorch.org/memory_viz](https://pytorch.org/memory_viz) 观察 timeline：

```bash
# forward-only
uv run python benchmarking.py \
  --model_size 2.7b --mode forward --seq_len 128 \
  --batch_size 4 --memory_profile

# full training step（forward + backward + optimizer step）
uv run python benchmarking.py \
  --model_size 2.7b --mode train --seq_len 128 \
  --batch_size 4 --memory_profile
```

输出文件：`memory_2.7b_forward_seq128.pickle` 和 `memory_2.7b_train_seq128.pickle`

---

### 问题 (b)：各 context length 的 peak memory

共 6 次（3 个 seq_len × 2 个 mode），在 memory_viz 的 "Peak Memory" 栏读数：

```bash
for SEQ in 128 256 512; do
  uv run python benchmarking.py \
    --model_size 2.7b --mode forward --seq_len $SEQ \
    --batch_size 4 --memory_profile

  uv run python benchmarking.py \
    --model_size 2.7b --mode train --seq_len $SEQ \
    --batch_size 4 --memory_profile
done
```

输出 6 个 pickle 文件，每个文件在 memory_viz 里读取 peak active memory（单位 MB）填入表格。

---

### 问题 (c)：Mixed Precision 对内存的影响

额外跑 2 次（forward + train 各一次），加 `--bf16`，与 (b) 中 seq_len=128 的结果对比：

```bash
uv run python benchmarking.py \
  --model_size 2.7b --mode forward --seq_len 128 \
  --batch_size 4 --bf16 --memory_profile

uv run python benchmarking.py \
  --model_size 2.7b --mode train --seq_len 128 \
  --batch_size 4 --bf16 --memory_profile
```

输出：`memory_2.7b_forward_seq128_bf16.pickle` 和 `memory_2.7b_train_seq128_bf16.pickle`

---

### 问题 (d)：Residual Stream Tensor 大小推导（FP32）

2.7B 模型参数（Table 1）：`d_model = 2560`，参考配置 `batch_size = 4`

Residual stream tensor 的形状为 `(batch_size, seq_len, d_model)`，FP32 每个元素 4 bytes：

```
size = batch_size × seq_len × d_model × 4 bytes

seq_len = 128:  4 × 128  × 2560 × 4 = 5,242,880  bytes = 5.0  MB
seq_len = 256:  4 × 256  × 2560 × 4 = 10,485,760 bytes = 10.0 MB
seq_len = 512:  4 × 512  × 2560 × 4 = 20,971,520 bytes = 20.0 MB
```

（除以 1024² = 1,048,576 得 MB，注意 PDF 要求用 1024² 而非 10⁶）

在完整的 forward pass 中，每一层结束后 residual stream 的值需要被保留用于 backward pass（pre-norm 结构中也是），所以理论上 GPU 上同时驻留 `num_layers + 1 = 33` 份这样的 tensor，约占：

```
seq_len = 128:  33 × 5.0  MB = 165  MB
seq_len = 256:  33 × 10.0 MB = 330  MB
seq_len = 512:  33 × 20.0 MB = 660  MB
```

这部分在 memory_viz 的 Active Memory Timeline 中可以验证。
