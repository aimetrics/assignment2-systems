---
name: Mixed Precision BF16 Plan
overview: 在 benchmarking.py 中新增 --bf16 CLI flag，通过 torch.autocast 为 forward pass 开启 BF16 mixed precision，backward pass 保持 FP32，不修改 model.py。
todos:
  - id: add-nullcontext-import
    content: 将 nullcontext 加入 contextlib import，删除 _nullctx 函数，全文替换为 nullcontext()
    status: completed
  - id: add-dtype-param
    content: 在 benchmark() 签名中新增 dtype 参数，并在函数内构造 autocast_ctx，修改 warmup 和 measure 循环
    status: completed
  - id: add-bf16-arg
    content: 在 parse_args() 中新增 --bf16 flag
    status: completed
  - id: update-main-bf16
    content: 在 main() 中打印 bf16 配置、传递 dtype 参数、将 bf16 加入 results 字典
    status: completed
isProject: false
---

# Mixed Precision BF16 方案

## 目标文件
- 只修改 [`benchmarking.py`](benchmarking.py)

## 改动清单

### 1. imports 与 _nullctx 清理
将现有的 `from contextlib import contextmanager` 扩展为：
```python
from contextlib import contextmanager, nullcontext
```

删除 `_nullctx` 函数定义（L72-75），并将全文所有 `_nullctx()` 替换为 `nullcontext()`。

### 2. `benchmark()` 函数签名
新增一个参数：
```python
def benchmark(
    model, x, y,
    warmup_steps, measure_steps, mode,
    use_nvtx=False,
    dtype: torch.dtype | None = None,   # 新增
):
```

`dtype=None` 表示 FP32（不使用 autocast），`dtype=torch.bfloat16` 表示 BF16 mixed precision。

### 3. `benchmark()` 函数体

在函数开头构造 autocast context：
```python
autocast_ctx = (
    torch.autocast(device=DEVICE, dtype=dtype)
    if dtype is not None
    else nullcontext()
)
```

在 warmup 和 measure 循环中，将 forward 计算包入 `autocast_ctx`：

- **forward mode**：
```python
with autocast_ctx:
    with torch.no_grad():
        _ = run_forward_step(model, x)
```

- **train mode**（autocast 只覆盖 forward，不覆盖 backward）：
```python
with nvtx.range("forward") if use_nvtx else nullcontext():
    with autocast_ctx:
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(...)
with nvtx.range("backward") if use_nvtx else nullcontext():
    loss.backward()   # 在 autocast 外，梯度自动在 FP32 累积
```

warmup 循环做同样的修改，保证 warmup 和 measure 使用相同的 precision 设置。

### 4. `parse_args()`
新增 `--bf16` flag：
```python
parser.add_argument(
    "--bf16",
    action="store_true",
    default=False,
    help="Enable BF16 mixed precision via torch.autocast",
)
```

### 5. `main()`
- 打印配置时加一行 `print(f"bf16           : {args.bf16}")`
- 调用 `benchmark()` 时传入：
```python
dtype=torch.bfloat16 if args.bf16 else None,
```

### 6. `results` 字典
加入 `"bf16": args.bf16` 字段，让结果表格能区分 FP32 与 BF16 行。

## 关键设计决策

- 删除 `_nullctx`，统一用标准库 `nullcontext()`，与 PDF 的提示一致，减少冗余
- autocast 只覆盖 forward（包含 loss 计算），不覆盖 backward，这是 PyTorch 官方推荐用法
- `dtype` 参数而非 `bool`，方便将来扩展到 FP16 等其他精度
- `contextmanager` import 保留（仍被 `_nullctx` 之外的代码用到时）；若删除 `_nullctx` 后无其他用途则一并移除
