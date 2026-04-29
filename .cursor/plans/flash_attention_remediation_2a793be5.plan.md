---
name: flash attention remediation
overview: 修复 FlashAttention-2 forward/backward 与 benchmarking 交付物之间的不一致，使实现更贴近 PDF 要求，并保留可验证的最小测试路径。计划只覆盖当前评审发现的问题，不引入额外 Triton backward 优化。
todos:
  - id: unify-backward
    content: 让 Triton autograd backward 复用公共 PyTorch+torch.compile backward helper，并保存 O/L
    status: completed
  - id: fix-forward-details
    content: 修正 forward 中 L dtype、causal mask 与 Triton key 越界 mask
    status: completed
  - id: repair-benchmark
    content: 收敛并修复 flash benchmark 入口、导入路径和异常隔离
    status: completed
  - id: verify
    content: 运行 CPU pytest、CUDA Triton pytest 与 H100 benchmark smoke/full sweep
    status: completed
isProject: false
---

# FlashAttention Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 `flash_forward`、`flash_backward`、`flash_benchmarking` 三个 Problem 的代码入口、保存张量、backward 公式和 benchmark 脚本与作业要求一致。

**Architecture:** 保留现有 `cs336_systems/flash_attention_pytorch.py` 和 `cs336_systems/flash_attention_triton.py` 分工：PyTorch 版用于 reference/debug，Triton 版只实现 fused forward，backward 统一复用 PyTorch+`torch.compile` 的重计算公式。Benchmark 统一到一个可运行脚本，避免两个入口互相引用不存在的模块。

**Tech Stack:** PyTorch autograd, `torch.compile`, Triton forward kernel, `triton.testing.do_bench`, pytest.

---

## 改造原因

- `flash_forward` 当前 PyTorch 版能通过测试，但 `L` 低精度保存和 causal 支持边界不利于后续复用；Triton 版没有保存 `O`，与 backward 题目要求的输入不一致。
- `flash_backward` 当前 PyTorch class 使用了指定公式，但 Triton class 的 backward 走 naive PyTorch autograd，能过小测试但没有使用 `Q,K,V,O,dO,L,D` 这条作业指定路径，也会在长序列 benchmark 中物化完整 attention matrix。
- `flash_benchmarking` 当前存在多个入口：`benchmark_attention.py` 不是这个 Problem；`benchmark_flash_attention.py` 引用不存在的 `cs336_systems.flash_attention`；`cs336_systems/flash_attention_benchmark.py` 又引用 `systems_core...`。需要收敛到一个明确、可运行、符合 sweep 要求的脚本。

## 目标文件结构

- Modify: `cs336_systems/flash_attention_pytorch.py`
  - 保留 tiled PyTorch forward。
  - 让 backward helper 成为 Triton class 也可复用的公共实现。
  - 可选增强：支持 causal mask，以便 PyTorch reference 与 benchmark 行为一致。

- Modify: `cs336_systems/flash_attention_triton.py`
  - 修正 forward 保存张量，保存 `L,Q,K,V,O` 或至少保存 `q,k,v,out,lse`。
  - backward 改成调用公共 compiled helper，而不是重新构图调用 `_attention_and_lse_torch`。
  - 检查 causal mask 和 key tile 越界 mask，避免小 seq 或非整 tile 时 padding 参与 softmax。

- Modify or Replace: `cs336_systems/flash_attention_benchmark.py`
  - 修正导入为当前包路径。
  - 使用 `get_flashattention_autograd_function_triton()`。
  - 保持 batch size 1、causal masking、seq length 128 到 65536、d 16 到 128、dtype bf16/fp32。
  - 单独捕获 PyTorch 与 Triton 的 OOM/RuntimeError，避免一个实现失败污染另一个实现的结果。

- Optional Modify: `benchmark_flash_attention.py`
  - 二选一：删除/弃用该入口，或改成调用 `cs336_systems.flash_attention_benchmark.main()`。
  - 原因：避免用户运行根目录脚本时遇到不存在模块导入错误。

- Tests: `tests/test_attention.py`
  - 不强制修改官方测试。
  - 如需本地增强，可增加小尺寸 causal/non-causal 测试，但执行时仍以官方测试为最终依据。

## Task 1: 统一 backward helper 的接口与精度

**Files:**
- Modify: `cs336_systems/flash_attention_pytorch.py`
- Modify: `cs336_systems/flash_attention_triton.py`

- [ ] 确认 `_flash_attention_backward_impl(Q,K,V,O,dO,L)` 是唯一的公式实现。

改造原因：PDF 明确要求 backward 使用 `Q,K,V,O,dO,L`，计算 `D = rowsum(O * dO)`，再按 Eq. 13-19 返回 `dQ,dK,dV`。当前 PyTorch class 符合，但 Triton class 没有复用。

- [ ] 在 Triton autograd forward 中保存 `out` 和 `lse`。

目标形态：

```python
ctx.is_causal = is_causal
ctx.save_for_backward(lse, q, k, v, out)
return out
```

- [ ] 在 Triton autograd backward 中调用公共 helper。

目标形态：

```python
lse, q, k, v, out = ctx.saved_tensors
dq, dk, dv = _flash_attention_backward_impl(q, k, v, out, do, lse)
return dq, dk, dv, None
```

- [ ] 对 causal=True 的 Triton backward 做明确决策。

保守实现：给公共 helper 增加 `is_causal` 参数，并在重算 `S` 后应用同样 causal mask。原因是 benchmark 和 Triton 测试都会使用 causal=True；如果 helper 不 mask，causal backward 会错。

## Task 2: 修正 forward 的 mask 与 dtype 细节

**Files:**
- Modify: `cs336_systems/flash_attention_pytorch.py`
- Modify: `cs336_systems/flash_attention_triton.py`

- [ ] PyTorch forward 中将 `L` 存为 `float32`。

原因：`L` 是 logsumexp，低精度保存会影响 backward 重计算概率，尤其 bf16/fp32 benchmark 对比时不应提前丢精度。

目标形态：

```python
L = torch.empty((batch_size, n_queries), device=Q.device, dtype=torch.float32)
```

- [ ] 给 PyTorch forward 增加可选 causal mask。

原因：虽然 PDF part (a) 说可忽略 `is_causal`，但统一 reference 行为可以减少后续 benchmark/debug 分叉。

核心逻辑：

```python
if is_causal:
    q_idx = torch.arange(q_start, q_start + q_tile_size, device=Q.device)[:, None]
    k_idx = torch.arange(k_start, k_start + k_tile.shape[1], device=Q.device)[None, :]
    s_ij = torch.where(q_idx >= k_idx, s_ij, torch.full_like(s_ij, -1e6))
```

- [ ] Triton forward 对 key tile 越界列显式 mask。

原因：`boundary_check` 对 load 做 padding，但 softmax 仍可能把 padding 列当作合法 key。当前测试用整 tile 可能掩盖问题。

核心逻辑：

```python
k_rows = tl.arange(0, K_TILE_SIZE) + key_start
valid_k = k_rows < N_KEYS
s = tl.where(valid_k[None, :], s, -1e6)
```

- [ ] causal mask 与 valid key mask 组合使用。

原因：causal 和越界是两个独立约束，不能互相覆盖。

## Task 3: 收敛 benchmark 入口

**Files:**
- Modify: `cs336_systems/flash_attention_benchmark.py`
- Optional Modify: `benchmark_flash_attention.py`

- [ ] 修正 `cs336_systems/flash_attention_benchmark.py` 的导入。

当前问题：

```python
from systems_core.flash_attention_triton import get_flashattention_autograd_function_triton
```

目标：

```python
from cs336_systems.flash_attention_triton import get_flashattention_autograd_function_triton
```

- [ ] 保留 `triton.testing.do_bench` 的 forward/backward/end-to-end 三项计时。

原因：PDF 明确要求使用 `triton.testing.do_bench` 并报告 forward、backward、end-to-end forward-backward latency。

- [ ] 将每个 implementation 的异常捕获拆开。

原因：PyTorch attention 在大 seq 长度更容易 OOM；不能因为 PyTorch OOM 就把 Triton FlashAttention 也标记为 OOM。

目标结构：

```python
for impl_name, forward_fn in implementations:
    try:
        f_ms, b_ms, e2e_ms = _bench_impl(...)
        rows.append(ok_row)
    except torch.cuda.OutOfMemoryError as exc:
        rows.append(oom_row_for_this_impl_only)
        torch.cuda.empty_cache()
    except RuntimeError as exc:
        rows.append(runtime_error_row_for_this_impl_only)
        torch.cuda.empty_cache()
```

- [ ] 处理根目录 `benchmark_flash_attention.py`。

建议：改成薄 wrapper，直接调用包内 benchmark 的 `main()`。原因是用户自然会从根目录运行脚本，当前导入不存在模块会失败。

目标形态：

```python
from cs336_systems.flash_attention_benchmark import main

if __name__ == "__main__":
    main()
```

## Task 4: 验证计划

**Files:**
- Verify only; no required code file changes.

- [ ] 本地 CPU 可执行测试。

命令：

```bash
uv run pytest -q tests/test_attention.py -k "pytorch"
```

期望：PyTorch forward/backward 通过。

- [ ] 在 CUDA/H100 环境执行官方 attention 测试。

命令：

```bash
uv run pytest -q tests/test_attention.py -k "flash_forward_pass_triton or flash_backward_triton"
```

期望：causal 和 non-causal Triton forward/backward 均通过。

- [ ] 在 CUDA/H100 环境执行小规模 benchmark smoke test。

命令：

```bash
uv run python -m cs336_systems.flash_attention_benchmark --seq-lens 128 256 --d-models 16 32 --dtypes bfloat16 float32 --warmup-ms 20 --rep-ms 50
```

期望：生成 CSV/Markdown，至少小规模组合不报导入错误。

- [ ] 在 H100 上执行完整 sweep。

命令：

```bash
uv run python -m cs336_systems.flash_attention_benchmark
```

期望：输出包含 PyTorch 和 Triton FlashAttention 的 forward、backward、end-to-end latency 表；大尺寸 PyTorch OOM 时 Triton 结果仍独立记录。

## 非目标

- 不在本轮实现 Triton backward kernel。原因：PDF 的 `flash_backward` 明确允许 PyTorch+`torch.compile`，Triton backward 属于 leaderboard/optional 优化。
- 不重构无关 DDP、optimizer sharding 或 earlier benchmarking 脚本。
- 不把 `benchmark_attention.py` 改造成 flash benchmark；它已有 `pytorch_attention` 和 `torch_compile` 职责。

## 风险与后续深究

- 长序列下 PyTorch+compiled backward 仍会物化 `S/P`，内存随 sequence length 二次增长；这满足基础题但不适合 leaderboard。
- `torch.compile` 在 macOS 使用 `aot_eager`，在 CUDA 使用 `inductor`，两端性能和可编译行为可能不同。
- Triton tile size 当前是手写规则，完整 sweep 可能需要按 `D` 和 seq length 做 autotune。
- Causal benchmark 可进一步优化：跳过完全 masked key tiles，减少无效计算。

## 自检

- `flash_forward` 覆盖：PyTorch tiled forward、Triton fused forward、causal mask、保存 L。
- `flash_backward` 覆盖：使用 `Q,K,V,O,dO,L`、计算 `D`、返回 `dQ,dK,dV`、`torch.compile` helper。
- `flash_benchmarking` 覆盖：`triton.testing.do_bench`、batch=1、causal、seq 128-65536、d 16-128、bf16/fp32、forward/backward/e2e 表格。