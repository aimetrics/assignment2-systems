---
name: flash-attention-plan
overview: 为 CS336 Assignment 2 的 flash_forward、flash_backward、flash_benchmarking 设计实现路径：新增 FlashAttention-2 模块、连接测试适配器，并补充 benchmark 脚本。
todos:
  - id: add-flash-module
    content: 新增 cs336_systems/flash_attention.py，包含 PyTorch tiled forward、Triton forward kernel 和 shared backward helper
    status: completed
  - id: wire-adapters
    content: 修改 tests/adapters.py 的两个 FlashAttention adapter，返回新增 autograd.Function 类
    status: completed
  - id: add-benchmark
    content: 新增 benchmark_flash_attention.py，用 triton.testing.do_bench 生成 FlashAttention 对比结果
    status: completed
  - id: verify-tests
    content: 运行 flash_forward、flash_backward 相关 pytest，并用小规模 benchmark 做 smoke test
    status: completed
isProject: false
---

# FlashAttention-2 实现方案

## 目标理解

作业这三题围绕一个逐步递进的目标：先用 PyTorch 写出可调试的 FlashAttention-2 forward，再把 forward 融合成 Triton kernel，backward 则用 PyTorch + `torch.compile` 按重计算公式实现，最后用 `triton.testing.do_bench` 对比普通 PyTorch attention 与 Triton FlashAttention 的 forward、backward、forward+backward 延迟。

关键要求来自 PDF：

- `flash_forward`：`autograd.Function.forward(ctx, Q, K, V, is_causal=False)`，返回 `O`，保存 `L, Q, K, V, O` 给 backward；PyTorch 版本先不需要 Triton；Triton 版本 launch grid 为 `(Tq, batch_size)`，每个 program 处理一个 batch 的一个 query tile；支持 optional causal mask，Triton 参数 `is_causal: tl.constexpr`。
- `flash_backward`：backward 不写 Triton，用 PyTorch + `torch.compile`；输入使用 `Q, K, V, O, dO, L`，先算 `D = rowsum(O * dO)`，再按 Eq. 13-19 重计算 `P = exp(S - L)` 并得到 `dQ, dK, dV`。
- `flash_benchmarking`：新增 benchmark 脚本，用 `triton.testing.do_bench` 比较 Triton FlashAttention 与普通 PyTorch attention，在 H100、batch size 1、causal mask 下 sweep `seq_len=128..65536`、`d=16..128`、`dtype in {bf16, fp32}`，报告 forward/backward/e2e latency。

## 推荐实现路径

我建议采用「独立模块 + 适配器导出 + 独立 benchmark」方案：

- 新增 [`cs336_systems/flash_attention.py](cs336_systems/flash_attention.py)`：集中放 PyTorch forward、Triton kernel、两个 `torch.autograd.Function` 类、compiled backward helper。
- 修改 [`tests/adapters.py](tests/adapters.py)`：让 `get_flashattention_autograd_function_pytorch()` 和 `get_flashattention_autograd_function_triton()` 返回新增类。
- 新增 [`benchmark_flash_attention.py](benchmark_flash_attention.py)`：专门完成 `flash_benchmarking`，避免继续扩大已有 [`benchmark_attention.py](benchmark_attention.py)`，后者已经覆盖 `pytorch_attention` 和 `torch_compile`。

备选方案有两个，但不推荐：

- 把 FlashAttention 加进 [`cs336_systems/weightedsum.py](cs336_systems/weightedsum.py)`：能复用 Triton 风格示例，但文件职责会混乱，且当前文件已有用户修改。
- 把 benchmark 合并进 [`benchmark_attention.py](benchmark_attention.py)`：复用参数解析较多，但该脚本已有 PyTorch/compile 任务逻辑，FlashAttention sweep 和 `do_bench` 生命周期不同，单独脚本更清晰。

## 核心算法设计

PyTorch forward 用和 Triton 一致的 tiled online softmax，作为可读、可调试参考：

1. 输入约定为 `Q, K, V` 形状 `(..., seq, d)`，测试中是 `(batch, seq, d)`；实现时先展平除最后两维外的 batch 前缀，变成 `(B, N, D)`，输出再 reshape 回去。
2. 对每个 query tile 初始化：`m=-inf`、`l=0`、`acc=0`。
3. 遍历 key/value tile，计算 `S = Q_i K_j^T / sqrt(D)`，causal 时对 `q_index < k_index` 写 `-1e6`。
4. 在线更新：`m_new=max(m,rowmax(S))`、`P_tilde=exp(S-m_new)`、`l_new=exp(m-m_new)*l + rowsum(P_tilde)`、`acc_new=exp(m-m_new)*acc + P_tilde @ V_j`。
5. 写 `O=acc/l` 和 `L=m+log(l)`。

Triton forward 复刻同一算法：

- kernel 签名按作业给出的 `flash_fwd_kernel(Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr, strides..., N_QUERIES, N_KEYS, scale, D, Q_TILE_SIZE, K_TILE_SIZE, is_causal)` 扩展。
- 使用 `tl.make_block_ptr` 读取 `Q_i`、循环读取 `K_j,V_j`，`tl.dot` 计算 scores 和 partial output。
- on-chip `m/l/acc` 全部用 `tl.float32`；`P_tilde` 在乘 `V` 前 cast 到 `V` dtype；写 `O` 前 cast 到输出 dtype，`L` 保持 fp32。
- 初始 tile size 保守选择 `Q_TILE_SIZE=16`、`K_TILE_SIZE=32` 或 `64`，满足作业至少 `16x16`，后续 benchmark 再按 `D` 调参。

Backward 用重计算公式，不保存 attention matrix：

- `D_vec = (O * dO).sum(dim=-1)`。
- `S = Q @ K.transpose(-2, -1) / sqrt(d)`，causal 时同样用 `-1e6` mask。
- `P = exp(S - L[..., None])`，不用 `torch.softmax`。
- `dV = P.transpose(-2, -1) @ dO`。
- `dP = dO @ V.transpose(-2, -1)`。
- `dS = P * (dP - D_vec[..., None])`。
- `dQ = dS @ K / sqrt(d)`，`dK = dS.transpose(-2, -1) @ Q / sqrt(d)`。

## 具体实施步骤

1. 在 [`cs336_systems/flash_attention.py](cs336_systems/flash_attention.py)` 中先写 PyTorch reference/tiled forward 和 `FlashAttentionPytorchFunction`，确保 `ctx.save_for_backward(L, Q, K, V, O)`，`ctx.is_causal = is_causal`。
2. 给同一模块加入 compiled backward helper，并让 PyTorch Function 的 `backward()` 返回 `dQ, dK, dV, None`。
3. 实现 `flash_fwd_kernel` 和 `FlashAttentionTritonFunction.forward()`，分配 `O=torch.empty_like(Q)`、`L=torch.empty(..., dtype=torch.float32)`，保存同样的 tensors。
4. 让 `FlashAttentionTritonFunction.backward()` 复用同一个 compiled backward helper，这正符合题目“forward Triton、backward PyTorch+compile”的要求。
5. 修改 [`tests/adapters.py](tests/adapters.py)` 两个 FlashAttention adapter，导入并返回新增类。
6. 新增 [`benchmark_flash_attention.py](benchmark_flash_attention.py)`：提供 CLI 参数、普通 PyTorch attention baseline、FlashAttention Triton variant、`do_bench` 的 forward/backward/e2e 三个闭包、OOM 捕获、CSV/Markdown 输出。
7. 验证顺序：
   - `uv run pytest -k test_flash_forward_pass_pytorch`
   - `uv run pytest -k test_flash_forward_pass_triton`
   - `uv run pytest -k test_flash_backward`
   - GPU 上用小 sweep 先 smoke test benchmark，例如 `--seq-lens 128 256 --d-models 16 64 --dtypes bf16 --warmup 10 --rep 50`。

## 风险与注意点

- `tests/test_attention.py` 会从 `o.grad_fn.saved_tensors` 中查找恰好一个形状为 `(batch, n_queries)` 的 tensor，所以保存 `L` 的形状必须精确，不能额外保存同形状临时量。
- 测试输入是 power-of-2 且至少 16，但 benchmark 要到 65536，普通 PyTorch baseline 很容易 OOM，benchmark 应记录 `oom` 而不是中断整个 sweep。
- causal mask 要在 PyTorch forward、Triton forward、backward 重计算三处保持一致，全部使用作业指定的 `-1e6` 而不是 `-inf`。
- `torch.compile` 在 CPU 上可能有额外开销，但测试规模小；如遇环境问题，可以把 compiled helper 包装成 lazy compile，并保留同一公式的直接 fallback，同时保证 CUDA 路径使用 compiled helper。