"""
weighted_sum.py
===============
Triton 实现的加权求和（matrix-vector product）算子。

数学定义：
    y_i = sum_j(w_j * X_ij)    即 y = X @ w

包含：
    - weighted_sum_fwd         Triton forward kernel
    - weighted_sum_backward    Triton backward kernel
    - WeightedSumFunc          torch.autograd.Function 包装
    - f_weightedsum            对外暴露的函数接口
"""

import torch
import triton
import triton.language as tl
from einops import rearrange


# ─────────────────────────────────────────────
#  工具函数
# ─────────────────────────────────────────────

def cdiv(a, b):
    """ceiling division"""
    return (a + b - 1) // b


# ─────────────────────────────────────────────
#  Forward Kernel
# ─────────────────────────────────────────────

@triton.jit
def weighted_sum_fwd(
    x_ptr, weight_ptr,          # 输入指针
    output_ptr,                 # 输出指针
    x_stride_row,               # X 行步长，通常 = D. D是列的维度.
    x_stride_dim,               # X 列步长，通常 = 1
    weight_stride_dim,          # w 步长，通常 = 1
    output_stride_row,          # y 步长，通常 = 1
    ROWS, D,                    # 运行时维度
    ROWS_TILE_SIZE: tl.constexpr,
    D_TILE_SIZE:    tl.constexpr,
):
    """
    每个 program instance 负责 ROWS_TILE_SIZE 行。
    内层循环沿 D 方向分 tile 遍历，累加到寄存器 output，最后写回 HBM。

    为什么 tile size 必须是 tl.constexpr：Triton 在编译 kernel 时需要静态决定寄存器分配大小（tl.zeros((ROWS_TILE_SIZE,), ...) 的大小必须编译期已知），所以 tile size 不能是运行时变量。
    """
    # Each instance will compute the weighted sum of a tile of rows of x
    # `tl.program_id` gives us a way to check which thread block we're running in
    row_tile_idx = tl.program_id(0)

    # ── Block Pointers ──────────────────────────────────────
    # Block pointers give us a way to select from an ND region of memory and move our selection around.
    # The block pointer must know:
    # - The pointer to the first element of the tensor, x_ptr
    # - The overall shape of the tensor to handle out-of-bounds access 
    # - The strides of each dimension to use the memory layout properly
    # - The ND coordinates of the starting block, i.e., "offsets"
    # - The block shape to use load/store at a time
    # - The order of the dimensions in memory from major to minor. axes (= np.argsort(strides)) for optimizations, especially useful on H100
    # Block Pointers机制: tl.make_block_ptr 是 Triton 的高级抽象，替代手动指针运算
    x_block_ptr = tl.make_block_ptr(
        x_ptr,
        shape=(ROWS, D),                            # x 的总形状（用于边界检查）
        strides=(x_stride_row, x_stride_dim),       # 步长（支持非 contiguous）
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0), # 当前 block 的起始坐标
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),  # 每次 load 的 tile 大小
        order=(1, 0),                               # 内存布局：dim1 是最 major（row-major）
    )
    weight_block_ptr = tl.make_block_ptr(
        weight_ptr,
        shape=(D,),
        strides=(weight_stride_dim,),
        offsets=(0,),
        block_shape=(D_TILE_SIZE,),
        order=(0,),
    )
    output_block_ptr = tl.make_block_ptr(
        output_ptr,
        shape=(ROWS,),
        strides=(output_stride_row,),
        offsets=(row_tile_idx * ROWS_TILE_SIZE,),
        block_shape=(ROWS_TILE_SIZE,),
        order=(0,),
    )

    # ── 累加器（寄存器，fp32 防精度损失）──────────────────────
    # Initialize a buffer to write to
    # NOTE:
    # 1. 在寄存器中分配一个长度 ROWS_TILE_SIZE 的 fp32 向量（不是 HBM，不是 shared memory）。
    #    这是 Triton 中 on-chip 计算的核心：累加发生在寄存器里，避免反复读写 HBM。
    # 2. dtype 选择：即使 X 和 w 是 fp16，累加器也用 fp32——经典的「compute in low precision, accumulate in high precision」，
    #    防止 1000 次小数累加误差爆炸
    output = tl.zeros((ROWS_TILE_SIZE,), dtype=tl.float32)

    # ── 沿 D 方向 tile 循环 ──────────────────────────────────
    for _ in range(tl.cdiv(D, D_TILE_SIZE)):
        # Load the current block pointer
        # Since ROWS_TILE_SIZE might not divide ROWS, and D_TILE_SIZE might not divide D
        # we need boundary checks for both dimensions. boundary_check=(0,1) 处理 ROWS / D 不整除 tile size 的情况，越界自动 pad 0
        # shape: (ROWS_TILE_SIZE, D_TILE_SIZE)
        row    = tl.load(x_block_ptr,      boundary_check=(0, 1), padding_option="zero")
        # shape: (D_TILE_SIZE,)
        weight = tl.load(weight_block_ptr, boundary_check=(0,),   padding_option="zero")
        # Compute the weighted sum of the row
        # weight[None,:] broadcast: (1, D_TILE_SIZE)
        # row * weight[None,:]:     (ROWS_TILE_SIZE, D_TILE_SIZE)
        # tl.sum(..., axis=1):      (ROWS_TILE_SIZE,)  ← 在 D 维度上 reduce
        output += tl.sum(row * weight[None, :], axis=1)
        # Move the pointers to the next tile
        # These are (rows, columns) coordinate deltas
        x_block_ptr      = x_block_ptr.advance((0, D_TILE_SIZE))        # 行方向不动、列方向前进 D_TILE_SIZE 
        weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE,))     # Move by D_TILE_SIZE

    # ── 写回 HBM ─────────────────────────────────────────────
    # Write output to the output block pointer (a single scalar per row)
    # Since ROWS_TILE_SIZE might not divide ROWS, we need boundary checks
    tl.store(output_block_ptr, output, boundary_check=(0,))


# ─────────────────────────────────────────────
#  Backward Kernel
# ─────────────────────────────────────────────

@triton.jit
def weighted_sum_backward(
    x_ptr, weight_ptr,              # 前向输入（backward 需要用到）
    grad_output_ptr,                # ∇y L，shape (NUM_ROWS,)
    grad_x_ptr,                     # 输出：∇X L，shape (NUM_ROWS, D)
    partial_grad_weight_ptr,        # 输出：∇w 局部结果，shape (n_row_tiles, D)
    stride_xr,  stride_xd,
    stride_wd,
    stride_gr,
    stride_gxr, stride_gxd,
    stride_gwb, stride_gwd,
    NUM_ROWS, D,
    ROWS_TILE_SIZE: tl.constexpr,
    D_TILE_SIZE:    tl.constexpr,
):
    """
    每个 program instance 负责一个行 tile。

    计算：
        ∇X[i,j]  = w[j] * ∇y[i]                （外积）
        ∇w[j]    = sum_i(X[i,j] * ∇y[i])        （加权列求和，分两阶段）

    ∇w 采用 partial reduction：
        kernel 内每个 instance 写 partial_grad_weight[row_tile_idx, :]
        Python 层再 torch.sum(axis=0) 完成全局 reduce
    """
    row_tile_idx = tl.program_id(0)
    n_row_tiles  = tl.num_programs(0)

    # ── Block Pointers（输入）────────────────────────────────
    grad_output_block_ptr = tl.make_block_ptr(
        grad_output_ptr,
        shape=(NUM_ROWS,), strides=(stride_gr,),
        offsets=(row_tile_idx * ROWS_TILE_SIZE,),
        block_shape=(ROWS_TILE_SIZE,),
        order=(0,),
    )
    x_block_ptr = tl.make_block_ptr(
        x_ptr,
        shape=(NUM_ROWS, D), strides=(stride_xr, stride_xd),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )
    weight_block_ptr = tl.make_block_ptr(
        weight_ptr,
        shape=(D,), strides=(stride_wd,),
        offsets=(0,),
        block_shape=(D_TILE_SIZE,),
        order=(0,),
    )

    # ── Block Pointers（输出）────────────────────────────────
    grad_x_block_ptr = tl.make_block_ptr(
        grad_x_ptr,
        shape=(NUM_ROWS, D), strides=(stride_gxr, stride_gxd),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )
    # partial_grad_weight：每个 instance 写第 row_tile_idx 行的一段
    partial_grad_weight_block_ptr = tl.make_block_ptr(
        partial_grad_weight_ptr,
        shape=(n_row_tiles, D), strides=(stride_gwb, stride_gwd),
        offsets=(row_tile_idx, 0),
        block_shape=(1, D_TILE_SIZE),
        order=(1, 0),
    )

    # ── 沿 D 方向 tile 循环 ──────────────────────────────────
    for _ in range(tl.cdiv(D, D_TILE_SIZE)):
        grad_output = tl.load(grad_output_block_ptr,
                              boundary_check=(0,), padding_option="zero")
        # shape: (ROWS_TILE_SIZE,)

        # ① 计算 ∇X tile：外积
        # shape: (D_TILE_SIZE,)
        weight      = tl.load(weight_block_ptr,
                               boundary_check=(0,), padding_option="zero")
        # grad_output[:, None] : (ROWS_TILE_SIZE, 1)
        # weight[None, :]      : (1, D_TILE_SIZE)
        # grad_x_row           : (ROWS_TILE_SIZE, D_TILE_SIZE)  ← 外积
        grad_x_row  = grad_output[:, None] * weight[None, :]
        tl.store(grad_x_block_ptr, grad_x_row, boundary_check=(0, 1))

        # ② 计算 ∇w 的局部贡献：在当前行 tile 内 reduce
        # shape: (ROWS_TILE_SIZE, D_TILE_SIZE)
        row             = tl.load(x_block_ptr,
                                   boundary_check=(0, 1), padding_option="zero")
        grad_weight_row = tl.sum(row * grad_output[:, None], axis=0, keep_dims=True)
        # row * grad_output[:, None] : (ROWS_TILE_SIZE, D_TILE_SIZE)
        # tl.sum(..., axis=0)        : (1, D_TILE_SIZE)   ← 对行 reduce
        tl.store(partial_grad_weight_block_ptr, grad_weight_row, boundary_check=(1,))

        # ── advance 所有 block pointer ────────────────────────
        x_block_ptr                   = x_block_ptr.advance((0, D_TILE_SIZE))
        weight_block_ptr              = weight_block_ptr.advance((D_TILE_SIZE,))
        partial_grad_weight_block_ptr = partial_grad_weight_block_ptr.advance((0, D_TILE_SIZE))
        grad_x_block_ptr              = grad_x_block_ptr.advance((0, D_TILE_SIZE))
        # grad_output_block_ptr 只有行维度，不随列循环 advance


# ─────────────────────────────────────────────
#  autograd.Function 包装
# ─────────────────────────────────────────────

class WeightedSumFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight):
        """
        参数
        ----
        ctx    : autograd 上下文对象，用于在 forward/backward 之间传递状态
        x      : (..., D)  任意前缀 batch 维
        weight : (D,)

        返回
        ----
        y : (...)  去掉最后一维的结果
        """
        D           = x.shape[-1]
        output_dims = x.shape[:-1]

        # 把任意前缀 batch 维拍平成 2D，简化 kernel
        input_shape = x.shape
        x = rearrange(x, "... d -> (...) d")   # (ROWS, D)

        # ── 校验 ────────────────────────────────────────────
        assert len(weight.shape) == 1 and weight.shape[0] == D, \
            f"weight shape {weight.shape} 与 D={D} 不匹配"
        assert x.is_cuda and weight.is_cuda, \
            "x 和 weight 必须在 CUDA 设备上"
        assert x.is_contiguous(), \
            "x 必须是 contiguous tensor，请先调用 .contiguous()"

        # ── 保存反向传播所需张量 ─────────────────────────────
        ctx.save_for_backward(x, weight)

        # ── Tile size（编译期常量）───────────────────────────
        ctx.D_TILE_SIZE    = triton.next_power_of_2(D) // 16        # Roughly 16 loops through the embedding dimension
        ctx.D_TILE_SIZE    = max(ctx.D_TILE_SIZE, 1)                # 至少为 1
        ctx.ROWS_TILE_SIZE = 16                                     # Each thread processes 16 batch elements at a time
        ctx.input_shape    = input_shape

        # ── 分配输出 ─────────────────────────────────────────
        n_rows = x.shape[0]
        # Need to initialize empty result tensor. Note that these elements are not necessarily 0!
        # 不初始化内存，节省一次写操作
        y = torch.empty(n_rows, device=x.device, dtype=torch.float32)

        # ── 调用 Triton kernel ───────────────────────────────
        # Launch our kernel with n instances in our 1D grid
        # when we invoke the Triton kernel with weighted_sum_fwd[(cdiv(n_rows, ctx.ROWS_TILE_SIZE),)], 
        # we define a so-called “launch grid” of thread blocks by passing the tuple (cdiv(n_rows, ctx.ROWS_TILE_SIZE),). 
        # Then, we can access the thread block index with tl.program_id(0) in our kernel
        weighted_sum_fwd[(cdiv(n_rows, ctx.ROWS_TILE_SIZE),)](
            x, weight, y,
            x.stride(0), x.stride(1),
            weight.stride(0),
            y.stride(0),
            ROWS=n_rows, D=D,
            ROWS_TILE_SIZE=ctx.ROWS_TILE_SIZE,
            D_TILE_SIZE=ctx.D_TILE_SIZE,
        )

        # 恢复原始 batch 形状
        return y.view(input_shape[:-1])

    @staticmethod
    def backward(ctx, grad_out):
        """
        参数
        ----
        grad_out : (...)  即 ∇y L，和 forward 输出形状相同

        返回
        ----
        grad_x      : (..., D)  ∇X L
        grad_weight : (D,)      ∇w L
        """
        x, weight      = ctx.saved_tensors          # x: (n_rows, D)
        ROWS_TILE_SIZE = ctx.ROWS_TILE_SIZE
        D_TILE_SIZE    = ctx.D_TILE_SIZE
        n_rows, D      = x.shape

        # grad_out 可能是多维的，拍平成 1D 和 x 对齐
        grad_out = grad_out.reshape(-1).contiguous()  # (n_rows,)

        # ── 分配输出 ─────────────────────────────────────────
        # partial_grad_weight : (n_row_tiles, D)
        #                        ↑ 每个 program instance 写一行
        partial_grad_weight = torch.empty(
            (cdiv(n_rows, ROWS_TILE_SIZE), D),
            device=x.device, dtype=x.dtype,
        )
        # grad_x              : (n_rows, D)   和 x 完全相同的形状
        grad_x = torch.empty_like(x)                  # (n_rows, D)

        # ── 调用 Triton kernel ───────────────────────────────
        weighted_sum_backward[(cdiv(n_rows, ROWS_TILE_SIZE),)](
            x, weight,
            grad_out,
            grad_x, partial_grad_weight,
            x.stride(0),                    x.stride(1),
            weight.stride(0),
            grad_out.stride(0),
            grad_x.stride(0),               grad_x.stride(1),
            partial_grad_weight.stride(0),  partial_grad_weight.stride(1),
            NUM_ROWS=n_rows, D=D,
            ROWS_TILE_SIZE=ROWS_TILE_SIZE,
            D_TILE_SIZE=D_TILE_SIZE,
        )

        # ── 全局 reduce：把各行 tile 的局部 ∇w 加起来 ────────
        grad_weight = partial_grad_weight.sum(axis=0)  # (n_row_tiles, D) → (D,)

        # 恢复 grad_x 的原始 batch 形状
        grad_x = grad_x.view(ctx.input_shape)

        return grad_x, grad_weight


# 对外暴露的函数接口，使用方式和 torch.nn.functional 一致
f_weightedsum = WeightedSumFunc.apply


# ─────────────────────────────────────────────
#  测试
# ─────────────────────────────────────────────

def ref(x, weight):
    """PyTorch 参考实现，用于对比验证"""
    return (x * weight).sum(dim=-1)


def test_forward_basic():
    """正确性：基本 case，整除 tile size"""
    print("=" * 50)
    print("测试 1：基本正确性（整除 tile size）")
    cases = [(16, 64), (32, 128), (64, 256)]
    for ROWS, D in cases:
        torch.manual_seed(0)
        x      = torch.randn(ROWS, D,  device="cuda", dtype=torch.float32)
        weight = torch.randn(D,        device="cuda", dtype=torch.float32)

        out_triton = f_weightedsum(x, weight)
        out_ref    = ref(x, weight)

        max_err = (out_triton - out_ref).abs().max().item()
        status  = "PASS" if max_err < 1e-4 else "FAIL"
        print(f"  [{status}] ROWS={ROWS:4d}, D={D:4d}  max_err={max_err:.2e}")


def test_forward_non_divisible():
    """边界：不整除 tile size"""
    print("=" * 50)
    print("测试 2：边界 case（不整除 tile size）")
    cases = [(17, 65), (1, 64), (64, 1), (1, 1), (100, 100)]
    for ROWS, D in cases:
        torch.manual_seed(42)
        x      = torch.randn(ROWS, D, device="cuda", dtype=torch.float32)
        weight = torch.randn(D,       device="cuda", dtype=torch.float32)

        out_triton = f_weightedsum(x, weight)
        out_ref    = ref(x, weight)

        max_err = (out_triton - out_ref).abs().max().item()
        status  = "PASS" if max_err < 1e-4 else "FAIL"
        print(f"  [{status}] ROWS={ROWS:4d}, D={D:4d}  max_err={max_err:.2e}")


def test_forward_batched():
    """高维 batch 输入"""
    print("=" * 50)
    print("测试 3：高维 batch 输入 (B, T, D)")
    B, T, D = 4, 32, 128
    torch.manual_seed(0)
    x      = torch.randn(B, T, D, device="cuda", dtype=torch.float32)
    weight = torch.randn(D,       device="cuda", dtype=torch.float32)

    out_triton = f_weightedsum(x, weight)   # 期望输出 (B, T)
    out_ref    = ref(x, weight)

    shape_ok = out_triton.shape == (B, T)
    max_err  = (out_triton - out_ref).abs().max().item()
    status   = "PASS" if (shape_ok and max_err < 1e-4) else "FAIL"
    print(f"  [{status}] 输入 {tuple(x.shape)} → 输出 {tuple(out_triton.shape)}  max_err={max_err:.2e}")


def test_gradcheck():
    """梯度正确性：用数值微分验证 backward"""
    print("=" * 50)
    print("测试 4：gradcheck（数值微分验证梯度）")
    torch.manual_seed(0)
    ROWS, D = 8, 16   # gradcheck 很慢，用小尺寸
    x = torch.randn(ROWS, D, device="cuda", dtype=torch.float64, requires_grad=True)
    w = torch.randn(D,       device="cuda", dtype=torch.float64, requires_grad=True)

    passed = torch.autograd.gradcheck(
        f_weightedsum, (x, w),
        eps=1e-6, atol=1e-4, rtol=1e-3,
    )
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] gradcheck ROWS={ROWS}, D={D}")


def test_backward_values():
    """验证梯度数值和参考实现一致"""
    print("=" * 50)
    print("测试 5：backward 梯度数值对比")
    torch.manual_seed(0)
    ROWS, D = 32, 64

    # ── Triton 实现 ──────────────────────────────────────────
    x_t = torch.randn(ROWS, D, device="cuda", requires_grad=True)
    w_t = torch.randn(D,       device="cuda", requires_grad=True)
    f_weightedsum(x_t, w_t).sum().backward()
    grad_x_triton = x_t.grad.clone()
    grad_w_triton = w_t.grad.clone()

    # ── PyTorch 参考 ─────────────────────────────────────────
    x_r = x_t.detach().clone().requires_grad_(True)
    w_r = w_t.detach().clone().requires_grad_(True)
    ref(x_r, w_r).sum().backward()
    grad_x_ref = x_r.grad.clone()
    grad_w_ref = w_r.grad.clone()

    err_x = (grad_x_triton - grad_x_ref).abs().max().item()
    err_w = (grad_w_triton - grad_w_ref).abs().max().item()
    status = "PASS" if (err_x < 1e-4 and err_w < 1e-4) else "FAIL"
    print(f"  [{status}] ∇X max_err={err_x:.2e}  ∇w max_err={err_w:.2e}")


def test_performance():
    """性能：确认 kernel 在 GPU 上正常运行并计时"""
    print("=" * 50)
    print("测试 6：性能（前向 100 次平均耗时）")
    ROWS, D = 4096, 1024
    x      = torch.randn(ROWS, D, device="cuda", dtype=torch.float32)
    weight = torch.randn(D,       device="cuda", dtype=torch.float32)

    # 热身
    for _ in range(5):
        f_weightedsum(x, weight)
    torch.cuda.synchronize()

    import time
    t0 = time.perf_counter()
    for _ in range(100):
        f_weightedsum(x, weight)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    ms = (t1 - t0) / 100 * 1000
    print(f"  [INFO] ROWS={ROWS}, D={D}  平均 {ms:.3f} ms/call")


# ─────────────────────────────────────────────
#  main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("❌ 未检测到 CUDA，请在有 GPU 的环境下运行")
        exit(1)

    print(f"设备：{torch.cuda.get_device_name(0)}")
    print()

    test_forward_basic()
    test_forward_non_divisible()
    test_forward_batched()
    test_backward_values()
    test_gradcheck()
    test_performance()

    print()
    print("=" * 50)
    print("全部测试完成")