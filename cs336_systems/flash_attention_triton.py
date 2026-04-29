from __future__ import annotations

import math
from typing import Type

import torch

from cs336_systems.flash_attention_pytorch import _flash_attention_backward_impl


def _attention_and_lse_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
):
    d = q.shape[-1]
    scale = 1.0 / math.sqrt(d)
    scores = torch.einsum("bqd,bkd->bqk", q, k) * scale
    if is_causal:
        n_queries = q.shape[-2]
        n_keys = k.shape[-2]
        q_idx = torch.arange(n_queries, device=q.device)[:, None]
        k_idx = torch.arange(n_keys, device=q.device)[None, :]
        scores = torch.where(q_idx >= k_idx, scores, torch.full_like(scores, -1e6))
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("bqk,bkd->bqd", probs, v)
    lse = torch.logsumexp(scores, dim=-1)
    return out, lse

def _build_triton_impl() -> Type[torch.autograd.Function]:
    import triton  # type: ignore[import-not-found]
    import triton.language as tl  # type: ignore[import-not-found]

    @triton.jit
    def flash_fwd_kernel(
        Q_ptr, K_ptr, V_ptr,
        O_ptr, L_ptr,
        stride_qb, stride_qq, stride_qd,
        stride_kb, stride_kk, stride_kd,
        stride_vb, stride_vk, stride_vd,
        stride_ob, stride_oq, stride_od,
        stride_lb, stride_lq,
        N_QUERIES, N_KEYS,
        scale,
        D: tl.constexpr,
        Q_TILE_SIZE: tl.constexpr,
        K_TILE_SIZE: tl.constexpr,
        is_causal: tl.constexpr,
    ):
        query_tile_index = tl.program_id(0)
        batch_index = tl.program_id(1)

        Q_block_ptr = tl.make_block_ptr(
            Q_ptr + batch_index * stride_qb,
            shape=(N_QUERIES, D),
            strides=(stride_qq, stride_qd),
            offsets=(query_tile_index * Q_TILE_SIZE, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )
        K_block_ptr = tl.make_block_ptr(
            K_ptr + batch_index * stride_kb,
            shape=(N_KEYS, D),
            strides=(stride_kk, stride_kd),
            offsets=(0, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )
        V_block_ptr = tl.make_block_ptr(
            V_ptr + batch_index * stride_vb,
            shape=(N_KEYS, D),
            strides=(stride_vk, stride_vd),
            offsets=(0, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )
        O_block_ptr = tl.make_block_ptr(
            O_ptr + batch_index * stride_ob,
            shape=(N_QUERIES, D),
            strides=(stride_oq, stride_od),
            offsets=(query_tile_index * Q_TILE_SIZE, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )

        q = tl.load(Q_block_ptr, boundary_check=(0, 1))
        m = tl.full((Q_TILE_SIZE,), -float("inf"), dtype=tl.float32)
        l = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
        o = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
        q_rows = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)

        for key_start in range(0, N_KEYS, K_TILE_SIZE):
            k = tl.load(K_block_ptr, boundary_check=(0, 1))
            v = tl.load(V_block_ptr, boundary_check=(0, 1))
            s = tl.dot(q, tl.trans(k)) * scale

            k_rows = tl.arange(0, K_TILE_SIZE) + key_start
            valid_k = k_rows < N_KEYS
            s = tl.where(valid_k[None, :], s, -1e6)

            if is_causal:
                causal_mask = q_rows[:, None] >= k_rows[None, :]
                s = tl.where(causal_mask, s, -1e6)

            m_new = tl.maximum(m, tl.max(s, axis=1))
            alpha = tl.exp(m - m_new)
            p = tl.exp(s - m_new[:, None])
            l = alpha * l + tl.sum(p, axis=1)
            o = o * alpha[:, None]
            o = tl.dot(p.to(v.dtype), v, acc=o)
            m = m_new

            K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
            V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))

        o = o / l[:, None]
        lse = m + tl.log(l)

        tl.store(O_block_ptr, o.to(O_block_ptr.type.element_ty), boundary_check=(0, 1))
        l_ptrs = L_ptr + batch_index * stride_lb + q_rows * stride_lq
        q_mask = q_rows < N_QUERIES
        tl.store(l_ptrs, lse, mask=q_mask)

    class FlashAttentionAutogradFunctionTriton(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            is_causal: bool = False,
        ):
            if q.device.type != "cuda":
                out, lse = _attention_and_lse_torch(q, k, v, is_causal)
                ctx.is_causal = is_causal
                ctx.save_for_backward(lse, q, k, v, out)
                return out

            q = q.contiguous()
            k = k.contiguous()
            v = v.contiguous()
            batch_size, n_queries, d = q.shape
            n_keys = k.shape[1]
            out = torch.empty((batch_size, n_queries, d), device=q.device, dtype=q.dtype)
            lse = torch.empty((batch_size, n_queries), device=q.device, dtype=torch.float32)
            scale = 1.0 / math.sqrt(d)

            if n_queries >= 4096:
                q_tile_size = 128
                k_tile_size = 128
            elif n_queries >= 1024:
                q_tile_size = 128
                k_tile_size = 64
            else:
                q_tile_size = 64
                k_tile_size = 64

            grid = (triton.cdiv(n_queries, q_tile_size), batch_size)
            flash_fwd_kernel[grid](
                q, k, v, out, lse,
                q.stride(0), q.stride(1), q.stride(2),
                k.stride(0), k.stride(1), k.stride(2),
                v.stride(0), v.stride(1), v.stride(2),
                out.stride(0), out.stride(1), out.stride(2),
                lse.stride(0), lse.stride(1),
                n_queries, n_keys,
                scale,
                D=d,
                Q_TILE_SIZE=q_tile_size,
                K_TILE_SIZE=k_tile_size,
                is_causal=is_causal,
            )
            ctx.is_causal = is_causal
            ctx.save_for_backward(lse, q, k, v, out)
            return out

        @staticmethod
        def backward(ctx, do: torch.Tensor):
            lse, q, k, v, out = ctx.saved_tensors
            dq, dk, dv = _flash_attention_backward_impl(q, k, v, out, do, lse, ctx.is_causal)
            return dq, dk, dv, None

    return FlashAttentionAutogradFunctionTriton

def get_flashattention_autograd_function_triton() -> Type[torch.autograd.Function]:
    return _build_triton_impl()

try:
    FlashAttentionAutogradFunctionTriton = get_flashattention_autograd_function_triton()
except (ModuleNotFoundError, ImportError):
    import warnings
    warnings.warn(
        "Triton is not available (Apple M4 Pro / macOS does not support Triton). "
        "Falling back to pure PyTorch implementation for FlashAttentionAutogradFunctionTriton.",
        RuntimeWarning,
        stacklevel=2,
    )

    class FlashAttentionAutogradFunctionTriton(torch.autograd.Function):  # type: ignore[no-redef]
        @staticmethod
        def forward(ctx, q, k, v, is_causal=False):
            out, lse = _attention_and_lse_torch(q, k, v, is_causal)
            ctx.is_causal = is_causal
            ctx.save_for_backward(lse, q, k, v, out)
            return out

        @staticmethod
        def backward(ctx, do):
            lse, q, k, v, out = ctx.saved_tensors
            dq, dk, dv = _flash_attention_backward_impl(q, k, v, out, do, lse, ctx.is_causal)
            return dq, dk, dv, None
