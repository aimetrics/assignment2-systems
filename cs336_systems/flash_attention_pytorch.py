from __future__ import annotations

import math
import platform

import torch

_compile_backend = "aot_eager" if platform.system() == "Darwin" else "inductor"


@torch.compile(backend=_compile_backend)
def _flash_attention_backward_impl(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    O: torch.Tensor,
    dO: torch.Tensor,
    L: torch.Tensor,
    is_causal: bool = False,
):
    d_model = Q.shape[-1]
    scale = 1.0 / math.sqrt(d_model)

    q_fp32 = Q.to(torch.float32)
    k_fp32 = K.to(torch.float32)
    v_fp32 = V.to(torch.float32)
    o_fp32 = O.to(torch.float32)
    do_fp32 = dO.to(torch.float32)
    l_fp32 = L.to(torch.float32)

    S = torch.einsum("bqd,bkd->bqk", q_fp32, k_fp32) * scale
    if is_causal:
        n_queries = Q.shape[-2]
        n_keys = K.shape[-2]
        q_idx = torch.arange(n_queries, device=Q.device)[:, None]
        k_idx = torch.arange(n_keys, device=Q.device)[None, :]
        S = torch.where(q_idx >= k_idx, S, torch.full_like(S, -1e6))
    P = torch.exp(S - l_fp32.unsqueeze(-1))

    dV = torch.einsum("bqk,bqd->bkd", P, do_fp32)
    dP = torch.einsum("bqd,bkd->bqk", do_fp32, v_fp32)

    D = torch.sum(do_fp32 * o_fp32, dim=-1, keepdim=True)
    dS = P * (dP - D)

    dQ = torch.einsum("bqk,bkd->bqd", dS, k_fp32) * scale
    dK = torch.einsum("bqk,bqd->bkd", dS, q_fp32) * scale

    return dQ.to(Q.dtype), dK.to(K.dtype), dV.to(V.dtype)


class FlashAttention2PyTorchFunction(torch.autograd.Function):
    """Pure PyTorch FlashAttention-2 forward pass (Section 1.3.2(a))."""

    TILE_Q = 32
    TILE_K = 32

    @staticmethod
    def forward(ctx, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
        batch_size, n_queries, d_model = Q.shape
        n_keys = K.shape[1]
        scale = 1.0 / math.sqrt(d_model)

        O = torch.empty_like(Q)
        L = torch.empty((batch_size, n_queries), device=Q.device, dtype=torch.float32)

        for q_start in range(0, n_queries, FlashAttention2PyTorchFunction.TILE_Q):
            q_end = q_start + FlashAttention2PyTorchFunction.TILE_Q
            q_tile = Q[:, q_start:q_end, :]
            q_tile_size = q_tile.shape[1]

            # Running online-softmax state for this query tile.
            m_i = torch.full((batch_size, q_tile_size), -float("inf"), device=Q.device, dtype=torch.float32)
            l_i = torch.zeros((batch_size, q_tile_size), device=Q.device, dtype=torch.float32)
            o_i = torch.zeros((batch_size, q_tile_size, d_model), device=Q.device, dtype=torch.float32)

            for k_start in range(0, n_keys, FlashAttention2PyTorchFunction.TILE_K):
                k_end = k_start + FlashAttention2PyTorchFunction.TILE_K
                k_tile = K[:, k_start:k_end, :]
                v_tile = V[:, k_start:k_end, :]

                s_ij = torch.einsum("bqd,bkd->bqk", q_tile, k_tile).to(torch.float32) * scale
                if is_causal:
                    q_idx = torch.arange(q_start, q_start + q_tile_size, device=Q.device)[:, None]
                    k_idx = torch.arange(k_start, k_start + k_tile.shape[1], device=Q.device)[None, :]
                    s_ij = torch.where(q_idx >= k_idx, s_ij, torch.full_like(s_ij, -1e6))
                m_ij = s_ij.max(dim=-1).values
                m_new = torch.maximum(m_i, m_ij)

                alpha = torch.exp(m_i - m_new)
                p_tilde = torch.exp(s_ij - m_new.unsqueeze(-1))

                l_i = alpha * l_i + p_tilde.sum(dim=-1)
                o_i = alpha.unsqueeze(-1) * o_i + torch.einsum("bqk,bkd->bqd", p_tilde, v_tile.to(torch.float32))
                m_i = m_new

            o_i = o_i / l_i.unsqueeze(-1)
            O[:, q_start:q_end, :] = o_i.to(O.dtype)
            L[:, q_start:q_end] = m_i + torch.log(l_i)

        ctx.is_causal = is_causal
        ctx.save_for_backward(L, Q, K, V, O)
        return O

    @staticmethod
    def backward(ctx, dO: torch.Tensor):
        L, Q, K, V, O = ctx.saved_tensors
        dQ, dK, dV = _flash_attention_backward_impl(Q, K, V, O, dO, L, ctx.is_causal)
        return dQ, dK, dV, None
