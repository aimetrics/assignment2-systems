from __future__ import annotations

import argparse
import csv
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

try:
    import triton  # type: ignore[import-not-found]
except ModuleNotFoundError:
    triton = None

from cs336_systems.flash_attention import FlashAttentionTritonFunction


@dataclass
class BenchmarkRow:
    variant: str
    seq_len: int
    d_model: int
    dtype: str
    status: str
    forward_ms: float | None
    backward_ms: float | None
    forward_backward_ms: float | None
    note: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark PyTorch attention against Triton FlashAttention-2.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seq-lens", nargs="+", type=int, default=[128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536])
    parser.add_argument("--d-models", nargs="+", type=int, default=[16, 32, 64, 128])
    parser.add_argument("--dtypes", nargs="+", choices=["bf16", "fp32"], default=["bf16", "fp32"])
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--rep", type=int, default=1000)
    parser.add_argument("--csv-path", type=Path, default=Path("flash_attention_benchmark_results.csv"))
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def pytorch_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = True) -> torch.Tensor:
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if is_causal:
        mask = torch.arange(q.shape[-2], device=q.device)[:, None] >= torch.arange(k.shape[-2], device=q.device)[None, :]
        scores = torch.where(mask, scores, -1e6)
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)


def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = True) -> torch.Tensor:
    return FlashAttentionTritonFunction.apply(q, k, v, is_causal)


def bench_one(
    *,
    fn,
    q_base: torch.Tensor,
    k_base: torch.Tensor,
    v_base: torch.Tensor,
    grad_out: torch.Tensor,
    warmup: int,
    rep: int,
) -> tuple[float, float, float]:
    if triton is None:
        raise RuntimeError("Triton is required to run flash attention benchmarking.")

    def make_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = q_base.detach().clone().requires_grad_(True)
        k = k_base.detach().clone().requires_grad_(True)
        v = v_base.detach().clone().requires_grad_(True)
        return q, k, v

    q, k, v = make_inputs()

    def forward_only() -> None:
        fn(q_base, k_base, v_base, True)

    forward_ms = triton.testing.do_bench(forward_only, warmup=warmup, rep=rep)

    def backward_only() -> None:
        nonlocal q, k, v
        out = fn(q, k, v, True)
        out.backward(grad_out)
        q.grad = None
        k.grad = None
        v.grad = None

    # Run once before timing so autograd/compile setup does not dominate the measured region.
    backward_only()
    backward_ms = triton.testing.do_bench(backward_only, warmup=warmup, rep=rep)

    def forward_backward() -> None:
        q_e2e, k_e2e, v_e2e = make_inputs()
        out = fn(q_e2e, k_e2e, v_e2e, True)
        out.backward(grad_out)

    forward_backward_ms = triton.testing.do_bench(forward_backward, warmup=warmup, rep=rep)
    return forward_ms, backward_ms, forward_backward_ms


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("flash_benchmarking requires a CUDA device.")
    if triton is None:
        raise RuntimeError("flash_benchmarking requires Triton, but it is not installed.")

    rows: list[BenchmarkRow] = []
    torch.manual_seed(0)

    for dtype_name in args.dtypes:
        dtype = dtype_from_name(dtype_name)
        for seq_len in args.seq_lens:
            for d_model in args.d_models:
                print(f"[flash] seq_len={seq_len} d={d_model} dtype={dtype_name}", flush=True)
                try:
                    q_base = torch.randn(1, seq_len, d_model, device=device, dtype=dtype)
                    k_base = torch.randn(1, seq_len, d_model, device=device, dtype=dtype)
                    v_base = torch.randn(1, seq_len, d_model, device=device, dtype=dtype)
                    grad_out = torch.randn_like(q_base)

                    for variant, fn in (("pytorch", pytorch_attention), ("triton_flash", flash_attention)):
                        try:
                            fwd, bwd, e2e = bench_one(
                                fn=fn,
                                q_base=q_base,
                                k_base=k_base,
                                v_base=v_base,
                                grad_out=grad_out,
                                warmup=args.warmup,
                                rep=args.rep,
                            )
                            rows.append(BenchmarkRow(variant, seq_len, d_model, dtype_name, "ok", fwd, bwd, e2e, ""))
                        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                            torch.cuda.empty_cache()
                            status = "oom" if isinstance(exc, torch.cuda.OutOfMemoryError) else "runtime_error"
                            rows.append(BenchmarkRow(variant, seq_len, d_model, dtype_name, status, None, None, None, str(exc).split("\n")[0]))
                finally:
                    torch.cuda.empty_cache()

    args.csv_path.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    print("\n| variant | seq_len | d_model | dtype | status | forward ms | backward ms | fwd+bwd ms |")
    print("|---|---:|---:|---|---|---:|---:|---:|")
    for row in rows:
        if row.status != "ok":
            print(f"| {row.variant} | {row.seq_len} | {row.d_model} | {row.dtype} | {row.status} | - | - | - |")
        else:
            print(
                f"| {row.variant} | {row.seq_len} | {row.d_model} | {row.dtype} | ok "
                f"| {row.forward_ms:.3f} | {row.backward_ms:.3f} | {row.forward_backward_ms:.3f} |"
            )
    print(f"\nWrote {len(rows)} rows to {args.csv_path}")


if __name__ == "__main__":
    main()
