from __future__ import annotations

import argparse
import csv
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

try:
    import triton.testing as triton_testing  # type: ignore[import-not-found]
except ModuleNotFoundError:
    triton_testing = None

from cs336_systems.flash_attention_triton import get_flashattention_autograd_function_triton


@dataclass
class BenchmarkRow:
    implementation: str
    dtype: str
    seq_len: int
    d_model: int
    forward_ms: float | None
    backward_ms: float | None
    end_to_end_ms: float | None
    status: str
    note: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Triton FlashAttention vs regular PyTorch attention.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup-ms", type=int, default=200)
    parser.add_argument("--rep-ms", type=int, default=800)
    parser.add_argument("--seq-lens", nargs="+", type=int, default=[2**i for i in range(7, 17)])
    parser.add_argument("--d-models", nargs="+", type=int, default=[16, 32, 64, 128])
    parser.add_argument("--dtypes", nargs="+", default=["bfloat16", "float32"])
    parser.add_argument("--csv-path", type=Path, default=Path("artifacts/systems/flash_attention_benchmark_results.csv"))
    parser.add_argument("--markdown-path", type=Path, default=Path("artifacts/systems/flash_attention_benchmark_results.md"))
    return parser.parse_args()


def _causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.einsum("bqd,bkd->bqk", q, k) * scale
    n_queries = q.shape[-2]
    n_keys = k.shape[-2]
    q_idx = torch.arange(n_queries, device=q.device)[:, None]
    k_idx = torch.arange(n_keys, device=q.device)[None, :]
    scores = torch.where(q_idx >= k_idx, scores, torch.full_like(scores, -1e6))
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("bqk,bkd->bqd", probs, v)


def _parse_dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


def _bench_impl(
    *,
    impl_name: str,
    forward_fn,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    warmup_ms: int,
    rep_ms: int,
) -> tuple[float, float, float]:
    if triton_testing is None:
        raise RuntimeError("Triton is required to run flash attention benchmarking.")

    q_fwd = q.detach().clone()
    k_fwd = k.detach().clone()
    v_fwd = v.detach().clone()

    def run_forward():
        with torch.no_grad():
            forward_fn(q_fwd, k_fwd, v_fwd)

    forward_ms = triton_testing.do_bench(run_forward, warmup=warmup_ms, rep=rep_ms)

    q_bwd = q.detach().clone().requires_grad_(True)
    k_bwd = k.detach().clone().requires_grad_(True)
    v_bwd = v.detach().clone().requires_grad_(True)
    out = forward_fn(q_bwd, k_bwd, v_bwd)

    def run_backward_only():
        if q_bwd.grad is not None:
            q_bwd.grad = None
            k_bwd.grad = None
            v_bwd.grad = None
        out.backward(do, retain_graph=True)

    backward_ms = triton_testing.do_bench(run_backward_only, warmup=warmup_ms, rep=rep_ms)

    def run_end_to_end():
        q_e2e = q.detach().clone().requires_grad_(True)
        k_e2e = k.detach().clone().requires_grad_(True)
        v_e2e = v.detach().clone().requires_grad_(True)
        out_e2e = forward_fn(q_e2e, k_e2e, v_e2e)
        out_e2e.backward(do)

    end_to_end_ms = triton_testing.do_bench(run_end_to_end, warmup=warmup_ms, rep=rep_ms)

    if q_bwd.grad is not None:
        q_bwd.grad = None
        k_bwd.grad = None
        v_bwd.grad = None

    return forward_ms, backward_ms, end_to_end_ms


def _to_markdown(rows: list[BenchmarkRow]) -> str:
    header = (
        "| implementation | dtype | seq_len | d_model | forward_ms | backward_ms | end_to_end_ms | status | note |\n"
        "|---|---|---:|---:|---:|---:|---:|---|---|"
    )
    lines = [header]
    for row in rows:
        lines.append(
            "| "
            f"{row.implementation} | {row.dtype} | {row.seq_len} | {row.d_model} | "
            f"{row.forward_ms if row.forward_ms is not None else 'NA'} | "
            f"{row.backward_ms if row.backward_ms is not None else 'NA'} | "
            f"{row.end_to_end_ms if row.end_to_end_ms is not None else 'NA'} | "
            f"{row.status} | {row.note} |"
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("This benchmark must run on CUDA.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Run this script on a machine with an NVIDIA GPU (ideally an H100).")
    if triton_testing is None:
        raise RuntimeError("Triton is required to run flash attention benchmarking.")
    if args.batch_size != 1:
        raise ValueError("Assignment benchmark requires batch size 1.")

    flash_triton_cls = get_flashattention_autograd_function_triton()

    rows: list[BenchmarkRow] = []
    for dtype_name in args.dtypes:
        dtype = _parse_dtype(dtype_name)
        for seq_len in args.seq_lens:
            for d_model in args.d_models:
                print(f"Running dtype={dtype_name}, seq={seq_len}, d_model={d_model} ...", flush=True)
                q = torch.randn(args.batch_size, seq_len, d_model, device=device, dtype=dtype)
                k = torch.randn(args.batch_size, seq_len, d_model, device=device, dtype=dtype)
                v = torch.randn(args.batch_size, seq_len, d_model, device=device, dtype=dtype)
                do = torch.randn(args.batch_size, seq_len, d_model, device=device, dtype=dtype)
                for impl_name, forward_fn in [
                    ("pytorch_attention", _causal_attention),
                    ("triton_flashattention2", lambda qq, kk, vv: flash_triton_cls.apply(qq, kk, vv, True)),
                ]:
                    try:
                        f_ms, b_ms, e2e_ms = _bench_impl(
                            impl_name=impl_name,
                            forward_fn=forward_fn,
                            q=q,
                            k=k,
                            v=v,
                            do=do,
                            warmup_ms=args.warmup_ms,
                            rep_ms=args.rep_ms,
                        )
                        rows.append(
                            BenchmarkRow(
                                implementation=impl_name,
                                dtype=dtype_name,
                                seq_len=seq_len,
                                d_model=d_model,
                                forward_ms=float(f_ms),
                                backward_ms=float(b_ms),
                                end_to_end_ms=float(e2e_ms),
                                status="ok",
                                note="",
                            )
                        )
                    except torch.cuda.OutOfMemoryError as exc:
                        note = str(exc).splitlines()[0]
                        rows.append(
                            BenchmarkRow(
                                implementation=impl_name,
                                dtype=dtype_name,
                                seq_len=seq_len,
                                d_model=d_model,
                                forward_ms=None,
                                backward_ms=None,
                                end_to_end_ms=None,
                                status="oom",
                                note=note,
                            )
                        )
                        torch.cuda.empty_cache()
                    except RuntimeError as exc:
                        note = str(exc).splitlines()[0]
                        rows.append(
                            BenchmarkRow(
                                implementation=impl_name,
                                dtype=dtype_name,
                                seq_len=seq_len,
                                d_model=d_model,
                                forward_ms=None,
                                backward_ms=None,
                                end_to_end_ms=None,
                                status="runtime_error",
                                note=note,
                            )
                        )
                        torch.cuda.empty_cache()

    args.csv_path.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_path.parent.mkdir(parents=True, exist_ok=True)

    with args.csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    args.markdown_path.write_text(_to_markdown(rows) + "\n")

    gpu_name = torch.cuda.get_device_name(device)
    print(f"GPU: {gpu_name}")
    print(f"Wrote {len(rows)} rows to {args.csv_path}")
    print(f"Wrote markdown table to {args.markdown_path}")


if __name__ == "__main__":
    main()