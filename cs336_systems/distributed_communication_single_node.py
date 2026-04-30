from __future__ import annotations

import argparse
import csv
import os
import socket
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


MB = 1024 * 1024
DTYPE_BYTES = 4  # float32


@dataclass
class BenchmarkResult:
    backend: str
    device_type: str
    world_size: int
    size_mb: int
    warmup_iters: int
    timed_iters: int
    rank_mean_ms: float
    rank_min_ms: float
    rank_max_ms: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark single-node multi-process all-reduce across backends, tensor sizes, and world sizes."
    )
    parser.add_argument("--backends", nargs="+", default=["gloo", "nccl"], choices=["gloo", "nccl"])
    parser.add_argument("--sizes-mb", nargs="+", type=int, default=[1, 10, 100, 1024])
    parser.add_argument("--world-sizes", nargs="+", type=int, default=[2, 4, 6])
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--timed-iters", type=int, default=20)
    parser.add_argument("--master-addr", type=str, default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument("--csv-out", type=Path, default=Path("artifacts/systems/distributed_communication_single_node.csv"))
    parser.add_argument(
        "--table-out", type=Path, default=Path("artifacts/systems/distributed_communication_single_node_table.md")
    )
    parser.add_argument(
        "--plot-out", type=Path, default=Path("artifacts/systems/distributed_communication_single_node_plot.png")
    )
    parser.add_argument(
        "--allow-cpu-fallback-for-nccl",
        action="store_true",
        help="If set, quietly skips NCCL rows when CUDA is unavailable instead of raising.",
    )
    return parser.parse_args()


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return int(sock.getsockname()[1])


def _device_for_rank(backend: str, rank: int) -> torch.device:
    if backend == "nccl":
        if not torch.cuda.is_available():
            raise RuntimeError("NCCL requested, but CUDA is not available.")
        n_gpus = torch.cuda.device_count()
        if rank >= n_gpus:
            raise RuntimeError(f"World size requires rank {rank}, but only {n_gpus} GPU(s) are available.")
        torch.cuda.set_device(rank)
        return torch.device(f"cuda:{rank}")
    return torch.device("cpu")


def _run_worker(
    rank: int,
    world_size: int,
    backend: str,
    size_mb: int,
    warmup_iters: int,
    timed_iters: int,
    master_addr: str,
    master_port: int,
    result_path: str,
) -> None:
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)

    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    try:
        device = _device_for_rank(backend, rank)
        n_elements = (size_mb * MB) // DTYPE_BYTES
        data = torch.randn(n_elements, dtype=torch.float32, device=device)

        for _ in range(warmup_iters):
            dist.all_reduce(data, op=dist.ReduceOp.SUM, async_op=False)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        timings: list[float] = []
        for _ in range(timed_iters):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            dist.all_reduce(data, op=dist.ReduceOp.SUM, async_op=False)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t1 = time.perf_counter()
            timings.append((t1 - t0) * 1000.0)

        rank_mean_ms = sum(timings) / len(timings)
        gathered_means: list[float] = [0.0 for _ in range(world_size)]
        dist.all_gather_object(gathered_means, rank_mean_ms)

        if rank == 0:
            min_ms = min(gathered_means)
            max_ms = max(gathered_means)
            avg_ms = sum(gathered_means) / len(gathered_means)
            result = BenchmarkResult(
                backend=backend,
                device_type=device.type,
                world_size=world_size,
                size_mb=size_mb,
                warmup_iters=warmup_iters,
                timed_iters=timed_iters,
                rank_mean_ms=avg_ms,
                rank_min_ms=min_ms,
                rank_max_ms=max_ms,
            )
            out_file = Path(result_path)
            with out_file.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(asdict(result).keys()))
                writer.writeheader()
                writer.writerow(asdict(result))
    finally:
        dist.destroy_process_group()


def _run_case(
    *,
    backend: str,
    world_size: int,
    size_mb: int,
    warmup_iters: int,
    timed_iters: int,
    master_addr: str,
    master_port: int,
) -> BenchmarkResult:
    token = f"{os.getpid()}_{time.time_ns()}_{backend}_{world_size}_{size_mb}"
    out_file = Path(f"/tmp/allreduce_bench_{token}.csv")
    if out_file.exists():
        out_file.unlink()

    mp.spawn(
        _run_worker,
        args=(world_size, backend, size_mb, warmup_iters, timed_iters, master_addr, master_port, str(out_file)),
        nprocs=world_size,
        join=True,
    )

    if not out_file.exists():
        raise RuntimeError(f"Could not find result file for backend={backend}, world_size={world_size}, size_mb={size_mb}")

    with out_file.open() as f:
        row = next(csv.DictReader(f))
    out_file.unlink(missing_ok=True)

    return BenchmarkResult(
        backend=row["backend"],
        device_type=row["device_type"],
        world_size=int(row["world_size"]),
        size_mb=int(row["size_mb"]),
        warmup_iters=int(row["warmup_iters"]),
        timed_iters=int(row["timed_iters"]),
        rank_mean_ms=float(row["rank_mean_ms"]),
        rank_min_ms=float(row["rank_min_ms"]),
        rank_max_ms=float(row["rank_max_ms"]),
    )


def _write_outputs(results: list[BenchmarkResult], args: argparse.Namespace) -> None:
    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    args.table_out.parent.mkdir(parents=True, exist_ok=True)
    args.plot_out.parent.mkdir(parents=True, exist_ok=True)

    with args.csv_out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))

    lines = [
        "| backend | device | world_size | size_mb | mean_ms | min_rank_ms | max_rank_ms |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            f"| {result.backend} | {result.device_type} | {result.world_size} | {result.size_mb} | "
            f"{result.rank_mean_ms:.3f} | {result.rank_min_ms:.3f} | {result.rank_max_ms:.3f} |"
        )
    args.table_out.write_text("\n".join(lines) + "\n")

    backends = sorted(set(r.backend for r in results))
    fig, axes = plt.subplots(1, len(backends), figsize=(6 * len(backends), 5), sharey=True)
    if len(backends) == 1:
        axes = [axes]

    for idx, backend in enumerate(backends):
        ax = axes[idx]
        backend_rows = [r for r in results if r.backend == backend]
        for world_size in sorted(set(r.world_size for r in backend_rows)):
            ws_rows = sorted((r for r in backend_rows if r.world_size == world_size), key=lambda r: r.size_mb)
            ax.plot([r.size_mb for r in ws_rows], [r.rank_mean_ms for r in ws_rows], marker="o", label=f"world={world_size}")
        ax.set_xscale("log")
        ax.set_xlabel("Tensor size (MB)")
        ax.set_title(f"{backend.upper()} ({backend_rows[0].device_type})")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()
    axes[0].set_ylabel("All-reduce latency (ms)")
    fig.suptitle("Single-node all-reduce benchmark")
    fig.tight_layout()
    fig.savefig(args.plot_out, dpi=150)


def main() -> None:
    args = parse_args()
    mp.set_start_method("spawn", force=True)

    if "nccl" in args.backends and not torch.cuda.is_available() and not args.allow_cpu_fallback_for_nccl:
        raise RuntimeError(
            "NCCL backend requested but CUDA is unavailable. Re-run with --allow-cpu-fallback-for-nccl to skip NCCL."
        )

    results: list[BenchmarkResult] = []

    for backend in args.backends:
        if backend == "nccl" and not torch.cuda.is_available():
            print("Skipping NCCL: CUDA unavailable.", flush=True)
            continue

        for world_size in args.world_sizes:
            if backend == "nccl" and world_size > torch.cuda.device_count():
                print(
                    f"Skipping NCCL world_size={world_size}: only {torch.cuda.device_count()} GPU(s) are available.",
                    flush=True,
                )
                continue

            for size_mb in args.sizes_mb:
                port = _pick_free_port() if args.master_port == 29500 else args.master_port
                print(
                    f"Running backend={backend}, world_size={world_size}, size_mb={size_mb}, "
                    f"warmup={args.warmup_iters}, timed={args.timed_iters}",
                    flush=True,
                )
                result = _run_case(
                    backend=backend,
                    world_size=world_size,
                    size_mb=size_mb,
                    warmup_iters=args.warmup_iters,
                    timed_iters=args.timed_iters,
                    master_addr=args.master_addr,
                    master_port=port,
                )
                results.append(result)

    if not results:
        raise RuntimeError("No benchmark runs were executed. Check backend/device/world-size settings.")

    _write_outputs(results, args)
    print(f"Wrote CSV: {args.csv_out}")
    print(f"Wrote table: {args.table_out}")
    print(f"Wrote plot: {args.plot_out}")


if __name__ == "__main__":
    main()