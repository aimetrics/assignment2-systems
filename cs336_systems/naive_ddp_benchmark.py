from __future__ import annotations

import argparse
import csv
import os
import socket
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from cs336_basics.model import TransformerLM
from cs336_basics.optimizer import AdamW
from systems_core.ddp_bucketed import DDPBucketed
from ddp_overlap_individual_parameters import DDPOverlapIndividualParameters


@dataclass
class StepMetrics:
    step: int
    total_step_ms: float
    comm_ms: float
    comm_ratio: float


@dataclass
class SummaryMetrics:
    backend: str
    device: str
    world_size: int
    precision: str
    optimizer: str
    batch_size_global: int
    context_length: int
    vocab_size: int
    d_model: int
    num_layers: int
    num_heads: int
    d_ff: int
    warmup_steps: int
    measure_steps: int
    comm_strategy: str
    mean_total_step_ms: float
    mean_comm_ms: float
    mean_comm_ratio: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark naive DDP training step time and communication proportion.")
    parser.add_argument("--backend", choices=["gloo", "nccl"], default="nccl")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measure-steps", type=int, default=10)
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--optimizer", choices=["sgd", "adamw"], default="sgd")
    parser.add_argument(
        "--comm-strategy",
        choices=["individual", "flat", "overlap_individual", "overlap_bucketed"],
        default="individual",
        help=(
            "Gradient communication strategy: all-reduce each parameter tensor, "
            "one flattened tensor, overlap compute+communication with async per-parameter all-reduce, "
            "or overlap using bucketed async all-reduce."
        ),
    )
    parser.add_argument("--bucket-size-mb", type=float, default=25.0, help="Bucket size (MB) for overlap_bucketed.")

    # XL defaults from experiment context used in this repository.
    parser.add_argument("--batch-size", type=int, default=2, help="Global batch size across all ranks.")
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=50_257)
    parser.add_argument("--d-model", type=int, default=2560)
    parser.add_argument("--num-layers", type=int, default=32)
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--d-ff", type=int, default=10_240)
    parser.add_argument("--rope-theta", type=float, default=10_000.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1337)

    parser.add_argument("--master-addr", type=str, default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument("--csv-out", type=Path, default=Path("artifacts/systems/naive_ddp_benchmark_steps.csv"))
    parser.add_argument("--summary-out", type=Path, default=Path("artifacts/systems/naive_ddp_benchmark_summary.md"))
    parser.add_argument("--enable-nvtx", action="store_true", help="Emit NVTX ranges for Nsight Systems traces.")
    return parser.parse_args()


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return int(sock.getsockname()[1])


def _dtype_from_precision(precision: str) -> torch.dtype:
    if precision == "fp32":
        return torch.float32
    if precision == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported precision: {precision}")


def _device_for_rank(args: argparse.Namespace, rank: int) -> torch.device:
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        if rank >= torch.cuda.device_count():
            raise RuntimeError(f"rank={rank} needs GPU, but only {torch.cuda.device_count()} GPU(s) available.")
        torch.cuda.set_device(rank)
        return torch.device(f"cuda:{rank}")
    return torch.device("cpu")


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _nvtx_range_push(enabled: bool, name: str) -> None:
    if enabled and torch.cuda.is_available():
        torch.cuda.nvtx.range_push(name)


def _nvtx_range_pop(enabled: bool) -> None:
    if enabled and torch.cuda.is_available():
        torch.cuda.nvtx.range_pop()


def _allreduce_gradients_and_measure(model: torch.nn.Module, world_size: int, device: torch.device) -> float:
    comm_s = 0.0
    for param in model.parameters():
        if param.grad is None:
            continue
        _sync(device)
        t0 = time.perf_counter()
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=False)
        _sync(device)
        t1 = time.perf_counter()
        param.grad /= world_size
        comm_s += t1 - t0
    return comm_s * 1000.0


def _allreduce_flattened_gradients_and_measure(model: torch.nn.Module, world_size: int, device: torch.device) -> float:
    "minimal_ddp_flat_benchmarking"
    grads = [param.grad for param in model.parameters() if param.grad is not None]
    if not grads:
        return 0.0

    flat_grad = _flatten_dense_tensors(grads)
    _sync(device)
    t0 = time.perf_counter()
    dist.all_reduce(flat_grad, op=dist.ReduceOp.SUM, async_op=False)
    _sync(device)
    t1 = time.perf_counter()
    flat_grad /= world_size

    synced = _unflatten_dense_tensors(flat_grad, grads)
    for original_grad, synced_grad in zip(grads, synced):
        original_grad.copy_(synced_grad)

    return (t1 - t0) * 1000.0


def _worker(rank: int, args: argparse.Namespace, output_csv: str) -> None:
    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = str(args.master_port)
    dist.init_process_group(backend=args.backend, rank=rank, world_size=args.world_size)

    try:
        torch.manual_seed(args.seed)
        device = _device_for_rank(args, rank)
        dtype = _dtype_from_precision(args.precision)

        model = TransformerLM(
            vocab_size=args.vocab_size,
            context_length=args.context_length,
            d_model=args.d_model,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            d_ff=args.d_ff,
            rope_theta=args.rope_theta,
        ).to(device=device, dtype=dtype)

        ddp_model: DDPOverlapIndividualParameters | DDPBucketed | None = None
        train_module: torch.nn.Module
        if args.comm_strategy == "overlap_individual":
            ddp_model = DDPOverlapIndividualParameters(model)
            train_module = ddp_model
        elif args.comm_strategy == "overlap_bucketed":
            ddp_model = DDPBucketed(model, bucket_size_mb=args.bucket_size_mb)
            train_module = ddp_model
        else:
            # Keep model init in sync.
            for p in model.parameters():
                dist.broadcast(p.data, src=0)
            train_module = model

        if args.optimizer == "adamw":
            optimizer = AdamW(model.parameters(), lr=args.lr)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
        local_bs = args.batch_size // args.world_size
        if args.batch_size % args.world_size != 0:
            raise ValueError("Global batch size must be divisible by world size.")

        total_steps = args.warmup_steps + args.measure_steps
        measured: list[StepMetrics] = []

        for step in range(total_steps):
            tokens_global = torch.randint(
                low=0,
                high=args.vocab_size,
                size=(args.batch_size, args.context_length),
                device=device,
                dtype=torch.long,
            )
            labels_global = torch.randint(
                low=0,
                high=args.vocab_size,
                size=(args.batch_size, args.context_length),
                device=device,
                dtype=torch.long,
            )
            start = rank * local_bs
            end = start + local_bs
            tokens = tokens_global[start:end]
            labels = labels_global[start:end]

            if args.comm_strategy == "overlap_bucketed":
                assert ddp_model is not None
                ddp_model.start_train_batch()

            optimizer.zero_grad(set_to_none=True)
            _sync(device)
            t0 = time.perf_counter()

            try:
                _nvtx_range_push(args.enable_nvtx, "forward")
                logits = train_module(tokens)
                _nvtx_range_pop(args.enable_nvtx)

                _nvtx_range_push(args.enable_nvtx, "backward")
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
                loss.backward()
                _nvtx_range_pop(args.enable_nvtx)

                _nvtx_range_push(args.enable_nvtx, "grad_sync")
                if args.comm_strategy in {"overlap_individual", "overlap_bucketed"}:
                    _sync(device)
                    c0 = time.perf_counter()
                    assert ddp_model is not None
                    ddp_model.finish_gradient_synchronization()
                    _sync(device)
                    c1 = time.perf_counter()
                    comm_ms = (c1 - c0) * 1000.0
                elif args.comm_strategy == "flat":
                    comm_ms = _allreduce_flattened_gradients_and_measure(model, args.world_size, device)
                else:
                    comm_ms = _allreduce_gradients_and_measure(model, args.world_size, device)
                _nvtx_range_pop(args.enable_nvtx)

                _nvtx_range_push(args.enable_nvtx, "optimizer_step")
                optimizer.step()
                _nvtx_range_pop(args.enable_nvtx)
            except torch.OutOfMemoryError as exc:
                raise RuntimeError(
                    "CUDA OOM during benchmark step. Try a smaller --batch-size/--context-length "
                    "or use --optimizer sgd."
                ) from exc

            _sync(device)
            t1 = time.perf_counter()

            if step >= args.warmup_steps:
                total_ms = (t1 - t0) * 1000.0
                measured.append(
                    StepMetrics(
                        step=step - args.warmup_steps,
                        total_step_ms=total_ms,
                        comm_ms=comm_ms,
                        comm_ratio=comm_ms / total_ms if total_ms > 0 else 0.0,
                    )
                )

        gathered: list[list[dict[str, float | int]]] = [list() for _ in range(args.world_size)]
        dist.all_gather_object(gathered, [asdict(m) for m in measured])

        if rank == 0:
            per_step: list[StepMetrics] = []
            for step_idx in range(args.measure_steps):
                total_vals = [float(gathered[r][step_idx]["total_step_ms"]) for r in range(args.world_size)]
                comm_vals = [float(gathered[r][step_idx]["comm_ms"]) for r in range(args.world_size)]
                total_ms = sum(total_vals) / len(total_vals)
                comm_ms = sum(comm_vals) / len(comm_vals)
                per_step.append(
                    StepMetrics(
                        step=step_idx,
                        total_step_ms=total_ms,
                        comm_ms=comm_ms,
                        comm_ratio=comm_ms / total_ms if total_ms > 0 else 0.0,
                    )
                )

            out_path = Path(output_csv)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(asdict(per_step[0]).keys()))
                writer.writeheader()
                for row in per_step:
                    writer.writerow(asdict(row))
    finally:
        dist.destroy_process_group()


def _write_summary(args: argparse.Namespace, per_step_csv: Path, summary_md: Path) -> None:
    rows: list[StepMetrics] = []
    with per_step_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                StepMetrics(
                    step=int(row["step"]),
                    total_step_ms=float(row["total_step_ms"]),
                    comm_ms=float(row["comm_ms"]),
                    comm_ratio=float(row["comm_ratio"]),
                )
            )

    summary = SummaryMetrics(
        backend=args.backend,
        device=args.device,
        world_size=args.world_size,
        precision=args.precision,
        optimizer=args.optimizer,
        batch_size_global=args.batch_size,
        context_length=args.context_length,
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        warmup_steps=args.warmup_steps,
        measure_steps=args.measure_steps,
        comm_strategy=args.comm_strategy,
        mean_total_step_ms=sum(r.total_step_ms for r in rows) / len(rows),
        mean_comm_ms=sum(r.comm_ms for r in rows) / len(rows),
        mean_comm_ratio=sum(r.comm_ratio for r in rows) / len(rows),
    )

    summary_md.parent.mkdir(parents=True, exist_ok=True)
    summary_md.write_text(
        "\n".join(
            [
                "# Naive DDP Benchmark Summary",
                "",
                "## Setup",
                f"- backend/device: {summary.backend}/{summary.device}",
                f"- world_size: {summary.world_size}",
                f"- precision: {summary.precision}",
                f"- optimizer: {summary.optimizer}",
                f"- communication strategy: {summary.comm_strategy}",
                (
                    f"- bucket size (MB): {args.bucket_size_mb}"
                    if args.comm_strategy == "overlap_bucketed"
                    else "- bucket size (MB): n/a"
                ),
                f"- global batch size: {summary.batch_size_global}",
                f"- context_length: {summary.context_length}",
                f"- model: d_model={summary.d_model}, num_layers={summary.num_layers}, num_heads={summary.num_heads}, d_ff={summary.d_ff}",
                f"- warmup_steps={summary.warmup_steps}, measure_steps={summary.measure_steps}",
                "",
                "## Results",
                f"- mean step time: {summary.mean_total_step_ms:.3f} ms",
                f"- mean gradient communication time: {summary.mean_comm_ms:.3f} ms",
                f"- mean communication proportion: {100.0 * summary.mean_comm_ratio:.2f}%",
                "",
                "## Commentary",
                (
                    "The overlap_individual strategy issues async per-parameter all-reduce during backward, "
                    "so part of communication can be hidden behind compute."
                ),
                (
                    "Compare mean step times across individual / flat / overlap_individual runs under the same setup "
                    "to quantify the benefit of overlap."
                ),
            ]
        )
        + "\n"
    )


def main() -> None:
    args = parse_args()
    if args.backend == "nccl" and args.device != "cuda":
        raise ValueError("NCCL requires --device=cuda")
    if args.world_size < 1:
        raise ValueError("world_size must be >=1")

    if args.master_port == 29500:
        args.master_port = _pick_free_port()

    mp.set_start_method("spawn", force=True)
    mp.spawn(_worker, args=(args, str(args.csv_out)), nprocs=args.world_size, join=True)

    _write_summary(args, args.csv_out, args.summary_out)
    print(f"Wrote step-level metrics to {args.csv_out}")
    print(f"Wrote summary to {args.summary_out}")


if __name__ == "__main__":
    main()