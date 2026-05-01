from __future__ import annotations

import argparse
import json
import os
import socket
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from cs336_basics.model import TransformerLM
from cs336_systems.sharded_optimizer import ShardedOptimizer


@dataclass
class RankMetrics:
    rank: int
    init_peak_bytes: int
    before_step_peak_bytes: int
    after_step_peak_bytes: int
    mean_iter_ms: float
    param_bytes: int
    grad_bytes: int
    optimizer_state_bytes: int


@dataclass
class RunSummary:
    mode: str
    world_size: int
    device: str
    warmup_steps: int
    measure_steps: int
    max_init_peak_bytes: int
    max_before_step_peak_bytes: int
    max_after_step_peak_bytes: int
    mean_iter_ms: float
    per_rank: list[RankMetrics]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile memory and step time with/without optimizer state sharding.")
    parser.add_argument("--mode", choices=["non_sharded", "sharded"], required=True)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--backend", choices=["nccl", "gloo"], default="nccl")
    parser.add_argument("--master-addr", type=str, default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=0)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--measure-steps", type=int, default=5)

    # Standard experiment-style XL defaults used in this repository.
    parser.add_argument("--batch-size", type=int, default=2, help="Global batch size across all ranks.")
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=50_257)
    parser.add_argument("--d-model", type=int, default=2560)
    parser.add_argument("--num-layers", type=int, default=32)
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--d-ff", type=int, default=10_240)
    parser.add_argument("--rope-theta", type=float, default=10_000.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--out", type=Path, default=Path("artifacts/systems/optimizer_state_sharding_accounting.json"))
    return parser.parse_args()


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return int(sock.getsockname()[1])


def _device_for_rank(rank: int) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("This script requires CUDA GPUs.")
    if rank >= torch.cuda.device_count():
        raise RuntimeError(f"rank={rank} needs GPU, but only {torch.cuda.device_count()} GPU(s) available.")
    torch.cuda.set_device(rank)
    return torch.device(f"cuda:{rank}")


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _param_bytes(model: torch.nn.Module) -> int:
    return sum(_tensor_bytes(p) for p in model.parameters())


def _grad_bytes(model: torch.nn.Module) -> int:
    return sum(_tensor_bytes(p.grad) for p in model.parameters() if p.grad is not None)


def _optimizer_state_bytes(optimizer: torch.optim.Optimizer) -> int:
    total = 0
    for param_state in optimizer.state.values():
        for value in param_state.values():
            if torch.is_tensor(value):
                total += _tensor_bytes(value)
    return total


def _worker(rank: int, args: argparse.Namespace, master_port: int) -> None:
    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    dist.init_process_group(backend=args.backend, rank=rank, world_size=args.world_size)

    try:
        device = _device_for_rank(rank)
        torch.manual_seed(args.seed)

        if args.batch_size % args.world_size != 0:
            raise ValueError("Global batch size must be divisible by world size.")
        local_bs = args.batch_size // args.world_size

        torch.cuda.reset_peak_memory_stats(device)

        model = TransformerLM(
            vocab_size=args.vocab_size,
            context_length=args.context_length,
            d_model=args.d_model,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            d_ff=args.d_ff,
            rope_theta=args.rope_theta,
        ).to(device=device, dtype=torch.float32)

        for p in model.parameters():
            dist.broadcast(p.data, src=0)

        if args.mode == "sharded":
            optimizer: torch.optim.Optimizer = ShardedOptimizer(
                model.parameters(),
                torch.optim.AdamW,
                lr=args.lr,
                weight_decay=args.weight_decay,
            )
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        _sync(device)
        init_peak_bytes = torch.cuda.max_memory_allocated(device)

        before_step_peak_bytes = 0
        after_step_peak_bytes = 0
        step_ms: list[float] = []

        total_steps = args.warmup_steps + args.measure_steps
        for step in range(total_steps):
            optimizer.zero_grad(set_to_none=True)

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

            _sync(device)
            t0 = time.perf_counter()
            logits = model(tokens)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            loss.backward()
            _sync(device)

            before_step_peak_bytes = max(before_step_peak_bytes, torch.cuda.max_memory_allocated(device))

            optimizer.step()
            _sync(device)

            after_step_peak_bytes = max(after_step_peak_bytes, torch.cuda.max_memory_allocated(device))
            t1 = time.perf_counter()

            if step >= args.warmup_steps:
                step_ms.append((t1 - t0) * 1000.0)

        local_optimizer = optimizer
        if isinstance(optimizer, ShardedOptimizer) and optimizer._local_optimizer is not None:
            local_optimizer = optimizer._local_optimizer

        rank_metrics = RankMetrics(
            rank=rank,
            init_peak_bytes=init_peak_bytes,
            before_step_peak_bytes=before_step_peak_bytes,
            after_step_peak_bytes=after_step_peak_bytes,
            mean_iter_ms=sum(step_ms) / len(step_ms),
            param_bytes=_param_bytes(model),
            grad_bytes=_grad_bytes(model),
            optimizer_state_bytes=_optimizer_state_bytes(local_optimizer),
        )

        gathered: list[dict] = [None for _ in range(args.world_size)]
        dist.all_gather_object(gathered, asdict(rank_metrics))

        if rank == 0:
            per_rank = [RankMetrics(**item) for item in gathered]
            summary = RunSummary(
                mode=args.mode,
                world_size=args.world_size,
                device="cuda",
                warmup_steps=args.warmup_steps,
                measure_steps=args.measure_steps,
                max_init_peak_bytes=max(m.init_peak_bytes for m in per_rank),
                max_before_step_peak_bytes=max(m.before_step_peak_bytes for m in per_rank),
                max_after_step_peak_bytes=max(m.after_step_peak_bytes for m in per_rank),
                mean_iter_ms=sum(m.mean_iter_ms for m in per_rank) / len(per_rank),
                per_rank=per_rank,
            )
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(asdict(summary), indent=2))
            print(json.dumps(asdict(summary), indent=2))
    finally:
        dist.barrier()
        dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    if args.master_port == 0:
        args.master_port = _pick_free_port()
    mp.spawn(_worker, args=(args, args.master_port), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()