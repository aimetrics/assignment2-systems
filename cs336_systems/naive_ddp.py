from __future__ import annotations

import argparse
import os
import socket
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


@dataclass
class Config:
    world_size: int
    backend: str
    device: str
    num_steps: int
    global_batch_size: int
    input_dim: int
    hidden_dim: int
    output_dim: int
    lr: float
    seed: int
    master_addr: str
    master_port: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Naive DDP training via all-reduce over individual parameter gradients.")
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--backend", choices=["gloo", "nccl"], default="gloo")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--num-steps", type=int, default=25)
    parser.add_argument("--global-batch-size", type=int, default=64)
    parser.add_argument("--input-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--output-dim", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--master-addr", type=str, default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=29500)
    return parser.parse_args()


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return int(sock.getsockname()[1])


def _get_device(config: Config, rank: int) -> torch.device:
    if config.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but CUDA is unavailable.")
        if rank >= torch.cuda.device_count():
            raise RuntimeError(
                f"world_size={config.world_size} requires rank {rank}, but only {torch.cuda.device_count()} GPU(s) exist."
            )
        torch.cuda.set_device(rank)
        return torch.device(f"cuda:{rank}")
    return torch.device("cpu")


def build_model(input_dim: int, hidden_dim: int, output_dim: int) -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden_dim),
        torch.nn.Tanh(),
        torch.nn.Linear(hidden_dim, output_dim),
    )


def make_synthetic_batch(
    step: int,
    global_batch_size: int,
    input_dim: int,
    output_dim: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed + step)
    x = torch.randn(global_batch_size, input_dim, generator=g)
    y = torch.randn(global_batch_size, output_dim, generator=g)
    return x.to(device), y.to(device)


def train_single_process(config: Config) -> dict[str, torch.Tensor]:
    torch.manual_seed(config.seed)
    device = torch.device("cpu") if config.device == "cpu" else torch.device("cuda:0")
    if config.device == "cuda":
        torch.cuda.set_device(0)

    model = build_model(config.input_dim, config.hidden_dim, config.output_dim).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr)
    loss_fn = torch.nn.MSELoss(reduction="mean")

    for step in range(config.num_steps):
        x, y = make_synthetic_batch(
            step,
            config.global_batch_size,
            config.input_dim,
            config.output_dim,
            config.seed,
            device,
        )
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()

    return {name: param.detach().cpu().clone() for name, param in model.named_parameters()}


def allreduce_gradients_individually(model: torch.nn.Module, world_size: int) -> None:
    for param in model.parameters():
        if param.grad is None:
            continue
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=False)
        param.grad /= world_size


def distributed_worker(rank: int, config: Config, reference_path: str, result_path: str) -> None:
    os.environ["MASTER_ADDR"] = config.master_addr
    os.environ["MASTER_PORT"] = str(config.master_port)
    dist.init_process_group(backend=config.backend, rank=rank, world_size=config.world_size)

    try:
        torch.manual_seed(config.seed)
        device = _get_device(config, rank)
        model = build_model(config.input_dim, config.hidden_dim, config.output_dim).to(device)

        # Initial parameter sync: rank0 -> everyone.
        for param in model.parameters():
            dist.broadcast(param.data, src=0)

        optimizer = torch.optim.SGD(model.parameters(), lr=config.lr)
        loss_fn = torch.nn.MSELoss(reduction="mean")

        for step in range(config.num_steps):
            x, y = make_synthetic_batch(
                step,
                config.global_batch_size,
                config.input_dim,
                config.output_dim,
                config.seed,
                device,
            )
            local_batch = config.global_batch_size // config.world_size
            start = rank * local_batch
            end = start + local_batch
            x_local, y_local = x[start:end], y[start:end]

            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x_local), y_local)
            loss.backward()
            allreduce_gradients_individually(model, config.world_size)
            optimizer.step()

            if device.type == "cuda":
                torch.cuda.synchronize(device)

        if rank == 0:
            torch.save({name: p.detach().cpu() for name, p in model.named_parameters()}, result_path)

        dist.barrier()

        if rank == 0:
            ref_state = torch.load(reference_path, map_location="cpu")
            ddp_state = torch.load(result_path, map_location="cpu")

            max_abs_diff = 0.0
            for name, ref_param in ref_state.items():
                diff = (ref_param - ddp_state[name]).abs().max().item()
                max_abs_diff = max(max_abs_diff, diff)
            print(f"Max |single_process - naive_ddp| across parameters: {max_abs_diff:.8f}")
    finally:
        dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    if args.global_batch_size % args.world_size != 0:
        raise ValueError("global_batch_size must be divisible by world_size.")
    if args.backend == "nccl" and args.device != "cuda":
        raise ValueError("NCCL backend requires --device cuda.")

    cfg = Config(
        world_size=args.world_size,
        backend=args.backend,
        device=args.device,
        num_steps=args.num_steps,
        global_batch_size=args.global_batch_size,
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        lr=args.lr,
        seed=args.seed,
        master_addr=args.master_addr,
        master_port=_pick_free_port() if args.master_port == 29500 else args.master_port,
    )

    if cfg.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA training but CUDA is unavailable.")

    reference_path = f"/tmp/naive_ddp_reference_{os.getpid()}.pt"
    result_path = f"/tmp/naive_ddp_result_{os.getpid()}.pt"

    ref_state = train_single_process(cfg)
    torch.save(ref_state, reference_path)

    mp.set_start_method("spawn", force=True)
    mp.spawn(
        distributed_worker,
        args=(cfg, reference_path, result_path),
        nprocs=cfg.world_size,
        join=True,
    )

    # Final explicit check in parent process.
    ref_state_final = torch.load(reference_path, map_location="cpu")
    ddp_state_final = torch.load(result_path, map_location="cpu")
    for name, ref_param in ref_state_final.items():
        if not torch.allclose(ref_param, ddp_state_final[name], rtol=1e-5, atol=1e-6):
            max_diff = (ref_param - ddp_state_final[name]).abs().max().item()
            raise AssertionError(f"Parameter mismatch for {name}: max_abs_diff={max_diff:.8f}")

    print("✅ Naive DDP verification passed: distributed parameters match single-process training.")


if __name__ == "__main__":
    main()