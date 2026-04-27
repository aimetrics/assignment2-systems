# benchmark.py
#
# CS336 Assignment 2
# Problem (benchmarking_script)
#
# 功能：
# 1. 根据超参数初始化 Transformer 模型
# 2. 随机生成 batch 数据
# 3. 支持 warmup + 正式计时
# 4. 支持：
#    - forward only
#    - forward + backward
# 5. 每一步都调用 torch.cuda.synchronize()
#
# 使用示例：
#
# Forward only:
# uv run python benchmark.py \
#   --model_size small \
#   --seq_len 512 \
#   --batch_size 4 \
#   --warmup_steps 5 \
#   --measure_steps 10 \
#   --mode forward
#
# Forward + Backward:
# uv run python benchmark.py \
#   --model_size small \
#   --seq_len 512 \
#   --batch_size 4 \
#   --warmup_steps 5 \
#   --measure_steps 10 \
#   --mode train
#

import argparse
import statistics
import pandas as pd
import timeit
from dataclasses import dataclass

import torch
import torch.nn as nn

# 你自己的 basics transformer
from cs336_basics.model import TransformerLM


VOCAB_SIZE = 10_000
if torch.cuda.is_available():
    _DEFAULT_DEVICE = "cuda"
elif torch.backends.mps.is_available():
    _DEFAULT_DEVICE = "mps"
else:
    _DEFAULT_DEVICE = "cpu"
DEVICE = _DEFAULT_DEVICE  # may be overridden by --device arg


def synchronize():
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elif DEVICE == "mps":
        torch.mps.synchronize()

@dataclass
class ModelConfig:
    d_model: int
    d_ff: int
    num_layers: int
    num_heads: int


MODEL_SIZES = {
    "small": ModelConfig(
        d_model=768,
        d_ff=3072,
        num_layers=12,
        num_heads=12,
    ),
    "medium": ModelConfig(
        d_model=1024,
        d_ff=4096,
        num_layers=24,
        num_heads=16,
    ),
    "large": ModelConfig(
        d_model=1280,
        d_ff=5120,
        num_layers=36,
        num_heads=20,
    ),
    "xl": ModelConfig(
        d_model=1600,
        d_ff=6400,
        num_layers=48,
        num_heads=25,
    ),
    "2.7b": ModelConfig(
        d_model=2560,
        d_ff=10240,
        num_layers=32,
        num_heads=32,
    ),
}


def build_model(
    cfg: ModelConfig,
    seq_len: int,
):
    """
    根据 assignment 中要求初始化 Transformer 模型
    """
    model = TransformerLM(
        vocab_size=VOCAB_SIZE,
        context_length=seq_len,
        d_model=cfg.d_model,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        d_ff=cfg.d_ff,
    )
    return model.to(DEVICE)


def make_batch(
    batch_size: int,
    seq_len: int,
):
    """
    随机生成 token batch
    """
    x = torch.randint(
        low=0,
        high=VOCAB_SIZE,
        size=(batch_size, seq_len),
        device=DEVICE,
        dtype=torch.long,
    )

    # next-token prediction target
    y = torch.randint(
        low=0,
        high=VOCAB_SIZE,
        size=(batch_size, seq_len),
        device=DEVICE,
        dtype=torch.long,
    )

    return x, y


def run_forward_step(
    model: nn.Module,
    x: torch.Tensor,
):
    logits = model(x)
    return logits


def run_train_step(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
):
    logits = model(x)

    # cross entropy
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        y.reshape(-1),
    )

    loss.backward()
    return loss


def benchmark(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    warmup_steps: int,
    measure_steps: int,
    mode: str,
):
    """
    mode:
        - forward
        - train
    """

    assert mode in ["forward", "train"]

    timings = []

    # -------------------------
    # Warmup
    # -------------------------
    for _ in range(warmup_steps):
        model.zero_grad(set_to_none=True)

        if mode == "forward":
            with torch.no_grad():
                _ = run_forward_step(model, x)
        else:
            _ = run_train_step(model, x, y)

        synchronize()

    # -------------------------
    # Measurement
    # -------------------------
    for _ in range(measure_steps):
        model.zero_grad(set_to_none=True)

        start = timeit.default_timer()

        if mode == "forward":
            with torch.no_grad():
                _ = run_forward_step(model, x)
        else:
            _ = run_train_step(model, x, y)

        synchronize()

        end = timeit.default_timer()
        timings.append(end - start)

    mean_time = statistics.mean(timings)
    std_time = statistics.stdev(timings) if len(timings) > 1 else 0.0

    return timings, mean_time, std_time


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_size",
        type=str,
        default="2.7b",
        choices=list(MODEL_SIZES.keys()),
    )

    parser.add_argument(
        "--seq_len",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--measure_steps",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["forward", "train"],
        help="forward = inference only, train = forward + backward",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=_DEFAULT_DEVICE,
        help="Device to run on: cpu, cuda, mps (default: auto-detected, prefers cuda over cpu)",
    )

    return parser.parse_args()



def main():
    global DEVICE
    args = parse_args()
    DEVICE = args.device

    cfg = MODEL_SIZES[args.model_size]

    print("=" * 80, flush=True)
    print("Benchmark Configuration", flush=True)
    print(f"device         : {DEVICE}", flush=True)
    print(f"model_size     : {args.model_size}", flush=True)
    print(f"seq_len        : {args.seq_len}", flush=True)
    print(f"batch_size     : {args.batch_size}", flush=True)
    print(f"warmup_steps   : {args.warmup_steps}", flush=True)
    print(f"measure_steps  : {args.measure_steps}", flush=True)
    print(f"mode           : {args.mode}", flush=True)
    print("=" * 80, flush=True)

    model = build_model(
        cfg=cfg,
        seq_len=args.seq_len,
    )

    x, y = make_batch(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
    )

    timings, mean_time, std_time = benchmark(
        model=model,
        x=x,
        y=y,
        warmup_steps=args.warmup_steps,
        measure_steps=args.measure_steps,
        mode=args.mode,
    )

    print("\nPer-step timings (seconds):")
    for i, t in enumerate(timings):
        print(f"step {i:02d}: {t:.6f}")

    print("\nSummary")
    print(f"mean: {mean_time:.6f} sec")
    print(f"std : {std_time:.6f} sec")

    results = [
        {
            "device": DEVICE,
            "warmup_steps": args.warmup_steps,
            "measure_steps": args.measure_steps,
            "model_size": args.model_size,
            "mode": args.mode,
            "seq_len": args.seq_len,
            "batch_size": args.batch_size,
            "mean_time_sec": round(mean_time, 4),
            "std_time_sec": round(std_time, 4),
        }
    ]

    df = pd.DataFrame(results)
    table = df.to_string(index=False)
    print("\nResults Table:")
    print(table)

    with open("benchmark_results.md", "a") as f:
        f.write(table + "\n")

if __name__ == "__main__":
    main()
