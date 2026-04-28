"""Benchmark PyTorch attention and whole-model torch.compile.

Covers:
  - Problem (pytorch_attention): vanilla attention across d_model x seq_len grid
  - Problem (torch_compile)(a): compiled attention vs vanilla, same grid
  - Problem (torch_compile)(b): compiled whole Transformer vs vanilla

Usage:
  # (pytorch_attention) Vanilla attention only:
  uv run python benchmark_attention.py

  # (torch_compile a) Vanilla + compiled attention comparison:
  uv run python benchmark_attention.py --compare-compile

  # (torch_compile b) Whole-model compile comparison:
  uv run python benchmark_attention.py --compare-compile-model

  # Both compile benchmarks at once:
  uv run python benchmark_attention.py --compare-compile --compare-compile-model
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from timeit import default_timer

import torch
import torch.nn.functional as F

from cs336_basics.model import TransformerLM

VOCAB_SIZE = 10_000

MODEL_CONFIGS: dict[str, dict] = {
    "small":  {"d_model": 768,  "d_ff": 3072,  "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096,  "num_layers": 24, "num_heads": 16},
    "large":  {"d_model": 1280, "d_ff": 5120,  "num_layers": 36, "num_heads": 20},
    "xl":     {"d_model": 1600, "d_ff": 6400,  "num_layers": 48, "num_heads": 25},
    "2.7b":   {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}


@dataclass
class BenchmarkRow:
    variant: str
    d_model: int
    seq_len: int
    status: str
    forward_ms_mean: float | None
    backward_ms_mean: float | None
    memory_before_backward_mib_mean: float | None
    memory_before_backward_mib_max: float | None
    note: str


@dataclass
class ModelBenchmarkRow:
    variant: str
    model_size: str
    mode: str
    seq_len: int
    batch_size: int
    status: str
    mean_ms: float | None
    std_ms: float | None
    note: str


class AttentionModule(torch.nn.Module):
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scale = 1.0 / math.sqrt(q.shape[-1])
        scores = (q @ k.transpose(-2, -1)) * scale
        scores = torch.where(mask, scores, float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        return probs @ v


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark attention (pytorch_attention, torch_compile a & b).",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp32")

    attn = parser.add_argument_group("Attention benchmark (pytorch_attention + torch_compile a)")
    attn.add_argument("--batch-size", type=int, default=8)
    attn.add_argument("--d-models", nargs="+", type=int, default=[16, 32, 64, 128])
    attn.add_argument("--seq-lens", nargs="+", type=int, default=[256, 1024, 4096, 8192, 16384])
    attn.add_argument("--warmup-iters", type=int, default=10)
    attn.add_argument("--measure-iters", type=int, default=100)
    attn.add_argument("--compare-compile", action="store_true",
                      help="Also benchmark torch.compile(attention_module).")
    attn.add_argument("--compile-mode", default="default",
                      choices=["default", "reduce-overhead", "max-autotune"],
                      help="Mode for torch.compile. Used with --compare-compile.")
    attn.add_argument("--skip-attention", action="store_true",
                      help="Skip the attention-level benchmark (useful to run only --compare-compile-model).")
    attn.add_argument("--csv-path", type=Path,
                      default=Path("pytorch_attention_benchmark_results.csv"))

    model = parser.add_argument_group("Whole-model compile benchmark (torch_compile b)")
    model.add_argument("--compare-compile-model", action="store_true",
                       help="Benchmark torch.compile on the full Transformer model.")
    model.add_argument("--model-sizes", nargs="+", default=list(MODEL_CONFIGS.keys()),
                       choices=list(MODEL_CONFIGS.keys()),
                       help="Model sizes to benchmark.")
    model.add_argument("--model-seq-len", type=int, default=512)
    model.add_argument("--model-batch-size", type=int, default=4)
    model.add_argument("--model-warmup", type=int, default=5)
    model.add_argument("--model-measure", type=int, default=10)
    model.add_argument("--model-csv-path", type=Path,
                       default=Path("torch_compile_model_benchmark_results.csv"))

    return parser.parse_args()


def get_dtype(name: str) -> torch.dtype:
    if name == "fp32":
        return torch.float32
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


def synchronize(device: torch.device) -> None:
    # 等待 GPU 上所有未完成的 CUDA 操作全部执行完毕，然后才让 CPU 继续往下走
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def build_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    idx = torch.arange(seq_len, device=device)
    return idx[None, :, None] >= idx[None, None, :]


def benchmark_variant(
    *,
    attn_module: torch.nn.Module,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    grad_out: torch.Tensor,
    mask: torch.Tensor,
    warmup_iters: int,
    measure_iters: int,
    device: torch.device,
) -> tuple[float, float, float, float]:
    for _ in range(warmup_iters):
        out = attn_module(q, k, v, mask)
        synchronize(device)
        out.backward(grad_out, retain_graph=False)
        synchronize(device)
        q.grad = None
        k.grad = None
        v.grad = None

    forward_times_ms: list[float] = []
    backward_times_ms: list[float] = []
    mem_before_backward_mib: list[float] = []
    for _ in range(measure_iters):
        t0 = default_timer()
        out = attn_module(q, k, v, mask)
        synchronize(device)
        t1 = default_timer()
        forward_times_ms.append((t1 - t0) * 1000.0)

        mem_before_backward_mib.append(torch.cuda.memory_allocated(device) / (1024.0**2))

        t2 = default_timer()
        out.backward(grad_out, retain_graph=False)
        synchronize(device)
        t3 = default_timer()
        backward_times_ms.append((t3 - t2) * 1000.0)

        q.grad = None
        k.grad = None
        v.grad = None

    return (
        sum(forward_times_ms) / len(forward_times_ms),
        sum(backward_times_ms) / len(backward_times_ms),
        sum(mem_before_backward_mib) / len(mem_before_backward_mib),
        max(mem_before_backward_mib),
    )


def benchmark_model_variant(
    *,
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    mode: str,
    warmup: int,
    measure: int,
    device: torch.device,
) -> tuple[float, float]:
    """Benchmark a Transformer model for forward-only or full train step.

    Returns (mean_ms, std_ms).
    """
    optimizer = torch.optim.AdamW(model.parameters()) if mode == "train" else None

    for _ in range(warmup):
        model.zero_grad(set_to_none=True)
        if mode == "forward":
            with torch.no_grad():
                model(x)
        else:
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss.backward()
            optimizer.step()
        synchronize(device)

    times_ms: list[float] = []
    for _ in range(measure):
        model.zero_grad(set_to_none=True)
        synchronize(device)
        t0 = default_timer()
        if mode == "forward":
            with torch.no_grad():
                model(x)
        else:
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss.backward()
            optimizer.step()
        synchronize(device)
        times_ms.append((default_timer() - t0) * 1000.0)

    mean = statistics.mean(times_ms)
    std = statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0
    return mean, std


def run_attention_benchmark(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> None:
    """Problem (pytorch_attention) + Problem (torch_compile)(a)."""
    rows: list[BenchmarkRow] = []

    for d_model in args.d_models:
        for seq_len in args.seq_lens:
            print(f"[attention] d_model={d_model}, seq_len={seq_len} ...", flush=True)
            try:
                mask = build_causal_mask(seq_len, device)
                base_q = torch.randn(args.batch_size, seq_len, d_model, device=device, dtype=dtype)
                base_k = torch.randn(args.batch_size, seq_len, d_model, device=device, dtype=dtype)
                base_v = torch.randn(args.batch_size, seq_len, d_model, device=device, dtype=dtype)
                base_grad_out = torch.randn(args.batch_size, seq_len, d_model, device=device, dtype=dtype)

                variants: list[tuple[str, torch.nn.Module]] = [("vanilla", AttentionModule())]
                if args.compare_compile:
                    compiled = torch.compile(AttentionModule(), mode=args.compile_mode)
                    variants.append(("compiled", compiled))

                for variant_name, module in variants:
                    q = base_q.detach().clone().requires_grad_(True)
                    k = base_k.detach().clone().requires_grad_(True)
                    v = base_v.detach().clone().requires_grad_(True)
                    grad_out = base_grad_out.detach().clone()

                    fwd_ms, bwd_ms, mem_mean, mem_max = benchmark_variant(
                        attn_module=module,
                        q=q, k=k, v=v, grad_out=grad_out, mask=mask,
                        warmup_iters=args.warmup_iters,
                        measure_iters=args.measure_iters,
                        device=device,
                    )
                    rows.append(BenchmarkRow(
                        variant=variant_name, d_model=d_model, seq_len=seq_len,
                        status="ok", forward_ms_mean=fwd_ms, backward_ms_mean=bwd_ms,
                        memory_before_backward_mib_mean=mem_mean,
                        memory_before_backward_mib_max=mem_max,
                        note="",
                    ))

            except torch.cuda.OutOfMemoryError as exc:
                err = str(exc).split("\n")[0]
                for v in (["vanilla", "compiled"] if args.compare_compile else ["vanilla"]):
                    rows.append(BenchmarkRow(
                        variant=v, d_model=d_model, seq_len=seq_len, status="oom",
                        forward_ms_mean=None, backward_ms_mean=None,
                        memory_before_backward_mib_mean=None,
                        memory_before_backward_mib_max=None,
                        note=err,
                    ))
                torch.cuda.empty_cache()

            except RuntimeError as exc:
                err = str(exc).split("\n")[0]
                for v in (["vanilla", "compiled"] if args.compare_compile else ["vanilla"]):
                    rows.append(BenchmarkRow(
                        variant=v, d_model=d_model, seq_len=seq_len, status="runtime_error",
                        forward_ms_mean=None, backward_ms_mean=None,
                        memory_before_backward_mib_mean=None,
                        memory_before_backward_mib_max=None,
                        note=err,
                    ))
                torch.cuda.empty_cache()

    if not rows:
        print("No attention benchmark rows produced.")
        return

    args.csv_path.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    print("\nAttention benchmark results (vanilla):")
    print("| d_model | seq_len | status | forward (ms) | backward (ms) | mem before bwd (MiB) mean | mem before bwd (MiB) max |")
    print("|---:|---:|---|---:|---:|---:|---:|")
    for r in rows:
        if r.variant != "vanilla":
            continue
        if r.status != "ok":
            print(f"| {r.d_model} | {r.seq_len} | {r.status} | - | - | - | - |")
        else:
            print(
                f"| {r.d_model} | {r.seq_len} | ok "
                f"| {r.forward_ms_mean:.3f} | {r.backward_ms_mean:.3f} "
                f"| {r.memory_before_backward_mib_mean:.1f} | {r.memory_before_backward_mib_max:.1f} |"
            )

    if args.compare_compile:
        print("\nAttention torch.compile comparison (mean ms):")
        print("| d_model | seq_len | vanilla fwd | compiled fwd | vanilla bwd | compiled bwd |")
        print("|---:|---:|---:|---:|---:|---:|")
        for d_model in args.d_models:
            for seq_len in args.seq_lens:
                van = next((r for r in rows if r.variant == "vanilla" and r.d_model == d_model and r.seq_len == seq_len), None)
                com = next((r for r in rows if r.variant == "compiled" and r.d_model == d_model and r.seq_len == seq_len), None)
                if van is None or com is None:
                    continue
                if van.status != "ok" or com.status != "ok":
                    print(f"| {d_model} | {seq_len} | {van.status} | {com.status} | - | - |")
                else:
                    print(
                        f"| {d_model} | {seq_len} "
                        f"| {van.forward_ms_mean:.3f} | {com.forward_ms_mean:.3f} "
                        f"| {van.backward_ms_mean:.3f} | {com.backward_ms_mean:.3f} |"
                    )
        print(f"compile_mode={args.compile_mode}")

    print(f"Wrote {len(rows)} attention rows to {args.csv_path}")


def run_model_compile_benchmark(args: argparse.Namespace, device: torch.device) -> None:
    """Problem (torch_compile)(b): compile entire Transformer model."""
    rows: list[ModelBenchmarkRow] = []

    for size_name in args.model_sizes:
        cfg = MODEL_CONFIGS[size_name]
        for mode in ("forward", "train"):
            print(f"\n[model compile] size={size_name}, mode={mode} ...", flush=True)
            for variant_name in ("vanilla", "compiled"):
                try:
                    torch.cuda.empty_cache()
                    model = TransformerLM(
                        vocab_size=VOCAB_SIZE,
                        context_length=args.model_seq_len,
                        **cfg,
                    ).to(device)

                    if variant_name == "compiled":
                        model = torch.compile(model)

                    x = torch.randint(0, VOCAB_SIZE, (args.model_batch_size, args.model_seq_len), device=device)
                    y = torch.randint(0, VOCAB_SIZE, (args.model_batch_size, args.model_seq_len), device=device)

                    mean_ms, std_ms = benchmark_model_variant(
                        model=model, x=x, y=y, mode=mode,
                        warmup=args.model_warmup, measure=args.model_measure,
                        device=device,
                    )
                    print(f"  {variant_name:>10s}  {mean_ms:8.1f} ± {std_ms:5.1f} ms")
                    rows.append(ModelBenchmarkRow(
                        variant=variant_name, model_size=size_name, mode=mode,
                        seq_len=args.model_seq_len, batch_size=args.model_batch_size,
                        status="ok", mean_ms=mean_ms, std_ms=std_ms, note="",
                    ))

                    del model, x, y
                    torch.cuda.empty_cache()

                except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                    err_status = "oom" if isinstance(exc, torch.cuda.OutOfMemoryError) else "runtime_error"
                    err = str(exc).split("\n")[0]
                    print(f"  {variant_name:>10s}  {err_status}: {err}")
                    rows.append(ModelBenchmarkRow(
                        variant=variant_name, model_size=size_name, mode=mode,
                        seq_len=args.model_seq_len, batch_size=args.model_batch_size,
                        status=err_status, mean_ms=None, std_ms=None, note=err,
                    ))
                    torch.cuda.empty_cache()

    if not rows:
        print("No model compile benchmark rows produced.")
        return

    args.model_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with args.model_csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    print("\nWhole-model torch.compile benchmark:")
    print("| model | mode | vanilla (ms) | compiled (ms) | speedup |")
    print("|---|---|---:|---:|---:|")
    for size_name in args.model_sizes:
        for mode in ("forward", "train"):
            van = next((r for r in rows if r.variant == "vanilla" and r.model_size == size_name and r.mode == mode and r.status == "ok"), None)
            com = next((r for r in rows if r.variant == "compiled" and r.model_size == size_name and r.mode == mode and r.status == "ok"), None)
            if van is None or com is None:
                van_s = f"{van.mean_ms:.1f}" if van and van.status == "ok" else "OOM/err"
                com_s = f"{com.mean_ms:.1f}" if com and com.status == "ok" else "OOM/err"
                print(f"| {size_name} | {mode} | {van_s} | {com_s} | - |")
            else:
                speedup = van.mean_ms / com.mean_ms if com.mean_ms > 0 else float("inf")
                print(f"| {size_name} | {mode} | {van.mean_ms:.1f} | {com.mean_ms:.1f} | {speedup:.2f}x |")

    print(f"Wrote {len(rows)} model rows to {args.model_csv_path}")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = get_dtype(args.dtype)

    if device.type != "cuda":
        raise RuntimeError("This benchmark is designed for CUDA; use --device cuda on a GPU machine.")

    torch.manual_seed(0)

    if not args.skip_attention:
        run_attention_benchmark(args, device, dtype)

    if args.compare_compile_model:
        run_model_compile_benchmark(args, device)


if __name__ == "__main__":
    '''
    Problem (pytorch_attention): vanilla attention benchmark
    uv run python benchmark_attention.py
    - batch=8, d_model ∈ [16,32,64,128] × seq_len ∈ [256,1024,4096,8192,16384]
    - 100 次 forward 计时 + backward 前内存 + 100 次 backward 计时
    - OOM 捕获，输出 Markdown 表格 + CSV


    Problem (torch_compile)(a): compiled vs vanilla attention
    uv run python benchmark_attention.py --compare-compile
    - 同样的配置，增加一列 compiled attention 对比

    Problem (torch_compile)(b): 整个 Transformer 模型 compiled vs vanilla
    uv run python benchmark_attention.py --compare-compile-model

    - 遍历 Table 1 所有 model size (small/medium/large/xl/2.7b)
    - 每个 size 跑 forward-only 和 train (forward+backward+optimizer) 两种模式
    - vanilla vs torch.compile(model) 对比，输出含 speedup 的表格 + CSV

    全部一起跑
    uv run python benchmark_attention.py --compare-compile --compare-compile-model

    只跑模型级别（跳过 attention）
    uv run python benchmark_attention.py --skip-attention --compare-compile-model 
    '''
    main()
