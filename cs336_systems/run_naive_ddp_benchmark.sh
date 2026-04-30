#!/usr/bin/env bash
set -euo pipefail

# One-shot benchmark for Problem (naive_ddp_benchmarking)
# Default setup: 1 node x 2 GPUs, XL model config.

OUT_DIR=${OUT_DIR:-artifacts/systems/naive_ddp_benchmarking}
mkdir -p "${OUT_DIR}"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

uv run python naive_ddp_benchmark.py \
  --backend nccl \
  --device cuda \
  --world-size 2 \
  --precision bf16 \
  --optimizer sgd \
  --warmup-steps 5 \
  --measure-steps 10 \
  --batch-size 2 \
  --context-length 128 \
  --vocab-size 50257 \
  --d-model 2560 \
  --num-layers 32 \
  --num-heads 32 \
  --d-ff 10240 \
  --csv-out "${OUT_DIR}/naive_ddp_benchmark_steps.csv" \
  --summary-out "${OUT_DIR}/naive_ddp_benchmark_summary.md"

echo "[DONE] outputs:"
echo "  - ${OUT_DIR}/naive_ddp_benchmark_steps.csv"
echo "  - ${OUT_DIR}/naive_ddp_benchmark_summary.md"