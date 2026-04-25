#!/usr/bin/env bash

set -euo pipefail

MODEL_SIZES=(small medium large xl 2.7b)
MODES=(forward train)
WARMUP_STEPS=(5)

SEQ_LEN=512
BATCH_SIZE=4
MEASURE_STEPS=10

for mode in "${MODES[@]}"; do
  for model_size in "${MODEL_SIZES[@]}"; do
    for warmup_steps in "${WARMUP_STEPS[@]}"; do
      echo "============================================================"
      echo "Running model_size=${model_size} mode=${mode} warmup_steps=${warmup_steps}"
      echo "============================================================"

      uv run python benchmarking.py \
        --model_size "${model_size}" \
        --seq_len "${SEQ_LEN}" \
        --batch_size "${BATCH_SIZE}" \
        --warmup_steps "${warmup_steps}" \
        --measure_steps "${MEASURE_STEPS}" \
        --mode "${mode}"
    done
  done
done
