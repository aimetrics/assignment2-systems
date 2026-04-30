#!/usr/bin/env bash
set -euo pipefail

# Run all required settings for Problem: distributed_communication_single_node
# in ONE command and emit ONE csv/table/plot.

SIZES=(1 10 100 1024)
WORLD_SIZES=(2 4 6)
WARMUP_ITERS=${WARMUP_ITERS:-5}
TIMED_ITERS=${TIMED_ITERS:-20}
OUT_DIR=${OUT_DIR:-artifacts/systems/distributed_communication_single_node}

mkdir -p "${OUT_DIR}"
echo "[INFO] Output directory: ${OUT_DIR}"
echo "[INFO] warmup=${WARMUP_ITERS}, timed=${TIMED_ITERS}"

CUDA_AVAILABLE=$(uv run python - <<'PY'
import torch
print(1 if torch.cuda.is_available() else 0)
PY
)

BACKENDS=(gloo)
if [[ "${CUDA_AVAILABLE}" == "1" ]]; then
  BACKENDS+=(nccl)
  GPU_COUNT=$(uv run python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)
  echo "[INFO] CUDA available, GPUs=${GPU_COUNT}. Will include NCCL."
else
  echo "[WARN] CUDA is not available. Running only Gloo."
fi

echo "[RUN] One-shot sweep over all requested settings"
uv run python distributed_communication_single_node.py \
  --backends "${BACKENDS[@]}" \
  --world-sizes "${WORLD_SIZES[@]}" \
  --sizes-mb "${SIZES[@]}" \
  --warmup-iters "${WARMUP_ITERS}" \
  --timed-iters "${TIMED_ITERS}" \
  --allow-cpu-fallback-for-nccl \
  --csv-out "${OUT_DIR}/all_settings.csv" \
  --table-out "${OUT_DIR}/all_settings.md" \
  --plot-out "${OUT_DIR}/all_settings.png"

echo "[DONE] Wrote:"
echo "  - ${OUT_DIR}/all_settings.csv"
echo "  - ${OUT_DIR}/all_settings.md"
echo "  - ${OUT_DIR}/all_settings.png"