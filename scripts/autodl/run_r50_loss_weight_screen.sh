#!/usr/bin/env bash
set -Eeuo pipefail

# Two controlled CLRNet-R50 screens for the false-positive hypothesis.  The
# only model/training change is model.head.cfg.cls_loss_weight; preprocessing,
# split, initialization, seed, and checkpoint/evaluation policy stay fixed.

PROJECT_ROOT="${HARDLANE_PROJECT_ROOT:-/hy-tmp/lane-detection-challenge}"
OUTPUT_ROOT="${HARDLANE_OUTPUT_ROOT:-/hy-tmp/lane-outputs}"
PYTHON_BIN="${HARDLANE_PYTHON:-/usr/local/miniconda3/envs/py39/bin/python}"

for name in HARDLANE_PROJECT_ROOT HARDLANE_DATA_ROOT UNLANEDET_ROOT \
  HARDLANE_WEIGHTS_ROOT HARDLANE_OUTPUT_ROOT; do
  value="${!name:-}"
  if [[ -z "$value" || "$value" != /* ]]; then
    echo "ERROR: $name must be set to an absolute AutoDL path" >&2
    exit 2
  fi
done

project="$HARDLANE_PROJECT_ROOT"
runner="$project/scripts/autodl/run_training.py"
replay="$project/scripts/autodl/evaluate_selected.py"

for spec in "3.0:screen_r50_clsweight3_15ep" "4.0:screen_r50_clsweight4_15ep"; do
  weight="${spec%%:*}"
  run_name="${spec#*:}"
  "$PYTHON_BIN" "$runner" \
    --model clrnet_r50 \
    --epochs 15 \
    --run-name "$run_name" \
    --auto-resume \
    --skip-if-complete \
    --override "model.head.cfg.cls_loss_weight=$weight"
  "$PYTHON_BIN" "$replay" \
    --run-dir "$OUTPUT_ROOT/runs/$run_name" \
    --skip-if-complete
done

echo "R50 classification-loss screen complete; evidence is under $OUTPUT_ROOT/runs"
