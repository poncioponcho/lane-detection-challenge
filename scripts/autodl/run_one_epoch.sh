#!/usr/bin/env bash
set -Eeuo pipefail

model_name="${1:-}"
case "$model_name" in
  clrnet_r50|adnet_r34) ;;
  *)
    echo "usage: $0 {clrnet_r50|adnet_r34}" >&2
    exit 2
    ;;
esac

for name in HARDLANE_PROJECT_ROOT HARDLANE_DATA_ROOT UNLANEDET_ROOT \
  HARDLANE_WEIGHTS_ROOT HARDLANE_OUTPUT_ROOT; do
  value="${!name:-}"
  if [[ -z "$value" || "$value" != /* ]]; then
    echo "ERROR: $name must be set to an absolute AutoDL path" >&2
    exit 2
  fi
done

python_bin="${HARDLANE_PYTHON:-python}"
"$python_bin" "$HARDLANE_PROJECT_ROOT/scripts/autodl/run_training.py" \
  --model "$model_name" \
  --epochs 1 \
  --run-name "gate_${model_name}_1ep" \
  --auto-resume \
  --skip-if-complete
