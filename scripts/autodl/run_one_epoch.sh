#!/usr/bin/env bash
set -Eeuo pipefail

model_name="${1:-}"
case "$model_name" in
  clrnet_r50)
    config_name="clrnet_r50_hardlane.py"
    iterations=525
    ;;
  adnet_r34)
    config_name="adnet_r34_hardlane.py"
    iterations=525
    ;;
  *)
    echo "usage: $0 {clrnet_r50|adnet_r34}" >&2
    exit 2
    ;;
esac

for name in HARDLANE_PROJECT_ROOT HARDLANE_DATA_ROOT UNLANEDET_ROOT HARDLANE_WEIGHTS_ROOT; do
  value="${!name:-}"
  if [[ -z "$value" || "$value" != /* ]]; then
    echo "ERROR: $name must be set to an absolute AutoDL path" >&2
    exit 2
  fi
done

python_bin="${HARDLANE_PYTHON:-python}"
output_base="${HARDLANE_OUTPUT_ROOT:-$HARDLANE_PROJECT_ROOT/outputs/autodl}"
run_dir="$output_base/one_epoch/$model_name"
mkdir -p "$run_dir"

cd "$UNLANEDET_ROOT"
"$python_bin" tools/train_net.py \
  --config-file "$HARDLANE_PROJECT_ROOT/configs/unlanedet/$config_name" \
  --num-gpus 1 \
  "train.max_iter=$iterations" \
  "train.eval_period=$iterations" \
  "train.checkpointer.period=$iterations" \
  "train.output_dir=$run_dir" \
  "dataloader.evaluator.output_basedir=$run_dir/val" \
  2>&1 | tee "$run_dir/train.log"

test -f "$run_dir/val/diagnostic_metric.json"
echo "AutoDL one-epoch train+val complete: $run_dir"
