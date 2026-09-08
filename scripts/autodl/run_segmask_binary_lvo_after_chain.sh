#!/usr/bin/env bash
set -Eeuo pipefail

# Queue an isolated auxiliary-segmentation screen behind the already running
# architecture chain.  The original checkout is never changed: this script
# uses a detached worktree at MASK_PROJECT_ROOT and a fresh output tree.

CHAIN_PROJECT="${CHAIN_PROJECT:-/hy-tmp/lane-detection-challenge}"
MASK_PROJECT_ROOT="${MASK_PROJECT_ROOT:-/hy-tmp/lane-detection-challenge-maskbinary}"
DATA_ROOT="${HARDLANE_DATA_ROOT:-/hy-tmp/datasets/HardLane/Lane}"
UNLANEDET_ROOT="${UNLANEDET_ROOT:-/hy-tmp/UnLanedet}"
WEIGHTS_ROOT="${HARDLANE_WEIGHTS_ROOT:-/hy-tmp/weights}"
OUTPUT_ROOT="${HARDLANE_OUTPUT_ROOT:-/hy-tmp/lane-outputs}"
MANIFESTS_ROOT="${MANIFESTS_ROOT:-$OUTPUT_ROOT/lvo_plain_v1_15ep_20260909/manifests}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-$OUTPUT_ROOT/lvo_maskbinary_15ep_20260909}"
PYTHON_BIN="${HARDLANE_PYTHON:-/usr/local/miniconda3/envs/py39/bin/python}"
STATUS="${STATUS:-$OUTPUT_ROOT/lvo_maskbinary_15ep_20260909.launch.status}"
LOG="${LOG:-$OUTPUT_ROOT/lvo_maskbinary_15ep_20260909.launch.log}"

write_status() {
    printf '%s\n' "$1" > "$STATUS"
}

on_error() {
    code=$?
    write_status "failed:$code"
    exit "$code"
}
trap on_error ERR

for path in "$MASK_PROJECT_ROOT" "$DATA_ROOT" "$UNLANEDET_ROOT" "$WEIGHTS_ROOT" "$OUTPUT_ROOT" "$MANIFESTS_ROOT"; do
    [ -e "$path" ] || { echo "missing required path: $path" >&2; exit 2; }
done
[ ! -e "$EXPERIMENT_ROOT" ] || {
    echo "refusing to overwrite existing experiment: $EXPERIMENT_ROOT" >&2
    exit 3
}

mkdir -p "$OUTPUT_ROOT"
write_status waiting
printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ) binary seg-mask LVO queued" >> "$LOG"

# Wait for the pre-existing chain, including its merge-sidecar helper, to
# leave the GPU.  The path-qualified pattern does not match this worktree.
while pgrep -f "$CHAIN_PROJECT/scripts/autodl/lvo_video_runner.py" >/dev/null 2>&1 \
   || pgrep -f "$CHAIN_PROJECT/scripts/merge_lvo_oof_scores.py" >/dev/null 2>&1; do
    sleep 60
done

# Do not launch while an orphaned CUDA child remains after a watcher exits.
while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | awk 'NF')" ]; do
    sleep 30
done

export HARDLANE_PROJECT_ROOT="$MASK_PROJECT_ROOT"
export HARDLANE_DATA_ROOT="$DATA_ROOT"
export UNLANEDET_ROOT="$UNLANEDET_ROOT"
export HARDLANE_WEIGHTS_ROOT="$WEIGHTS_ROOT"
export HARDLANE_OUTPUT_ROOT="$OUTPUT_ROOT"

write_status running
printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ) binary seg-mask LVO start" >> "$LOG"

if "$PYTHON_BIN" "$MASK_PROJECT_ROOT/scripts/autodl/lvo_video_runner.py" \
    --mode all \
    --project-root "$MASK_PROJECT_ROOT" \
    --data-root "$DATA_ROOT" \
    --unlanedet-root "$UNLANEDET_ROOT" \
    --weights-root "$WEIGHTS_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --experiment-root "$EXPERIMENT_ROOT" \
    --manifests-root "$MANIFESTS_ROOT" \
    --python-bin "$PYTHON_BIN" \
    --base-config "$MASK_PROJECT_ROOT/configs/unlanedet/clrnet_r50_hardlane.py" \
    --input-width 800 \
    --input-height 320 \
    --cut-height 180 \
    --eval-workers 0 \
    --checkpoint-max-to-keep 1 \
    --iterations-per-epoch 525 \
    --resume \
    --skip-complete \
    --override "model.head.cfg.seg_mask_mode='binary_union'" \
    >> "$LOG" 2>&1; then
    write_status complete
    printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ) binary seg-mask LVO complete" >> "$LOG"
else
    code=$?
    write_status "failed:$code"
    printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ) binary seg-mask LVO failed:$code" >> "$LOG"
    exit "$code"
fi
