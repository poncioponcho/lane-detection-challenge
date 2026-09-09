#!/usr/bin/env bash
set -uo pipefail

# Queue two additional full-71-clip seeds behind the currently running
# VAT/mask screen. This is an external launcher: it does not modify the
# training checkout while the existing chain is active, and it never uses
# testA/testB images for training.

PROJECT_ROOT="${PROJECT_ROOT:-/hy-tmp/lane-detection-challenge}"
DATA_ROOT="${DATA_ROOT:-/hy-tmp/datasets/HardLane/Lane}"
UNLANEDET_ROOT="${UNLANEDET_ROOT:-/hy-tmp/UnLanedet}"
WEIGHTS_ROOT="${WEIGHTS_ROOT:-/hy-tmp/weights}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/hy-tmp/lane-outputs}"
PYTHON_BIN="${PYTHON_BIN:-/usr/local/miniconda3/envs/py39/bin/python}"
CHAIN_PID="${CHAIN_PID:-723018}"

ALL71_MANIFEST="$OUTPUT_ROOT/experiments_manifest_train_all71.jsonl"
CHAIN_STATUS="$OUTPUT_ROOT/vat_lowweight_mask_chain_20260909.status"
STATUS="$OUTPUT_ROOT/full71_seed45_46_after_chain.status"
LOG="$OUTPUT_ROOT/full71_seed45_46_after_chain.log"
RUNS="$OUTPUT_ROOT/full71_seed45_46_after_chain.runs"

log() {
    printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "$LOG"
}

set_status() {
    printf '%s\n' "$1" > "$STATUS"
}

mkdir -p "$OUTPUT_ROOT"
: > "$RUNS"
set_status waiting_for_vat_mask_chain
log "queued behind chain pid=$CHAIN_PID status=$CHAIN_STATUS"

# Wait for the external chain shell itself. Its status may be a failure
# state; either way its children must be gone before this launcher uses CUDA.
while [ "$CHAIN_PID" -gt 0 ] && kill -0 "$CHAIN_PID" 2>/dev/null; do
    sleep 60
done

drain_deadline=$(( $(date +%s) + 7200 ))
while true; do
    gpu_procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | awk 'NF' | wc -l | tr -d ' ')
    lvo_procs=$(pgrep -f '[l]vo_video_runner.py' 2>/dev/null || true)
    training_procs=$(pgrep -f '[t]ools/train_net.py' 2>/dev/null || true)
    if [ "${gpu_procs:-0}" -eq 0 ] && [ -z "$lvo_procs" ] && [ -z "$training_procs" ]; then
        break
    fi
    if [ "$(date +%s)" -gt "$drain_deadline" ]; then
        set_status aborted_gpu_drain_timeout
        log "GPU/process drain timed out: gpu=$gpu_procs lvo=$lvo_procs train=$training_procs"
        exit 25
    fi
    sleep 30
done

for path in "$PROJECT_ROOT/scripts/autodl/run_training.py" "$ALL71_MANIFEST"; do
    if [ ! -e "$path" ]; then
        set_status aborted_missing_input
        log "missing required path: $path"
        exit 4
    fi
done

export HARDLANE_PROJECT_ROOT="$PROJECT_ROOT"
export HARDLANE_DATA_ROOT="$DATA_ROOT"
export UNLANEDET_ROOT="$UNLANEDET_ROOT"
export HARDLANE_WEIGHTS_ROOT="$WEIGHTS_ROOT"
export HARDLANE_OUTPUT_ROOT="$OUTPUT_ROOT"
export HARDLANE_PYTHON="$PYTHON_BIN"

set_status running
log "chain ended with status=$(cat "$CHAIN_STATUS" 2>/dev/null || echo missing); starting full seeds"

for seed in 45 46; do
    name="all71_seed${seed}_clrnet_r50_36ep"
    log "start $name seed=$seed"
    if "$PYTHON_BIN" "$PROJECT_ROOT/scripts/autodl/run_training.py" \
        --model clrnet_r50 \
        --run-name "$name" \
        --epochs 36 \
        --seed "$seed" \
        --train-manifest "$ALL71_MANIFEST" \
        --iters-per-epoch 592 \
        --eval-every-epochs 6 \
        --max-to-keep 2 \
        --skip-if-complete \
        --auto-resume >> "$LOG" 2>&1; then
        printf '%s:ok\n' "$name" >> "$RUNS"
        log "done $name"
    else
        code=$?
        printf '%s:failed:%s\n' "$name" "$code" >> "$RUNS"
        log "FAILED $name exit=$code"
    fi
done

if grep -q ':failed:' "$RUNS" 2>/dev/null; then
    set_status complete_with_failures
else
    set_status complete
fi
log "full seed45/46 queue complete"
