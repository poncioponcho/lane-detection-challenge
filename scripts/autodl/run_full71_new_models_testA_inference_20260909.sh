#!/usr/bin/env bash
set -uo pipefail

# After the full-71 training queues finish, export testA predictions for the
# new full-data checkpoints. This is inference only: no test image is used for
# training and this script never submits to A/B榜.
PROJECT_ROOT=/hy-tmp/lane-detection-challenge
DATA_ROOT=/hy-tmp/datasets/HardLane/Lane
UNLANEDET_ROOT=/hy-tmp/UnLanedet
WEIGHTS_ROOT=/hy-tmp/weights
OUTPUT_ROOT=/hy-tmp/lane-outputs
PYTHON_BIN=/usr/local/miniconda3/envs/py39/bin/python

TRAIN_STATUS=$OUTPUT_ROOT/all71_os_hardknown_x5_seed48_clrnet_r50_36ep_20260909.status
TRAIN_SCRIPT=$OUTPUT_ROOT/run_full71_os_hardknown_x5_after_queue_20260909.sh
STATUS=$OUTPUT_ROOT/full71_new_models_testA_inference_20260909.status
LOG=$OUTPUT_ROOT/full71_new_models_testA_inference_20260909.log
RUNS=$OUTPUT_ROOT/full71_new_models_testA_inference_20260909.runs
LOCK=$OUTPUT_ROOT/.full71_new_models_testA_inference_20260909.lock
WAIT_TIMEOUT_SECONDS=$((48 * 3600))

log() {
    printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "$LOG"
}

set_status() {
    printf '%s\n' "$1" > "$STATUS"
}

mkdir -p "$OUTPUT_ROOT"
if ! mkdir "$LOCK" 2>/dev/null; then
    owner=
    if [ -f "$LOCK/pid" ]; then
        owner=$(sed -n '1p' "$LOCK/pid" 2>/dev/null || true)
    fi
    if [ -n "$owner" ] && kill -0 "$owner" 2>/dev/null \
        && ps -p "$owner" -o args= 2>/dev/null \
        | grep -Fq 'run_full71_new_models_testA_inference_20260909.sh'; then
        exit 0
    fi
    if [ -f "$STATUS" ]; then
        state=$(sed -n '1p' "$STATUS" 2>/dev/null || true)
        case "$state" in
            complete|complete_with_failures|failed|aborted_*) exit 0 ;;
        esac
    fi
    rm -f "$LOCK/pid" 2>/dev/null || true
    rmdir "$LOCK" 2>/dev/null || exit 0
    mkdir "$LOCK" 2>/dev/null || exit 0
fi
printf '%s\n' "$$" > "$LOCK/pid"
trap 'rm -f "$LOCK/pid" 2>/dev/null; rmdir "$LOCK" 2>/dev/null || true' EXIT

touch "$RUNS"
set_status waiting_for_full71_training
log "queued behind train status=$TRAIN_STATUS script=$TRAIN_SCRIPT"

deadline=$(( $(date +%s) + WAIT_TIMEOUT_SECONDS ))
while true; do
    training_state=$(sed -n '1p' "$TRAIN_STATUS" 2>/dev/null || true)
    case "$training_state" in
        complete|complete_with_failures|failed|aborted_*)
            log "training queue reached terminal state=$training_state"
            break
            ;;
    esac
    # The terminal dependency is the x5 launcher.  The former check only
    # looked for the x3 name, so inference could abort in the interval after
    # x3 finished but while x5 was still training.
    training_pids=$(pgrep -f '[r]un_full71_os_hardknown_x5_after_queue_20260909.sh' 2>/dev/null || true)
    if [ -z "$training_pids" ] && [ -n "$training_state" ]; then
        set_status aborted_training_queue_missing
        log "training queue launcher absent with nonterminal state=$training_state"
        exit 23
    fi
    if [ "$(date +%s)" -gt "$deadline" ]; then
        set_status aborted_timeout_waiting_training
        log "timed out waiting for training queue (last=$training_state)"
        exit 24
    fi
    sleep 60
done

set_status waiting_for_gpu_drain
drain_deadline=$(( $(date +%s) + $((3 * 3600)) ))
while true; do
    gpu_procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
        | awk 'NF' | wc -l | tr -d ' ')
    train_procs=$(pgrep -f '[t]ools/train_net.py' 2>/dev/null || true)
    lvo_procs=$(pgrep -f '[l]vo_video_runner.py' 2>/dev/null || true)
    if [ -z "$gpu_procs" ]; then gpu_procs=0; fi
    if [ "$gpu_procs" -eq 0 ] && [ -z "$train_procs" ] && [ -z "$lvo_procs" ]; then
        break
    fi
    if [ "$(date +%s)" -gt "$drain_deadline" ]; then
        set_status aborted_gpu_drain_timeout
        log "GPU/process drain timed out: gpu=$gpu_procs train=$train_procs lvo=$lvo_procs"
        exit 25
    fi
    sleep 30
done

export HARDLANE_PROJECT_ROOT=$PROJECT_ROOT
export HARDLANE_DATA_ROOT=$DATA_ROOT
export UNLANEDET_ROOT=$UNLANEDET_ROOT
export HARDLANE_WEIGHTS_ROOT=$WEIGHTS_ROOT
export HARDLANE_OUTPUT_ROOT=$OUTPUT_ROOT
export HARDLANE_PYTHON=$PYTHON_BIN

for name in \
    all71_seed45_clrnet_r50_36ep \
    all71_seed46_clrnet_r50_36ep \
    all71_os_hardknown_x3_seed47_clrnet_r50_36ep_20260909 \
    all71_os_hardknown_x5_seed48_clrnet_r50_36ep_20260909; do
    run_dir=$OUTPUT_ROOT/runs/$name
    evidence=$run_dir/run_evidence.json
    if [ ! -f "$evidence" ]; then
        printf '%s:missing_run_evidence\n' "$name" >> "$RUNS"
        log "skip $name: run evidence missing"
        continue
    fi
    state=$(
        "$PYTHON_BIN" - "$evidence" <<'PY'
import json
import sys
from pathlib import Path

value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(value.get("status", "invalid"))
PY
    )
    if [ "$state" != pass ]; then
        printf '%s:not_pass:%s\n' "$name" "$state" >> "$RUNS"
        log "skip $name: run evidence status=$state"
        continue
    fi

    output_dir=$run_dir/testA_infer
    set_status running_$name
    log "start testA inference name=$name"
    if "$PYTHON_BIN" "$PROJECT_ROOT/scripts/autodl/infer_testA.py" \
        --run-dir "$run_dir" \
        --split testA \
        --output-dir "$output_dir" \
        --conf-threshold 0.50 \
        --skip-if-complete >> "$LOG" 2>&1; then
        printf '%s:ok\n' "$name" >> "$RUNS"
        log "done testA inference name=$name"
    else
        code=$?
        printf '%s:failed:%s\n' "$name" "$code" >> "$RUNS"
        log "FAILED testA inference name=$name exit=$code"
    fi
done

if grep -q ':failed:' "$RUNS" 2>/dev/null; then
    set_status complete_with_failures
else
    set_status complete
fi
log "new full71 model inference queue complete"
