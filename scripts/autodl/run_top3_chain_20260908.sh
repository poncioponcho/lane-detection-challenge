#!/usr/bin/env bash
set -uo pipefail

# §36.4 T1 chain: 63-clip seeds 43/44, then full-71-clip seeds 42/43/44,
# all 36 epochs, sequential on the single RTX 3090.  Every stage launches
# through scripts/autodl/run_training.py so the launches.jsonl / run_evidence
# discipline applies.  While this chain runs, do not push new commits to the
# instance repo (cross-commit recovery guard).  Checkpoint retention is capped
# at 6 periodic checkpoints (~2.3 GB per run) because 36-checkpoint retention
# across 5 runs would exhaust the /hy-tmp disk.

PROJECT_ROOT=/hy-tmp/lane-detection-challenge
UNLANEDET_ROOT=/hy-tmp/UnLanedet
DATA_ROOT=/hy-tmp/datasets/HardLane/Lane
WEIGHTS_ROOT=/hy-tmp/weights
OUTPUT_ROOT=/hy-tmp/lane-outputs
PYTHON_BIN=/usr/local/miniconda3/envs/py39/bin/python
ALL71_MANIFEST="$OUTPUT_ROOT/experiments_manifest_train_all71.jsonl"

export HARDLANE_PROJECT_ROOT="$PROJECT_ROOT"
export HARDLANE_DATA_ROOT="$DATA_ROOT"
export HARDLANE_WEIGHTS_ROOT="$WEIGHTS_ROOT"
export HARDLANE_OUTPUT_ROOT="$OUTPUT_ROOT"
export HARDLANE_PYTHON="$PYTHON_BIN"

STATUS="$OUTPUT_ROOT/top3_chain_20260908.status"
LOG="$OUTPUT_ROOT/top3_chain_20260908.log"
RUNS="$OUTPUT_ROOT/top3_chain_20260908.runs"

write_status() { printf '%s\n' "$1" > "$STATUS"; }
log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "$LOG"; }

for path in "$PROJECT_ROOT" "$UNLANEDET_ROOT" "$DATA_ROOT" "$WEIGHTS_ROOT" \
            "$ALL71_MANIFEST" \
            "$PROJECT_ROOT/scripts/autodl/run_training.py"; do
    [ -e "$path" ] || { echo "missing required path: $path" >&2; exit 4; }
done

unlanedet_head=$(git -C "$UNLANEDET_ROOT" rev-parse HEAD)
[ "$unlanedet_head" = 03921844220adb2e65c840de2d9759478d5c3d4c ] || {
    echo "unexpected UnLanedet HEAD: $unlanedet_head" >&2
    exit 5
}

run_one() {
    name=$1; seed=$2; shift 2
    log "start $name seed=$seed extra_args=$*"
    if "$PYTHON_BIN" "$PROJECT_ROOT/scripts/autodl/run_training.py" \
        --model clrnet_r50 --run-name "$name" --epochs 36 \
        --seed "$seed" --max-to-keep 6 \
        --skip-if-complete --auto-resume \
        "$@" >> "$LOG" 2>&1; then
        log "done $name"
        printf '%s:ok\n' "$name" >> "$RUNS"
    else
        code=$?
        log "FAILED $name exit=$code"
        printf '%s:failed:%s\n' "$name" "$code" >> "$RUNS"
    fi
}

write_status running
log "chain start"
run_one seed43_clrnet_r50_36ep 43
run_one seed44_clrnet_r50_36ep 44
run_one all71_seed42_clrnet_r50_36ep 42 \
    --train-manifest "$ALL71_MANIFEST" --iters-per-epoch 592 --eval-every-epochs 6
run_one all71_seed43_clrnet_r50_36ep 43 \
    --train-manifest "$ALL71_MANIFEST" --iters-per-epoch 592 --eval-every-epochs 6
run_one all71_seed44_clrnet_r50_36ep 44 \
    --train-manifest "$ALL71_MANIFEST" --iters-per-epoch 592 --eval-every-epochs 6
write_status complete
log "chain complete"
