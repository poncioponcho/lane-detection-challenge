#!/usr/bin/env bash
set -uo pipefail

# Extra consensus candidates selected from the frozen full-71 validation
# replay.  The original queue uses max_dx=15; this isolated queue adds the
# validated wider matching band (30/35 px) without overwriting any output or
# uploading a submission.  It is CPU-only and waits for inference plus the
# original candidate queue to reach a terminal state.

PROJECT_ROOT=/hy-tmp/lane-detection-challenge
DATA_ROOT=/hy-tmp/datasets/HardLane/Lane
UNLANEDET_ROOT=/hy-tmp/UnLanedet
WEIGHTS_ROOT=/hy-tmp/weights
OUTPUT_ROOT=/hy-tmp/lane-outputs
PYTHON_BIN=/usr/local/miniconda3/envs/py39/bin/python

INFER_STATUS=$OUTPUT_ROOT/full71_new_models_testA_inference_20260909.status
INFER_SCRIPT=$OUTPUT_ROOT/run_full71_new_models_testA_inference_20260909.sh
ORIGINAL_STATUS=$OUTPUT_ROOT/ensemble_candidates_20260909.status
ORIGINAL_SCRIPT=$OUTPUT_ROOT/run_full71_ensemble_candidates_after_inference_20260909.sh
ROOT=$OUTPUT_ROOT/runs/ensemble_candidates_dx30_20260909
STATUS=$OUTPUT_ROOT/ensemble_candidates_dx30_20260909.status
LOG=$OUTPUT_ROOT/ensemble_candidates_dx30_20260909.log
RUNS=$OUTPUT_ROOT/ensemble_candidates_dx30_20260909.runs
LOCK=$OUTPUT_ROOT/.ensemble_candidates_dx30_20260909.lock
WAIT_TIMEOUT_SECONDS=$((48 * 3600))

BASE=$OUTPUT_ROOT/runs
PRED42=$BASE/all71_seed42_clrnet_r50_36ep/testA_infer/testA/predictions
SCORE42=$BASE/all71_seed42_clrnet_r50_36ep/testA_infer/testA/prediction_scores.json
PRED43=$BASE/all71_seed43_clrnet_r50_36ep/testA_infer/testA/predictions
SCORE43=$BASE/all71_seed43_clrnet_r50_36ep/testA_infer/testA/prediction_scores.json
PRED44=$BASE/all71_seed44_clrnet_r50_36ep/testA_infer/testA/predictions
SCORE44=$BASE/all71_seed44_clrnet_r50_36ep/testA_infer/testA/prediction_scores.json
PRED45=$BASE/all71_seed45_clrnet_r50_36ep/testA_infer/testA/predictions
SCORE45=$BASE/all71_seed45_clrnet_r50_36ep/testA_infer/testA/prediction_scores.json
PRED46=$BASE/all71_seed46_clrnet_r50_36ep/testA_infer/testA/predictions
SCORE46=$BASE/all71_seed46_clrnet_r50_36ep/testA_infer/testA/prediction_scores.json
PREDX3=$BASE/all71_os_hardknown_x3_seed47_clrnet_r50_36ep_20260909/testA_infer/testA/predictions
SCOREX3=$BASE/all71_os_hardknown_x3_seed47_clrnet_r50_36ep_20260909/testA_infer/testA/prediction_scores.json
PREDX5=$BASE/all71_os_hardknown_x5_seed48_clrnet_r50_36ep_20260909/testA_infer/testA/predictions
SCOREX5=$BASE/all71_os_hardknown_x5_seed48_clrnet_r50_36ep_20260909/testA_infer/testA/prediction_scores.json

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
        | grep -Fq 'run_full71_ensemble_candidates_dx30_after_inference_20260909.sh'; then
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
set_status waiting_for_inference
log "queued behind inference=$INFER_STATUS original_candidates=$ORIGINAL_STATUS"

deadline=$(( $(date +%s) + WAIT_TIMEOUT_SECONDS ))
while true; do
    inference_state=$(sed -n '1p' "$INFER_STATUS" 2>/dev/null || true)
    original_state=$(sed -n '1p' "$ORIGINAL_STATUS" 2>/dev/null || true)
    inference_done=0
    original_done=0
    case "$inference_state" in
        complete|complete_with_failures|failed|aborted_*) inference_done=1 ;;
    esac
    case "$original_state" in
        complete|complete_with_failures|failed|aborted_*) original_done=1 ;;
    esac
    if [ "$inference_done" -eq 1 ] && [ "$original_done" -eq 1 ]; then
        break
    fi
    if [ "$inference_done" -eq 0 ] && [ -z "$(pgrep -f '[r]un_full71_new_models_testA_inference_20260909.sh' 2>/dev/null || true)" ]; then
        set_status aborted_inference_queue_missing
        log "inference launcher absent with nonterminal state=$inference_state"
        exit 23
    fi
    if [ "$original_done" -eq 0 ] && [ -z "$(pgrep -f '[r]un_full71_ensemble_candidates_after_inference_20260909.sh' 2>/dev/null || true)" ]; then
        set_status aborted_original_queue_missing
        log "original candidate launcher absent with nonterminal state=$original_state"
        exit 23
    fi
    if [ "$(date +%s)" -gt "$deadline" ]; then
        set_status aborted_timeout_waiting_upstream
        log "timed out waiting upstream inference=$inference_state original=$original_state"
        exit 24
    fi
    sleep 60
done

set_status validating_inputs
for path in "$PROJECT_ROOT/scripts/autodl/consensus_fusion.py" \
            "$PROJECT_ROOT/scripts/autodl/package_testA_submit.py" \
            "$PROJECT_ROOT" "$DATA_ROOT" "$UNLANEDET_ROOT"; do
    if [ ! -e "$path" ]; then
        set_status aborted_missing_input
        log "missing required input: $path"
        exit 4
    fi
done

export HARDLANE_PROJECT_ROOT=$PROJECT_ROOT
export HARDLANE_DATA_ROOT=$DATA_ROOT
export UNLANEDET_ROOT=$UNLANEDET_ROOT
export HARDLANE_WEIGHTS_ROOT=$WEIGHTS_ROOT
export HARDLANE_OUTPUT_ROOT=$OUTPUT_ROOT
export HARDLANE_PYTHON=$PYTHON_BIN

check_pair() {
    pred=$1
    score=$2
    [ -d "$pred" ] && [ -f "$score" ] || return 1
    count=$(find "$pred" -type f -name '*.lines.txt' | wc -l | tr -d ' ')
    [ "$count" -eq 900 ]
}

spec_new4=
if check_pair "$PRED45" "$SCORE45" \
    && check_pair "$PRED46" "$SCORE46" \
    && check_pair "$PREDX3" "$SCOREX3" \
    && check_pair "$PREDX5" "$SCOREX5"; then
    spec_new4="--seed $PRED45=$SCORE45 --seed $PRED46=$SCORE46 --seed $PREDX3=$SCOREX3 --seed $PREDX5=$SCOREX5"
else
    log "new4 inputs incomplete; new4 candidates skipped"
fi

spec_old3=
if check_pair "$PRED42" "$SCORE42" \
    && check_pair "$PRED43" "$SCORE43" \
    && check_pair "$PRED44" "$SCORE44"; then
    spec_old3="--seed $PRED42=$SCORE42 --seed $PRED43=$SCORE43 --seed $PRED44=$SCORE44"
else
    log "old3 inputs incomplete; old3 candidates skipped"
fi

spec_all7=
if check_pair "$PRED42" "$SCORE42" \
    && check_pair "$PRED43" "$SCORE43" \
    && check_pair "$PRED44" "$SCORE44" \
    && check_pair "$PRED45" "$SCORE45" \
    && check_pair "$PRED46" "$SCORE46" \
    && check_pair "$PREDX3" "$SCOREX3" \
    && check_pair "$PREDX5" "$SCOREX5"; then
    spec_all7="--seed $PRED42=$SCORE42 --seed $PRED43=$SCORE43 --seed $PRED44=$SCORE44 --seed $PRED45=$SCORE45 --seed $PRED46=$SCORE46 --seed $PREDX3=$SCOREX3 --seed $PREDX5=$SCOREX5"
else
    log "all7 inputs incomplete; all7 candidates skipped"
fi

mkdir -p "$ROOT"
run_candidate() {
    name=$1
    quorum=$2
    max_dx=$3
    sources=$4
    if [ -z "$sources" ]; then
        printf '%s:skipped_missing_inputs\n' "$name" >> "$RUNS"
        return 0
    fi
    out=$ROOT/$name/predictions
    anchor=$BASE/all71_seed42_clrnet_r50_36ep
    if [ -f "$ROOT/$name/submit_testA.package_evidence.json" ]; then
        printf '%s:already_complete\n' "$name" >> "$RUNS"
        return 0
    fi
    mkdir -p "$ROOT/$name"
    log "start fusion candidate=$name quorum=$quorum max_dx=$max_dx"
    if ! "$PYTHON_BIN" "$PROJECT_ROOT/scripts/autodl/consensus_fusion.py" \
        $sources --out "$out" --quorum "$quorum" --max-dx "$max_dx" \
        --geometry best --emit-scores >> "$LOG" 2>&1; then
        printf '%s:fusion_failed\n' "$name" >> "$RUNS"
        log "fusion failed candidate=$name"
        return 0
    fi
    output_count=$(find "$out" -type f -name '*.lines.txt' | wc -l | tr -d ' ')
    if [ "$output_count" -ne 900 ] || [ ! -f "$out/prediction_scores.json" ]; then
        printf '%s:fusion_incomplete:%s\n' "$name" "$output_count" >> "$RUNS"
        log "fusion output incomplete candidate=$name count=$output_count"
        return 0
    fi
    if ! "$PYTHON_BIN" "$PROJECT_ROOT/scripts/autodl/package_testA_submit.py" \
        --run-dir "$anchor" --split testA \
        --prediction-root "$out" \
        --output-zip "$ROOT/$name/submit_testA.zip" >> "$LOG" 2>&1; then
        printf '%s:package_failed\n' "$name" >> "$RUNS"
        log "package verification failed candidate=$name"
        return 0
    fi
    printf '%s:ok\n' "$name" >> "$RUNS"
    log "candidate verified name=$name"
}

for dx in 30 35; do
    run_candidate "old3_q2_dx${dx}_best" 2 "$dx" "$spec_old3"
    run_candidate "old3_q3_dx${dx}_best" 3 "$dx" "$spec_old3"
    run_candidate "new4_q2_dx${dx}_best" 2 "$dx" "$spec_new4"
    run_candidate "new4_q3_dx${dx}_best" 3 "$dx" "$spec_new4"
    run_candidate "all7_q2_dx${dx}_best" 2 "$dx" "$spec_all7"
    run_candidate "all7_q3_dx${dx}_best" 3 "$dx" "$spec_all7"
    run_candidate "all7_q4_dx${dx}_best" 4 "$dx" "$spec_all7"
    run_candidate "all7_q5_dx${dx}_best" 5 "$dx" "$spec_all7"
    run_candidate "all7_q6_dx${dx}_best" 6 "$dx" "$spec_all7"
done

if grep -q ':fusion_failed\|:package_failed' "$RUNS" 2>/dev/null; then
    set_status complete_with_failures
else
    set_status complete
fi
log "extra dx30/dx35 candidate queue complete"
