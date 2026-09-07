#!/usr/bin/env bash
set -Eeuo pipefail

# Risk-on matched-FOV high-resolution screen.
#
# Candidate: 960x384 + cut_height=180.  The effective source view remains
# 1366x540 (the production view), while the network receives a 2.5:1 tensor;
# this isolates the resolution/quantization hypothesis from the rejected
# 960x480 + cut0 crop/FOV combination.  The runner uses the same 8-video LVO,
# seed, adapted checkpoint, Oracle, and 15-epoch schedule as the R50 reference.

PROJECT_ROOT="${HARDLANE_PROJECT_ROOT:?HARDLANE_PROJECT_ROOT must be set}"
DATA_ROOT="${HARDLANE_DATA_ROOT:?HARDLANE_DATA_ROOT must be set}"
UNLANEDET_ROOT="${UNLANEDET_ROOT:?UNLANEDET_ROOT must be set}"
WEIGHTS_ROOT="${HARDLANE_WEIGHTS_ROOT:?HARDLANE_WEIGHTS_ROOT must be set}"
OUTPUT_ROOT="${HARDLANE_OUTPUT_ROOT:?HARDLANE_OUTPUT_ROOT must be set}"
PYTHON_BIN="${HARDLANE_PYTHON:-/usr/local/miniconda3/envs/py39/bin/python}"
MANIFESTS_ROOT="${HARDLANE_LVO_MANIFESTS_ROOT:-$OUTPUT_ROOT/lvo_clrnet_r50_15ep_20260904/manifests}"
EXPERIMENT_ROOT="${HARDLANE_RISKON_RESOLUTION_ROOT:-$OUTPUT_ROOT/risk_on_res960x384_lvo_20260907}"
RESUME="${RESUME:-0}"

BUILDER="$PROJECT_ROOT/scripts/autodl/build_resolution_config.py"
RUNNER="$PROJECT_ROOT/scripts/autodl/lvo_video_runner.py"
BASE_CONFIG="$PROJECT_ROOT/configs/unlanedet/clrnet_r50_hardlane.py"
DERIVED_CONFIG="$EXPERIMENT_ROOT/clrnet_r50_hardlane_960x384_cut180.py"
STATUS="$EXPERIMENT_ROOT/status"
LOG="$EXPERIMENT_ROOT/run.log"

for path in "$PROJECT_ROOT" "$DATA_ROOT" "$UNLANEDET_ROOT" "$WEIGHTS_ROOT" \
            "$OUTPUT_ROOT" "$MANIFESTS_ROOT" "$BUILDER" "$RUNNER" \
            "$BASE_CONFIG"; do
    [ -e "$path" ] || { echo "missing required path: $path" >&2; exit 2; }
done

if [ "$RESUME" != 1 ] && [ -e "$EXPERIMENT_ROOT" ]; then
    echo "refusing to overwrite existing experiment: $EXPERIMENT_ROOT" >&2
    exit 3
fi
mkdir -p "$EXPERIMENT_ROOT"

export HARDLANE_PROJECT_ROOT="$PROJECT_ROOT"
export HARDLANE_DATA_ROOT="$DATA_ROOT"
export HARDLANE_WEIGHTS_ROOT="$WEIGHTS_ROOT"
export HARDLANE_OUTPUT_ROOT="$OUTPUT_ROOT"

write_status() {
    printf '%s\n' "$1" > "$STATUS"
}

on_error() {
    code=$?
    write_status "failed:$code"
    printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ) failed:$code" >> "$LOG"
    exit "$code"
}
trap on_error ERR

write_status preparing
"$PYTHON_BIN" "$BUILDER" \
    --base "$BASE_CONFIG" \
    --output "$DERIVED_CONFIG" \
    --width 960 --height 384 --cut-height 180 >> "$LOG" 2>&1
grep -Fq 'img_w = 960' "$DERIVED_CONFIG"
grep -Fq 'img_h = 384' "$DERIVED_CONFIG"
grep -Fq 'cut_height = 180' "$DERIVED_CONFIG"

command=(
    "$PYTHON_BIN" "$RUNNER" --mode all
    --project-root "$PROJECT_ROOT"
    --data-root "$DATA_ROOT"
    --unlanedet-root "$UNLANEDET_ROOT"
    --weights-root "$WEIGHTS_ROOT"
    --output-root "$OUTPUT_ROOT"
    --experiment-root "$EXPERIMENT_ROOT"
    --manifests-root "$MANIFESTS_ROOT"
    --python-bin "$PYTHON_BIN"
    --base-config "$DERIVED_CONFIG"
    --input-width 960 --input-height 384 --cut-height 180
    --eval-workers 0
    --skip-complete
)
if [ "$RESUME" = 1 ]; then
    command+=(--resume)
fi

write_status running
"${command[@]}" >> "$LOG" 2>&1
write_status complete
printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ) risk-on resolution LVO complete" >> "$LOG"
