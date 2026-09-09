#!/usr/bin/env bash
set -uo pipefail

# Replace the already falsified VAT=2000 branch with a controlled low-weight
# screen.  This launcher is intentionally external to the training checkout:
# it can wait across sessions without changing the repository while the
# current plain/CLRerNet/ConvNeXt LVO chain is running.
#
# Sequence after the ConvNeXt LVO marker:
#   1. screen VAT weights 5, 10, and 25 on the fixed 6300-image val split;
#   2. run VAT LVO only when the best screen clears +0.5pp over baseline;
#   3. run the independent binary segmentation-mask LVO screen.
#
# No test image is used for training.  The binary worktree and every output
# tree remain isolated from the original checkout.

set_status() {
    printf '%s\n' "$1" > "$STATUS"
}

log() {
    printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "$LOG"
}

PROJECT_ROOT="${PROJECT_ROOT:-/hy-tmp/lane-detection-challenge}"
MASK_PROJECT_ROOT="${MASK_PROJECT_ROOT:-/hy-tmp/lane-detection-challenge-maskbinary}"
DATA_ROOT="${DATA_ROOT:-/hy-tmp/datasets/HardLane/Lane}"
UNLANEDET_ROOT="${UNLANEDET_ROOT:-/hy-tmp/UnLanedet}"
WEIGHTS_ROOT="${WEIGHTS_ROOT:-/hy-tmp/weights}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/hy-tmp/lane-outputs}"
PYTHON_BIN="${PYTHON_BIN:-/usr/local/miniconda3/envs/py39/bin/python}"
CHAIN_PID="${CHAIN_PID:-716026}"

CONVNEXT_MARKER="$OUTPUT_ROOT/lvo_clrernet_convnext_tiny_15ep_20260909/experiment/lvo_training_complete.json"
BASELINE_METRICS="$OUTPUT_ROOT/runs/screen_clrnet_r50_15ep/metrics.json"
VAT_SCREEN_RESULTS="$OUTPUT_ROOT/vat_lowweight_screen_20260909.tsv"
VAT_LVO_ROOT="$OUTPUT_ROOT/lvo_vat_lowweight_15ep_20260909/experiment"
MASK_LVO_ROOT="$OUTPUT_ROOT/lvo_maskbinary_15ep_20260909/experiment"
STATUS="$OUTPUT_ROOT/vat_lowweight_mask_chain_20260909.status"
LOG="$OUTPUT_ROOT/vat_lowweight_mask_chain_20260909.log"
WAIT_TIMEOUT_SECONDS="${WAIT_TIMEOUT_SECONDS:-$((36 * 3600))}"

mkdir -p "$OUTPUT_ROOT"
set_status waiting_for_convnext
log "low-weight VAT + binary-mask chain queued; convnext marker=$CONVNEXT_MARKER"

deadline=$(( $(date +%s) + WAIT_TIMEOUT_SECONDS ))
while [ ! -f "$CONVNEXT_MARKER" ]; do
    if [ "${CHAIN_PID:-0}" -gt 0 ] && ! kill -0 "$CHAIN_PID" 2>/dev/null; then
        if ! pgrep -f "$PROJECT_ROOT/scripts/autodl/lvo_video_runner.py" >/dev/null 2>&1; then
            set_status aborted_convnext_chain_lost
            log "convnext marker absent and chain pid $CHAIN_PID is gone"
            exit 23
        fi
    fi
    if [ "$(date +%s)" -gt "$deadline" ]; then
        set_status aborted_timeout_waiting_convnext
        log "timed out waiting for ConvNeXt marker"
        exit 24
    fi
    sleep 60
done
log "convnext LVO marker present"

# The marker is written by the runner before the surrounding chain performs
# score-sidecar merge.  Wait for both the merge and all CUDA children to end.
set_status waiting_for_gpu_drain
drain_deadline=$(( $(date +%s) + $((2 * 3600)) ))
while true; do
    gpu_procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | awk 'NF' | wc -l | tr -d ' ')
    lvo_procs=$(pgrep -f "$PROJECT_ROOT/scripts/autodl/lvo_video_runner.py" 2>/dev/null || true)
    merge_procs=$(pgrep -f "$PROJECT_ROOT/scripts/merge_lvo_oof_scores.py" 2>/dev/null || true)
    if [ "${gpu_procs:-0}" -eq 0 ] && [ -z "$lvo_procs" ] && [ -z "$merge_procs" ]; then
        break
    fi
    if [ "$(date +%s)" -gt "$drain_deadline" ]; then
        set_status aborted_gpu_drain_timeout
        log "GPU/process drain timed out: gpu=$gpu_procs lvo=$lvo_procs merge=$merge_procs"
        exit 25
    fi
    sleep 30
done

export HARDLANE_PROJECT_ROOT="$PROJECT_ROOT"
export HARDLANE_DATA_ROOT="$DATA_ROOT"
export UNLANEDET_ROOT="$UNLANEDET_ROOT"
export HARDLANE_WEIGHTS_ROOT="$WEIGHTS_ROOT"
export HARDLANE_OUTPUT_ROOT="$OUTPUT_ROOT"
export HARDLANE_PYTHON="$PYTHON_BIN"

baseline_f1="0.78396"
if [ -f "$BASELINE_METRICS" ]; then
    baseline_f1=$(
        "$PYTHON_BIN" - "$BASELINE_METRICS" <<'PY'
import json
import sys
from pathlib import Path

best = None
for raw in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    if not raw.strip():
        continue
    row = json.loads(raw)
    if "F1" in row:
        value = float(row["F1"])
        best = value if best is None else max(best, value)
print(f"{best:.9f}" if best is not None else "0.783960000")
PY
    )
fi
log "baseline screen best F1=$baseline_f1"

: > "$VAT_SCREEN_RESULTS"
set_status vat_screens_running

weights=("5.0" "10.0" "25.0")
names=(
    "vat5_clrnet_r50_15ep_20260909"
    "vat10_clrnet_r50_15ep_20260909"
    "vat25_clrnet_r50_15ep_20260909"
)

for index in "${!weights[@]}"; do
    weight="${weights[$index]}"
    name="${names[$index]}"
    log "VAT standard-val screen start name=$name weight=$weight"
    if "$PYTHON_BIN" "$PROJECT_ROOT/scripts/autodl/run_training.py" \
        --model clrnet_r50_vat \
        --run-name "$name" \
        --epochs 15 \
        --seed 42 \
        --eval-workers 0 \
        --max-to-keep 4 \
        --skip-if-complete \
        --auto-resume \
        --override "model.vat_weight=$weight" \
        >> "$LOG" 2>&1; then
        metrics="$OUTPUT_ROOT/runs/$name/metrics.json"
        best=$(
            "$PYTHON_BIN" - "$metrics" <<'PY'
import json
import sys
from pathlib import Path

values = []
for raw in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    if raw.strip():
        row = json.loads(raw)
        if "F1" in row:
            values.append(float(row["F1"]))
print(f"{max(values):.9f}" if values else "nan")
PY
        )
        printf '%s\t%s\t%s\n' "$name" "$weight" "$best" >> "$VAT_SCREEN_RESULTS"
        log "VAT standard-val screen done name=$name weight=$weight best=$best"
    else
        code=$?
        printf '%s\t%s\tfailed:%s\n' "$name" "$weight" "$code" >> "$VAT_SCREEN_RESULTS"
        log "VAT standard-val screen failed name=$name weight=$weight exit=$code"
    fi
done

selection=$(
    "$PYTHON_BIN" - "$VAT_SCREEN_RESULTS" "$baseline_f1" <<'PY'
import math
import sys

results = []
for raw in open(sys.argv[1], encoding="utf-8"):
    fields = raw.rstrip("\n").split("\t")
    if len(fields) != 3:
        continue
    name, weight, score = fields
    try:
        score = float(score)
        weight = float(weight)
    except ValueError:
        continue
    if math.isfinite(score):
        results.append((score, weight, name))
if not results:
    print("NONE")
else:
    score, weight, name = max(results)
    print(f"{name}\t{weight:.1f}\t{score:.9f}\t{score - float(sys.argv[2]):.9f}")
PY
)
log "VAT screen selection: $selection"

best_weight=""
best_name=""
best_delta="-inf"
if [ "$selection" != "NONE" ]; then
    IFS=$'\t' read -r best_name best_weight best_score best_delta <<EOF
$selection
EOF
fi

if [ -n "$best_weight" ] && "$PYTHON_BIN" - "$best_delta" <<'PY'
import sys
raise SystemExit(0 if float(sys.argv[1]) >= 0.005 else 1)
PY
then
    log "VAT screen clears +0.5pp: name=$best_name weight=$best_weight delta=$best_delta"
    set_status vat_lvo_running
    mkdir -p "$VAT_LVO_ROOT"
    if "$PYTHON_BIN" "$PROJECT_ROOT/scripts/autodl/lvo_video_runner.py" \
        --mode all \
        --project-root "$PROJECT_ROOT" \
        --data-root "$DATA_ROOT" \
        --unlanedet-root "$UNLANEDET_ROOT" \
        --weights-root "$WEIGHTS_ROOT" \
        --output-root "$OUTPUT_ROOT" \
        --experiment-root "$VAT_LVO_ROOT" \
        --manifests-root "$OUTPUT_ROOT/lvo_plain_v1_15ep_20260909/manifests" \
        --python-bin "$PYTHON_BIN" \
        --base-config "$PROJECT_ROOT/configs/unlanedet/clrnet_r50_hardlane_vat.py" \
        --input-width 800 \
        --input-height 320 \
        --cut-height 180 \
        --eval-workers 0 \
        --checkpoint-max-to-keep 1 \
        --iterations-per-epoch 525 \
        --resume \
        --skip-complete \
        --override "model.vat_weight=$best_weight" \
        >> "$LOG" 2>&1; then
        "$PYTHON_BIN" "$PROJECT_ROOT/scripts/merge_lvo_oof_scores.py" \
            --experiment-root "$VAT_LVO_ROOT" >> "$LOG" 2>&1 || log "VAT LVO score merge failed"
        log "VAT low-weight LVO stage finished"
    else
        code=$?
        log "VAT low-weight LVO stage failed exit=$code"
    fi
else
    set_status vat_lvo_skipped
    log "no low-weight VAT screen cleared +0.5pp; skip VAT LVO"
fi

# Independent segmentation-mask experiment.  It is intentionally launched
# after the low-weight branch, so no two CUDA jobs can contend for the 3090.
set_status binary_mask_lvo_running
if [ -e "$MASK_LVO_ROOT/lvo_training_complete.json" ]; then
    log "binary-mask LVO already complete; preserving existing result"
else
    mkdir -p "$MASK_LVO_ROOT"
    if "$PYTHON_BIN" "$MASK_PROJECT_ROOT/scripts/autodl/lvo_video_runner.py" \
        --mode all \
        --project-root "$MASK_PROJECT_ROOT" \
        --data-root "$DATA_ROOT" \
        --unlanedet-root "$UNLANEDET_ROOT" \
        --weights-root "$WEIGHTS_ROOT" \
        --output-root "$OUTPUT_ROOT" \
        --experiment-root "$MASK_LVO_ROOT" \
        --manifests-root "$OUTPUT_ROOT/lvo_plain_v1_15ep_20260909/manifests" \
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
        "$PYTHON_BIN" "$MASK_PROJECT_ROOT/scripts/merge_lvo_oof_scores.py" \
            --experiment-root "$MASK_LVO_ROOT" >> "$LOG" 2>&1 || log "binary-mask score merge failed"
        log "binary-mask LVO stage finished"
    else
        code=$?
        set_status binary_mask_lvo_failed
        log "binary-mask LVO stage failed exit=$code"
        exit "$code"
    fi
fi

set_status complete
log "low-weight VAT + binary-mask chain complete"
