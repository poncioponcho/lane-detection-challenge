#!/usr/bin/env bash
set -uo pipefail

# §36 T3: CLRerNet architecture screen on the single 3090.
#
# Runs AFTER phase-2 (phase2_vat_20260908.sh).  Sequence:
#   1. wait for the phase-2 status to leave "running"/"waiting" (any exit
#      state — complete, complete_with_failures or aborted_* — frees the
#      GPU for T3; the merge below carries the VAT commits too if phase-2
#      died before its own merge), aborting on timeout or on a dead
#      phase-2 process;
#   2. wait for the GPU compute queue to drain (guards against orphaned
#      training children if the phase-2 watcher itself was killed);
#   3. drop untracked VAT smoke copies IF (and only if) still untracked,
#      then merge bundle 292d559 -> HEAD with --ff-only;
#   4. regenerate the weight-probe + dataloader/loss smoke evidence for
#      the new HEAD (the launcher refuses stale evidence);
#   5. run the CLRerNet smoke (geometry patch, dynamic assign, AMP
#      backward);
#   6. launch one 15-epoch screen: clrernet_r50_15ep on the frozen 63-clip
#      LVO split (baseline: screen_clrnet_r50_15ep, F1 0.7840);
#   7. print the F1 comparison table into the T3 log.
#
# This script must live OUTSIDE the repo (it merges the repo's git HEAD),
# run it as: bash /hy-tmp/lane-outputs/t3_clrernet_20260908.sh

PROJECT_ROOT=/hy-tmp/lane-detection-challenge
UNLANEDET_ROOT=/hy-tmp/UnLanedet
DATA_ROOT=/hy-tmp/datasets/HardLane/Lane
WEIGHTS_ROOT=/hy-tmp/weights
OUTPUT_ROOT=/hy-tmp/lane-outputs
PYTHON_BIN=/usr/local/miniconda3/envs/py39/bin/python
# Same bundle file phase-2 merges from: it carries 292d559 -> HEAD (VAT fix
# + T3 commits).  If phase-2 already merged, this ff-merge is a no-op; if
# phase-2 died before its merge, this is what brings the code in.
BUNDLE=/hy-tmp/bundles/hardlane_292d559_to_HEAD.bundle
MERGE_REQUIRED_PATHS="src/integrations/unlanedet_clrernet.py configs/unlanedet/clrernet_r50_hardlane.py scripts/autodl/smoke_clrernet_training.py scripts/autodl/run_training.py scripts/autodl/validate_run.py"
PHASE2_STATUS="$OUTPUT_ROOT/phase2_vat_20260908.status"
PHASE2_TIMEOUT_SECONDS=$((40 * 3600))
GPU_DRAIN_TIMEOUT_SECONDS=$((2 * 3600))

export HARDLANE_PROJECT_ROOT="$PROJECT_ROOT"
export HARDLANE_DATA_ROOT="$DATA_ROOT"
export UNLANEDET_ROOT="$UNLANEDET_ROOT"
export HARDLANE_WEIGHTS_ROOT="$WEIGHTS_ROOT"
export HARDLANE_OUTPUT_ROOT="$OUTPUT_ROOT"
export HARDLANE_PYTHON="$PYTHON_BIN"

STATUS="$OUTPUT_ROOT/t3_clrernet_20260908.status"
LOG="$OUTPUT_ROOT/t3_clrernet_20260908.log"
RUNS="$OUTPUT_ROOT/t3_clrernet_20260908.runs"

write_status() { printf '%s\n' "$1" > "$STATUS"; }
log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "$LOG"; }

for path in "$PROJECT_ROOT" "$BUNDLE" "$PHASE2_STATUS"; do
    [ -e "$path" ] || { echo "missing required path: $path" >&2; exit 4; }
done

write_status waiting
log "t3 start: waiting for phase2"

deadline=$(( $(date +%s) + PHASE2_TIMEOUT_SECONDS ))
while true; do
    phase2=$(cat "$PHASE2_STATUS" 2>/dev/null || printf 'missing')
    case "$phase2" in
        running|waiting) : ;;
        *) break ;;
    esac
    if ! pgrep -f phase2_vat_20260908.sh > /dev/null; then
        log "phase2 watcher process is gone (last status: $phase2)"
        break
    fi
    if [ "$(date +%s)" -gt "$deadline" ]; then
        write_status aborted_timeout
        log "phase2 did not leave running within ${PHASE2_TIMEOUT_SECONDS}s (last: $phase2)"
        exit 5
    fi
    sleep 300
done
log "phase2 finished with status: $phase2"

# Drain: never launch onto an occupied GPU (orphaned children, etc.).
gpu_deadline=$(( $(date +%s) + GPU_DRAIN_TIMEOUT_SECONDS ))
while true; do
    procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
    if [ "$procs" -eq 0 ]; then
        break
    fi
    if [ "$(date +%s)" -gt "$gpu_deadline" ]; then
        write_status aborted_gpu_busy
        log "GPU still busy after ${GPU_DRAIN_TIMEOUT_SECONDS}s (compute procs: $procs)"
        exit 5
    fi
    sleep 120
done
log "GPU idle"

# Merge the committed T3 work.  The three VAT files below exist as
# untracked copies only when phase-2 died BEFORE its merge; after phase-2's
# merge they are tracked and must NOT be rm'd (a dirty worktree makes the
# launcher's clean-tree guard fail).
cd "$PROJECT_ROOT" || { write_status aborted_cd; exit 6; }
for f in src/integrations/unlanedet_vat.py \
         configs/unlanedet/clrnet_r50_hardlane_vat.py \
         scripts/autodl/smoke_vat_training.py; do
    if [ -e "$f" ] && ! git cat-file -e "HEAD:$f" 2>/dev/null; then
        rm -f "$f"
        log "removed untracked $f"
    fi
done
git fetch "$BUNDLE" HEAD || { write_status aborted_fetch; exit 6; }
git merge --ff-only FETCH_HEAD || { write_status aborted_merge; exit 6; }
head=$(git rev-parse HEAD)
for required in $MERGE_REQUIRED_PATHS; do
    if ! git cat-file -e "HEAD:$required" 2>/dev/null; then
        write_status aborted_merge_incomplete
        log "post-merge HEAD is missing tracked $required"
        exit 6
    fi
done
log "merged to $head"

# Prerequisite evidence must be regenerated at the new HEAD or the launcher
# refuses to start (stale-evidence guard).
# The pinned UnLanedet modelzoo configs resolve "config/common/train.py"
# relative to CWD, so the probe must run from UNLANEDET_ROOT (same as
# run_pipeline.sh does).
if ! (cd "$UNLANEDET_ROOT" && "$PYTHON_BIN" "$PROJECT_ROOT/scripts/autodl/probe_weights.py" \
        --no-download --output "$OUTPUT_ROOT/weight_probe.json") >> "$LOG" 2>&1; then
    write_status aborted_probe
    log "weight probe failed"
    exit 7
fi
log "weight probe regenerated"
if ! "$PYTHON_BIN" "$PROJECT_ROOT/scripts/autodl/smoke_dataloader_and_loss.py" \
        --output "$OUTPUT_ROOT/smoke/dataloader_loss_smoke.json" >> "$LOG" 2>&1; then
    write_status aborted_dataloader_smoke
    log "dataloader/loss smoke failed"
    exit 7
fi
log "dataloader/loss smoke regenerated"

if ! "$PYTHON_BIN" "$PROJECT_ROOT/scripts/autodl/smoke_clrernet_training.py" >> "$LOG" 2>&1; then
    write_status aborted_clrernet_smoke
    log "CLRerNet smoke failed"
    exit 7
fi
log "CLRerNet smoke passed"

run_one() {
    model=$1; name=$2; seed=$3; shift 3
    log "start $name model=$model seed=$seed extra_args=$*"
    if "$PYTHON_BIN" "$PROJECT_ROOT/scripts/autodl/run_training.py" \
        --model "$model" --run-name "$name" --epochs 15 \
        --seed "$seed" --max-to-keep 4 \
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
run_one clrernet_r50 clrernet_r50_15ep 42

"$PYTHON_BIN" - "$OUTPUT_ROOT" >> "$LOG" 2>&1 <<'PYEOF'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
names = [
    "screen_clrnet_r50_15ep",
    "clrernet_r50_15ep",
]
print("=== t3 screen F1 summary (best over epochs / final) ===")
for name in names:
    metrics = root / "runs" / name / "metrics.json"
    if not metrics.is_file():
        print(f"{name}: MISSING {metrics}")
        continue
    best = final = None
    for line in metrics.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if "F1" not in entry:
            continue
        f1 = float(entry["F1"])
        iteration = int(entry.get("iteration", 0))
        if final is None or iteration >= final[1]:
            final = (f1, iteration)
        if best is None or f1 > best[0]:
            best = (f1, iteration)
    if best is None:
        print(f"{name}: no F1 observations")
    else:
        print(f"{name}: best={best[0]:.5f}@{best[1]} final={final[0]:.5f}@{final[1]}")
PYEOF

if grep -q failed "$RUNS" 2>/dev/null; then
    write_status complete_with_failures
else
    write_status complete
fi
log "t3 complete"
