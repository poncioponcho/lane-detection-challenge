#!/usr/bin/env bash
set -uo pipefail

# §36 T4 phase-2: VAT / hard-clip oversampling screens on the single 3090.
#
# Runs AFTER the T1 five-run chain (top3_chain_20260908).  Sequence:
#   1. wait for the chain status file to read "complete" (abort on failures);
#   2. drop the untracked VAT smoke copies, merge bundle 292d559 -> HEAD;
#   3. regenerate the weight-probe + dataloader/loss smoke evidence for the
#      new HEAD (the launcher refuses stale evidence);
#   4. re-run the VAT smoke from the committed copies;
#   5. launch three 15-epoch screens on the frozen 63-clip LVO split
#      (baseline: screen_clrnet_r50_15ep, F1 0.7840):
#        vat2000_clrnet_r50_15ep     VAT only, vat_weight=2000
#        os_clrnet_r50_15ep          hard-clip oversampling only (bottom-25%
#                                    OOF-F1 clips x3, 9500-row manifest,
#                                    iters-per-epoch held at 525 so the
#                                    optimization budget matches baseline)
#        vat2000_os_clrnet_r50_15ep  VAT + oversampling (the plan's bet)
#   6. print an F1 comparison table into the phase-2 log.
#
# This script must live OUTSIDE the repo (it merges the repo's git HEAD),
# run it as: bash /hy-tmp/lane-outputs/phase2_vat_20260908.sh

PROJECT_ROOT=/hy-tmp/lane-detection-challenge
UNLANEDET_ROOT=/hy-tmp/UnLanedet
DATA_ROOT=/hy-tmp/datasets/HardLane/Lane
WEIGHTS_ROOT=/hy-tmp/weights
OUTPUT_ROOT=/hy-tmp/lane-outputs
PYTHON_BIN=/usr/local/miniconda3/envs/py39/bin/python
BUNDLE=/hy-tmp/bundles/hardlane_292d559_to_HEAD.bundle
# The merge must bring these paths in as tracked files (a pinned commit hash
# cannot live inside its own commit, so file presence is the check).
MERGE_REQUIRED_PATHS="src/integrations/unlanedet_vat.py configs/unlanedet/clrnet_r50_hardlane_vat.py scripts/autodl/smoke_vat_training.py scripts/autodl/run_training.py"
OS_MANIFEST="$OUTPUT_ROOT/experiments_manifest_train_os_v1_seed42_hard25_x3.jsonl"
CHAIN_STATUS="$OUTPUT_ROOT/top3_chain_20260908.status"
CHAIN_TIMEOUT_SECONDS=$((26 * 3600))

export HARDLANE_PROJECT_ROOT="$PROJECT_ROOT"
export HARDLANE_DATA_ROOT="$DATA_ROOT"
export UNLANEDET_ROOT="$UNLANEDET_ROOT"
export HARDLANE_WEIGHTS_ROOT="$WEIGHTS_ROOT"
export HARDLANE_OUTPUT_ROOT="$OUTPUT_ROOT"
export HARDLANE_PYTHON="$PYTHON_BIN"

STATUS="$OUTPUT_ROOT/phase2_vat_20260908.status"
LOG="$OUTPUT_ROOT/phase2_vat_20260908.log"
RUNS="$OUTPUT_ROOT/phase2_vat_20260908.runs"

write_status() { printf '%s\n' "$1" > "$STATUS"; }
log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "$LOG"; }

for path in "$PROJECT_ROOT" "$BUNDLE" "$OS_MANIFEST" "$CHAIN_STATUS"; do
    [ -e "$path" ] || { echo "missing required path: $path" >&2; exit 4; }
done

write_status waiting
log "phase2 start: waiting for chain"

deadline=$(( $(date +%s) + CHAIN_TIMEOUT_SECONDS ))
while true; do
    chain=$(cat "$CHAIN_STATUS" 2>/dev/null || printf 'missing')
    case "$chain" in
        complete) break ;;
        complete_with_failures)
            write_status aborted_chain_failures
            log "chain finished with failures; refusing to switch HEAD under review"
            exit 4 ;;
        *) : ;;
    esac
    if [ "$(date +%s)" -gt "$deadline" ]; then
        write_status aborted_timeout
        log "chain did not complete within ${CHAIN_TIMEOUT_SECONDS}s (last: $chain)"
        exit 5
    fi
    sleep 300
done
log "chain complete"

# Merge the committed VAT work.  The three files below exist as untracked
# copies only when an earlier attempt died BEFORE its merge; once tracked at
# HEAD they must NOT be rm'd (a dirty worktree breaks the ff-merge and the
# launcher's clean-tree guard on relaunch).
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

if ! "$PYTHON_BIN" "$PROJECT_ROOT/scripts/autodl/smoke_vat_training.py" >> "$LOG" 2>&1; then
    write_status aborted_vat_smoke
    log "VAT smoke failed"
    exit 7
fi
log "VAT smoke passed"

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
run_one clrnet_r50_vat vat2000_clrnet_r50_15ep 42
run_one clrnet_r50 os_clrnet_r50_15ep 42 \
    --train-manifest "$OS_MANIFEST" --iters-per-epoch 525
run_one clrnet_r50_vat vat2000_os_clrnet_r50_15ep 42 \
    --train-manifest "$OS_MANIFEST" --iters-per-epoch 525

"$PYTHON_BIN" - "$OUTPUT_ROOT" >> "$LOG" 2>&1 <<'PYEOF'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
names = [
    "screen_clrnet_r50_15ep",
    "vat2000_clrnet_r50_15ep",
    "os_clrnet_r50_15ep",
    "vat2000_os_clrnet_r50_15ep",
]
print("=== phase2 screen F1 summary (best over epochs / final) ===")
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
log "phase2 complete"
