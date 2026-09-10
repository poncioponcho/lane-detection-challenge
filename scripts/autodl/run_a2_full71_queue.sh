#!/usr/bin/env bash
# §37 A2: full-71 production-recipe 36ep queue — seeds 42/101/202/303.
#
# ALL FOUR seeds retrain here: the 09-09 instance expiry destroyed the T1
# all71 weights (only their testA/val predictions survive locally).  Names
# follow the T1 convention all71_seed<NN>_clrnet_r50_36ep so downstream
# inventory/selection globs stay uniform.
#
# Recovery contract (instance was down 09-09 22:4x .. TBD):
#   1. wait until NO codex-chain processes remain (lvo_video_runner /
#      train_net.py) AND the GPU compute queue is empty (24h budget) —
#      never preempt a live foreign chain;
#   2. validate prerequisites (manifest, run_training, adapted weights);
#   3. launch the three runs sequentially with --skip-if-complete
#      --auto-resume (each ~2.2h on the 3090 at 0.44s/it);
#   4. print the 4-model inventory for the B-package selection step.
#
# All71 recipe is EXACTLY the T1 one (run_top3_chain_20260908.sh):
#   --model clrnet_r50 --epochs 36 --iters-per-epoch 592
#   --eval-every-epochs 6 --max-to-keep 6 --train-manifest $ALL71_MANIFEST
#
# Lives OUTSIDE the repo (instance repo HEAD is diverged; do not ff-merge).
# Run: nohup bash run_a2_full71_queue.sh > a2_full71_queue.nohup.log 2>&1 &

set -uo pipefail

PROJECT_ROOT=/hy-tmp/lane-detection-challenge
OUTPUT_ROOT=/hy-tmp/lane-outputs
PYTHON_BIN=/usr/local/miniconda3/envs/py39/bin/python
ALL71_MANIFEST="$OUTPUT_ROOT/experiments_manifest_train_all71.jsonl"
STATUS="$OUTPUT_ROOT/a2_full71_queue.status"
LOG="$OUTPUT_ROOT/a2_full71_queue.log"
RUNS="$OUTPUT_ROOT/a2_full71_queue.runs"

export HARDLANE_PROJECT_ROOT="$PROJECT_ROOT"
export HARDLANE_DATA_ROOT=/hy-tmp/datasets/HardLane/Lane
export UNLANEDET_ROOT=/hy-tmp/UnLanedet
export HARDLANE_WEIGHTS_ROOT=/hy-tmp/weights
export HARDLANE_OUTPUT_ROOT="$OUTPUT_ROOT"
export HARDLANE_PYTHON="$PYTHON_BIN"

write_status() { printf '%s\n' "$1" > "$STATUS"; }
log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "$LOG"; }

# --- 1. foreign-chain + GPU drain (24h budget) -----------------------------
write_status waiting_gpu
log "A2 queue start: waiting for foreign chains + GPU"
deadline=$(( $(date +%s) + 24 * 3600 ))
while true; do
    procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
    chain=$(pgrep -fc 'lvo_video_runner|tools/train_net.py|overnight_optimizer|a2_full71_queue' 2>/dev/null || printf 0)
    # the pgrep above matches our own script name too; subtract via a second probe
    self=$(pgrep -fc 'a2_full71_queue' 2>/dev/null || printf 0)
    foreign=$(( chain - self ))
    if [ "$procs" -eq 0 ] && [ "$foreign" -le 0 ]; then
        break
    fi
    if [ "$(date +%s)" -gt "$deadline" ]; then
        write_status aborted_gpu_busy
        log "GPU/chain still busy after 24h; standing down"
        exit 5
    fi
    sleep 120
done
log "GPU idle and no foreign chain; launching"

# --- 2. prerequisites -------------------------------------------------------
for path in "$ALL71_MANIFEST" "$PROJECT_ROOT/scripts/autodl/run_training.py" \
            "$UNLANEDET_ROOT/tools/train_net.py"; do
    [ -e "$path" ] || { write_status aborted_prereq; log "missing: $path"; exit 4; }
done

# --- 3. four runs (≈9h total on the 3090 at ~0.44s/it) ----------------------
write_status running
run_one() {
    name=$1; seed=$2
    log "start $name seed=$seed"
    if "$PYTHON_BIN" "$PROJECT_ROOT/scripts/autodl/run_training.py" \
        --model clrnet_r50 --run-name "$name" --epochs 36 \
        --seed "$seed" --max-to-keep 6 \
        --skip-if-complete --auto-resume \
        --train-manifest "$ALL71_MANIFEST" --iters-per-epoch 592 \
        --eval-every-epochs 6 >> "$LOG" 2>&1; then
        log "done $name"
        printf '%s:ok\n' "$name" >> "$RUNS"
    else
        code=$?
        log "FAILED $name exit=$code"
        printf '%s:failed:%s\n' "$name" "$code" >> "$RUNS"
    fi
}
run_one all71_seed42_clrnet_r50_36ep 42
run_one all71_seed101_clrnet_r50_36ep 101
run_one all71_seed202_clrnet_r50_36ep 202
run_one all71_seed303_clrnet_r50_36ep 303

# --- 4. inventory for the B-package selection -------------------------------
if grep -q failed "$RUNS" 2>/dev/null; then
    write_status complete_with_failures
else
    write_status complete
fi
log "=== A2 pool inventory (B-package candidates) ==="
for d in "$OUTPUT_ROOT/runs"/all71_seed42_clrnet_r50_36ep \
         "$OUTPUT_ROOT/runs"/all71_seed101_clrnet_r50_36ep \
         "$OUTPUT_ROOT/runs"/all71_seed202_clrnet_r50_36ep \
         "$OUTPUT_ROOT/runs"/all71_seed303_clrnet_r50_36ep; do
    if [ -e "$d/last_checkpoint" ]; then
        log "READY  $d  ckpt=$(cat "$d/last_checkpoint")"
    else
        log "MISSING $d"
    fi
done
log "A2 queue complete; next: OS 三门 Oracle (CPU) -> B-package selection"
