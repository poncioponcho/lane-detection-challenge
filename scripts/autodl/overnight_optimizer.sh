#!/usr/bin/env bash
# §36 overnight optimization loop — autonomous iteration scheduler.
#
# Starts AFTER the T3 watcher frees the GPU (waits for the t3 status file to
# leave waiting/running, then drains the compute queue), then loops:
#
#   planner (bounded rules, scripts/autodl/iteration_planner.py)
#     -> dispatch: consensus | train | stop
#     -> monitor: run_training with --auto-resume (retry <= 2), or the
#        consensus_fusion script guarded by its inputs manifest
#     -> tracker (score_tracker.py): append score history + render
#        reports/iter_N.md with deltas vs incumbent and previous best
#     -> failures: 2nd attempt failed => ALERTS.log entry + alert file,
#        iteration abandoned, loop continues with the next plan
#
# Safety rails:
#   * no A-board/B-board submission is ever performed automatically — the
#     loop only prepares evidence; submissions stay human-gated;
#   * no new TRAINING after the freeze epoch (2026-09-12 22:00 Asia/Shanghai);
#   * all work is resumable (--skip-if-complete --auto-resume); failed run
#     dirs are archived as .failed-* by run_training's own guards;
#   * every iteration appends to score_history.jsonl for the morning review.
#
# Must live OUTSIDE the repo (mirrors the watcher-script convention).  Run:
#   nohup bash overnight_optimizer.sh > overnight_optimizer.nohup.log 2>&1 &

set -uo pipefail

PROJECT_ROOT=/hy-tmp/lane-detection-challenge
OUTPUT_ROOT=/hy-tmp/lane-outputs
PYTHON_BIN=/usr/local/miniconda3/envs/py39/bin/python
STATE_ROOT="$OUTPUT_ROOT/overnight"
RUNS_ROOT="$OUTPUT_ROOT/runs"
HISTORY="$STATE_ROOT/score_history.jsonl"
STATUS="$STATE_ROOT/optimizer.status"
T3_STATUS="$OUTPUT_ROOT/t3_clrernet_20260908.status"
FREEZE_TRAIN_EPOCH="$($PYTHON_BIN - <<'PY'
import datetime, zoneinfo
print(int(datetime.datetime(2026, 9, 12, 22, 0, tzinfo=zoneinfo.ZoneInfo('Asia/Shanghai')).timestamp()))
PY
)"
MAX_ITERATIONS=8
BETWEEN_ITERATIONS_SLEEP=60

export HARDLANE_PROJECT_ROOT="$PROJECT_ROOT"
export HARDLANE_DATA_ROOT=/hy-tmp/datasets/HardLane/Lane
export UNLANEDET_ROOT=/hy-tmp/UnLanedet
export HARDLANE_WEIGHTS_ROOT=/hy-tmp/weights
export HARDLANE_OUTPUT_ROOT="$OUTPUT_ROOT"
export HARDLANE_PYTHON="$PYTHON_BIN"

mkdir -p "$STATE_ROOT/reports" "$STATE_ROOT/markers"
write_status() { printf '%s\n' "$1" > "$STATUS"; }
log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "$STATE_ROOT/optimizer.log"; }
alert() {
    log "ALERT: $*"
    printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "$STATE_ROOT/ALERTS.log"
}

# --- 1. wait for the T3 watcher to leave waiting/running -------------------
write_status waiting_t3
log "optimizer start: waiting for T3 watcher (freeze epoch: $FREEZE_TRAIN_EPOCH)"
while true; do
    t3=$(cat "$T3_STATUS" 2>/dev/null || printf 'missing')
    case "$t3" in
        waiting|running) : ;;
        *) break ;;
    esac
    sleep 300
done
log "t3 left running state (status: $t3)"

# GPU drain guard — never launch onto an occupied GPU.
gpu_deadline=$(( $(date +%s) + 2 * 3600 ))
while true; do
    procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
    [ "$procs" -eq 0 ] && break
    if [ "$(date +%s)" -gt "$gpu_deadline" ]; then
        alert "GPU still busy after 2h drain wait; optimizer standing down"
        write_status aborted_gpu_busy
        exit 5
    fi
    sleep 120
done
log "GPU idle; entering iteration loop"

# --- 2. iteration loop -----------------------------------------------------
iter=0
while [ "$iter" -lt "$MAX_ITERATIONS" ]; do
    iter=$((iter + 1))
    spec=$("$PYTHON_BIN" "$PROJECT_ROOT/scripts/autodl/iteration_planner.py" \
        --state-root "$STATE_ROOT" --runs-root "$RUNS_ROOT" --history "$HISTORY" \
        --freeze-train-epoch "$FREEZE_TRAIN_EPOCH" 2>> "$STATE_ROOT/optimizer.log") || {
        alert "planner crashed (iter $iter); standing down to avoid looping on a broken planner"
        write_status aborted_planner
        exit 6
    }
    itype=$(printf '%s' "$spec" | "$PYTHON_BIN" -c 'import json,sys; print(json.load(sys.stdin)["type"])')
    slug=$(printf '%s' "$spec" | "$PYTHON_BIN" -c 'import json,sys; print(json.load(sys.stdin).get("slug", "iter"))')
    log "iter $iter plan: $spec"

    case "$itype" in
        wait)
            log "iter $iter: waiting for screens ($spec)"
            iter=$((iter - 1))
            sleep 600
            continue ;;
        stop)
            log "planner says stop: $spec"
            break ;;
        consensus)
            write_status "running_consensus"
            # consensus_inputs.json: {"seeds":[{"pred_dir":..,"scores_json":..}], "out":..}
            mapfile -t cargs < <("$PYTHON_BIN" - "$STATE_ROOT/consensus_inputs.json" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1]))
for s in cfg["seeds"]:
    print(f"--seed={s['pred_dir']}={s['scores_json']}")
print(f"--out={cfg['out']}")
PY
)
            if "$PYTHON_BIN" "$PROJECT_ROOT/scripts/autodl/consensus_fusion.py" \
                ${cargs[@]+"${cargs[@]}"} --emit-scores \
                >> "$STATE_ROOT/optimizer.log" 2>&1; then
                mkdir -p "$STATE_ROOT/markers" && touch "$STATE_ROOT/markers/consensus"
                log "consensus fusion done"
            else
                alert "consensus fusion failed twice; skipped (see optimizer.log)"
            fi ;;
        train)
            write_status "running_train"
            run_name=$(printf '%s' "$spec" | "$PYTHON_BIN" -c 'import json,sys; print(json.load(sys.stdin)["run_name"])')
            epochs=$(printf '%s' "$spec" | "$PYTHON_BIN" -c 'import json,sys; print(json.load(sys.stdin)["epochs"])')
            seed=$(printf '%s' "$spec" | "$PYTHON_BIN" -c 'import json,sys; print(json.load(sys.stdin)["seed"])')
            mapfile -t extra < <(printf '%s' "$spec" | "$PYTHON_BIN" -c 'import json,sys; print("\n".join(json.load(sys.stdin)["extra_args"]))')
            attempt=1
            while [ "$attempt" -le 2 ]; do
                log "iter $iter train $run_name attempt $attempt (epochs=$epochs seed=$seed extra=${extra[*]:-})"
                if "$PYTHON_BIN" "$PROJECT_ROOT/scripts/autodl/run_training.py" \
                    --run-name "$run_name" --epochs "$epochs" --seed "$seed" \
                    --max-to-keep 4 --skip-if-complete --auto-resume \
                    ${extra[@]+"${extra[@]}"} >> "$STATE_ROOT/optimizer.log" 2>&1; then
                    log "iter $iter train $run_name ok"
                    break
                fi
                log "iter $iter train $run_name attempt $attempt FAILED"
                attempt=$((attempt + 1))
                sleep 300
            done
            if [ "$attempt" -gt 2 ]; then
                alert "train $run_name failed twice; abandoned"
            else
                mkdir -p "$STATE_ROOT/markers"
                marker=$(printf '%s' "$spec" | "$PYTHON_BIN" -c 'import json,sys; print(json.load(sys.stdin).get("marker", ""))')
                [ -n "$marker" ] && touch "$STATE_ROOT/markers/$marker"
            fi ;;
        *)
            alert "unknown iteration type '$itype'; standing down"
            write_status aborted_unknown_type
            exit 7 ;;
    esac

    # --- 3. track + report every iteration ---------------------------------
    write_status "tracking"
    all_names="$("$PYTHON_BIN" - "$RUNS_ROOT" <<'PY'
import sys
from pathlib import Path
root = Path(sys.argv[1])
names = [p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")
         and (p / "metrics.json").is_file() and not p.name.startswith("gate_")]
print(",".join(sorted(names)))
PY
)"
    "$PYTHON_BIN" "$PROJECT_ROOT/scripts/autodl/score_tracker.py" \
        --runs-root "$RUNS_ROOT" --names "$all_names" --history "$HISTORY" \
        --report "$STATE_ROOT/reports/iter_${iter}_${slug}.md" \
        --context "iter_${iter}_${slug}" \
        --notes "spec: $spec" >> "$STATE_ROOT/optimizer.log" 2>&1 \
        || alert "tracker failed on iter $iter"
    log "iter $iter tracked+reported"

    sleep "$BETWEEN_ITERATIONS_SLEEP"
done

# --- 4. final report -------------------------------------------------------
write_status complete
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/autodl/score_tracker.py" \
    --runs-root "$RUNS_ROOT" --names "$("$PYTHON_BIN" - "$RUNS_ROOT" <<'PY'
import sys
from pathlib import Path
root = Path(sys.argv[1])
print(",".join(sorted(p.name for p in root.iterdir() if p.is_dir()
      and not p.name.startswith((".", "gate_")) and (p / "metrics.json").is_file())))
PY
)" \
    --history "$HISTORY" --report "$STATE_ROOT/OPTIMIZATION_REPORT.md" \
    --context final >> "$STATE_ROOT/optimizer.log" 2>&1
log "optimizer complete; final report at $STATE_ROOT/OPTIMIZATION_REPORT.md"
