#!/usr/bin/env bash
set -Eeuo pipefail

# Local continuation for the already-authorized D -> C sequence.  It never
# starts CUDA work: the remote C guard starts eval-only export after D, while
# this process only samples, downloads the finished C artifacts, and runs the
# CPU-side frozen-Oracle scan.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SSH_KEY="/Users/seyonmacbook/.ssh/lane_id"
REMOTE="root@i-1.gpushare.com"
SSH_PORT="34529"
REMOTE_OUT="/hy-tmp/lane-outputs"
D_EXP="${REMOTE_OUT}/lvo_clrnet_r50_36ep_20260905"
C_EXP="${REMOTE_OUT}/lvo_clrnet_r50_36ep_c_export_20260906"
LOCAL_C="${ROOT}/outputs/lvo_clrnet_r50_36ep_c_export_20260906"
SCAN_DIR="${ROOT}/outputs/lvo_conf_checkpoint_scan_20260906_probability"
MONITOR="${ROOT}/outputs/overnight_monitor_2026-09-06_probability.jsonl"
LOG="${ROOT}/outputs/lvo36_completion_watch_20260906.log"
ORACLE_PYTHON="/private/tmp/lane-oracle-py312/bin/python"
LANE_PYTHON="/Users/seyonmacbook/.workbuddy/binaries/python/envs/lane/bin/python"
LOCK_DIR="${ROOT}/outputs/.lvo36_completion_watch.lock"

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    echo "watcher already running: $LOCK_DIR" >&2
    exit 0
fi
cleanup() {
    rmdir "$LOCK_DIR"
}
trap cleanup EXIT

SSH_OPTS=(-i "$SSH_KEY" -p "$SSH_PORT" -o BatchMode=yes -o ConnectTimeout=15)
SCP_OPTS=(-i "$SSH_KEY" -P "$SSH_PORT" -o BatchMode=yes -o ConnectTimeout=15)

remote() {
    ssh "${SSH_OPTS[@]}" "$REMOTE" "$1"
}

# Read all monitoring fields through one SSH connection.  Repeated short SSH
# connections can be refused transiently by the instance-side gateway.
remote_snapshot() {
    ssh "${SSH_OPTS[@]}" "$REMOTE" '
        out=/hy-tmp/lane-outputs
        d=$(cat "$out/lvo36.status" 2>/dev/null || echo missing)
        c=$(cat "$out/lvo36_c_export_20260906.status" 2>/dev/null || echo pending)
        gpu=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | tr -d "\n")
        [ -n "$gpu" ] || gpu=unavailable
        last_iter=$(find /hy-tmp/lane-outputs/lvo_clrnet_r50_36ep_20260905/runs \
            -mindepth 2 -maxdepth 2 -name metrics.json -type f -exec tail -n1 {} + 2>/dev/null \
            | sed -n "s/.*iteration[^0-9]*\([0-9][0-9]*\).*/\1/p" | sort -n | tail -n1)
        [ -n "$last_iter" ] || last_iter=0
        printf "%s\t%s\t%s\t%s\n" "$d" "$c" "$gpu" "$last_iter"
    '
}

read_snapshot() {
    local snapshot
    if ! snapshot="$(remote_snapshot 2>/dev/null)"; then
        d_status=ssh_failed
        c_status=ssh_failed
        gpu=unavailable
        last_iter=0
        return 0
    fi
    IFS=$'\t' read -r d_status c_status gpu last_iter <<< "$snapshot"
    [[ -n "$d_status" ]] || d_status=ssh_failed
    [[ -n "$c_status" ]] || c_status=ssh_failed
    [[ -n "$gpu" ]] || gpu=unavailable
    [[ -n "$last_iter" ]] || last_iter=0
}

sample() {
    local note
    read_snapshot
    note="D=${d_status}; C=${c_status}"
    printf '{"ts_utc":"%s","status":"%s","gpu":"%s","last_iter":%s,"note":"%s"}\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$d_status" "$gpu" "$last_iter" "$note" >> "$MONITOR"
    printf '%s d=%s c=%s gpu=%s iter=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$d_status" "$c_status" "$gpu" "$last_iter" >> "$LOG"
}

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) watcher start" >> "$LOG"

while :; do
    sample
    case "$d_status" in
        ssh_failed)
            echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) transient SSH failure; retrying" >> "$LOG"
            sleep 60
            continue
            ;;
        failed:*|missing)
            echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) stopping: D=${d_status}" >> "$LOG"
            exit 1
            ;;
    esac
    case "$c_status" in
        ssh_failed)
            echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) transient SSH failure for C; retrying" >> "$LOG"
            sleep 60
            continue
            ;;
        failed:*)
            echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) stopping: C=${c_status}" >> "$LOG"
            exit 1
            ;;
        complete)
            break
            ;;
    esac
    sleep 1200
done

if [[ -e "$LOCAL_C" ]]; then
    echo "refusing to overwrite existing local C export: $LOCAL_C" >> "$LOG"
    exit 2
fi
remote_ready=0
for attempt in 1 2 3 4 5; do
    if remote "test -f ${C_EXP}/midpoint/prediction_scores.json && test -f ${C_EXP}/final/prediction_scores.json && test \"\$(find ${C_EXP}/midpoint/predictions -type f -name '*.lines.txt' | wc -l)\" -eq 7100 && test \"\$(find ${C_EXP}/final/predictions -type f -name '*.lines.txt' | wc -l)\" -eq 7100"; then
        remote_ready=1
        break
    fi
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) C ready check failed (attempt ${attempt}/5); retrying" >> "$LOG"
    sleep 60
done
if [[ "$remote_ready" -ne 1 ]]; then
    echo "C export did not pass the remote readiness check" >> "$LOG"
    exit 5
fi
scp "${SCP_OPTS[@]}" -r "${REMOTE}:${C_EXP}" "${ROOT}/outputs/" >> "$LOG" 2>&1

[[ -x "$ORACLE_PYTHON" ]] || { echo "missing official Oracle Python: $ORACLE_PYTHON" >> "$LOG"; exit 3; }
if [[ ! -x "$LANE_PYTHON" ]]; then
    LANE_PYTHON="python3"
fi
if [[ -e "$SCAN_DIR" ]]; then
    echo "refusing to overwrite existing scan directory: $SCAN_DIR" >> "$LOG"
    exit 4
fi

"$LANE_PYTHON" "$ROOT/scripts/scan_lvo_conf_checkpoints.py" \
    --manifest "$ROOT/data/processed/manifest_train.jsonl" \
    --gt-dir "$ROOT/outputs/a_gt_train_20260905/Lane/anno_txt" \
    --official-python "$ORACLE_PYTHON" \
    --output-dir "$SCAN_DIR" \
    --input "midpoint=${LOCAL_C}/midpoint/predictions=${LOCAL_C}/midpoint/prediction_scores.json" \
    --input "final=${LOCAL_C}/final/predictions=${LOCAL_C}/final/prediction_scores.json" \
    --threshold 0.30 --threshold 0.35 --threshold 0.40 --threshold 0.45 \
    --threshold 0.50 --threshold 0.55 --threshold 0.60 --threshold 0.65 \
    --bootstrap 10000 --seed 42 >> "$LOG" 2>&1

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) C export downloaded and Oracle scan complete" >> "$LOG"
