#!/usr/bin/env bash
set -uo pipefail

# Queue a second hard-clip oversampling strength as a diversity member. It
# waits for the x3 full-71 run, reuses only its validated 16-clip list, and
# trains an isolated x5/seed48 model with the same 36-epoch optimization
# budget. No test image is used for training and no submission is performed.

PROJECT_ROOT=/hy-tmp/lane-detection-challenge
DATA_ROOT=/hy-tmp/datasets/HardLane/Lane
UNLANEDET_ROOT=/hy-tmp/UnLanedet
WEIGHTS_ROOT=/hy-tmp/weights
OUTPUT_ROOT=/hy-tmp/lane-outputs
PYTHON_BIN=/usr/local/miniconda3/envs/py39/bin/python

UPSTREAM_STATUS=$OUTPUT_ROOT/all71_os_hardknown_x3_seed47_clrnet_r50_36ep_20260909.status
UPSTREAM_SCRIPT=$OUTPUT_ROOT/run_full71_os_hardknown_after_queue.sh
BASE_MANIFEST=$OUTPUT_ROOT/experiments_manifest_train_all71.jsonl
X3_META=$OUTPUT_ROOT/experiments_manifest_train_all71_knownhard_x3.metadata.json
DERIVED_MANIFEST=$OUTPUT_ROOT/experiments_manifest_train_all71_knownhard_x5.jsonl
DERIVED_META=$OUTPUT_ROOT/experiments_manifest_train_all71_knownhard_x5.metadata.json

RUN_NAME=all71_os_hardknown_x5_seed48_clrnet_r50_36ep_20260909
RUN_DIR=$OUTPUT_ROOT/runs/$RUN_NAME
STATUS=$OUTPUT_ROOT/$RUN_NAME.status
LOG=$OUTPUT_ROOT/$RUN_NAME.log
RUNS=$OUTPUT_ROOT/$RUN_NAME.runs
LOCK=$OUTPUT_ROOT/.$RUN_NAME.lock
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
        | grep -Fq 'run_full71_os_hardknown_x5_after_queue_20260909.sh'; then
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
set_status waiting_for_x3
log "queued behind x3 status=$UPSTREAM_STATUS script=$UPSTREAM_SCRIPT"

deadline=$(( $(date +%s) + WAIT_TIMEOUT_SECONDS ))
while true; do
    upstream=$(sed -n '1p' "$UPSTREAM_STATUS" 2>/dev/null || true)
    case "$upstream" in
        complete|complete_with_failures|failed|aborted_*)
            log "x3 queue reached terminal state=$upstream"
            break
            ;;
    esac
    upstream_pids=$(pgrep -f '[r]un_full71_os_hardknown_after_queue.sh' 2>/dev/null || true)
    if [ -z "$upstream_pids" ] && [ -n "$upstream" ]; then
        set_status aborted_x3_missing
        log "x3 launcher absent with nonterminal state=$upstream"
        exit 23
    fi
    if [ -z "$upstream_pids" ] && [ -z "$upstream" ]; then
        set_status aborted_x3_status_missing
        log "x3 status and launcher both absent"
        exit 23
    fi
    if [ "$(date +%s)" -gt "$deadline" ]; then
        set_status aborted_timeout_waiting_x3
        log "timed out waiting for x3 (last=$upstream)"
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

for path in "$PROJECT_ROOT/scripts/autodl/run_training.py" \
            "$BASE_MANIFEST" "$X3_META" "$PROJECT_ROOT" "$DATA_ROOT" \
            "$UNLANEDET_ROOT" "$WEIGHTS_ROOT"; do
    if [ ! -e "$path" ]; then
        set_status aborted_missing_input
        log "missing required path: $path"
        exit 4
    fi
done

if [ -e "$DERIVED_MANIFEST" ]; then
    if ! "$PYTHON_BIN" - "$DERIVED_MANIFEST" "$DERIVED_META" <<'PY' >> "$LOG" 2>&1
import hashlib
import json
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
meta = Path(sys.argv[2])
value = json.loads(meta.read_text(encoding="utf-8"))
digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
if value.get("status") != "pass" or value.get("output_manifest_sha256") != digest:
    raise SystemExit("existing x5 manifest failed metadata/SHA validation")
if value.get("hard_clip_count") != 16 or value.get("repeat_factor") != 5:
    raise SystemExit("existing x5 manifest has unexpected shape")
if value.get("output_manifest_rows") != 13500:
    raise SystemExit("existing x5 manifest has unexpected row count")
print(json.dumps({"status": "reuse", "sha256": digest, "rows": 13500}))
PY
    then
        set_status aborted_bad_derived_manifest
        log "existing x5 manifest is invalid; refusing to overwrite it"
        exit 6
    fi
else
    temp_manifest=$DERIVED_MANIFEST.tmp.$$
    temp_meta=$DERIVED_META.tmp.$$
    rm -f "$temp_manifest" "$temp_meta"
    if ! "$PYTHON_BIN" - "$BASE_MANIFEST" "$X3_META" "$temp_manifest" "$temp_meta" <<'PY' >> "$LOG" 2>&1
import hashlib
import json
import sys
from pathlib import Path

base_path, x3_meta_path, output_path, meta_path = map(Path, sys.argv[1:])

base = [
    json.loads(line)
    for line in base_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
x3 = json.loads(x3_meta_path.read_text(encoding="utf-8"))
hard_clips = set(x3.get("hard_clips", []))
if len(base) != 7100 or len(hard_clips) != 16:
    raise SystemExit(f"unexpected x3/base shape: base={len(base)} hard={len(hard_clips)}")
if x3.get("status") != "pass" or x3.get("repeat_factor") != 3:
    raise SystemExit("x3 metadata is not a validated x3 manifest")
if any(row["clip_id"] in hard_clips for row in base) is False:
    raise SystemExit("none of the x3 hard clips occurs in full-71 base")

hard_base_rows = sum(row["clip_id"] in hard_clips for row in base)
if hard_base_rows != 1600:
    raise SystemExit(f"unexpected hard rows in full-71 base: {hard_base_rows}")
output_rows = []
for row in base:
    output_rows.extend([row] * (5 if row["clip_id"] in hard_clips else 1))
if len(output_rows) != 13500:
    raise SystemExit(f"unexpected x5 row count: {len(output_rows)}")

output_path.write_text(
    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output_rows),
    encoding="utf-8",
)
output_sha = hashlib.sha256(output_path.read_bytes()).hexdigest()
metadata = {
    "status": "pass",
    "protocol": "full71_base_plus_known_hard_clips_x5",
    "base_manifest": str(base_path),
    "base_manifest_sha256": hashlib.sha256(base_path.read_bytes()).hexdigest(),
    "base_manifest_rows": len(base),
    "source_x3_metadata": str(x3_meta_path),
    "source_x3_metadata_sha256": hashlib.sha256(x3_meta_path.read_bytes()).hexdigest(),
    "output_manifest": str(output_path),
    "output_manifest_sha256": output_sha,
    "output_manifest_rows": len(output_rows),
    "hard_base_rows": hard_base_rows,
    "hard_clip_count": len(hard_clips),
    "hard_clips": sorted(hard_clips),
    "repeat_factor": 5,
}
meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"status": "pass", "sha256": output_sha, "rows": len(output_rows),
                  "hard_clips": sorted(hard_clips)}, ensure_ascii=False))
PY
    then
        rm -f "$temp_manifest" "$temp_meta"
        set_status aborted_manifest_derivation
        log "failed to derive x5 manifest"
        exit 6
    fi
    mv "$temp_manifest" "$DERIVED_MANIFEST"
    mv "$temp_meta" "$DERIVED_META"
fi

export HARDLANE_PROJECT_ROOT=$PROJECT_ROOT
export HARDLANE_DATA_ROOT=$DATA_ROOT
export UNLANEDET_ROOT=$UNLANEDET_ROOT
export HARDLANE_WEIGHTS_ROOT=$WEIGHTS_ROOT
export HARDLANE_OUTPUT_ROOT=$OUTPUT_ROOT
export HARDLANE_PYTHON=$PYTHON_BIN

set_status running
log "starting $RUN_NAME with manifest=$DERIVED_MANIFEST rows=13500 iters_per_epoch=592"
if "$PYTHON_BIN" "$PROJECT_ROOT/scripts/autodl/run_training.py" \
    --model clrnet_r50 \
    --run-name "$RUN_NAME" \
    --epochs 36 \
    --seed 48 \
    --train-manifest "$DERIVED_MANIFEST" \
    --iters-per-epoch 592 \
    --eval-every-epochs 6 \
    --max-to-keep 2 \
    --skip-if-complete \
    --auto-resume >> "$LOG" 2>&1; then
    printf '%s:ok\n' "$RUN_NAME" >> "$RUNS"
    set_status complete
    log "run complete"
else
    code=$?
    printf '%s:failed:%s\n' "$RUN_NAME" "$code" >> "$RUNS"
    set_status failed
    log "run failed exit=$code"
    exit "$code"
fi
