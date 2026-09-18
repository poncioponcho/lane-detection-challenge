#!/usr/bin/env bash
set -uo pipefail

# Queue one independent full-71-clip run using the 16 clips already identified
# as worst by the frozen 15-epoch LVO OOF. The source oversampled manifest is
# read on the GPU host; no training manifest or image path is transferred from
# the workstation.
#
# This launcher is external to the training checkout. It waits for the
# existing seed45/46 queue, drains CUDA, derives a reproducible 10,300-row
# manifest, and launches exactly one 36-epoch seed47 run. It never uses
# testA/testB data and never touches incumbent artifacts.

PROJECT_ROOT=/hy-tmp/lane-detection-challenge
DATA_ROOT=/hy-tmp/datasets/HardLane/Lane
UNLANEDET_ROOT=/hy-tmp/UnLanedet
WEIGHTS_ROOT=/hy-tmp/weights
OUTPUT_ROOT=/hy-tmp/lane-outputs
PYTHON_BIN=/usr/local/miniconda3/envs/py39/bin/python

UPSTREAM_STATUS=$OUTPUT_ROOT/full71_seed45_46_after_chain.status
UPSTREAM_SCRIPT=$OUTPUT_ROOT/run_full71_seed45_46_after_chain_20260909.sh
BASE_MANIFEST=$OUTPUT_ROOT/experiments_manifest_train_all71.jsonl
KNOWN_SOURCE=$OUTPUT_ROOT/experiments_manifest_train_os_v1_seed42_hard25_x3.jsonl
DERIVED_MANIFEST=$OUTPUT_ROOT/experiments_manifest_train_all71_knownhard_x3.jsonl
DERIVED_META=$OUTPUT_ROOT/experiments_manifest_train_all71_knownhard_x3.metadata.json

RUN_NAME=all71_os_hardknown_x3_seed47_clrnet_r50_36ep_20260909
RUN_DIR=$OUTPUT_ROOT/runs/$RUN_NAME
STATUS=$OUTPUT_ROOT/$RUN_NAME.status
LOG=$OUTPUT_ROOT/$RUN_NAME.log
RUNS=$OUTPUT_ROOT/$RUN_NAME.runs
LOCK=$OUTPUT_ROOT/.$RUN_NAME.lock
SELF_MARKER=run_full71_os_hardknown_after_queue.sh
WAIT_TIMEOUT_SECONDS=$((48 * 3600))

log() {
    printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "$LOG"
}

set_status() {
    printf '%s\n' "$1" > "$STATUS"
}

mkdir -p "$OUTPUT_ROOT"

# mkdir is atomic. A stale lock is recoverable only when its recorded owner
# is no longer the same launcher; this permits --auto-resume after an SSH
# interruption without allowing two seed47 processes to coexist.
if ! mkdir "$LOCK" 2>/dev/null; then
    owner=
    if [ -f "$LOCK/pid" ]; then
        owner=$(sed -n '1p' "$LOCK/pid" 2>/dev/null || true)
    fi
    if [ -n "$owner" ] && kill -0 "$owner" 2>/dev/null \
        && ps -p "$owner" -o args= 2>/dev/null | grep -Fq "$SELF_MARKER"; then
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

if [ -f "$RUN_DIR/run_evidence.json" ] && [ -f "$STATUS" ]; then
    state=$(sed -n '1p' "$STATUS" 2>/dev/null || true)
    case "$state" in
        complete|complete_with_failures)
            log "run already marked terminal: $state"
            exit 0
            ;;
    esac
fi

touch "$RUNS"
set_status waiting_for_seed45_46
log "queued behind status=$UPSTREAM_STATUS upstream_script=$UPSTREAM_SCRIPT"

deadline=$(( $(date +%s) + WAIT_TIMEOUT_SECONDS ))
while true; do
    upstream=$(sed -n '1p' "$UPSTREAM_STATUS" 2>/dev/null || true)
    case "$upstream" in
        complete|complete_with_failures|aborted_*|failed)
            log "upstream reached terminal state=$upstream"
            break
            ;;
    esac

    # If the status writer disappeared without publishing a terminal state,
    # stop rather than racing a possibly orphaned seed45/46 chain. A later
    # explicit relaunch can inspect and recover the upstream queue safely.
    upstream_pids=$(pgrep -f '[r]un_full71_seed45_46_after_chain' 2>/dev/null || true)
    if [ -z "$upstream_pids" ] && [ -n "$upstream" ]; then
        set_status aborted_upstream_missing
        log "upstream launcher is absent with nonterminal state=$upstream; refusing to race it"
        exit 23
    fi
    if [ -z "$upstream_pids" ] && [ -z "$upstream" ]; then
        set_status aborted_upstream_status_missing
        log "upstream status and launcher are both absent; refusing to start"
        exit 23
    fi
    if [ "$(date +%s)" -gt "$deadline" ]; then
        set_status aborted_timeout_waiting_seed45_46
        log "timed out waiting for upstream state (last=$upstream)"
        exit 24
    fi
    sleep 60
done

set_status waiting_for_gpu_drain
drain_deadline=$(( $(date +%s) + $((3 * 3600)) ))
while true; do
    gpu_procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
        | awk 'NF' | wc -l | tr -d ' ')
    lvo_procs=$(pgrep -f '[l]vo_video_runner.py' 2>/dev/null || true)
    train_procs=$(pgrep -f '[t]ools/train_net.py' 2>/dev/null || true)
    merge_procs=$(pgrep -f '[m]erge_lvo_oof_scores.py' 2>/dev/null || true)
    if [ -z "$gpu_procs" ]; then gpu_procs=0; fi
    if [ "$gpu_procs" -eq 0 ] && [ -z "$lvo_procs" ] \
        && [ -z "$train_procs" ] && [ -z "$merge_procs" ]; then
        break
    fi
    if [ "$(date +%s)" -gt "$drain_deadline" ]; then
        set_status aborted_gpu_drain_timeout
        log "GPU/process drain timed out: gpu=$gpu_procs lvo=$lvo_procs train=$train_procs merge=$merge_procs"
        exit 25
    fi
    sleep 30
done

for path in "$PROJECT_ROOT/scripts/autodl/run_training.py" \
            "$BASE_MANIFEST" "$KNOWN_SOURCE" "$PROJECT_ROOT" \
            "$DATA_ROOT" "$UNLANEDET_ROOT" "$WEIGHTS_ROOT"; do
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
    raise SystemExit("existing derived manifest failed metadata/SHA validation")
if value.get("output_manifest_rows") != 10300 or value.get("hard_clip_count") != 16:
    raise SystemExit("existing derived manifest has unexpected shape")
print(json.dumps({"status": "reuse", "sha256": digest, "rows": 10300}))
PY
    then
        set_status aborted_bad_derived_manifest
        log "existing derived manifest is invalid; refusing to overwrite it"
        exit 6
    fi
else
    temp_manifest=$DERIVED_MANIFEST.tmp.$$
    temp_meta=$DERIVED_META.tmp.$$
    rm -f "$temp_manifest" "$temp_meta"
    if ! "$PYTHON_BIN" - "$BASE_MANIFEST" "$KNOWN_SOURCE" "$temp_manifest" "$temp_meta" <<'PY' >> "$LOG" 2>&1
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

base_path, source_path, output_path, meta_path = map(Path, sys.argv[1:])

def rows(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

base = rows(base_path)
source = rows(source_path)
if len(base) != 7100 or len(source) != 9500:
    raise SystemExit(f"unexpected input sizes: base={len(base)} source={len(source)}")

by_image = {}
counts = Counter()
for row in source:
    image_id = row["image_id"]
    counts[image_id] += 1
    canonical = json.dumps(
        row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    previous = by_image.setdefault(image_id, canonical)
    if previous != canonical:
        raise SystemExit(f"repeated image has non-identical rows: {image_id}")

if any(count not in (1, 3) for count in counts.values()):
    raise SystemExit("source oversampling contains a repeat count other than 1 or 3")
hard_images = {image_id for image_id, count in counts.items() if count == 3}
hard_clips = {row["clip_id"] for row in source if row["image_id"] in hard_images}
if len(hard_clips) != 16:
    raise SystemExit(f"expected 16 hard clips, found {len(hard_clips)}")

base_by_image = {row["image_id"]: row for row in base}
source_ids = {row["image_id"] for row in source}
if not source_ids.issubset(base_by_image):
    raise SystemExit("source oversampling image set is not a subset of full-71 base")
if any(image_id not in base_by_image for image_id in hard_images):
    raise SystemExit("hard image missing from full-71 base")
if any(
    row["clip_id"] in hard_clips and counts[row["image_id"]] != 3
    for row in base
):
    raise SystemExit("hard clip is not repeated uniformly")

output_rows = []
for row in base:
    repeats = 3 if row["clip_id"] in hard_clips else 1
    output_rows.extend([row] * repeats)
if len(output_rows) != 10300:
    raise SystemExit(f"unexpected derived row count: {len(output_rows)}")

output_path.write_text(
    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output_rows),
    encoding="utf-8",
)
output_sha = hashlib.sha256(output_path.read_bytes()).hexdigest()
metadata = {
    "status": "pass",
    "protocol": "full71_base_plus_known_hard_clips_x3",
    "base_manifest": str(base_path),
    "base_manifest_sha256": hashlib.sha256(base_path.read_bytes()).hexdigest(),
    "base_manifest_rows": len(base),
    "source_oversampled_manifest": str(source_path),
    "source_oversampled_manifest_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
    "source_oversampled_manifest_rows": len(source),
    "output_manifest": str(output_path),
    "output_manifest_sha256": output_sha,
    "output_manifest_rows": len(output_rows),
    "hard_clip_count": len(hard_clips),
    "hard_clips": sorted(hard_clips),
    "repeat_factor": 3,
}
meta_path.write_text(
    json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(
    json.dumps(
        {
            "status": "pass",
            "sha256": output_sha,
            "rows": len(output_rows),
            "hard_clips": sorted(hard_clips),
        },
        ensure_ascii=False,
    )
)
PY
    then
        rm -f "$temp_manifest" "$temp_meta"
        set_status aborted_manifest_derivation
        log "failed to derive known-hard manifest"
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
log "starting $RUN_NAME with manifest=$DERIVED_MANIFEST rows=10300 iters_per_epoch=592"
if "$PYTHON_BIN" "$PROJECT_ROOT/scripts/autodl/run_training.py" \
    --model clrnet_r50 \
    --run-name "$RUN_NAME" \
    --epochs 36 \
    --seed 47 \
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
