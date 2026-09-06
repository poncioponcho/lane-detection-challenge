#!/usr/bin/env bash
set -Eeuo pipefail

# Standalone D experiment.  It deliberately does not touch pipeline.status or
# the frozen v1 manifests/config.  Checkpoints are retained at the midpoint and
# final point so C can test checkpoint selection without retaining 36 files per
# fold.

PROJECT_ROOT=/hy-tmp/lane-detection-challenge
UNLANEDET_ROOT=/hy-tmp/UnLanedet
DATA_ROOT=/hy-tmp/datasets/HardLane/Lane
WEIGHTS_ROOT=/hy-tmp/weights
OUTPUT_ROOT=/hy-tmp/lane-outputs
SOURCE_EXP="$OUTPUT_ROOT/lvo_clrnet_r50_15ep_20260904"
EXP="$OUTPUT_ROOT/lvo_clrnet_r50_36ep_20260905"
MANIFESTS="$SOURCE_EXP/manifests"
CONFIG="$PROJECT_ROOT/configs/unlanedet/clrnet_r50_hardlane.py"
TRAIN_NET="$UNLANEDET_ROOT/tools/train_net.py"
PYTHON_BIN=/usr/local/miniconda3/envs/py39/bin/python
STATUS="$OUTPUT_ROOT/lvo36.status"
LOG="$OUTPUT_ROOT/lvo36.log"
RESUME="${RESUME:-0}"

# The normal pipeline exports these before invoking train_net.py.  D is a
# standalone experiment, so make the same contract explicit before LazyConfig
# is loaded.
export HARDLANE_PROJECT_ROOT="$PROJECT_ROOT"
export HARDLANE_DATA_ROOT="$DATA_ROOT"
export HARDLANE_WEIGHTS_ROOT="$WEIGHTS_ROOT"
export HARDLANE_OUTPUT_ROOT="$OUTPUT_ROOT"
export HARDLANE_PYTHON="$PYTHON_BIN"

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

write_status running
printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ) D start" >> "$LOG"

if [ "$(cat "$OUTPUT_ROOT/pipeline.status")" != "complete" ]; then
    echo "pipeline.status is not complete" >&2
    exit 2
fi
if [ "$RESUME" != 1 ] && [ -e "$EXP" ]; then
    echo "refusing to overwrite existing experiment: $EXP" >&2
    exit 3
fi
for path in "$PROJECT_ROOT" "$UNLANEDET_ROOT" "$DATA_ROOT" "$WEIGHTS_ROOT" \
            "$SOURCE_EXP" "$MANIFESTS" "$CONFIG" "$TRAIN_NET"; do
    [ -e "$path" ] || { echo "missing required path: $path" >&2; exit 4; }
done

project_head=$(git -C "$PROJECT_ROOT" rev-parse HEAD)
unlanedet_head=$(git -C "$UNLANEDET_ROOT" rev-parse HEAD)
[ "$unlanedet_head" = 03921844220adb2e65c840de2d9759478d5c3d4c ] || {
    echo "unexpected UnLanedet HEAD: $unlanedet_head" >&2
    exit 5
}
tracked_dirty=$(git -C "$PROJECT_ROOT" status --short --untracked-files=no)
if [ -n "$tracked_dirty" ]; then
    if [ "$RESUME" = 1 ] && [ "$tracked_dirty" = " M src/integrations/unlanedet_hardlane.py" ]; then
        printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ) resume preserving pre-existing tracked change: $tracked_dirty" >> "$LOG"
    else
        echo "project tracked worktree dirty: $tracked_dirty" >&2
        exit 6
    fi
fi

avail_kb=$(df -Pk "$OUTPUT_ROOT" | awk 'NR==2 {print $4}')
[ "$avail_kb" -ge 9000000 ] || {
    echo "less than 9GB available on output filesystem: ${avail_kb}KB" >&2
    exit 7
}

if [ "$RESUME" = 1 ]; then
    [ -f "$EXP/manifests/source_manifest_train.jsonl" ] || {
        echo "resume experiment is missing copied manifests: $EXP" >&2
        exit 12
    }
    [ -f "$EXP/protocol_remote.json" ] || {
        echo "resume experiment is missing protocol: $EXP" >&2
        exit 13
    }
else
    mkdir -p "$EXP/manifests" "$EXP/runs" "$EXP/oof/predictions"
    cp "$MANIFESTS/source_manifest_train.jsonl" "$EXP/manifests/source_manifest_train.jsonl"
    for fold_dir in "$MANIFESTS"/fold_*_v*; do
        cp -R "$fold_dir" "$EXP/manifests/"
    done

    SOURCE_SHA=$(sha256sum "$MANIFESTS/source_manifest_train.jsonl" | awk '{print $1}')
    CONFIG_SHA=$(sha256sum "$CONFIG" | awk '{print $1}')
    cat > "$EXP/protocol_remote.json" <<EOF
{
  "status": "running",
  "protocol": "leave-one-video-out",
  "model": "clrnet_r50",
  "epochs": 36,
  "batch_size": 12,
  "input": "800x320",
  "cut_height": 180,
  "conf_threshold": 0.4,
  "candidate_export_conf_threshold": 0.0,
  "checkpoint_policy": "retain midpoint and final checkpoints; no holdout selection during D",
  "checkpoint_period_epochs": 18,
  "checkpoint_max_to_keep": 3,
  "confidence_export": true,
  "score_schema_version": 1,
  "score_semantics": "positive_class_softmax_probability",
  "post_nms": true,
  "c_policy": "filter exported post-NMS candidates offline, then score with frozen Oracle",
  "project_git_head": "$project_head",
  "unlanedet_git_head": "$unlanedet_head",
  "source_manifest_sha256": "$SOURCE_SHA",
  "source_manifest_rows": 7100,
  "config_sha256": "$CONFIG_SHA",
  "started_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
fi

mkdir -p "$EXP/runs" "$EXP/oof/predictions"

for fold_dir in $(find "$MANIFESTS" -mindepth 1 -maxdepth 1 -type d -name 'fold_*_v*' | sort); do
    fold=$(basename "$fold_dir")
    train_manifest="$fold_dir/manifest_train.jsonl"
    holdout_manifest="$fold_dir/manifest_holdout.jsonl"
    train_rows=$(wc -l < "$train_manifest")
    holdout_rows=$(wc -l < "$holdout_manifest")
    iter_per_epoch=$((train_rows / 12))
    target_iter=$((iter_per_epoch * 36))
    checkpoint_period=$((iter_per_epoch * 18))
    run_dir="$EXP/runs/$fold"
    eval_dir="$run_dir/holdout_eval"
    state="$run_dir/fold_state.json"
    mkdir -p "$run_dir"
    printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ) fold=$fold train=$train_rows holdout=$holdout_rows target=$target_iter" >> "$LOG"

    checkpoint="$run_dir/model_final.pth"
    if [ ! -f "$checkpoint" ]; then
        "$PYTHON_BIN" "$TRAIN_NET" --config-file "$CONFIG" --num-gpus 1 \
            "train.max_iter=$target_iter" \
            "train.eval_period=0" \
            "train.checkpointer.period=$checkpoint_period" \
            "train.checkpointer.max_to_keep=3" \
            "train.output_dir='$run_dir'" \
            "dataloader.evaluator.output_basedir='$run_dir/train_eval'" \
            "dataloader.train.dataset.manifest_path='$train_manifest'" \
            "dataloader.train.dataset.split='train'" \
            "dataloader.test.dataset.manifest_path='$holdout_manifest'" \
            "dataloader.test.dataset.split='val'" \
            "model.head.cfg.test_parameters.conf_threshold=0.4" \
            train.seed=42 train.cudnn_benchmark=False \
            > "$run_dir/train.log" 2>&1
    fi

    [ -f "$checkpoint" ] || { echo "missing final checkpoint: $checkpoint" >&2; exit 8; }
    pred_root="$eval_dir/val/predictions"
    diagnostic="$eval_dir/val/diagnostic_metric.json"
    score_json="$eval_dir/val/prediction_scores.json"
    eval_ready=0
    if [ -d "$pred_root" ] && [ -f "$diagnostic" ] && [ -f "$score_json" ]; then
        prediction_count=$(find "$pred_root" -type f -name '*.lines.txt' | wc -l | tr -d ' ')
        if [ "$prediction_count" -eq "$holdout_rows" ] \
            && grep -Fq '"score_schema_version": 1' "$score_json" \
            && grep -Fq '"score_semantics": "positive_class_softmax_probability"' "$score_json" \
            && grep -Fq '"post_nms": true' "$score_json"; then
            eval_ready=1
        fi
    fi
    if [ "$eval_ready" -ne 1 ]; then
        if [ "$RESUME" = 1 ] && [ -e "$eval_dir" ]; then
            stale_eval="${eval_dir}.stale-incomplete-$(date -u +%Y%m%d-%H%M%S)"
            mv "$eval_dir" "$stale_eval"
        fi
        mkdir -p "$eval_dir"
        "$PYTHON_BIN" "$TRAIN_NET" --eval-only --config-file "$CONFIG" --num-gpus 1 \
            "train.init_checkpoint='$checkpoint'" \
            "train.output_dir='$eval_dir'" \
            "dataloader.evaluator.output_basedir='$eval_dir/val'" \
            "dataloader.test.dataset.manifest_path='$holdout_manifest'" \
            "dataloader.test.dataset.split='val'" \
            "model.head.cfg.test_parameters.conf_threshold=0.0" \
            train.seed=42 train.cudnn_benchmark=False \
            > "$eval_dir/eval.log" 2>&1
        pred_root="$eval_dir/val/predictions"
        diagnostic="$eval_dir/val/diagnostic_metric.json"
        score_json="$eval_dir/val/prediction_scores.json"
        [ -d "$pred_root" ] && [ -f "$diagnostic" ] && [ -f "$score_json" ] || {
            echo "missing evaluation evidence: $eval_dir" >&2
            exit 9
        }
        prediction_count=$(find "$pred_root" -type f -name '*.lines.txt' | wc -l | tr -d ' ')
        [ "$prediction_count" -eq "$holdout_rows" ] || {
            echo "prediction count mismatch for $fold: $prediction_count != $holdout_rows" >&2
            exit 10
        }
    fi
    sha=$(sha256sum "$checkpoint" | awk '{print $1}')
    cp "$diagnostic" "$run_dir/diagnostic_metric.json"
    cat > "$state" <<EOF
{
  "status": "pass",
  "fold": "$fold",
  "epochs": 36,
  "train_rows": $train_rows,
  "holdout_rows": $holdout_rows,
  "iterations_per_epoch": $iter_per_epoch,
  "target_max_iter": $target_iter,
  "checkpoint_period": $checkpoint_period,
  "checkpoint_sha256": "$sha",
  "checkpoint": "$checkpoint",
  "prediction_root": "$pred_root",
  "prediction_scores": "$score_json",
  "candidate_export_conf_threshold": 0.0,
  "prediction_count": $prediction_count,
  "finished_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF

    while IFS= read -r -d '' source_path; do
        rel=${source_path#"$pred_root/"}
        destination="$EXP/oof/predictions/$rel"
        mkdir -p "$(dirname "$destination")"
        if [ -e "$destination" ]; then
            cmp -s "$source_path" "$destination" || {
                echo "conflicting OOF destination: $destination" >&2
                exit 11
            }
        else
            cp "$source_path" "$destination"
        fi
    done < <(find "$pred_root" -type f -name '*.lines.txt' -print0)
    printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ) fold=$fold pass" >> "$LOG"
done

SOURCE_MANIFEST="$EXP/manifests/source_manifest_train.jsonl" \
OOF_ROOT="$EXP/oof/predictions" EXP_ROOT="$EXP" \
  "$PYTHON_BIN" - <<'PY'
import hashlib, json, math, os
from pathlib import Path

exp = Path(os.environ["EXP_ROOT"])
manifest = Path(os.environ["SOURCE_MANIFEST"])
pred_root = Path(os.environ["OOF_ROOT"])
records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
expected = {row["pred_rel_path"] for row in records}
actual = {path.relative_to(pred_root).as_posix() for path in pred_root.rglob("*.lines.txt")}
if expected != actual or len(expected) != 7100:
    raise SystemExit(f"OOF mismatch expected={len(expected)} actual={len(actual)}")
score_by_image = {}
score_semantics = "positive_class_softmax_probability"
score_schema_version = 1
for score_path in sorted((exp / "runs").glob("fold_*/holdout_eval/val/prediction_scores.json")):
    payload = json.loads(score_path.read_text(encoding="utf-8"))
    if payload.get("score_schema_version") != score_schema_version:
        raise SystemExit(f"unsupported score schema: {score_path}")
    if payload.get("score_semantics") != score_semantics:
        raise SystemExit(f"score semantics mismatch: {score_path}")
    if payload.get("score_range") != {"min": 0.0, "max": 1.0, "inclusive": True}:
        raise SystemExit(f"score range mismatch: {score_path}")
    if payload.get("post_nms") is not True:
        raise SystemExit(f"score sidecar is not post-NMS: {score_path}")
    fold_scores = payload.get("scores_by_image", {})
    if not isinstance(fold_scores, dict):
        raise SystemExit(f"score sidecar is not an object: {score_path}")
    overlap = set(score_by_image).intersection(fold_scores)
    if overlap:
        raise SystemExit(f"duplicate OOF scores: {sorted(overlap)[:3]}")
    score_by_image.update(fold_scores)
if set(score_by_image) != {row["image_id"] for row in records}:
    raise SystemExit(
        f"score sidecar mismatch expected={len(records)} actual={len(score_by_image)}"
    )
if any(not isinstance(scores, list) for scores in score_by_image.values()):
    raise SystemExit("score sidecar contains a non-list image entry")
for image_id, scores in score_by_image.items():
    for score in scores:
        if score is None or not math.isfinite(float(score)) or not 0.0 <= float(score) <= 1.0:
            raise SystemExit(f"non-probability score for {image_id}: {score!r}")
digest = hashlib.sha256()
for rel in sorted(actual):
    path = pred_root / rel
    file_hash = hashlib.sha256(path.read_bytes()).digest()
    digest.update(rel.encode())
    digest.update(b"\0")
    digest.update(file_hash)
    digest.update(b"\n")
value = {
    "status": "pass",
    "protocol": "leave-one-video-out",
    "source_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    "source_rows": len(records),
    "prediction_count": len(actual),
    "unique_image_count": len(actual),
    "each_image_once": True,
    "prediction_root": str(pred_root),
    "prediction_tree_sha256": digest.hexdigest(),
    "candidate_export_conf_threshold": 0.0,
    "score_schema_version": score_schema_version,
    "score_semantics": score_semantics,
    "score_range": {"min": 0.0, "max": 1.0, "inclusive": True},
    "post_nms": True,
    "prediction_score_count": len(score_by_image),
    "prediction_scores_path": str(exp / "oof" / "prediction_scores.json"),
    "created_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
}
(exp / "oof" / "oof_evidence.json").write_text(
    json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
(exp / "oof" / "prediction_scores.json").write_text(
    json.dumps(
        {
            "status": "pass",
            "images": len(score_by_image),
            "candidate_export_conf_threshold": 0.0,
            "score_schema_version": score_schema_version,
            "score_semantics": score_semantics,
            "score_range": {"min": 0.0, "max": 1.0, "inclusive": True},
            "post_nms": True,
            "scores_by_image": score_by_image,
        },
        ensure_ascii=False,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
protocol_path = exp / "protocol_remote.json"
protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
protocol.update({"status": "pass", "prediction_count": len(actual),
                "prediction_tree_sha256": digest.hexdigest(),
                "finished_utc": value["created_utc"]})
protocol_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY

write_status complete
printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ) D complete" >> "$LOG"
