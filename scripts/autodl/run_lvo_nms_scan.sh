#!/usr/bin/env bash
set -Eeuo pipefail

# Eval-only NMS screen on the completed 36ep LVO checkpoints. The output root
# defaults to /tmp because the persistent AutoDL data disk is near full; only
# the aggregated prediction trees need to be retained for later Oracle work.

PROJECT_ROOT="${HARDLANE_PROJECT_ROOT:-/hy-tmp/lane-detection-challenge}"
UNLANEDET_ROOT="${UNLANEDET_ROOT:-/hy-tmp/UnLanedet}"
DATA_ROOT="${HARDLANE_DATA_ROOT:-/hy-tmp/datasets/HardLane/Lane}"
WEIGHTS_ROOT="${HARDLANE_WEIGHTS_ROOT:-/hy-tmp/weights}"
D_EXP="${HARDLANE_D_EXP:-/hy-tmp/lane-outputs/lvo_clrnet_r50_36ep_20260905}"
OUTPUT_ROOT="${HARDLANE_NMS_OUTPUT_ROOT:-/tmp/lane-nms-scan-20260906}"
PYTHON_BIN="${HARDLANE_PYTHON:-/usr/local/miniconda3/envs/py39/bin/python}"
MANIFESTS="$D_EXP/manifests"
CONFIG="$PROJECT_ROOT/configs/unlanedet/clrnet_r50_hardlane.py"
TRAIN_NET="$UNLANEDET_ROOT/tools/train_net.py"
STATUS="$OUTPUT_ROOT/status"
LOG="$OUTPUT_ROOT/run.log"

export HARDLANE_PROJECT_ROOT="$PROJECT_ROOT"
export HARDLANE_DATA_ROOT="$DATA_ROOT"
export HARDLANE_WEIGHTS_ROOT="$WEIGHTS_ROOT"
export HARDLANE_OUTPUT_ROOT="$OUTPUT_ROOT"

[ ! -e "$OUTPUT_ROOT" ] || { echo "refusing to overwrite $OUTPUT_ROOT" >&2; exit 2; }
for path in "$PROJECT_ROOT" "$UNLANEDET_ROOT" "$DATA_ROOT" "$D_EXP" "$MANIFESTS" "$CONFIG" "$TRAIN_NET"; do
    [ -e "$path" ] || { echo "missing required path: $path" >&2; exit 3; }
done
mkdir -p "$OUTPUT_ROOT"
printf '%s\n' running > "$STATUS"

fail() {
    code=$?
    printf '%s\n' "failed:$code" > "$STATUS"
    printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ) failed:$code" >> "$LOG"
    exit "$code"
}
trap fail ERR

for spec in nms_topk8 nms_thres30; do
    case "$spec" in
        nms_topk8) nms_topk=8; nms_thres=50;;
        nms_thres30) nms_topk=12; nms_thres=30;;
        *) echo "unknown NMS spec: $spec" >&2; exit 4;;
    esac
    variant_root="$OUTPUT_ROOT/$spec"
    mkdir -p "$variant_root/predictions"
    for fold_dir in $(find "$MANIFESTS" -mindepth 1 -maxdepth 1 -type d -name 'fold_*_v*' | sort); do
        fold=$(basename "$fold_dir")
        checkpoint="$D_EXP/runs/$fold/model_final.pth"
        holdout_manifest="$fold_dir/manifest_holdout.jsonl"
        eval_dir="$variant_root/$fold"
        [ -f "$checkpoint" ] || { echo "missing checkpoint: $checkpoint" >&2; exit 5; }
        [ -f "$holdout_manifest" ] || { echo "missing manifest: $holdout_manifest" >&2; exit 6; }
        mkdir -p "$eval_dir"
        "$PYTHON_BIN" "$TRAIN_NET" --eval-only --config-file "$CONFIG" --num-gpus 1 \
            "train.init_checkpoint='$checkpoint'" \
            "train.output_dir='$eval_dir'" \
            "dataloader.evaluator.output_basedir='$eval_dir/val'" \
            "dataloader.test.dataset.manifest_path='$holdout_manifest'" \
            "dataloader.test.dataset.split='val'" \
            "model.head.cfg.test_parameters.conf_threshold=0.0" \
            "dataloader.evaluator.cfg.test_parameters.conf_threshold=0.0" \
            "model.head.cfg.test_parameters.nms_topk=$nms_topk" \
            "model.head.cfg.test_parameters.nms_thres=$nms_thres" \
            train.seed=42 train.cudnn_benchmark=False \
            > "$eval_dir/eval.log" 2>&1
        pred_root="$eval_dir/val/predictions"
        score_json="$eval_dir/val/prediction_scores.json"
        expected=$(wc -l < "$holdout_manifest")
        actual=$(find "$pred_root" -type f -name '*.lines.txt' | wc -l | tr -d ' ')
        [ "$expected" -eq "$actual" ] || {
            echo "prediction count mismatch $spec/$fold: $actual != $expected" >&2
            exit 7
        }
        grep -Fq '"score_schema_version": 1' "$score_json"
        grep -Fq '"score_semantics": "positive_class_softmax_probability"' "$score_json"
        grep -Fq '"candidate_export_conf_threshold": 0.0' "$score_json"
        while IFS= read -r -d '' source_path; do
            rel="${source_path#"$pred_root"/}"
            destination="$variant_root/predictions/$rel"
            mkdir -p "$(dirname "$destination")"
            [ ! -e "$destination" ] || { echo "duplicate destination: $destination" >&2; exit 8; }
            cp "$source_path" "$destination"
        done < <(find "$pred_root" -type f -name '*.lines.txt' -print0)
        printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ) $spec/$fold pass" >> "$LOG"
    done

    SPEC="$spec" VARIANT_ROOT="$variant_root" MANIFEST="$MANIFESTS/source_manifest_train.jsonl" \
      "$PYTHON_BIN" - <<'PY'
import json, math, os
from pathlib import Path

root = Path(os.environ["VARIANT_ROOT"])
manifest = Path(os.environ["MANIFEST"])
semantics = "positive_class_softmax_probability"
schema = 1
scores = {}
for path in sorted(root.glob("fold_*/val/prediction_scores.json")):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("score_schema_version") != schema or payload.get("score_semantics") != semantics:
        raise SystemExit(f"invalid score contract: {path}")
    if payload.get("score_range") != {"min": 0.0, "max": 1.0, "inclusive": True} or payload.get("post_nms") is not True:
        raise SystemExit(f"invalid score range/NMS contract: {path}")
    if float(payload.get("candidate_export_conf_threshold")) != 0.0:
        raise SystemExit(f"nonzero export threshold: {path}")
    fold_scores = payload.get("scores_by_image")
    if not isinstance(fold_scores, dict):
        raise SystemExit(f"invalid scores_by_image: {path}")
    overlap = scores.keys() & fold_scores.keys()
    if overlap:
        raise SystemExit(f"duplicate score image: {sorted(overlap)[:3]}")
    scores.update(fold_scores)
expected = {
    json.loads(line)["image_id"]
    for line in manifest.read_text(encoding="utf-8").splitlines()
    if line.strip()
}
if set(scores) != expected:
    raise SystemExit(f"score coverage mismatch: {len(scores)} != {len(expected)}")
for image_id, values in scores.items():
    if not isinstance(values, list) or any(
        value is None or not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0
        for value in values
    ):
        raise SystemExit(f"invalid probability scores: {image_id}")
(root / "prediction_scores.json").write_text(
    json.dumps({
        "status": "pass",
        "checkpoint": os.environ["SPEC"],
        "candidate_export_conf_threshold": 0.0,
        "score_schema_version": schema,
        "score_semantics": semantics,
        "score_range": {"min": 0.0, "max": 1.0, "inclusive": True},
        "post_nms": True,
        "images": len(scores),
        "scores_by_image": scores,
    }, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
PY
done

printf '%s\n' complete > "$STATUS"
printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ) NMS scan complete" >> "$LOG"
