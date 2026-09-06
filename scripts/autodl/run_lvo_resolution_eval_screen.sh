#!/usr/bin/env bash
set -Eeuo pipefail

# Eval-only high-resolution screen. It deliberately uses the completed
# 800x320-trained checkpoints and changes only inference preprocessing to
# 960x480/cut0. A positive result is only a screen signal; a full 960x480
# LVO training run would still be required before production use.

PROJECT_ROOT="${HARDLANE_PROJECT_ROOT:-/hy-tmp/lane-detection-challenge}"
UNLANEDET_ROOT="${UNLANEDET_ROOT:-/hy-tmp/UnLanedet}"
D_EXP="${HARDLANE_D_EXP:-/hy-tmp/lane-outputs/lvo_clrnet_r50_36ep_20260905}"
OUTPUT_ROOT="${HARDLANE_RESOLUTION_OUTPUT_ROOT:-/tmp/lane-resolution-eval-screen-20260906}"
PYTHON_BIN="${HARDLANE_PYTHON:-/usr/local/miniconda3/envs/py39/bin/python}"
BUILDER="$PROJECT_ROOT/scripts/autodl/build_resolution_config.py"
BASE_CONFIG="$PROJECT_ROOT/configs/unlanedet/clrnet_r50_hardlane.py"
TRAIN_NET="$UNLANEDET_ROOT/tools/train_net.py"
MANIFESTS="$D_EXP/manifests"
STATUS="$OUTPUT_ROOT/status"
LOG="$OUTPUT_ROOT/run.log"

[ ! -e "$OUTPUT_ROOT" ] || { echo "refusing to overwrite $OUTPUT_ROOT" >&2; exit 2; }
for path in "$PROJECT_ROOT" "$UNLANEDET_ROOT" "$D_EXP" "$MANIFESTS" \
            "$BUILDER" "$BASE_CONFIG" "$TRAIN_NET"; do
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

"$PYTHON_BIN" "$BUILDER" --base "$BASE_CONFIG" \
    --output "$OUTPUT_ROOT/clrnet_r50_hardlane_960x480_cut0.py" \
    --width 960 --height 480 --cut-height 0 >> "$LOG"
CONFIG="$OUTPUT_ROOT/clrnet_r50_hardlane_960x480_cut0.py"
grep -Fq 'img_w = 960' "$CONFIG"
grep -Fq 'img_h = 480' "$CONFIG"
grep -Fq 'cut_height = 0' "$CONFIG"

for fold_dir in $(find "$MANIFESTS" -mindepth 1 -maxdepth 1 -type d -name 'fold_*_v*' | sort); do
    fold=$(basename "$fold_dir")
    checkpoint="$D_EXP/runs/$fold/model_final.pth"
    holdout_manifest="$fold_dir/manifest_holdout.jsonl"
    eval_dir="$OUTPUT_ROOT/$fold"
    [ -f "$checkpoint" ] || { echo "missing checkpoint: $checkpoint" >&2; exit 4; }
    [ -f "$holdout_manifest" ] || { echo "missing manifest: $holdout_manifest" >&2; exit 5; }
    mkdir -p "$eval_dir"
    "$PYTHON_BIN" "$TRAIN_NET" --eval-only --config-file "$CONFIG" --num-gpus 1 \
        "train.init_checkpoint='$checkpoint'" \
        "train.output_dir='$eval_dir'" \
        "dataloader.evaluator.output_basedir='$eval_dir/val'" \
        "dataloader.test.dataset.manifest_path='$holdout_manifest'" \
        "dataloader.test.dataset.split='val'" \
        "model.head.cfg.test_parameters.conf_threshold=0.0" \
        "dataloader.evaluator.cfg.test_parameters.conf_threshold=0.0" \
        train.seed=42 train.cudnn_benchmark=False \
        > "$eval_dir/eval.log" 2>&1
    pred_root="$eval_dir/val/predictions"
    score_json="$eval_dir/val/prediction_scores.json"
    expected=$(wc -l < "$holdout_manifest")
    actual=$(find "$pred_root" -type f -name '*.lines.txt' | wc -l | tr -d ' ')
    [ "$expected" -eq "$actual" ] || {
        echo "prediction count mismatch $fold: $actual != $expected" >&2
        exit 6
    }
    grep -Fq '"score_schema_version": 1' "$score_json"
    grep -Fq '"score_semantics": "positive_class_softmax_probability"' "$score_json"
    grep -Fq '"post_nms": true' "$score_json"
    while IFS= read -r -d '' source_path; do
        rel="${source_path#"$pred_root"/}"
        destination="$OUTPUT_ROOT/predictions/$rel"
        mkdir -p "$(dirname "$destination")"
        [ ! -e "$destination" ] || { echo "duplicate destination: $destination" >&2; exit 7; }
        cp "$source_path" "$destination"
    done < <(find "$pred_root" -type f -name '*.lines.txt' -print0)
    printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ) $fold pass" >> "$LOG"
done

SPEC=resolution_960x480_cut0 OUTPUT_ROOT="$OUTPUT_ROOT" \
MANIFEST="$MANIFESTS/source_manifest_train.jsonl" "$PYTHON_BIN" - <<'PY'
import json, math, os
from pathlib import Path

root = Path(os.environ["OUTPUT_ROOT"])
manifest = Path(os.environ["MANIFEST"])
scores = {}
for path in sorted(root.glob("fold_*/val/prediction_scores.json")):
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "score_schema_version": 1,
        "score_semantics": "positive_class_softmax_probability",
        "post_nms": True,
        "candidate_export_conf_threshold": 0.0,
    }
    for key, value in required.items():
        if payload.get(key) != value:
            raise SystemExit(f"invalid score contract {key}: {path}")
    fold_scores = payload.get("scores_by_image")
    if not isinstance(fold_scores, dict) or scores.keys() & fold_scores.keys():
        raise SystemExit(f"invalid/duplicate score images: {path}")
    scores.update(fold_scores)
expected = {
    json.loads(line)["image_id"]
    for line in manifest.read_text(encoding="utf-8").splitlines()
    if line.strip()
}
if set(scores) != expected or len(scores) != 7100:
    raise SystemExit(f"score coverage mismatch: {len(scores)} != {len(expected)}")
for image_id, values in scores.items():
    if not isinstance(values, list) or any(
        value is None or not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0
        for value in values
    ):
        raise SystemExit(f"invalid scores: {image_id}")
(root / "prediction_scores.json").write_text(
    json.dumps({
        "status": "pass",
        "checkpoint": os.environ["SPEC"],
        "candidate_export_conf_threshold": 0.0,
        "score_schema_version": 1,
        "score_semantics": "positive_class_softmax_probability",
        "score_range": {"min": 0.0, "max": 1.0, "inclusive": True},
        "post_nms": True,
        "images": len(scores),
        "scores_by_image": scores,
    }, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
PY

printf '%s\n' complete > "$STATUS"
printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ) resolution eval screen complete" >> "$LOG"
