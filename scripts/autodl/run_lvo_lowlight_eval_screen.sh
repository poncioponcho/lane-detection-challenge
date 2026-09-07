#!/usr/bin/env bash
set -Eeuo pipefail

# Eval-only conditional low-light screen. It uses the completed 800x320,
# cut=180 checkpoints and applies the same deterministic trigger/curve in the
# dataset before train/eval transforms. A positive result is only a screen
# signal; a trained low-light variant would still require a separate LVO run.

PROJECT_ROOT="${HARDLANE_PROJECT_ROOT:-/tmp/lane-p0-project-58ca080}"
UNLANEDET_ROOT="${UNLANEDET_ROOT:-/tmp/lane-p0-unlanedet-58ca080-v2}"
DATA_ROOT="${HARDLANE_DATA_ROOT:-/hy-tmp/datasets/HardLane/Lane}"
WEIGHTS_ROOT="${HARDLANE_WEIGHTS_ROOT:-/hy-tmp/weights}"
D_EXP="${HARDLANE_D_EXP:-/hy-tmp/lane-outputs/lvo_clrnet_r50_36ep_20260905}"
OUTPUT_ROOT="${HARDLANE_LOWLIGHT_OUTPUT_ROOT:-/tmp/lane-lowlight-eval-screen-20260906}"
PYTHON_BIN="${HARDLANE_PYTHON:-/usr/local/miniconda3/envs/py39/bin/python}"
BUILDER="$PROJECT_ROOT/scripts/autodl/build_lowlight_config.py"
BASE_CONFIG="$PROJECT_ROOT/configs/unlanedet/clrnet_r50_hardlane.py"
TRAIN_NET="$UNLANEDET_ROOT/tools/train_net.py"
MANIFESTS="$D_EXP/manifests"
STATUS="$OUTPUT_ROOT/status"
LOG="$OUTPUT_ROOT/run.log"
LUMA_THRESHOLD="${HARDLANE_LOWLIGHT_LUMA_THRESHOLD:-42.0}"
GAMMA="${HARDLANE_LOWLIGHT_GAMMA:-0.85}"

[ ! -e "$OUTPUT_ROOT" ] || { echo "refusing to overwrite $OUTPUT_ROOT" >&2; exit 2; }
for path in "$PROJECT_ROOT" "$UNLANEDET_ROOT" "$D_EXP" "$MANIFESTS" \
            "$DATA_ROOT" "$WEIGHTS_ROOT" "$BUILDER" "$BASE_CONFIG" "$TRAIN_NET"; do
    [ -e "$path" ] || { echo "missing required path: $path" >&2; exit 3; }
done
mkdir -p "$OUTPUT_ROOT"
printf '%s\n' running > "$STATUS"

export HARDLANE_PROJECT_ROOT="$PROJECT_ROOT"
export HARDLANE_DATA_ROOT="$DATA_ROOT"
export HARDLANE_WEIGHTS_ROOT="$WEIGHTS_ROOT"
export HARDLANE_OUTPUT_ROOT="$OUTPUT_ROOT"
export PYTHONPATH="$UNLANEDET_ROOT:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

fail() {
    code=$?
    printf '%s\n' "failed:$code" > "$STATUS"
    printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ) failed:$code" >> "$LOG"
    exit "$code"
}
trap fail ERR

loaded_unlanedet_root=$("$PYTHON_BIN" -c 'import pathlib, unlanedet; print(pathlib.Path(unlanedet.__file__).resolve().parent.parent)')
[ "$loaded_unlanedet_root" = "$(cd "$UNLANEDET_ROOT" && pwd -P)" ] || {
    echo "wrong UnLanedet import root: $loaded_unlanedet_root" >&2
    exit 8
}

"$PYTHON_BIN" "$BUILDER" --base "$BASE_CONFIG" \
    --output "$OUTPUT_ROOT/clrnet_r50_hardlane_lowlight.py" \
    --luma-threshold "$LUMA_THRESHOLD" --gamma "$GAMMA" >> "$LOG"
CONFIG="$OUTPUT_ROOT/clrnet_r50_hardlane_lowlight.py"
grep -Fq '"enabled": True' "$CONFIG"
grep -Fq "\"luma_threshold\": $LUMA_THRESHOLD" "$CONFIG"
grep -Fq "\"gamma\": $GAMMA" "$CONFIG"

cat > "$OUTPUT_ROOT/protocol.json" <<EOF
{
  "status": "running",
  "protocol": "8-fold LVO eval-only conditional low-light screen",
  "model": "clrnet_r50",
  "checkpoint_experiment": "$D_EXP",
  "weights_root": "$WEIGHTS_ROOT",
  "input": "800x320",
  "cut_height": 180,
  "conditional_gamma": {
    "enabled": true,
    "luma_threshold": $LUMA_THRESHOLD,
    "gamma": $GAMMA,
    "roi": "raw BGR lower half, start=max(height//2, cut_height)"
  },
  "candidate_export_conf_threshold": 0.0,
  "score_semantics": "positive_class_softmax_probability",
  "post_nms": true
}
EOF

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

MANIFEST="$MANIFESTS/source_manifest_train.jsonl" OUTPUT_ROOT="$OUTPUT_ROOT" \
LUMA_THRESHOLD="$LUMA_THRESHOLD" GAMMA="$GAMMA" "$PYTHON_BIN" - <<'PY'
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
protocol_path = root / "protocol.json"
protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
protocol.update({
    "status": "pass",
    "images": len(scores),
    "prediction_count": len(list((root / "predictions").rglob("*.lines.txt"))),
    "luma_threshold": float(os.environ["LUMA_THRESHOLD"]),
    "gamma": float(os.environ["GAMMA"]),
})
protocol_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
(root / "prediction_scores.json").write_text(
    json.dumps({
        "status": "pass",
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
printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ) low-light eval screen complete" >> "$LOG"
