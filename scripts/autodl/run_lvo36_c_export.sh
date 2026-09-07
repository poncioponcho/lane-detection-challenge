#!/usr/bin/env bash
set -Eeuo pipefail

# C-stage eval-only export. It runs only after D is complete and does not
# retrain or modify any checkpoint/manifest/config. The patched evaluator
# exports post-NMS candidates and positive-class softmax probability scores at
# conf=0.0.  The probability unit is required by the offline scan contract.

PROJECT_ROOT=/hy-tmp/lane-detection-challenge
UNLANEDET_ROOT=/hy-tmp/UnLanedet
DATA_ROOT=/hy-tmp/datasets/HardLane/Lane
WEIGHTS_ROOT=/hy-tmp/weights
OUTPUT_ROOT=/hy-tmp/lane-outputs
D_EXP="$OUTPUT_ROOT/lvo_clrnet_r50_36ep_20260905"
# 20260905 C artifacts used the legacy raw-logit sidecar.  Use a new output
# namespace so a corrected probability export can never be mixed with them.
C_EXP="$OUTPUT_ROOT/lvo_clrnet_r50_36ep_c_export_20260906"
MANIFESTS="$D_EXP/manifests"
CONFIG="$PROJECT_ROOT/configs/unlanedet/clrnet_r50_hardlane.py"
TRAIN_NET="$UNLANEDET_ROOT/tools/train_net.py"
PYTHON_BIN=/usr/local/miniconda3/envs/py39/bin/python
STATUS="$OUTPUT_ROOT/lvo36_c_export_20260906.status"
LOG="$OUTPUT_ROOT/lvo36_c_export_20260906.log"

export HARDLANE_PROJECT_ROOT="$PROJECT_ROOT"
export HARDLANE_DATA_ROOT="$DATA_ROOT"
export HARDLANE_WEIGHTS_ROOT="$WEIGHTS_ROOT"
export HARDLANE_OUTPUT_ROOT="$OUTPUT_ROOT"
export HARDLANE_PYTHON="$PYTHON_BIN"

fail() {
    code=$?
    printf '%s\n' "failed:$code" > "$STATUS"
    printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ) failed:$code" >> "$LOG"
    exit "$code"
}
trap fail ERR

printf '%s\n' running > "$STATUS"
printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ) C raw export start" >> "$LOG"
[ "$(cat "$OUTPUT_ROOT/lvo36.status")" = complete ] || {
    echo "D is not complete" >&2
    exit 2
}
[ -f "$D_EXP/oof/oof_evidence.json" ] || {
    echo "D OOF evidence missing" >&2
    exit 3
}
[ ! -e "$C_EXP" ] || {
    echo "refusing to overwrite $C_EXP" >&2
    exit 4
}
for path in "$PROJECT_ROOT" "$UNLANEDET_ROOT" "$DATA_ROOT" "$WEIGHTS_ROOT" \
            "$D_EXP" "$MANIFESTS" "$CONFIG" "$TRAIN_NET"; do
    [ -e "$path" ] || { echo "missing required path: $path" >&2; exit 5; }
done
mkdir -p "$C_EXP"

for label in midpoint final; do
    mkdir -p "$C_EXP/$label/predictions"
    for fold_dir in $(find "$MANIFESTS" -mindepth 1 -maxdepth 1 -type d -name 'fold_*_v*' | sort); do
        fold=$(basename "$fold_dir")
        train_rows=$(wc -l < "$fold_dir/manifest_train.jsonl")
        iter_per_epoch=$((train_rows / 12))
        if [ "$label" = midpoint ]; then
            checkpoint_period=$((iter_per_epoch * 18))
            checkpoint_name=$(printf 'model_%07d.pth' "$checkpoint_period")
        else
            checkpoint_name=model_final.pth
        fi
        checkpoint="$D_EXP/runs/$fold/$checkpoint_name"
        # Detectron2's periodic checkpointer names the checkpoint for the
        # completed iteration with a zero-based file stem (e.g. period=9450
        # produces model_0009449.pth).  Accept that canonical file when the
        # one-based arithmetic name is absent; final remains model_final.pth.
        if [ "$label" = midpoint ] && [ ! -f "$checkpoint" ]; then
            zero_based_period=$((checkpoint_period - 1))
            zero_based_checkpoint="$D_EXP/runs/$fold/$(printf 'model_%07d.pth' "$zero_based_period")"
            if [ -f "$zero_based_checkpoint" ]; then
                checkpoint="$zero_based_checkpoint"
            fi
        fi
        [ -f "$checkpoint" ] || {
            echo "missing $label checkpoint: $checkpoint" >&2
            exit 6
        }
        holdout_manifest="$fold_dir/manifest_holdout.jsonl"
        eval_dir="$C_EXP/$label/$fold"
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
        [ -d "$pred_root" ] && [ -f "$score_json" ] || {
            echo "raw export evidence missing: $eval_dir" >&2
            exit 7
        }
        expected=$(wc -l < "$holdout_manifest")
        actual=$(find "$pred_root" -type f -name '*.lines.txt' | wc -l | tr -d ' ')
        [ "$expected" -eq "$actual" ] || {
            echo "prediction count mismatch $label/$fold: $actual != $expected" >&2
            exit 8
        }
        while IFS= read -r -d '' source_path; do
            rel=$(printf '%s' "$source_path" | sed "s#^$pred_root/##")
            destination="$C_EXP/$label/predictions/$rel"
            mkdir -p "$(dirname "$destination")"
            [ ! -e "$destination" ] || {
                echo "duplicate destination: $destination" >&2
                exit 9
            }
            cp "$source_path" "$destination"
        done < <(find "$pred_root" -type f -name '*.lines.txt' -print0)
        printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ) $label/$fold pass" >> "$LOG"
    done
done

for label in midpoint final; do
    LABEL="$label" C_EXP="$C_EXP" MANIFEST="$MANIFESTS/source_manifest_train.jsonl" \
      "$PYTHON_BIN" - <<'PY'
import json, math, os
from pathlib import Path

label = os.environ["LABEL"]
root = Path(os.environ["C_EXP"])
manifest = Path(os.environ["MANIFEST"])
score_by_image = {}
score_semantics = "positive_class_softmax_probability"
score_schema_version = 1
for path in sorted((root / label).glob("fold_*/val/prediction_scores.json")):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("score_schema_version") != score_schema_version:
        raise SystemExit(f"unsupported score schema: {path}")
    if payload.get("score_semantics") != score_semantics:
        raise SystemExit(f"score semantics mismatch: {path}")
    if payload.get("score_range") != {"min": 0.0, "max": 1.0, "inclusive": True}:
        raise SystemExit(f"score range mismatch: {path}")
    if payload.get("post_nms") is not True:
        raise SystemExit(f"score sidecar is not post-NMS: {path}")
    if payload.get("candidate_export_conf_threshold") != 0.0:
        raise SystemExit(
            f"candidate export threshold is not 0.0: {path}: "
            f"{payload.get('candidate_export_conf_threshold')!r}"
        )
    fold_scores = payload.get("scores_by_image", {})
    if not isinstance(fold_scores, dict):
        raise SystemExit(f"score sidecar is not an object: {path}")
    overlap = set(score_by_image).intersection(fold_scores)
    if overlap:
        raise SystemExit(f"duplicate scores: {sorted(overlap)[:3]}")
    score_by_image.update(fold_scores)
records = [
    json.loads(line)
    for line in manifest.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
expected = {row["image_id"] for row in records}
if set(score_by_image) != expected:
    raise SystemExit(
        f"{label}: score coverage {len(score_by_image)} != {len(expected)}"
    )
for image_id, scores in score_by_image.items():
    if not isinstance(scores, list):
        raise SystemExit(f"{label}: non-list score entry for {image_id}")
    for score in scores:
        if score is None or not math.isfinite(float(score)) or not 0.0 <= float(score) <= 1.0:
            raise SystemExit(f"{label}: non-probability score for {image_id}: {score!r}")
(root / label / "prediction_scores.json").write_text(
    json.dumps(
        {
            "status": "pass",
            "checkpoint": label,
            "candidate_export_conf_threshold": 0.0,
            "score_schema_version": score_schema_version,
            "score_semantics": score_semantics,
            "score_range": {"min": 0.0, "max": 1.0, "inclusive": True},
            "post_nms": True,
            "images": len(score_by_image),
            "scores_by_image": score_by_image,
        },
        ensure_ascii=False,
        indent=2,
    ) + "\n",
    encoding="utf-8",
)
PY
done

printf '%s\n' complete > "$STATUS"
printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ) C raw export complete" >> "$LOG"
