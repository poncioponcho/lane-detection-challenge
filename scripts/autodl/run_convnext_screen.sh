#!/usr/bin/env bash
set -Eeuo pipefail

# One controlled ConvNeXt-Tiny 15-epoch screen. It uses the same HardLane
# split, preprocessing, head, seed, and probability sidecar contract as the
# frozen CLRNet-R50 baseline. The result is diagnostic only: it must pass the
# video-disjoint/Oracle gates before any production decision.

PROJECT_ROOT="${HARDLANE_PROJECT_ROOT:-/hy-tmp/lane-detection-challenge}"
UNLANEDET_ROOT="${UNLANEDET_ROOT:-/hy-tmp/UnLanedet}"
DATA_ROOT="${HARDLANE_DATA_ROOT:-/hy-tmp/datasets/HardLane/Lane}"
WEIGHTS_ROOT="${HARDLANE_WEIGHTS_ROOT:-/hy-tmp/weights}"
OUTPUT_ROOT="${HARDLANE_OUTPUT_ROOT:-/hy-tmp/lane-outputs}"
PYTHON_BIN="${HARDLANE_PYTHON:-/usr/local/miniconda3/envs/py39/bin/python}"
CONFIG="$PROJECT_ROOT/configs/unlanedet/clrnet_convnext_tiny_hardlane.py"
TRAIN_NET="$UNLANEDET_ROOT/tools/train_net.py"
CONVERTER="$PROJECT_ROOT/scripts/autodl/convert_convnext_weight.py"
RAW_WEIGHT="$WEIGHTS_ROOT/clrnet_convnext_culane.pth"
INIT_WEIGHT="$WEIGHTS_ROOT/adapted_clrnet_convnext_tiny_hardlane.pth"
RUN_DIR="$OUTPUT_ROOT/runs/convnext_tiny_15ep"
EVAL_DIR="$RUN_DIR/selected_best_eval"
STATUS="$OUTPUT_ROOT/convnext_tiny_screen_20260906.status"
LOG="$OUTPUT_ROOT/convnext_tiny_screen_20260906.log"

on_error() {
    code=$?
    printf '%s\n' "failed:$code" > "$STATUS"
    printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ) failed:$code" >> "$LOG"
    exit "$code"
}
trap on_error ERR

for path in "$PROJECT_ROOT" "$UNLANEDET_ROOT" "$DATA_ROOT" "$WEIGHTS_ROOT" "$OUTPUT_ROOT"; do
    [ -e "$path" ] || { echo "missing required path: $path" >&2; exit 2; }
done
[ -f "$CONFIG" ] || { echo "missing config: $CONFIG" >&2; exit 2; }
[ -f "$TRAIN_NET" ] || { echo "missing train_net: $TRAIN_NET" >&2; exit 2; }
[ -f "$CONVERTER" ] || { echo "missing converter: $CONVERTER" >&2; exit 2; }
[ "$(cat "$OUTPUT_ROOT/pipeline.status")" = complete ] || {
    echo "pipeline.status is not complete" >&2
    exit 3
}
[ -z "$(git -C "$PROJECT_ROOT" status --short --untracked-files=no)" ] || {
    echo "tracked project worktree is dirty" >&2
    exit 4
}
[ ! -e "$RUN_DIR" ] || {
    echo "refusing to overwrite existing run: $RUN_DIR" >&2
    exit 5
}

mkdir -p "$OUTPUT_ROOT"
printf '%s\n' running > "$STATUS"
printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ) ConvNeXt-Tiny screen start" > "$LOG"

export HARDLANE_PROJECT_ROOT="$PROJECT_ROOT"
export HARDLANE_DATA_ROOT="$DATA_ROOT"
export UNLANEDET_ROOT="$UNLANEDET_ROOT"
export HARDLANE_WEIGHTS_ROOT="$WEIGHTS_ROOT"
export HARDLANE_OUTPUT_ROOT="$OUTPUT_ROOT"
export HARDLANE_PYTHON="$PYTHON_BIN"

bash "$PROJECT_ROOT/scripts/autodl/setup_unlanedet.sh"

if [ ! -f "$RAW_WEIGHT" ]; then
    curl -fL --retry 3 --connect-timeout 20 \
        -o "$RAW_WEIGHT" \
        "https://github.com/zkyntu/UnLanedet/releases/download/Weights/clrnet_convnext_culane.pth"
fi
[ -s "$RAW_WEIGHT" ] || { echo "ConvNeXt source weight is empty" >&2; exit 6; }

if [ ! -f "$INIT_WEIGHT" ]; then
    "$PYTHON_BIN" "$CONVERTER" "$RAW_WEIGHT" "$INIT_WEIGHT"
fi
[ -s "$INIT_WEIGHT" ] || { echo "converted ConvNeXt weight is empty" >&2; exit 7; }

raw_sha="$(sha256sum "$RAW_WEIGHT" | awk '{print $1}')"
init_sha="$(sha256sum "$INIT_WEIGHT" | awk '{print $1}')"
printf '%s\n' "weight raw=$raw_sha adapted=$init_sha" >> "$LOG"

mkdir -p "$RUN_DIR"
cd "$UNLANEDET_ROOT"
"$PYTHON_BIN" "$TRAIN_NET" --config-file "$CONFIG" --num-gpus 1 \
    "train.init_checkpoint='$INIT_WEIGHT'" \
    "train.output_dir='$RUN_DIR'" \
    "dataloader.evaluator.output_basedir='$RUN_DIR/val'" \
    "train.max_iter=7875" \
    "train.eval_period=525" \
    "train.checkpointer.period=525" \
    "train.checkpointer.max_to_keep=3" \
    "train.seed=42" \
    "train.cudnn_benchmark=False" \
    > "$RUN_DIR/train.log" 2>&1

[ -f "$RUN_DIR/model_final.pth" ] || {
    echo "training completed without model_final.pth" >&2
    exit 8
}

mkdir -p "$EVAL_DIR"
"$PYTHON_BIN" "$TRAIN_NET" --eval-only --config-file "$CONFIG" --num-gpus 1 \
    "train.init_checkpoint='$RUN_DIR/model_final.pth'" \
    "train.output_dir='$EVAL_DIR'" \
    "dataloader.evaluator.output_basedir='$EVAL_DIR/val'" \
    "dataloader.test.dataset.manifest_path='$PROJECT_ROOT/data/processed/manifest_val_v1_seed42.jsonl'" \
    "dataloader.test.dataset.split='val'" \
    "model.head.cfg.test_parameters.conf_threshold=0.4" \
    "dataloader.evaluator.cfg.test_parameters.conf_threshold=0.4" \
    "train.seed=42" \
    "train.cudnn_benchmark=False" \
    > "$EVAL_DIR/eval.log" 2>&1

PRED_ROOT="$EVAL_DIR/val/predictions"
DIAGNOSTIC="$EVAL_DIR/val/diagnostic_metric.json"
SCORES="$EVAL_DIR/val/prediction_scores.json"
[ -d "$PRED_ROOT" ] && [ -f "$DIAGNOSTIC" ] && [ -f "$SCORES" ] || {
    echo "missing ConvNeXt evaluation evidence" >&2
    exit 9
}
prediction_count="$(find "$PRED_ROOT" -type f -name '*.lines.txt' | wc -l | tr -d ' ')"
[ "$prediction_count" -eq 800 ] || {
    echo "prediction count mismatch: $prediction_count != 800" >&2
    exit 10
}

RUN_DIR="$RUN_DIR" EVAL_DIR="$EVAL_DIR" CONFIG="$CONFIG" RAW_WEIGHT="$RAW_WEIGHT" \
INIT_WEIGHT="$INIT_WEIGHT" PROJECT_ROOT="$PROJECT_ROOT" PROJECT_HEAD="$(git -C "$PROJECT_ROOT" rev-parse HEAD)" \
PYTHON_BIN="$PYTHON_BIN" "$PYTHON_BIN" - <<'PY'
import hashlib
import json
import os
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


run_dir = Path(os.environ["RUN_DIR"])
eval_dir = Path(os.environ["EVAL_DIR"])
diagnostic = json.loads((eval_dir / "val/diagnostic_metric.json").read_text())
scores = json.loads((eval_dir / "val/prediction_scores.json").read_text())
metrics = []
metrics_path = run_dir / "metrics.json"
if metrics_path.is_file():
    for raw in metrics_path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if "F1" in value:
            metrics.append(value)
best = max(metrics, key=lambda value: float(value["F1"])) if metrics else None
final = metrics[-1] if metrics else None
evidence = {
    "status": "pass",
    "experiment": "convnext_tiny_15ep_screen",
    "model": "clrnet_convnext_tiny",
    "epochs": 15,
    "expected_max_iter": 7875,
    "completed_iteration": 7874,
    "project_git_head": os.environ["PROJECT_HEAD"],
    "config": os.environ["CONFIG"],
    "config_sha256": sha256(Path(os.environ["CONFIG"])),
    "raw_weight": os.environ["RAW_WEIGHT"],
    "raw_weight_sha256": sha256(Path(os.environ["RAW_WEIGHT"])),
    "adapted_weight": os.environ["INIT_WEIGHT"],
    "adapted_weight_sha256": sha256(Path(os.environ["INIT_WEIGHT"])),
    "final_checkpoint": str(run_dir / "model_final.pth"),
    "final_checkpoint_sha256": sha256(run_dir / "model_final.pth"),
    "diagnostic": diagnostic,
    "f1_history": {
        "observations": len(metrics),
        "best": best,
        "final": final,
    },
    "prediction_images": len(scores.get("scores_by_image", {})),
    "score_schema_version": scores.get("score_schema_version"),
    "score_semantics": scores.get("score_semantics"),
    "post_nms": scores.get("post_nms"),
    "candidate_export_conf_threshold": scores.get("candidate_export_conf_threshold"),
}
if evidence["prediction_images"] != 800:
    raise SystemExit("score sidecar does not cover 800 validation images")
if evidence["score_schema_version"] != 1:
    raise SystemExit("unexpected score schema")
if evidence["score_semantics"] != "positive_class_softmax_probability":
    raise SystemExit("unexpected score semantics")
if evidence["post_nms"] is not True:
    raise SystemExit("score sidecar is not post-NMS")
if evidence["candidate_export_conf_threshold"] != 0.4:
    raise SystemExit("screen eval did not use explicit conf=0.4")
(run_dir / "convnext_screen_evidence.json").write_text(
    json.dumps(evidence, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
PY

printf '%s\n' complete > "$STATUS"
printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ) ConvNeXt-Tiny screen complete" >> "$LOG"
