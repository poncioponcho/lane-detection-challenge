#!/usr/bin/env bash
# 夜间 soup 试验驱动 v2：6 候选在 val 800 图上的预测（infer_testA --split testA --manifest val）
# 产物: <run_or_stage>/val_pred_stage/testA/predictions → 拉回本地冻结 Oracle 评分
set -uo pipefail
export HARDLANE_PROJECT_ROOT=/hy-tmp/lane-detection-challenge
export HARDLANE_DATA_ROOT=/hy-tmp/datasets/HardLane/Lane
export UNLANEDET_ROOT=/hy-tmp/UnLanedet
export HARDLANE_WEIGHTS_ROOT=/hy-tmp/weights
export HARDLANE_OUTPUT_ROOT=/hy-tmp/lane-outputs
export HARDLANE_PYTHON=/usr/local/miniconda3/envs/py39/bin/python
export PYTHON_BIN=/usr/local/miniconda3/envs/py39/bin/python
P=$PYTHON_BIN
R=/hy-tmp/lane-outputs/runs
VAL_MANIFEST="$HARDLANE_PROJECT_ROOT/data/processed/manifest_val_v1_seed42.jsonl"

infer_val() {  # infer_val <run_dir_or_stage>
    local d=$1
    cd "$HARDLANE_PROJECT_ROOT"
    $P scripts/autodl/infer_testA.py --run-dir "$d" --split testA \
        --manifest "$VAL_MANIFEST" \
        --output-dir "$d/val_pred_stage" \
        --skip-if-complete \
        > "/hy-tmp/lane-outputs/infer_val2_$(basename "$d").log" 2>&1
    echo "$(basename "$d") exit=$? preds=$(find "$d/val_pred_stage/testA/predictions" -name '*.lines.txt' 2>/dev/null | wc -l)"
}

for s in 42 101 202 303; do
    infer_val "$R/all71_seed${s}_clrnet_r50_36ep"
done
infer_val "$R/soup4_uniform_stage"
infer_val "$R/soup5_with_incumbent_stage"

echo "ALL_VAL_INFERS_DONE"
cd /hy-tmp/lane-outputs && tar -czf /tmp/night_val_preds.tar.gz \
    runs/*/val_pred_stage/testA/predictions 2>/dev/null
md5sum /tmp/night_val_preds.tar.gz
