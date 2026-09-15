#!/usr/bin/env bash
# B-board step 1 (runs on the GPU instance): get testB images in place, build the
# manifest, run every base/support inference, and hand the trees back as one
# tarball. Post-processing and packing stay local because the frozen Oracle,
# prepare_submit.py and check_submission.py live there.
#
#   bash scripts/autodl/run_testB_infer.sh <lane_root>
#
# Prerequisites (see runbook_testB.md section 2):
#   * testB images already extracted under <lane_root>/JPEGImages/<clip>/<frame>.jpg
#     -- they are NOT on the instance by default; the user downloads them from
#     the competition platform and scps them up.
#   * instance HEAD == local HEAD (sync with a git bundle, not a pull).
#   * run_evidence.json present for every run dir listed below.
set -u
export HARDLANE_PROJECT_ROOT=/hy-tmp/lane-detection-challenge
export HARDLANE_DATA_ROOT=/hy-tmp/datasets/HardLane/Lane
export UNLANEDET_ROOT=/hy-tmp/UnLanedet
export HARDLANE_WEIGHTS_ROOT=/hy-tmp/weights
export HARDLANE_OUTPUT_ROOT=/hy-tmp/lane-outputs
export HARDLANE_PYTHON=/usr/local/miniconda3/envs/py39/bin/python
PY=$HARDLANE_PYTHON
R=$HARDLANE_OUTPUT_ROOT/runs
cd "$HARDLANE_PROJECT_ROOT" || exit 1

LANE_ROOT=${1:-$HARDLANE_DATA_ROOT}

echo "=== [0/4] sanity: testB images present? ==="
TOTAL=$(ls "$LANE_ROOT/JPEGImages" 2>/dev/null | grep -v _hflip | wc -l)
echo "clips (excluding _hflip) = $TOTAL   (train-only is 80; testB present adds 10)"
if [ "$TOTAL" -le 80 ]; then
  echo "!! testB clips not found under $LANE_ROOT/JPEGImages -- fetch them first."
  echo "!! Continuing anyway so the manifest step reports its own error."
fi

echo "=== [1/4] build manifest ==="
$PY scripts/build_testB_manifest.py \
  --lane-root "$LANE_ROOT" \
  --output data/processed/manifest_testB.jsonl || exit 2
wc -l data/processed/manifest_testB.jsonl

echo "=== [2/4] run inference (base + supports), conf from the testA picks ==="
# ⚠️ This list is the set of run dirs that ACTUALLY exists on the instance
#    (2026-09-15 verified). seed43/seed44/clrernet_15ep do NOT exist here.
#    swa4 is derived from the base, so it is a candidate model, not a support vote.
while read -r run conf tag; do
  echo "--- $tag (conf $conf) ---"
  $PY scripts/autodl/infer_testA.py \
    --run-dir "$R/$run" --split testB --conf-threshold "$conf" \
    --output-dir "$HARDLANE_OUTPUT_ROOT/testB_$tag" --skip-if-complete
done <<'LIST'
all71_seed42_clrnet_r50_54ep            0.50 base54
all71_seed42_clrnet_r50_54ep            0.55 base54_c55
all71_seed42_clrnet_r50_36ep            0.50 seed42
all71_seed101_clrnet_r50_36ep           0.50 seed101
all71_seed202_clrnet_r50_36ep           0.50 seed202
all71_seed303_clrnet_r50_36ep           0.50 seed303
all71_seed42_clrernet_r50_36ep          0.50 clrernet36
all71_seed42_clrnet_r50_cut400_36ep     0.35 cut400
all71_seed42_clrnet_r50_hires_36ep      0.40 hires
swa4_54ep_stage                         0.50 swa4
all71_seed42_clrnet_r50_occlude_36ep    0.50 occlude
LIST

echo "=== [3/4] line counts per tree ==="
for t in base54 base54_c55 seed42 seed101 seed202 seed303 clrernet36 cut400 hires swa4 occlude; do
  p=$HARDLANE_OUTPUT_ROOT/testB_$t/testB/predictions
  n=$(find $p -name '*.lines.txt' 2>/dev/null | wc -l)
  l=$(find $p -name '*.lines.txt' -exec cat {} + 2>/dev/null | grep -c .)
  echo "$t files=$n lanes=$l"
done

echo "=== [4/4] f<450 decision statistic (margin rule, runbook section 4) ==="
$PY - "$HARDLANE_OUTPUT_ROOT/testB_base54/testB/predictions" <<'PY'
import sys, pathlib
root = pathlib.Path(sys.argv[1])
tops = []
for f in root.rglob("*.lines.txt"):
    for line in f.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        v = [float(t) for t in line.split()]
        tops.append(min(v[1::2]))
if not tops:
    print("NO LANES -- check the inference step")
    raise SystemExit(1)
far = sum(1 for t in tops if t < 450) / len(tops)
import statistics
print(f"lanes={len(tops)}  f<450={far:.4f}  top_p50={statistics.median(tops):.1f}")
if far > 0.10:
    print("=> train-shaped geometry: use TRIM MARGIN 40")
else:
    print("=> testA-shaped geometry: use TRIM MARGIN 0   (40 would be a no-op)")
PY

echo "=== [5/5] stage + pack for transfer ==="
# The LOCAL packaging script (build_testB_candidates.sh) also reads
# manifest_testB.jsonl, so it must travel with the trees -- otherwise it would
# silently fall back to a stale/absent manifest. The list file goes too so the
# official precheck can run locally without rebuilding it.
STAGE=/hy-tmp/testB_stage
rm -rf "$STAGE"; mkdir -p "$STAGE"
cp -r $HARDLANE_OUTPUT_ROOT/testB_* "$STAGE"/ 2>/dev/null
mkdir -p "$STAGE/data_processed"
cp data/processed/manifest_testB.jsonl "$STAGE/data_processed/"
[ -f data/processed/manifest_testB.list.txt ] && \
  cp data/processed/manifest_testB.list.txt "$STAGE/data_processed/"
cd "$STAGE" || exit 1
tar czf /hy-tmp/testB_bundle.tgz .
ls -la /hy-tmp/testB_bundle.tgz
echo
echo "scp -i ~/.ssh/lane_id -P <PORT> root@<HOST>:/hy-tmp/testB_bundle.tgz <local>/"
echo "then: tar xzf testB_bundle.tgz -C outputs/testB_$(date +%Y%m%d)   # trees"
echo "      cp outputs/testB_$(date +%Y%m%d)/data_processed/manifest_testB* data/processed/"
