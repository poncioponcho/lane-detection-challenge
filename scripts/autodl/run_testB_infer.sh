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
# 2026-09-17: the platform DOES ship an explicit list -- 测试集B.zip contains
# testB/data/testB.txt (1000 rows, verified byte-identical to the testB.txt in
# the B-board sample package). Prefer it over deriving from the tree, because
# src/data/manifest.py states "the official list order is part of the evaluation
# contract" and the derived order is NOT the official one: the official clip
# order is 0,1093,120,220,285,385,893,943,1373,1473 while derive_list sorts
# lexicographically (0,120,285,385,893,943,1093,1373,1473,220). Same 1000 rows,
# different enumeration.
OFFICIAL=$LANE_ROOT/data/testB.txt
if [ -f "$OFFICIAL" ]; then
  echo "using official list: $OFFICIAL ($(wc -l < "$OFFICIAL") rows)"
  $PY scripts/build_testB_manifest.py \
    --lane-root "$LANE_ROOT" --official-list "$OFFICIAL" \
    --output data/processed/manifest_testB.jsonl || exit 2
  # Hand the exact official list to the local precheck: check_submission.py takes
  # the leading-slash form, which is how the official file is written already.
  cp "$OFFICIAL" data/processed/manifest_testB.list.txt
else
  echo "!! official list absent ($OFFICIAL) -- falling back to deriving from the tree"
  $PY scripts/build_testB_manifest.py \
    --lane-root "$LANE_ROOT" \
    --output data/processed/manifest_testB.jsonl || exit 2
fi
wc -l data/processed/manifest_testB.jsonl

echo "=== [2/4] run inference (base + supports), conf from the testA picks ==="
# ⚠️ This list is the set of run dirs that ACTUALLY exists on the instance
#    (2026-09-15 verified). seed43/seed44/clrernet_15ep do NOT exist here.
#    swa4 is derived from the base, so it is a candidate model, not a support vote.
#
# Why base54_c35 exists (see action_testB_20260916.md section 7):
# On the 36ep LVO OOF, conf 0.40 beat 0.50 by +0.417pp, so this tree was added
# to keep that option open. It was then cross-checked on an INDEPENDENT model
# (15ep LVO clsweight3, same OOF protocol and GT) and the effect vanished:
# +0.001pp. So the +0.417pp is model-specific noise, not a general law -- do
# NOT expect it on the 54ep incumbent. The pass is kept only because it costs
# ~1 min of GPU and BASE_TAG=base54_c35 can switch the floor without a re-run;
# there is no measured reason to prefer it. Default stays conf 0.50.
#
# 2026-09-17: these explanations used to live INSIDE the heredoc below. A `#`
# line inside a heredoc is DATA, not a comment: `while read` consumed each one
# as a model row, producing "conf Why" / "conf On" / "conf there" usage errors.
# Kept out of the heredoc now, and the loop also skips a leading `#` defensively.
FAILED_TAGS=""
TAGS_SEEN=""
while read -r run conf tag; do
  [ -z "${tag:-}" ] && continue
  case "$tag" in \#*) continue ;; esac
  echo "--- $tag (conf $conf) ---"
  if $PY scripts/autodl/infer_testA.py \
    --run-dir "$R/$run" --split testB --conf-threshold "$conf" \
    --output-dir "$HARDLANE_OUTPUT_ROOT/testB_$tag" --skip-if-complete; then
    TAGS_SEEN="$TAGS_SEEN $tag"
  else
    echo "!! inference FAILED for $tag"
    FAILED_TAGS="$FAILED_TAGS $tag"
  fi
done <<'LIST'
all71_seed42_clrnet_r50_54ep            0.50 base54
all71_seed42_clrnet_r50_54ep            0.55 base54_c55
all71_seed42_clrnet_r50_54ep            0.35 base54_c35
all71_seed42_clrnet_r50_36ep            0.50 seed42
all71_seed101_clrnet_r50_36ep           0.50 seed101
all71_seed202_clrnet_r50_36ep           0.50 seed202
all71_seed303_clrnet_r50_36ep           0.50 seed303
all71_seed42_clrernet_r50_36ep          0.50 clrernet36
all71_seed42_clrnet_r50_cut400_36ep     0.35 cut400
all71_seed42_clrnet_r50_hires_36ep      0.40 hires
swa4_54ep_stage                         0.50 swa4
all71_seed42_clrnet_r50_occlude_36ep    0.50 occlude
all71_seed101_clrnet_r50_54ep           0.50 s101_54ep
LIST

# 2026-09-17: the loop above deliberately does not abort on the first failure
# (one bad tree must not cost the whole window), but the SCRIPT MUST NOT exit 0
# afterwards -- on 9/17 every single tree was blocked by the dirty-worktree
# guard, the script still returned rc=0, and the watcher treated a 20 KB bundle
# of empty prediction dirs as a success. Count the products and fail loudly.
echo "=== [2b/4] verify every tree actually produced 1000 files ==="
NFILE_ERR=0
for t in $TAGS_SEEN; do
  p=$HARDLANE_OUTPUT_ROOT/testB_$t/testB/predictions
  n=$(find "$p" -name '*.lines.txt' 2>/dev/null | wc -l | tr -d ' ')
  l=$(find "$p" -name '*.lines.txt' -exec cat {} + 2>/dev/null | grep -c . || true)
  if [ "$n" -ne 1000 ]; then
    echo "!! $t produced $n/1000 prediction files (lanes=$l)"
    NFILE_ERR=1
  else
    echo "ok $t files=$n lanes=$l"
  fi
done
if [ -n "$FAILED_TAGS" ] || [ "$NFILE_ERR" -ne 0 ]; then
  echo "!! ABORTING: failures=[$FAILED_TAGS] incomplete=$NFILE_ERR"
  echo "!! Do NOT treat any bundle from this run as usable."
  exit 3
fi

echo "=== [3/4] line counts per tree ==="
for t in base54 base54_c55 seed42 seed101 seed202 seed303 clrernet36 cut400 hires swa4 occlude s101_54ep; do
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
