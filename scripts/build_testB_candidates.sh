#!/usr/bin/env bash
# B-board step 2 (runs locally): turn the testB prediction trees into verified
# submission packages, in shot order, using the frozen Oracle environment.
#
#   bash scripts/build_testB_candidates.sh <path-to-testB_trees.tgz>
#
# It prints one line per package (lanes + precheck result) and finishes with the
# absolute path and a <=50-char note for each shot. Nothing is uploaded: handing
# a package to the platform is a manual step that needs the user's sign-off.
set -u
cd "/Users/seyonmacbook/WorkBuddy/恶劣场景下的车道线检测挑战赛" || exit 1
PY=/Users/seyonmacbook/.workbuddy/binaries/python/envs/lane-oracle-py312/bin/python
TREES=$1
DAY=$(date +%Y%m%d)
OUT=outputs/testB_${DAY}
BUILD=$OUT/build
MANIFEST=data/processed/manifest_testB.jsonl

[ -f "$TREES" ] || { echo "usage: $0 <testB_trees.tgz>"; exit 1; }
[ -f "$MANIFEST" ] || { echo "missing $MANIFEST -- generate it on the instance"; exit 1; }

mkdir -p "$BUILD"
tar xzf "$TREES" -C "$OUT"
ls -d "$OUT"/testB_*

B=$OUT/testB_base54/testB/predictions
[ -d "$B" ] || { echo "base tree missing at $B"; exit 1; }

echo "=== official pre-check list ==="
$PY - "$MANIFEST" /tmp/testB_list_$DAY.txt <<'PY'
import json, sys
recs = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
open(sys.argv[2], "w").write("".join("/" + r["image_path"] + "\n" for r in recs))
print("list rows:", len(recs))
PY

echo "=== margin decision (f<450, runbook section 4) ==="
M=$($PY - "$B" <<'PY'
import sys, pathlib, statistics
root = pathlib.Path(sys.argv[1])
tops = [min(float(t) for t in [float(x) for x in line.split()][1::2])
        for f in root.rglob("*.lines.txt") for line in f.read_text().splitlines() if line.strip()]
far = sum(1 for t in tops if t < 450) / max(1, len(tops))
print(40 if far > 0.10 else 0)
PY
)
echo "chosen trim margin M = $M"

pack () {  # pack <name> <srcdir>
  n=$1; s=$2
  $PY scripts/apply_bottom_trim_testA.py --src "$s" --dst "$BUILD/${n}_trim" --margin "$M" >/dev/null
  $PY src/submit/prepare_submit.py --raw-pred-dir "$BUILD/${n}_trim" \
    --canonical-dir "$BUILD/${n}_canon" \
    --out-zip "outputs/submit_testB_${n}.zip" \
    --manifest "$MANIFEST" \
    --report "outputs/reports/prepare_submit_testB_${n}.json" >/dev/null
  $PY src/eval/official_oracle/check_submission.py \
    --zip_path "outputs/submit_testB_${n}.zip" \
    --list_path /tmp/testB_list_$DAY.txt >/dev/null 2>&1 \
    && echo "  $n: OK" || echo "  $n: PRECHECK FAILED"
}

echo "=== shot 1: base54 + trim($M) ==="
pack shot1_base54 "$B"

echo "=== shot 3/2 inputs: consensus at >=60% agreement ==="
cat > /tmp/testB_supports_$DAY.json <<JSON
[
 {"name": "seed42_36ep",  "path": "$(pwd)/$OUT/testB_seed42/testB/predictions"},
 {"name": "seed101_36ep", "path": "$(pwd)/$OUT/testB_seed101/testB/predictions"},
 {"name": "seed202_36ep", "path": "$(pwd)/$OUT/testB_seed202/testB/predictions"},
 {"name": "seed303_36ep", "path": "$(pwd)/$OUT/testB_seed303/testB/predictions"},
 {"name": "clrernet_36ep","path": "$(pwd)/$OUT/testB_clrernet36/testB/predictions"},
 {"name": "cut400_36ep",  "path": "$(pwd)/$OUT/testB_cut400/testB/predictions"},
 {"name": "hires_36ep",   "path": "$(pwd)/$OUT/testB_hires/testB/predictions"}
]
JSON
# 7 independent supports; the gate is an AGREEMENT FRACTION, not a vote count.
# 60% of 7 -> --min-support 4 (57%) is the practical choice; 5 (71%) is stricter.
$PY scripts/build_testA_consensus_union_20260913.py \
  --base "$B" --dst "$BUILD/cons" --min-support 4 --average \
  --supports-json /tmp/testB_supports_$DAY.json | tail -4
$PY scripts/filter_short_lanes.py --src "$BUILD/cons" --base "$B" \
  --dst "$BUILD/cons_f80" --min-span 80
pack shot3_consensus "$BUILD/cons_f80"

echo "=== shot 2: union(swa4, consensus) -- strictly dominates swa4 ==="
$PY scripts/apply_bottom_trim_testA.py \
  --src "$OUT/testB_swa4/testB/predictions" --dst "$BUILD/swa4_trim" --margin "$M" >/dev/null
$PY scripts/build_union_pair_20260915.py \
  --a "$BUILD/swa4_trim" --b "$BUILD/shot3_consensus_trim" --dst "$BUILD/uni"
pack shot2_uni_swa4_cons "$BUILD/uni"

echo "=== shot 4: conf0.55 (testA-shaped) or margin hedge (train-shaped) ==="
if [ "$M" = "0" ]; then
  pack shot4_conf55 "$OUT/testB_base54_c55/testB/predictions"
else
  $PY scripts/apply_bottom_trim_testA.py --src "$B" --dst "$BUILD/base_m0" --margin 0 >/dev/null
  $PY src/submit/prepare_submit.py --raw-pred-dir "$BUILD/base_m0" \
    --canonical-dir "$BUILD/base_m0_canon" \
    --out-zip "outputs/submit_testB_shot4_margin0.zip" \
    --manifest "$MANIFEST" \
    --report "outputs/reports/prepare_submit_testB_shot4_margin0.json" >/dev/null
  $PY src/eval/official_oracle/check_submission.py \
    --zip_path "outputs/submit_testB_shot4_margin0.zip" \
    --list_path /tmp/testB_list_$DAY.txt >/dev/null 2>&1 \
    && echo "  shot4_margin0: OK (hedge against M=$M)" \
    || echo "  shot4_margin0: PRECHECK FAILED"
fi

echo "=== shot 5: cut400 conf0.35 + span80 ==="
$PY scripts/filter_short_lanes.py \
  --src "$OUT/testB_cut400/testB/predictions" \
  --base "$B" --dst "$BUILD/cut400_f80" --min-span 80
pack shot5_cut400 "$BUILD/cut400_f80"

echo
echo "=== READY (hand these to the user; submission needs explicit sign-off) ==="
for z in outputs/submit_testB_*.zip; do
  l=$($PY -c "
import zipfile
n=0
with zipfile.ZipFile('$z') as zf:
    for nm in zf.namelist():
        if nm.endswith('.lines.txt'):
            n+=sum(1 for x in zf.read(nm).decode().splitlines() if x.strip())
print(n)")
  echo "  $(pwd)/$z   lanes=$l"
done
echo
echo "Shot order: 1 base54 -> 2 union -> 3 consensus -> 4 hedge -> 5 cut400 -> 6 adaptive"
echo "Suggested notes (<=50 chars):"
echo "  1: 54ep conf0.50 trim$M"
echo "  2: swa4 union consensus trim$M"
echo "  3: consensus 60pct span80 trim$M"
echo "  4: $([ "$M" = "0" ] && echo '54ep conf0.55 trim0' || echo '54ep conf0.50 trim0 hedge')"
echo "  5: cut400 conf0.35 span80 trim$M"
