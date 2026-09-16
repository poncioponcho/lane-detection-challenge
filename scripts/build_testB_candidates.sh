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

[ -f "$TREES" ] || { echo "usage: $0 <testB_bundle.tgz>"; exit 1; }

mkdir -p "$BUILD"
tar xzf "$TREES" -C "$OUT"
# the bundle also carries the manifest + list produced on the instance
if [ -f "$OUT/data_processed/manifest_testB.jsonl" ]; then
  cp "$OUT/data_processed/manifest_testB.jsonl" "$MANIFEST"
  echo "manifest installed from bundle: $MANIFEST"
fi
[ -f "$OUT/data_processed/manifest_testB.list.txt" ] && \
  cp "$OUT/data_processed/manifest_testB.list.txt" "/tmp/testB_list_$DAY.txt"
ls -d "$OUT"/testB_*

# Which tree plays the floor. Defaults to conf=0.50, which is what the A-board
# 0.73574 incumbent used and the only operating point with a measured anchor.
# Override to spend the low-confidence sweep:  BASE_TAG=base54_cNN bash ... <tgz>
BASE_TAG=${BASE_TAG:-base54}
B=$OUT/testB_${BASE_TAG}/testB/predictions
[ -d "$B" ] || { echo "base tree missing at $B (BASE_TAG=$BASE_TAG)"; exit 1; }
[ -f "$MANIFEST" ] || { echo "missing $MANIFEST -- transfer it with the bundle"; exit 1; }
NLINES=$(wc -l < "$MANIFEST")
echo "manifest present: $MANIFEST ($NLINES rows)"
[ "$NLINES" -ge 1000 ] || echo "!! only $NLINES rows -- expected >=1000 for testB"

echo "=== official pre-check list ==="
if [ -f "/tmp/testB_list_$DAY.txt" ]; then
  echo "using list from bundle: /tmp/testB_list_$DAY.txt ($(wc -l < /tmp/testB_list_$DAY.txt) rows)"
else
  $PY - "$MANIFEST" /tmp/testB_list_$DAY.txt <<'PY'
import json, sys
recs = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
open(sys.argv[2], "w").write("".join("/" + r["image_path"] + "\n" for r in recs))
print("list rows:", len(recs))
PY
fi

echo "=== margin decision (f<450, runbook section 4) ==="
M=$($PY - "$B" <<'PY'
import sys, pathlib, statistics
root = pathlib.Path(sys.argv[1])
tops = [min(float(t) for t in [float(x) for x in line.split()][1::2])
        for f in root.rglob("*.lines.txt") for line in f.read_text().splitlines() if line.strip()]
far = sum(1 for t in tops if t < 450) / max(1, len(tops))
# 2026-09-16: the threshold was a hard-coded 10%. The two branches are wildly
# asymmetric, so the break-even point is far lower than that:
#   train-shaped lanes  -> margin 40 beats margin 0 by +2.25pp (measured on OOF)
#   testA-shaped lanes  -> margin 0 beats margin 40 by only +0.069pp (A board)
# break-even fraction p:  p*2.25 == (1-p)*0.069  ->  p = 2.98%
# So switch at 3%, not 10%: it can only ever cost 0.07pp and can gain 2.25pp.
import sys as _s
print("f<450 = %.4f (%d/%d)" % (far, sum(1 for t in tops if t < 450), len(tops)),
      file=_s.stderr)
print(40 if far > 0.03 else 0)
PY
)
echo "chosen trim margin M = $M"

pack () {  # pack <name> <srcdir>
  n=$1; s=$2
  if [ ! -d "$s" ] || [ -z "$(find "$s" -name '*.lines.txt' -print -quit 2>/dev/null)" ]; then
    echo "  $n: SKIPPED (source tree empty or missing: $s)"
    return 1
  fi
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

# Same as pack() but with an explicit trim margin, for shots that deliberately
# disagree with the global decision.
pack_m () {  # pack_m <name> <srcdir> <margin>
  n=$1; s=$2; m=$3
  if [ ! -d "$s" ] || [ -z "$(find "$s" -name '*.lines.txt' -print -quit 2>/dev/null)" ]; then
    echo "  $n: SKIPPED (source tree empty or missing: $s)"
    return 1
  fi
  $PY scripts/apply_bottom_trim_testA.py --src "$s" --dst "$BUILD/${n}_trim" --margin "$m" >/dev/null
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

count_lanes () {  # count_lanes <tree>
  find "$1" -name '*.lines.txt' -print0 2>/dev/null \
    | xargs -0 cat 2>/dev/null | grep -cve '^[[:space:]]*$'
}

# A consensus shot that adds ~nothing is just the floor resubmitted; say so
# loudly instead of burning a slot. (2026-09-16 dry run: 6 supports at k=4 is a
# 67% gate, not 57%, and it added exactly 0 lanes.)
warn_if_no_addition () {  # warn_if_no_addition <name> <tree> <base_lanes>
  n=$(count_lanes "$2")
  added=$(( n - $3 ))
  echo "  $1: lanes=$n (added vs floor: $added)"
  if [ "$added" -le 0 ]; then
    echo "  !! $1 adds NO lanes -- identical to the floor. Do NOT spend a slot on it."
  fi
}

echo "=== shot 1: base54 + trim($M) ==="
pack shot1_base54 "$B"
BASE_LANES=$(find "$BUILD/shot1_base54_trim" -name '*.lines.txt' -print0 2>/dev/null \
  | xargs -0 cat 2>/dev/null | grep -cve '^[[:space:]]*$')
echo "  base lanes after trim: $BASE_LANES"

echo "=== support trees: which ones actually made it through inference? ==="
# 2026-09-16 修：支撑树写死 7 棵，但只要有 1 棵推理失败，
# build_testA_consensus_union_20260913.py 会**静默产出空树**（files=0），
# 后面 filter → pack 全失败却不中断 → 白扔 3 发额度。演练实测到这个行为。
# 现在：只把真实存在的树写进 json，并按**同意率**（不是票数）重算门槛。
$PY - "$OUT" "$DAY" <<'PY'
import json, sys, pathlib
out, day = pathlib.Path(sys.argv[1]), sys.argv[2]
spec = [("seed42_36ep", "testB_seed42"), ("seed101_36ep", "testB_seed101"),
        ("seed202_36ep", "testB_seed202"), ("seed303_36ep", "testB_seed303"),
        ("clrernet_36ep", "testB_clrernet36"), ("cut400_36ep", "testB_cut400"),
        ("hires_36ep", "testB_hires"),
        # 2026-09-16: measured on testA that 9 supports beat 7 at the same
        # agreement fraction (56% -> 24 added lanes vs 14). Both run-dirs exist.
        ("occlude_36ep", "testB_occlude"), ("s101_54ep", "testB_s101_54ep")]
present, missing = [], []
for name, d in spec:
    p = out / d / "testB/predictions"
    if p.is_dir() and any(p.rglob("*.lines.txt")):
        present.append({"name": name, "path": str(p.resolve())})
    else:
        missing.append(name)
json.dump(present, open(f"/tmp/testB_supports_{day}.json", "w"), indent=1)
print(f"  support trees: present={len(present)} missing={missing}")
if missing:
    print("  !! WARNING: agreement fractions recomputed on the available set")
print(len(present))
PY
M_SUP=$($PY - "$OUT" <<'PY'
import sys, pathlib
out = pathlib.Path(sys.argv[1])
print(sum(1 for d in ("testB_seed42", "testB_seed101", "testB_seed202", "testB_seed303",
                     "testB_clrernet36", "testB_cut400", "testB_hires",
                     "testB_occlude", "testB_s101_54ep")
          if (out / d / "testB/predictions").is_dir()
          and any((out / d / "testB/predictions").rglob("*.lines.txt"))))
PY
)
# round(frac * m), not ceil -- ceil is too strict when a tree is missing and the
# consensus then adds nothing at all (dry run: 6 supports at k=4 is a 67% gate
# and produced added_kept=0, i.e. the floor resubmitted under another name).
# 0.571/0.714/0.857 of 7 -> 4/5/6; of 6 -> 3/4/5 (50%/67%/83%).
# 2026-09-16: the ladder is re-pinned to the points actually measured on testA
# with the production support set. 44/56/67% of 9 -> 4/5/6, of 7 -> 3/4/5.
# Above ~70% the consensus adds nothing at all (measured 0).
K4=$(( (M_SUP * 440 + 500) / 1000 ))
K5=$(( (M_SUP * 560 + 500) / 1000 ))
K6=$(( (M_SUP * 670 + 500) / 1000 ))
[ "$K4" -lt 2 ] && K4=2
[ "$K5" -lt 3 ] && K5=3
[ "$K6" -lt 3 ] && K6=3
echo "  available supports=$M_SUP -> min-support k: 44%=$K4  56%=$K5  67%=$K6"

build_cons () {  # build_cons <dst> <k>
  $PY scripts/build_testA_consensus_union_20260913.py \
    --base "$B" --dst "$1" --min-support "$2" --average \
    --supports-json /tmp/testB_supports_$DAY.json | tail -2
}

echo "=== shot 3: consensus at >=56% agreement + span80 (best measured EV) ==="
if [ "$M_SUP" -ge 4 ]; then
  # K5 (56%), not K4: the union shot is built on top of this tree, and the
  # 56% point prices highest (+0.036pp vs +0.027pp on the measured ladder).
  build_cons "$BUILD/cons" "$K5"
  $PY scripts/filter_short_lanes.py --src "$BUILD/cons" --base "$B" \
    --dst "$BUILD/cons_f80" --min-span 80
  pack shot3_consensus "$BUILD/cons_f80"
  warn_if_no_addition shot3_consensus "$BUILD/cons_f80" "$BASE_LANES"
else
  echo "  shot3: SKIPPED (only $M_SUP support trees -- gate is meaningless)"
fi

echo "=== shot 2: union(swa4, consensus) -- strictly dominates swa4 ==="
$PY scripts/apply_bottom_trim_testA.py \
  --src "$OUT/testB_swa4/testB/predictions" --dst "$BUILD/swa4_trim" --margin "$M" >/dev/null
$PY scripts/build_union_pair_20260915.py \
  --a "$BUILD/swa4_trim" --b "$BUILD/shot3_consensus_trim" --dst "$BUILD/uni"
pack shot2_uni_swa4_cons "$BUILD/uni"

echo "=== shot 4: consensus at >=44% agreement (testA-shaped) or margin hedge (train-shaped) ==="
# 2026-09-16 改注：原为 conf0.55。夜间用校准后的印证率→真线率拟合给它定了价：
# conf55 删掉的 69 条线 corr=0.333 → r_est=0.384 > 盈亏线 0.3675 → **估计 -0.039pp**（此前按 r≈0.30
# 估成 +0.16pp 是错的）。已知/疑似为负的不该占 max 名额，换成同曲线上更纯的一点：7 树 k=5(71%)。
if [ "$M" = "0" ]; then
  if [ "$M_SUP" -ge 4 ]; then
    build_cons "$BUILD/cons_k5" "$K4"
    $PY scripts/filter_short_lanes.py --src "$BUILD/cons_k5" --base "$B" \
      --dst "$BUILD/cons_k5_f80" --min-span 80
    pack shot4_cons_k5 "$BUILD/cons_k5_f80"
  warn_if_no_addition shot4_cons_k5 "$BUILD/cons_k5_f80" "$BASE_LANES"
  else
    echo "  shot4: SKIPPED (only $M_SUP support trees)"
  fi
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

echo "=== shot 5: consensus at >=86% agreement -- the purest point on the ladder ==="
# 2026-09-16 改注：原为 union(incumbent, occlude)。夜间定价：
# occlude 增量的 119 条 corr=0.227 → r_est=0.352 < 盈亏线 0.3675 → **-0.067pp**（负）。
# 单支撑 union 的增量线印证率普遍低（swa4 0.222 / occlude 0.227 / soupB 0.177），
# 只有**共识门控**的增量线印证率高（gate6 0.519）。所以第 5 注改成同意率阶梯的最纯一端。
if [ "$M_SUP" -ge 4 ]; then
  build_cons "$BUILD/cons_k6" "$K6"
  $PY scripts/filter_short_lanes.py --src "$BUILD/cons_k6" --base "$B" \
    --dst "$BUILD/cons_k6_f80" --min-span 80
  pack shot5_cons_k6 "$BUILD/cons_k6_f80"
  warn_if_no_addition shot5_cons_k6 "$BUILD/cons_k6_f80" "$BASE_LANES"
else
  echo "  shot5: SKIPPED (only $M_SUP support trees)"
fi

echo "=== shot 6: geometry hedge -- deliberately trim with the OTHER margin ==="
# Why this shot exists even when the f<450 statistic says not to trim.
#
# The two candidate geometries are wildly asymmetric (runbook section 4):
#   train-shaped testB -> margin 40 beats margin 0 by up to +2.25pp (OOF)
#   testA-shaped testB -> margin 40 costs only 0.069pp, because with every lane
#                         starting near y=564 the cut lands past y=719 and the
#                         rule degenerates into a no-op.
# The 3% gate optimises EXPECTED value for a single submission. But the B board
# takes the MAX over 6 shots, and expectation is the wrong objective there: this
# is the last-ranked shot, so the 0.069pp it can lose costs essentially nothing,
# while it removes the one remaining multi-pp way to be wrong -- misreading testB
# geometry and leaving +2.25pp unclaimed. Buy the option every time.
if [ "$M" = "0" ]; then
  # Prefer carrying the CONSENSUS tree here, not the bare base: if the repair
  # lands, this shot collects both effects (added lanes + trimmed overshoot).
  # If it does not, it degrades into a copy of shot4 -- the acknowledged cost
  # of buying the option.
  if [ "$M_SUP" -ge 4 ] && [ -d "$BUILD/cons_k5_f80" ]; then
    pack_m shot6_cons_k5_m40 "$BUILD/cons_k5_f80" 40
  else
    pack_m shot6_margin40 "$B" 40
  fi
else
  echo "  shot6: not needed (M=$M already; shot4 carries the margin-0 hedge)"
fi

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
echo "Shot order: 1 base54 -> 2 union(swa4, cons 56%) -> 3 cons 56% -> 4 cons 44%/hedge -> 5 cons 67% -> 6 geometry hedge"
echo
# Official rule section 4: "every team may submit at most 3 times PER DAY".
# The B board spans 9/16 and 9/17, and the daily quota does not roll over --
# 9/16 went unused because the images never arrived. So budget for 3, not 6.
echo "!! QUOTA: official rule says 3 submissions PER DAY, non-cumulative."
echo "!! 9/16's 3 were never spent and did NOT roll over -> assume only 3 today."
echo "!! If the platform shows 3 left, submit ONLY: shot1, shot3, shot6 (in that order)."
echo "!! See docs/action_testB_20260916.md section 2.4 for the 6-shot fallback."
echo "Suggested notes (<=50 chars):"
echo "  1: 54ep conf0.50 trim$M"
echo "  2: swa4 union consensus trim$M"
echo "  3: consensus 56pct span80 trim$M"
echo "  4: $([ "$M" = "0" ] && echo "consensus 44pct span80 trim0" || echo '54ep conf0.50 trim0 hedge')"
echo "  5: consensus 67pct span80 trim$M"
echo "  6: $([ "$M" = "0" ] && echo 'cons 44pct span80 trim40 (hedge)' || echo 'unused, shot4 is the margin-0 hedge')"
echo
echo "REMINDER: if any shot printed SKIPPED, do not submit a placeholder -- fix or drop the slot."
