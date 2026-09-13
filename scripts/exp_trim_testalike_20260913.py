#!/usr/bin/env python3
"""Find the optimal bottom-trim depth for testA-like geometry, locally.

Why this exists
---------------
testA (54ep) predicted lanes all start at y >= 549 (median 564, span 155)
while train OOF lanes start at 244..564 (median 434, span 210). That is why
the trim rule measured +4.224pp on the full OOF but only +0.069pp on testA:
the rule was tuned on a geometry the A board does not have.

Train still contains clips whose GT horizon matches testA (video
v444733262 and v576104564 have GT top 532..587). Restricting the honest OOF
to those clips gives a local, GT-labelled proxy for testA geometry, so the
trim depth can be optimised without spending an A-board submission.

Honesty note: the OOF predictions for those clips come from the LVO folds
that held their video out, so the proxy stays video-disjoint.
"""
from __future__ import annotations

import glob
import json
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from data.manifest import read_manifest  # noqa: E402
from eval.oracle_runner import run_official_eval  # noqa: E402
from exp_bottom_trim_20260913 import build_variant, OFFICIAL_PY  # noqa: E402

GT_GLOB = "data/raw/**/*.jpg.json"
OOF = PROJECT_ROOT / "outputs/lvo_clrnet_r50_15ep_20260904/oof/predictions"
GT_DIR = PROJECT_ROOT / "outputs/a_gt_train_20260905/Lane/anno_txt"
WORK = PROJECT_ROOT / "outputs/exp_trim_testalike_20260913"
MIN_GT_TOP = 530.0


def clip_gt_top() -> dict[str, float]:
    per: dict[str, list[float]] = {}
    for f in glob.glob(str(PROJECT_ROOT / GT_GLOB), recursive=True):
        clip = f.split("/")[-2]
        d = json.load(open(f))
        lanes = d.get("annotations", {}).get("lane", [])
        if not lanes:
            continue
        tops = []
        for ln in lanes:
            pts = ln.get("points", ln) if isinstance(ln, dict) else ln
            ys = [p[1] for p in pts]
            if ys:
                tops.append(min(ys))
        if tops:
            per.setdefault(clip, []).append(statistics.median(tops))
    return {c: statistics.median(v) for c, v in per.items()}


def main() -> None:
    tops = clip_gt_top()
    selected = sorted(c for c, t in tops.items() if t >= MIN_GT_TOP)
    print(f"testA-like clips (GT top median >= {MIN_GT_TOP:.0f}): {len(selected)}")
    for c in selected:
        print(f"  {c}  gtTop={tops[c]:.0f}")
    if not selected:
        raise SystemExit("no testA-like clip found")

    all_records = read_manifest(PROJECT_ROOT / "data/processed/manifest_train.jsonl")
    keep = set(selected)
    # The Oracle validates manifest ordering, so the subset must be renumbered
    # from zero (same trick evaluate_lvo_video_oof.py uses for video subsets).
    records = [replace(r, order=i)
               for i, r in enumerate(r for r in all_records
                                     if getattr(r, "clip_id", None) in keep)]
    print(f"subset images: {len(records)}")

    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "subset.json").write_text(
        json.dumps({"min_gt_top": MIN_GT_TOP, "clips": selected,
                    "images": len(records)}, indent=1), encoding="utf-8")

    variants: list[tuple[str, str, float]] = [("baseline", "none", 0.0)]
    for m in (0.0, -10.0, -20.0, -30.0, -40.0):
        variants.append((f"gtcond{m:+.0f}", "gt_cond", m))
    for f in (0.90, 0.85, 0.80):
        variants.append((f"span{int(f*100)}", "span_frac", f))

    results = []
    for name, rule, param in variants:
        pred_dir = OOF if rule == "none" else WORK / name
        if rule != "none":
            build_variant(OOF, pred_dir, rule, param)
        t0 = time.time()
        g = run_official_eval(
            pred_dir, GT_DIR, records, official_python=OFFICIAL_PY,
            per_clip=False,
        ).to_dict()["global"]
        print(f"{name:>10}: TP={g['tp']} FP={g['fp']} FN={g['fn']} "
              f"F1={g['f1']:.6f}  ({time.time()-t0:.0f}s)", flush=True)
        results.append({"variant": name, "rule": rule, "param": param, **g})

    base = next(r for r in results if r["variant"] == "baseline")
    print(f"\n=== delta vs baseline (F1={base['f1']:.6f}) ===")
    for r in sorted(results[1:], key=lambda x: -x["f1"]):
        print(f"{r['variant']:>10}: F1={r['f1']:.6f}  dF1={(r['f1']-base['f1'])*100:+.3f}pp")
    (WORK / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
