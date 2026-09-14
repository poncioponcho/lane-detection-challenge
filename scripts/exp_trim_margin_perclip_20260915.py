#!/usr/bin/env python3
"""Per-clip trim-margin calibration on the honest 15ep LVO OOF (2026-09-15).

Why this exists
---------------
Bottom-trim is the only geometric repair that has ever migrated to the target
domain, and it is the single largest lever in the B-board plan: full OOF says
margin +40 is worth +4.224pp while testA says margin 0 is the peak (+0.069pp).
The runbook's decision rule for testB is a *global* one -- "if the fraction of
predicted tops below y=450 exceeds 10%, the set is train-shaped, use +40" --
which silently assumes testB is homogeneous.  B is ten clips; if some clips are
train-shaped and some are testA-shaped, one global margin leaves several pp on
the table.

This script answers, with honest OOF evidence:

  1. what the margin->F1 curve looks like over the whole OOF;
  2. whether the *best margin actually varies by clip*;
  3. whether a cheap per-clip statistic (the top<450 fraction, computable on
     unlabeled testB without any GT) predicts which margin that clip wants.

Everything is computed in one pass per margin with the frozen official scorer's
own mask/IoU/Hungarian primitives.  Global numbers are diagnostics; any number
promoted to a candidate decision is re-confirmed with ``run_official_eval``.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import numpy as np  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

from data.manifest import read_manifest  # noqa: E402
from exp_bottom_trim_20260913 import cut_bottom, gt_bottom_for_top, parse_line  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "outputs/lvo_clrnet_r50_15ep_20260904/oof/predictions"
GT_DIR = PROJECT_ROOT / "outputs/a_gt_train_20260905/Lane/anno_txt"
MANIFEST = PROJECT_ROOT / "data/processed/manifest_train.jsonl"
WORK = PROJECT_ROOT / "outputs/exp_trim_margin_perclip_20260915"
OFFICIAL_PY = "/Users/seyonmacbook/.workbuddy/binaries/python/envs/lane-oracle-py312/bin/python"
WIDTH = 30
IOU_THR = 0.5


def load_oracle():
    p = PROJECT_ROOT / "src" / "eval" / "official_oracle" / "score.py"
    spec = importlib.util.spec_from_file_location("official_oracle_score", str(p))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ORACLE = load_oracle()


def read_lines(path: Path):
    if not path.is_file():
        return []
    out = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        pts = parse_line(raw)
        cleaned = [pts[0]]
        for p in pts[1:]:
            if p != cleaned[-1]:
                cleaned.append(p)
        if len(cleaned) >= 2:
            out.append(cleaned)
    return out


def safe_interp(lanes):
    out = []
    for pts in lanes:
        try:
            out.append(ORACLE.interp_lane(pts))
        except Exception:
            out.append(None)
    return out


def apply_margin(lanes, margin):
    out = []
    for pts in lanes:
        top = min(p[1] for p in pts)
        cut = cut_bottom(pts, gt_bottom_for_top(top) + margin)
        if len(cut) >= 2:
            out.append(cut)
    return out


def score_pair(preds, gts):
    """Official per-image counts: [tp, fp, fn]."""
    if not preds or not gts:
        return 0, len(preds), len(gts)
    ip = safe_interp(preds)
    ig = safe_interp(gts)
    kp = [i for i, a in enumerate(ip) if a is not None]
    kg = [j for j, a in enumerate(ig) if a is not None]
    preds = [preds[i] for i in kp]
    gts = [gts[j] for j in kg]
    ip = [ip[i] for i in kp]
    ig = [ig[j] for j in kg]
    if not preds or not gts:
        return 0, len(preds), len(gts)
    pm = [ORACLE.draw_lane_mask(a, WIDTH) for a in ip]
    gm = [ORACLE.draw_lane_mask(a, WIDTH) for a in ig]
    ious = np.zeros((len(preds), len(gts)))
    for i, a in enumerate(pm):
        for j, b in enumerate(gm):
            u = (a | b).sum()
            if u:
                ious[i, j] = (a & b).sum() / u
    ri, ci = linear_sum_assignment(1 - ious)
    tp = int((ious[ri, ci] > IOU_THR).sum())
    return tp, len(preds) - tp, len(gts) - tp


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--margins", default="0,20,40,60,80")
    ap.add_argument("--max-images", type=int, default=0)
    args = ap.parse_args()
    margins = [int(m) for m in args.margins.split(",")]

    records = read_manifest(MANIFEST)
    if args.max_images:
        records = records[: args.max_images]
    WORK.mkdir(parents=True, exist_ok=True)

    # per-margin per-clip counters
    per: dict[int, dict[str, list]] = {
        m: defaultdict(lambda: [0, 0, 0]) for m in margins}
    # cheap unlabeled statistic per clip: fraction of predicted tops above the
    # horizon candidate (a low top means the lane starts far away = train shape)
    clip_tops: dict[str, list] = defaultdict(list)
    t0 = time.time()

    for idx, rec in enumerate(records):
        gts = read_lines(GT_DIR / rec.pred_rel_path)
        raw = read_lines(SRC / rec.pred_rel_path)
        for pts in raw:
            clip_tops[rec.clip_id].append(min(p[1] for p in pts))
        for m in margins:
            tp, fp, fn = score_pair(apply_margin(raw, m), gts)
            c = per[m][rec.clip_id]
            c[0] += tp
            c[1] += fp
            c[2] += fn
        if (idx + 1) % 1000 == 0:
            print(f"  {idx+1}/{len(records)} {time.time()-t0:.0f}s", flush=True)

    def f1(c):
        p, g = c[0] + c[1], c[0] + c[2]
        return 2 * c[0] / (p + g) if p + g else 0.0

    clips = sorted(clip_tops)
    global_res = {}
    for m in margins:
        tot = [sum(per[m][c][k] for c in clips) for k in range(3)]
        global_res[m] = {"TP": tot[0], "FP": tot[1], "FN": tot[2], "F1": f1(tot)}

    # per-clip: which margin wins, and does top<450 predict it?
    rows = []
    for c in clips:
        tops = clip_tops[c]
        far = sum(1 for t in tops if t < 450) / len(tops) if tops else 0.0
        scores = {m: f1(per[m][c]) for m in margins}
        best = max(scores, key=lambda m: scores[m])
        rows.append({
            "clip": c, "far_top_frac": round(far, 3),
            "top_p50": round(float(np.median(tops)), 1) if tops else None,
            "n_lanes": len(tops),
            "f1_by_margin": {str(m): round(scores[m], 4) for m in margins},
            "best_margin": best,
            "spread_pp": round((max(scores.values()) - min(scores.values())) * 100, 3),
        })

    base = global_res[margins[0]]["F1"]
    summary = {
        "margins": margins,
        "global_by_margin": {str(m): {"F1": global_res[m]["F1"],
                                      "TP": global_res[m]["TP"],
                                      "FP": global_res[m]["FP"],
                                      "FN": global_res[m]["FN"],
                                      "dF1_pp_vs_margin0": round((global_res[m]["F1"] - base) * 100, 4)}
                             for m in margins},
        "best_global_margin": max(global_res, key=lambda m: global_res[m]["F1"]),
        "n_clips": len(clips),
        "elapsed_s": round(time.time() - t0, 1),
    }
    # how often does the per-clip winner differ from the global winner?
    gw = summary["best_global_margin"]
    summary["clips_where_global_wins"] = sum(1 for r in rows if r["best_margin"] == gw)
    summary["clips_disagreeing"] = sum(1 for r in rows if r["best_margin"] != gw)
    # oracle per-clip selection (upper bound of a perfect per-clip rule)
    orc = sum(per[r["best_margin"]][r["clip"]][k] for r in rows for k in (0,)) , 0, 0
    tot_best = [sum(per[r["best_margin"]][r["clip"]][k] for r in rows) for k in range(3)]
    summary["oracle_perclip_F1"] = f1(tot_best)
    summary["oracle_perclip_dF1_pp_vs_global_best"] = round(
        (f1(tot_best) - global_res[gw]["F1"]) * 100, 4)
    # does the cheap statistic separate the two margin camps?
    for m in margins:
        sel = [r["far_top_frac"] for r in rows if r["best_margin"] == m]
        if sel:
            summary[f"far_top_frac_when_best_margin_{m}"] = {
                "n": len(sel), "median": round(float(np.median(sel)), 3),
                "min": round(min(sel), 3), "max": round(max(sel), 3)}

    (WORK / "perclip.json").write_text(
        json.dumps({"summary": summary, "clips": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\n=== per-clip (best margin | far_top_frac | spread pp) ===")
    for r in sorted(rows, key=lambda r: -r["far_top_frac"])[:20]:
        print(f"{r['clip']:26s} m={r['best_margin']:>3d} far={r['far_top_frac']:.3f} "
              f"top50={r['top_p50']} spread={r['spread_pp']:+.2f}pp")


if __name__ == "__main__":
    main()
