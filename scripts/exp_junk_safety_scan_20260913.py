#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Safety audit for candidate junk-lane designs.

The decompose probe assumes every injected lane is an unmatched FP.  That only
holds if max_j IoU(junk_mask, gt_lane_j) stays below the 0.5 gate for *every*
image.  This script measures that maximum directly, using the frozen oracle's
own interpolation and rasterisation, and reports the worst case.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCORE_PY = REPO / "src/eval/official_oracle/score.py"

CANDIDATES = {
    "A_top_horizontal": "0.0 4.0 1365.0 4.0",
    "B_bottom_horizontal": "0.0 715.0 1365.0 715.0",
    "C_mid_horizontal": "0.0 360.0 1365.0 360.0",
    "D_left_vertical": "4.0 4.0 4.0 715.0",
    "E_tiny_stub": "683.0 340.0 683.0 342.0",
}


def load_oracle():
    spec = importlib.util.spec_from_file_location("frozen_score", SCORE_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", required=True)
    ap.add_argument("--pred-dir", required=True, help="used only to enumerate image keys")
    ap.add_argument("--n-images", type=int, default=400)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    oracle = load_oracle()
    gt_dir = Path(args.gt_dir)
    pred_dir = Path(args.pred_dir)

    keys = []
    for clip in sorted(os.listdir(pred_dir)):
        cdir = pred_dir / clip
        if not cdir.is_dir():
            continue
        for fn in sorted(os.listdir(cdir)):
            if fn.endswith(".lines.txt"):
                keys.append((clip, fn))
    keys = keys[: args.n_images]

    stats = {name: {"max_iou": 0.0, "argmax_image": None, "n_above_0p1": 0, "n_above_0p3": 0}
             for name in CANDIDATES}
    masks = {name: oracle.draw_lane_mask(oracle.interp_lane(
        [(float(spec.split()[i]), float(spec.split()[i + 1]))
         for i in range(0, len(spec.split()), 2)]), oracle.LINE_WIDTH)
        for name, spec in CANDIDATES.items()}

    for clip, fn in keys:
        gt_path = gt_dir / clip / fn
        if not gt_path.exists():
            continue
        lanes = oracle.parse_lines_txt(str(gt_path))
        if not lanes:
            continue
        gt_masks = [oracle.draw_lane_mask(oracle.interp_lane(l), oracle.LINE_WIDTH) for l in lanes]
        for name, jmask in masks.items():
            best = 0.0
            for gm in gt_masks:
                union = int((jmask | gm).sum())
                if union:
                    iou = float((jmask & gm).sum()) / union
                    if iou > best:
                        best = iou
            s = stats[name]
            if best > s["max_iou"]:
                s["max_iou"], s["argmax_image"] = best, f"{clip}/{fn}"
            if best > 0.1:
                s["n_above_0p1"] += 1
            if best > 0.3:
                s["n_above_0p3"] += 1

    report = {"n_images": len(keys), "results": stats,
              "verdict": {n: ("SAFE" if s["max_iou"] < 0.2 else "REVIEW")
                          for n, s in stats.items()}}
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    for n, s in stats.items():
        print(f"{n:22s} max_iou={s['max_iou']:.3f}  >0.1:{s['n_above_0p1']:4d}  >0.3:{s['n_above_0p3']:4d}  worst={s['argmax_image']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
