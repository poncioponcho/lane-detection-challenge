#!/usr/bin/env python3
"""Build a consensus-gated union for testA (the only measured positive lever).

Rationale
---------
`docs/union_true_rate_20260913.md`: a single-support union adds lanes at
38.5% precision against a 38.9% break-even -- exactly neutral, so recall
cannot be bought that way.

`outputs/exp_union_consensus_20260913/result.json`: requiring TWO independent
support models to agree lifts the marginal true-rate to **0.5507** (276 added,
152 true) against a 0.3921 break-even, worth **+0.210pp** on the honest 63-clip
OOF subset. That is the first construction that clears F1/2.

This script applies the same construction to testA:
  * base   = the 54ep conf0.50 raw tree (the pack that scored 0.73574 after
             trim0), kept verbatim;
  * novel  = a support lane that does not explain any base lane
             (official rasterised mask IoU <= 0.5);
  * gated  = the novel lane is confirmed by >= --min-support OTHER support
             trees (also by mask IoU > 0.5);
  * then the frozen bottom-trim (margin 0) is applied to the whole union,
             exactly as in the incumbent pack.

Support order is frozen because dedup is order sensitive.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import numpy as np  # noqa: E402

from build_testA_variants_20260914 import (  # noqa: E402
    IOU_THRESHOLD,
    load_oracle,
    parse_line,
)
from exp_bottom_trim_20260913 import cut_bottom, fmt, gt_bottom_for_top  # noqa: E402

ORACLE = load_oracle()
_CACHE: dict[tuple, "np.ndarray"] = {}


def mask_of(points):
    key = tuple(points)
    hit = _CACHE.get(key)
    if hit is None:
        interp = ORACLE.interp_lane(list(points))
        hit = ORACLE.draw_lane_mask(interp, ORACLE.LINE_WIDTH)
        _CACHE[key] = hit
    return hit


def iou(a, b) -> float:
    ma, mb = mask_of(a), mask_of(b)
    union = int(np.logical_or(ma, mb).sum())
    if union == 0:
        return 0.0
    return int(np.logical_and(ma, mb).sum()) / union


def read_lanes(path: Path) -> list:
    if not path.is_file():
        return []
    return [parse_line(t) for t in path.read_text(encoding="utf-8").splitlines()
            if t.strip()]


# Frozen support order: most independent first so the retained representative
# comes from the most diverse model when several supports agree.
SUPPORTS = [
    ("hires_c40", "outputs/testA_hires/testA_hires_conf0.40/testA/predictions"),
    ("t05_36ep", "outputs/testA_support_trees/t05_36ep/testA/predictions"),
    ("clrernet_36ep", "outputs/testA_support_trees/clrernet_36ep/testA/predictions"),
    ("seed202_36ep", "outputs/testA_support_trees/seed202_36ep/testA/predictions"),
    ("seed303_36ep", "outputs/testA_support_trees/seed303_36ep/testA/predictions"),
    ("seed101_36ep", "outputs/testA_support_trees/seed101_36ep/testA/predictions"),
    ("seed42_36ep", "outputs/testA_full71_infer_20260909/all71_seed42_clrnet_r50_36ep/testA/predictions"),
    ("seed43_36ep", "outputs/testA_full71_infer_20260909/all71_seed43_clrnet_r50_36ep/testA/predictions"),
    ("seed44_36ep", "outputs/testA_full71_infer_20260909/all71_seed44_clrnet_r50_36ep/testA/predictions"),
    ("clrernet_15ep", "outputs/testA_full71_infer_20260909/clrernet_r50_15ep/testA/predictions"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True, type=Path)
    ap.add_argument("--dst", required=True, type=Path)
    ap.add_argument("--min-support", type=int, default=2,
                    help="how many OTHER support trees must confirm a novel lane")
    ap.add_argument("--trim-margin", type=float, default=0.0)
    args = ap.parse_args()

    base_root = args.base.resolve()
    dst = args.dst.resolve()
    if dst.exists():
        raise SystemExit(f"destination exists: {dst}")
    supports = [(n, (PROJECT_ROOT / p).resolve()) for n, p in SUPPORTS]
    missing = [n for n, p in supports if not p.is_dir()]
    if missing:
        raise SystemExit(f"missing support trees: {missing}")

    rels = sorted(str(p.relative_to(base_root)) for p in base_root.rglob("*.lines.txt"))
    added_total = 0
    novel_total = 0
    per_image = []

    for rel in rels:
        base_lanes = read_lanes(base_root / rel)
        base_masks = [mask_of(p) for p in base_lanes]
        # novel candidates per support tree
        novel_by_tree: list[list] = []
        for _, root in supports:
            novel = []
            for cand in read_lanes(root / rel):
                if any(iou(cand, b) > IOU_THRESHOLD for b in base_lanes):
                    continue
                novel.append(cand)
            novel_by_tree.append(novel)
        novel_total += sum(len(n) for n in novel_by_tree)

        kept = list(base_lanes)
        for ti, novel in enumerate(novel_by_tree):
            for cand in novel:
                # count the OTHER trees that also propose this lane
                support = 1  # itself
                for tj, other in enumerate(novel_by_tree):
                    if tj == ti:
                        continue
                    if any(iou(cand, o) > IOU_THRESHOLD for o in other):
                        support += 1
                if support < args.min_support:
                    continue
                # already added from an earlier tree?
                if any(iou(cand, k) > IOU_THRESHOLD for k in kept[len(base_lanes):]):
                    continue
                kept.append(cand)
                added_total += 1
        per_image.append(len(kept) - len(base_lanes))

        out_lines = []
        for pts in kept:
            ys = [p[1] for p in pts]
            top, bottom = min(ys), max(ys)
            cut = gt_bottom_for_top(top) + args.trim_margin
            if cut < bottom - 1.0:
                trimmed = cut_bottom(pts, cut)
                if not trimmed:
                    continue
                out_lines.append(fmt(trimmed))
            else:
                out_lines.append(fmt(pts))
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(("\n".join(out_lines) + "\n") if out_lines else "",
                       encoding="utf-8")

    report = {
        "base": str(base_root),
        "images": len(rels),
        "min_support": args.min_support,
        "trim_margin": args.trim_margin,
        "novel_candidates": novel_total,
        "added_lanes": added_total,
        "max_added_per_image": max(per_image) if per_image else 0,
        "images_gaining": sum(1 for n in per_image if n > 0),
        "iou_threshold": IOU_THRESHOLD,
        "support_order": [n for n, _ in supports],
    }
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
