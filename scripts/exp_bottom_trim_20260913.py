#!/usr/bin/env python3
"""Bottom-trim experiment on the honest 15ep LVO OOF (7100 images).

Motivation: CLRNet decode emits lanes whose far end (top) tracks GT almost
exactly (clip-level pearson 0.995, mean diff -1.6px) but whose near end is
pushed to the image bottom (62% of OOF lines end exactly at y=719) while GT
bottom P50 = 632. The extra near-field length dilutes the 30px-stroke IoU.

This script materialises bottom-trimmed variants of the OOF prediction tree
and scores every variant with the frozen official Oracle (global single call).
No per-clip calls: the global number is the arbitration quantity.
"""
from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.manifest import read_manifest  # noqa: E402
from eval.oracle_runner import run_official_eval  # noqa: E402

OFFICIAL_PY = (
    "/Users/seyonmacbook/.workbuddy/binaries/python/envs/lane-oracle-py312/bin/python"
)

# GT conditional bottom median (top bucket -> bottom), measured on 7100 train
# images in the 2026-09-13 span audit (see docs/span_bottom_trim_20260913.md).
GT_BOTTOM_BY_TOP = [
    (200, 396), (250, 460), (300, 512), (350, 563),
    (400, 615), (450, 710), (500, 712), (550, 712), (600, 713), (650, 713),
]


def gt_bottom_for_top(top: float) -> float:
    if top <= GT_BOTTOM_BY_TOP[0][0]:
        return GT_BOTTOM_BY_TOP[0][1]
    for (t0, b0), (t1, b1) in zip(GT_BOTTOM_BY_TOP, GT_BOTTOM_BY_TOP[1:]):
        if t0 <= top < t1:
            return b0 + (b1 - b0) * (top - t0) / (t1 - t0)
    return GT_BOTTOM_BY_TOP[-1][1]


def parse_line(text: str) -> list[tuple[float, float]]:
    vals = [float(v) for v in text.split()]
    return list(zip(vals[0::2], vals[1::2]))


def fmt(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.1f} {y:.1f}" for x, y in points)


def cut_bottom(points: list[tuple[float, float]], cut: float) -> list[tuple[float, float]]:
    """Keep the far part of the lane: drop points below (y >) the cut."""
    pts = sorted(points, key=lambda p: p[1])  # ascending y: far -> near
    kept = [p for p in pts if p[1] <= cut]
    if len(kept) < 2:
        return []
    # land the last point exactly on the cut so the polyline ends where we want
    if kept[-1][1] < cut:
        prev = kept[-1]
        nxt = next((p for p in pts if p[1] > cut), None)
        if nxt is not None and nxt[1] != prev[1]:
            r = (cut - prev[1]) / (nxt[1] - prev[1])
            kept.append((prev[0] + (nxt[0] - prev[0]) * r, cut))
    return kept


def build_variant(src: Path, dst: Path, rule: str, param: float) -> int:
    if dst.exists():
        shutil.rmtree(dst)
    n_lines = 0
    for path in sorted(src.rglob("*.lines.txt")):
        rel = path.relative_to(src)
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for text in path.read_text(encoding="utf-8").splitlines():
            if not text.strip():
                continue
            pts = parse_line(text)
            ys = [p[1] for p in pts]
            top, bottom = min(ys), max(ys)
            cut = None
            if rule == "fixed":
                cut = param
            elif rule == "span_frac":
                cut = top + (bottom - top) * param
            elif rule == "gt_cond":
                cut = gt_bottom_for_top(top) + param
            if cut is None or cut >= bottom - 1.0:
                rows.append(fmt(pts))
            else:
                kept = cut_bottom(pts, cut)
                if kept:
                    rows.append(fmt(kept))
            n_lines += 1
        out.write_text(("\n".join(rows) + "\n") if rows else "", encoding="utf-8")
    return n_lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--gt-dir", required=True, type=Path)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--work", required=True, type=Path)
    args = ap.parse_args()

    records = read_manifest(args.manifest)
    print(f"manifest rows: {len(records)}")

    variants: list[tuple[str, str, float]] = [("baseline", "none", 0.0)]
    for c in (632.0, 680.0, 712.0):
        variants.append((f"fixed{c:.0f}", "fixed", c))
    for f in (0.80, 0.90):
        variants.append((f"span{int(f*100)}", "span_frac", f))
    for m in (0.0, 40.0):
        variants.append((f"gtcond+{m:.0f}", "gt_cond", m))

    results = []
    for name, rule, param in variants:
        if rule == "none":
            pred_dir = args.src
        else:
            pred_dir = args.work / name
            n = build_variant(args.src, pred_dir, rule, param)
            print(f"  built {name}: {n} lines")
        t0 = time.time()
        res = run_official_eval(
            pred_dir, args.gt_dir, records,
            official_python=OFFICIAL_PY, per_clip=False,
        ).to_dict()
        g = res["global"]
        results.append({
            "variant": name, "rule": rule, "param": param,
            "tp": g["tp"], "fp": g["fp"], "fn": g["fn"], "f1": g["f1"],
            "seconds": round(time.time() - t0, 1),
        })
        print(f"{name:>12}: TP={g['tp']} FP={g['fp']} FN={g['fn']} "
              f"F1={g['f1']:.6f}  ({time.time()-t0:.0f}s)")

    base = next(r for r in results if r["variant"] == "baseline")
    print("\n=== delta vs baseline (pp) ===")
    for r in results:
        d = (r["f1"] - base["f1"]) * 100
        print(f"{r['variant']:>12}: F1={r['f1']:.6f}  dF1={d:+.3f}pp")
    args.work.mkdir(parents=True, exist_ok=True)
    (args.work / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
