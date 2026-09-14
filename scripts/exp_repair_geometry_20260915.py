#!/usr/bin/env python3
"""Score geometric *repair* variants on the honest 15ep LVO OOF (2026-09-15).

Why this exists
---------------
The near-miss forensics (``exp_near_miss_forensics_20260915.py``) found 2739
pairs at 0.15 < IoU <= 0.5 on this exact tree -- a +11.7pp ceiling if every one
could be repaired -- and ruled out any *global* affine correction
(k_scale = -0.0029/px against an rms lateral residual of 31px, i.e. noise).
What is left is per-lane geometry.  The official scorer builds an
**interpolating** spline (``s=0``) through whatever points we hand it, so the
shape of the drawn 30px stroke is decided by our sampling: dense, jittery
points draw a wobbly stroke; sparse points draw a smooth one.  GT is annotated
at a coarser cadence, so matching its effective smoothness should raise IoU on
the crowded 0.4-0.6 band.

Variants here are all *geometry-only* (no lanes added, none removed), so they
are directly comparable at fixed P:

  raw        control, byte-identical copy of the OOF tree
  trim0      frozen bottom-trim rule, margin 0  (production setting)
  trim40     frozen bottom-trim rule, margin +40 (OOF optimum)
  sub2/3/4   keep every 2nd/3rd/4th point (10/15/20px cadence), ends kept
  ma5/ma9    moving-average of x over 5/9 samples (25/45px window)
  quad/cubic least-squares x(y) refit at the original y positions
  trim0_sub4 / trim40_sub4   combinations of the two independent levers

Scoring is the frozen Oracle, one global call per variant, plus per-clip for
the paired video bootstrap.  Nothing here touches the competition platform.
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import numpy as np  # noqa: E402

from data.manifest import read_manifest  # noqa: E402
from eval.oracle_runner import run_official_eval  # noqa: E402
from exp_bottom_trim_20260913 import cut_bottom, fmt, gt_bottom_for_top, parse_line  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYS_PY = "/Users/seyonmacbook/.workbuddy/binaries/python/envs/lane-oracle-py312/bin/python"
OFFICIAL_PY = SYS_PY

SRC = PROJECT_ROOT / "outputs/lvo_clrnet_r50_15ep_20260904/oof/predictions"
GT_DIR = PROJECT_ROOT / "outputs/a_gt_train_20260905/Lane/anno_txt"
MANIFEST = PROJECT_ROOT / "data/processed/manifest_train.jsonl"
WORK = PROJECT_ROOT / "outputs/exp_repair_geometry_20260915"


def read_lines(path: Path):
    if not path.is_file():
        return []
    out = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if raw:
            pts = parse_line(raw)
            if len(pts) >= 2:
                out.append(pts)
    return out


def sort_pts(pts):
    return sorted(pts, key=lambda p: p[1])


def subsample(pts, k):
    pts = sort_pts(pts)
    if len(pts) <= 2 * k:
        return pts
    keep = pts[::k]
    if keep[-1] != pts[-1]:
        keep.append(pts[-1])
    return keep


def movavg(pts, w):
    pts = sort_pts(pts)
    n = len(pts)
    if n < w:
        return pts
    xs = np.array([p[0] for p in pts])
    ys = np.array([p[1] for p in pts])
    kernel = np.ones(w) / w
    pad = w // 2
    padded = np.concatenate([np.repeat(xs[0], pad), xs, np.repeat(xs[-1], pad)])
    sm = np.convolve(padded, kernel, mode="valid")
    return [(float(a), float(b)) for a, b in zip(sm, ys)]


def polyrefit(pts, deg):
    pts = sort_pts(pts)
    if len(pts) <= deg:
        return pts
    xs = np.array([p[0] for p in pts])
    ys = np.array([p[1] for p in pts])
    coef = np.polyfit(ys, xs, deg)
    sm = np.polyval(coef, ys)
    return [(float(a), float(b)) for a, b in zip(sm, ys)]


def trim(pts, margin):
    top = min(p[1] for p in pts)
    out = cut_bottom(pts, gt_bottom_for_top(top) + margin)
    return out if len(out) >= 2 else []


VARIANTS = {
    "raw": lambda p: sort_pts(p),
    "trim0": lambda p: trim(p, 0),
    "trim40": lambda p: trim(p, 40),
    "sub2": lambda p: subsample(p, 2),
    "sub3": lambda p: subsample(p, 3),
    "sub4": lambda p: subsample(p, 4),
    "ma5": lambda p: movavg(p, 5),
    "ma9": lambda p: movavg(p, 9),
    "quad": lambda p: polyrefit(p, 2),
    "cubic": lambda p: polyrefit(p, 3),
    "trim0_sub4": lambda p: subsample(trim(p, 0), 4),
    "trim40_sub4": lambda p: subsample(trim(p, 40), 4),
}


def materialise(name, fn):
    dst = WORK / name
    # NOTE: no rmtree here on purpose -- every variant writes exactly the same
    # 7100 rel-path set, so overwriting is sufficient and keeps the run inside
    # the workspace bulk-delete guard.
    dst.mkdir(parents=True, exist_ok=True)
    n_lines = 0
    for src_file in SRC.rglob("*.lines.txt"):
        rel = src_file.relative_to(SRC)
        out_file = dst / rel
        out_file.parent.mkdir(parents=True, exist_ok=True)
        lanes = []
        for pts in read_lines(src_file):
            new = fn(pts)
            if new and len(new) >= 2:
                lanes.append(new)
                n_lines += 1
        out_file.write_text(
            "".join(fmt(l) + "\n" for l in lanes), encoding="utf-8")
    return n_lines


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default="", help="comma-separated variant names")
    ap.add_argument("--no-per-clip", action="store_true",
                    help="global call only; use for the scan, then re-run the "
                         "winner with per-clip for the paired video bootstrap")
    args = ap.parse_args()

    WORK.mkdir(parents=True, exist_ok=True)
    records = read_manifest(MANIFEST)
    names = [n for n in VARIANTS if not args.only or n in args.only.split(",")]

    results = {}
    for name in names:
        t0 = time.time()
        n = materialise(name, VARIANTS[name])
        res = run_official_eval(
            WORK / name, GT_DIR, MANIFEST,
            official_python=OFFICIAL_PY, per_clip=not args.no_per_clip,
            output_path=WORK / f"oracle_{name}.json",
        )
        g = res.global_
        results[name] = {
            "lanes": n, "TP": g.tp, "FP": g.fp, "FN": g.fn,
            "precision": g.precision, "recall": g.recall, "f1": g.f1,
            "seconds": round(time.time() - t0, 1),
        }
        print(f"{name:12s} lanes={n:6d} F1={g.f1:.6f} P={g.precision:.4f} "
              f"R={g.recall:.4f} TP={g.tp} FP={g.fp} FN={g.fn} "
              f"({results[name]['seconds']}s)", flush=True)
        (WORK / "results.json").write_text(
            json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    base = results.get("raw", {}).get("f1")
    if base:
        print("\n=== delta vs raw (pp) ===")
        for name, r in sorted(results.items(), key=lambda kv: -kv[1]["f1"]):
            print(f"{name:12s} {(r['f1'] - base) * 100:+.4f} pp")


if __name__ == "__main__":
    main()
