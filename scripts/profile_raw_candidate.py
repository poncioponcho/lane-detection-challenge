#!/usr/bin/env python3
"""Profile a *raw* prediction tree against the incumbent submission pack.

The 2026-09-15 night profiling script only reads packaged zips. New models
arrive as raw trees (``<dir>/testA/predictions/<clip>/<frame>.lines.txt``), so
this wrapper trims them with the production rule and then reuses the same
containment ledger:

  kept      incumbent lanes reproduced at IoU > 0.5
  dropped   incumbent lanes the candidate no longer has
  novel     candidate lanes that explain no incumbent lane

Economics (F1 = 2TP/(P+G), theta = F1/2 = 0.3675 on testA, P+G = 5819.8):
  deleting n lanes at true rate r pays only if r < theta
  adding   n lanes at true rate r pays only if r > theta

Usage:
  python scripts/profile_raw_candidate.py --src <rawdir> --name occlude [--margin 0]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import zipfile
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

from exp_bottom_trim_20260913 import build_variant  # noqa: E402

WIDTH = 30
IOU_THR = 0.5
THETA = 0.3675
P_PLUS_G = 5819.8


def load_oracle():
    p = ROOT / "src" / "eval" / "official_oracle" / "score.py"
    spec = importlib.util.spec_from_file_location("official_oracle_score", str(p))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ORACLE = load_oracle()


def read_zip(zip_path: Path):
    out = {}
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not name.endswith(".lines.txt"):
                continue
            rel = name.split("submit/", 1)[-1]
            lanes = []
            for raw in zf.read(name).decode("utf-8").splitlines():
                raw = raw.strip()
                if not raw:
                    continue
                t = raw.split()
                pts = [(float(t[i]), float(t[i + 1])) for i in range(0, len(t) - 1, 2)]
                if len(pts) >= 2:
                    lanes.append(pts)
            out[rel] = lanes
    return out


def read_tree(root: Path):
    out = {}
    for f in sorted(root.rglob("*.lines.txt")):
        rel = f"{f.parent.name}/{f.name}"
        lanes = []
        for raw in f.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            t = raw.split()
            pts = [(float(t[i]), float(t[i + 1])) for i in range(0, len(t) - 1, 2)]
            if len(pts) >= 2:
                lanes.append(pts)
        out[rel] = lanes
    return out


def geom_stats(all_lanes):
    tops, spans, bottoms = [], [], []
    for lanes in all_lanes.values():
        for pts in lanes:
            ys = [p[1] for p in pts]
            tops.append(min(ys))
            bottoms.append(max(ys))
            spans.append(max(ys) - min(ys))
    t, s, b = np.array(tops), np.array(spans), np.array(bottoms)
    far = float((t < 450).mean())
    return {
        "lanes": len(t),
        "empty_images": sum(1 for v in all_lanes.values() if not v),
        "top_p5_p50_p95": [round(float(np.percentile(t, q)), 1) for q in (5, 50, 95)],
        "span_p5_p50_p95": [round(float(np.percentile(s, q)), 1) for q in (5, 50, 95)],
        "bottom_eq_719_pct": round(100.0 * float((b >= 718.5).mean()), 1),
        "f_lt_450_pct": round(100.0 * far, 2),
    }


def masks_of(lanes):
    out = []
    for pts in lanes:
        try:
            out.append(ORACLE.draw_lane_mask(ORACLE.interp_lane(pts), WIDTH))
        except Exception:
            out.append(None)
    return out


def containment(base, cand):
    kept = dropped = novel = 0
    for rel, bl in base.items():
        if not bl:
            continue
        cl = cand.get(rel, [])
        bm, cm = masks_of(bl), masks_of(cl)
        used = set()
        for a in bm:
            if a is None:
                dropped += 1
                continue
            best, best_j = 0.0, -1
            for j, b in enumerate(cm):
                if b is None or j in used:
                    continue
                u = (a | b).sum()
                if u:
                    v = (a & b).sum() / u
                    if v > best:
                        best, best_j = v, j
            if best > IOU_THR:
                kept += 1
                used.add(best_j)
            else:
                dropped += 1
    for rel, cl in cand.items():
        if not cl:
            continue
        bl = base.get(rel, [])
        if not bl:
            novel += len(cl)
            continue
        bm = masks_of(bl)
        for b in masks_of(cl):
            if b is None:
                continue
            best = 0.0
            for a in bm:
                if a is None:
                    continue
                u = (a | b).sum()
                if u:
                    best = max(best, (a & b).sum() / u)
            if best <= IOU_THR:
                novel += 1
    return kept, dropped, novel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="raw tree: <dir>/testA/predictions")
    ap.add_argument("--name", required=True)
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--no-trim", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    src = Path(args.src)
    if not src.is_dir():
        sys.exit(f"src missing: {src}")

    tree = read_tree(src)
    if not args.no_trim:
        dst = ROOT / "outputs/testA_night_20260915" / f"build_{args.name}_m{int(args.margin)}"
        if dst.exists():
            dst = ROOT / "outputs/testA_night_20260915" / (
                f"build_{args.name}_m{int(args.margin)}_{int(time.time())}")
        build_variant(src, dst, "gt_cond", args.margin)
        tree = read_tree(dst)

    incumbent = ROOT / "outputs/submit_testA_54ep_trim0.zip"
    base = read_zip(incumbent)
    g = geom_stats(tree)
    kept, dropped, novel = containment(base, tree)
    n_base = sum(len(v) for v in base.values())

    # break-even true rates implied by the economics
    r_del = THETA  # deleting pays if dropped true rate < theta
    r_add = THETA  # adding pays if novel true rate > theta
    print(f"incumbent lanes={n_base}")
    print(f"{args.name}: lanes={g['lanes']} empty={g['empty_images']} "
          f"top_p50={g['top_p5_p50_p95'][1]} span_p50={g['span_p5_p50_p95'][1]} "
          f"f<450={g['f_lt_450_pct']}% bottom719={g['bottom_eq_719_pct']}%")
    print(f"  kept={kept} dropped={dropped} novel={novel} "
          f"kept_pct={round(100.0*kept/max(1,kept+dropped),2)} delta_lanes={g['lanes']-n_base}")
    print(f"  break-even: dropped true rate must be < {r_del:.4f}; "
          f"novel true rate must be > {r_add:.4f}")

    outp = Path(args.out) if args.out else (
        ROOT / "outputs/testA_night_20260915" / f"profile_{args.name}.json")
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps({"name": args.name, "src": str(src), "margin": args.margin,
                                "geom": g,
                                "vs_incumbent": {"kept": kept, "dropped": dropped,
                                                 "novel": novel, "base_lanes": n_base}},
                               indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  wrote {outp}")


if __name__ == "__main__":
    main()
