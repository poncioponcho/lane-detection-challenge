#!/usr/bin/env python3
"""Profile the 2026-09-15 night candidates against the incumbent pack.

testA has no ground truth, so a candidate cannot be *scored* offline -- but it
can be *compared*.  This script reuses the frozen official rasteriser to build,
for every candidate, the geometric profile plus a containment ledger against
the incumbent package (``submit_testA_54ep_trim0.zip``, 2664 lanes, F1 0.73574):

  kept      incumbent lanes reproduced at IoU > 0.5
  dropped   incumbent lanes the candidate no longer has
  novel     candidate lanes that explain no incumbent lane

The three numbers bound what the candidate can be.  A candidate that keeps
~100% and adds nothing is a pure re-localisation of the same lanes (its whole
delta comes from IoU moving across the 0.5 gate).  One that drops lanes is
making a *deletion* bet, which only pays if the dropped set is more than
1 - F1/2 = 63.2% false.  One that adds lanes is making an *addition* bet, which
only pays above F1/2 = 36.75% true.  Nothing here touches the platform.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import numpy as np  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NIGHT = PROJECT_ROOT / "outputs/testA_night_20260915"
WORK = PROJECT_ROOT / "outputs/exp_night_candidate_profile_20260915"
WIDTH = 30
IOU_THR = 0.5


def load_oracle():
    p = PROJECT_ROOT / "src" / "eval" / "official_oracle" / "score.py"
    spec = importlib.util.spec_from_file_location("official_oracle_score", str(p))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ORACLE = load_oracle()


def read_tree_lines(path: Path):
    out = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        toks = raw.split()
        pts = [(float(toks[i]), float(toks[i + 1])) for i in range(0, len(toks) - 1, 2)]
        clean = [pts[0]]
        for p in pts[1:]:
            if p != clean[-1]:
                clean.append(p)
        if len(clean) >= 2:
            out.append(clean)
    return out


def read_zip(zip_path: Path):
    """{rel: [lane, ...]} from a packaged submit zip (submit/<clip>/<frame>.lines.txt)."""
    out = {}
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not name.endswith(".lines.txt"):
                continue
            rel = name.split("submit/", 1)[-1]
            text = zf.read(name).decode("utf-8")
            lanes = []
            for raw in text.splitlines():
                raw = raw.strip()
                if not raw:
                    continue
                toks = raw.split()
                pts = [(float(toks[i]), float(toks[i + 1])) for i in range(0, len(toks) - 1, 2)]
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
    if not tops:
        return {}
    t = np.array(tops)
    s = np.array(spans)
    b = np.array(bottoms)
    return {
        "lanes": len(t),
        "empty_images": sum(1 for v in all_lanes.values() if not v),
        "top_p5_p50_p95": [round(float(np.percentile(t, q)), 1) for q in (5, 50, 95)],
        "span_p5_p50_p95": [round(float(np.percentile(s, q)), 1) for q in (5, 50, 95)],
        "bottom_eq_719_pct": round(100.0 * float((b >= 718.5).mean()), 1),
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
    """kept / dropped / novel, by official rasterised IoU."""
    kept = dropped = novel = 0
    for rel, bl in base.items():
        if not bl:
            continue
        cl = cand.get(rel, [])
        bm = masks_of(bl)
        cm = masks_of(cl)
        used = set()
        for i, a in enumerate(bm):
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
                    v = (a & b).sum() / u
                    best = max(best, v)
            if best <= IOU_THR:
                novel += 1
    return kept, dropped, novel


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    incumbent_zip = PROJECT_ROOT / "outputs/submit_testA_54ep_trim0.zip"
    base = read_zip(incumbent_zip)
    print(f"incumbent {incumbent_zip.name}: images={len(base)} "
          f"lanes={sum(len(v) for v in base.values())}")

    cands = {}
    # packaged candidates (already trimmed + canonicalised)
    for zp in sorted(PROJECT_ROOT.glob("outputs/submit_testA_night_*_m0.zip")):
        cands[zp.stem.replace("submit_testA_night_", "")] = read_zip(zp)
    for zp in sorted(PROJECT_ROOT.glob("outputs/submit_testA_night_conf*_m0.zip")):
        cands[zp.stem.replace("submit_testA_night_", "")] = read_zip(zp)

    results = {}
    for name, tree in cands.items():
        g = geom_stats(tree)
        kept, dropped, novel = containment(base, tree)
        n = g.get("lanes", 0)
        results[name] = {
            **g,
            "vs_incumbent": {
                "kept": kept, "dropped": dropped, "novel": novel,
                "kept_pct": round(100.0 * kept / max(1, kept + dropped), 2),
                "delta_lanes": n - sum(len(v) for v in base.values()),
            },
        }
        print(f"{name:22s} lanes={n:5d} kept={kept:5d} dropped={dropped:4d} "
              f"novel={novel:4d} top_p50={g.get('top_p5_p50_p95', [None]*3)[1]} "
              f"span_p50={g.get('span_p5_p50_p95', [None]*3)[1]}", flush=True)

    (WORK / "profile.json").write_text(
        json.dumps({"incumbent": geom_stats(base), "candidates": results},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {WORK / 'profile.json'}")


if __name__ == "__main__":
    main()
