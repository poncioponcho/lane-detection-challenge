#!/usr/bin/env python3
"""Price the 96 novel lanes of `uni_s101_54ep` (incumbent 2664 + s101_54ep novel).

Same machinery as `calibrate_corroboration_20260915.py`, but it only builds the
pseudo-GT on the image rels that actually carry novel lanes, which keeps the
cost at seconds instead of minutes.

Corroboration -> true rate fit (two A-board anchors, 2026-09-15):
    r = 0.2816 + 0.3083 * corroboration

Economics (F1 = 2TP/(P+G), theta = F1/2 on testA):
    dF1 = 2/(P+G) * n * (r - theta)
"""
from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path

sys.dont_write_bytecode = True
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

WIDTH = 30
IOU_THR = 0.5
MIN_TREES = 7
U = 2.0 / 5819.8
THETA = 0.73574 / 2

# only locally complete (900/900) support trees
SUPPORTS = [
    "outputs/testA_hires/testA_hires_conf0.40/testA/predictions",
    "outputs/testA_support_trees/t05_36ep/testA/predictions",
    "outputs/testA_support_trees/seed202_36ep/testA/predictions",
    "outputs/testA_support_trees/seed303_36ep/testA/predictions",
    "outputs/testA_support_trees/seed101_36ep/testA/predictions",
    "outputs/testA_full71_infer_20260909/all71_seed42_clrnet_r50_36ep/testA/predictions",
    "outputs/testA_full71_infer_20260909/all71_seed43_clrnet_r50_36ep/testA/predictions",
    "outputs/testA_full71_infer_20260909/all71_seed44_clrnet_r50_36ep/testA/predictions",
]

INCUMBENT = "outputs/submit_testA_54ep_trim0.zip"
UNION = "outputs/testA_night_20260915/build_uni_s101_54ep_m0"

A, B = 0.2816, 0.3083


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
        t = raw.split()
        pts = [(float(t[i]), float(t[i + 1])) for i in range(0, len(t) - 1, 2)]
        clean = [pts[0]]
        for p in pts[1:]:
            if p != clean[-1]:
                clean.append(p)
        if len(clean) >= 2:
            out.append(clean)
    return out


def read_zip(zip_path: Path):
    out = {}
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not name.endswith(".lines.txt"):
                continue
            rel = name.split("submit/", 1)[-1]
            out[rel] = read_lines_text(zf.read(name).decode("utf-8"))
    return out


def read_lines_text(text: str):
    out = []
    for raw in text.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        t = raw.split()
        pts = [(float(t[i]), float(t[i + 1])) for i in range(0, len(t) - 1, 2)]
        if len(pts) >= 2:
            out.append(pts)
    return out


def mask_of(lane):
    try:
        return ORACLE.draw_lane_mask(ORACLE.interp_lane(lane), WIDTH)
    except Exception:
        return None


def iou(a, b):
    u = (a | b).sum()
    return (a & b).sum() / u if u else 0.0


def main() -> None:
    inc = read_zip(PROJECT_ROOT / INCUMBENT)
    uroot = PROJECT_ROOT / UNION

    # novel = union lanes that no incumbent lane explains
    novel = {}
    for f in sorted(uroot.rglob("*.lines.txt")):
        rel = f"{f.parent.name}/{f.name}"
        bm = [m for m in (mask_of(l) for l in inc.get(rel, [])) if m is not None]
        keep = []
        for lane in read_lines(f):
            m = mask_of(lane)
            if m is None or any(iou(b, m) > IOU_THR for b in bm):
                continue
            keep.append(m)
        if keep:
            novel[rel] = keep
    n_novel = sum(len(v) for v in novel.values())
    print(f"novel lanes={n_novel} images={len(novel)}")

    # pseudo-GT restricted to those rels
    roots = [PROJECT_ROOT / p for p in SUPPORTS]
    hit = tot = 0
    for rel, masks in novel.items():
        lanes, owners = [], []
        for ti, root in enumerate(roots):
            f = root / rel
            if f.is_file():
                for lane in read_lines(f):
                    lanes.append(lane)
                    owners.append(ti)
        clusters = []
        for oi, lane in enumerate(lanes):
            m = mask_of(lane)
            if m is None:
                continue
            done = False
            for c in clusters:
                if iou(c["mask"], m) > IOU_THR:
                    c["trees"].add(owners[oi])
                    done = True
                    break
            if not done:
                clusters.append({"mask": m, "trees": {owners[oi]}})
        g = [c["mask"] for c in clusters if len(c["trees"]) >= MIN_TREES]
        for m in masks:
            tot += 1
            if any(iou(m, x) > IOU_THR for x in g):
                hit += 1

    corr = hit / tot if tot else 0.0
    r = A + B * corr
    df1 = U * tot * (r - THETA)
    print(f"corroboration={corr:.4f}  hit={hit}/{tot}")
    print(f"r_est = {A} + {B} * {corr:.4f} = {r:.4f}   (break-even {THETA:.4f})")
    print(f"dF1_est = {df1*100:+.4f} pp over {tot} added lanes")

    outp = PROJECT_ROOT / "outputs/testA_night_20260915/price_uni_s101_54ep.json"
    outp.write_text(json.dumps({
        "candidate": "uni_s101_54ep",
        "union_dir": UNION,
        "novel_lanes": n_novel,
        "corroboration": round(corr, 4),
        "r_est": round(r, 4),
        "break_even": round(THETA, 4),
        "dF1_est_pp": round(df1 * 100, 4),
        "n_support_trees": len(SUPPORTS),
    }, indent=2), encoding="utf-8")
    print(f"wrote {outp}")


if __name__ == "__main__":
    main()
