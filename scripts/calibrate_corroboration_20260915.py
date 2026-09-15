#!/usr/bin/env python3
"""Calibrate the pseudo-GT corroboration index against the 2026-09-15 readings.

Three A-board shots returned real numbers today:

    probe_g4  (incumbent + 84 lanes, nothing removed)   0.73609   (+0.035pp)
    soupB     (215 added / 51 dropped)                  0.73195   (-0.379pp)
    cut400    (a different backbone)                    0.72802   (-0.772pp)

probe_g4 is a *pure addition*, so its score inverts cleanly to a true rate for
the lanes the 40%-agreement gate adds:

    r = theta + dF1 / (u * n),  u = 2/(P+G), theta = F1/2
      = 0.3679 + 0.00035 / (3.4365e-04 * 84) = 0.3800

That, plus the 20%-agreement point already measured on 9/14 (r = 0.3414), gives
the testA agreement->true-rate curve and shows a looser gate is not worth using.

This script pushes the same idea one step further. The corroboration rate
(fraction of a lane set that coincides with a >=7/9 support-tree cluster) was
already computed for every candidate, but its absolute calibration failed
because the pseudo-GT carries false lanes of its own. Now there are two lane
sets whose true rate is independently known from a score, so the corroboration
index can be *fitted* rather than assumed:

    r  ~=  a + b * corroboration

and then applied to lane sets we never scored -- notably the 69 lanes that the
conf0.55 threshold removes, which decides whether that candidate (and a
conf0.55 + consensus combination) is worth a B-board slot.

Everything here is an ESTIMATE built on a two-point fit. It is reported as such.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
import zipfile
from pathlib import Path

sys.dont_write_bytecode = True
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402

WIDTH = 30
IOU_THR = 0.5
MIN_TREES = 7
U = 2.0 / 5819.8
THETA = 0.73574 / 2

WORK = PROJECT_ROOT / "outputs/exp_calibrated_corroboration_20260915"

SUPPORTS = [
    "outputs/testA_hires/testA_hires_conf0.40/testA/predictions",
    "outputs/testA_support_trees/t05_36ep/testA/predictions",
    "outputs/testA_support_trees/seed202_36ep/testA/predictions",
    "outputs/testA_support_trees/seed303_36ep/testA/predictions",
    "outputs/testA_support_trees/seed101_36ep/testA/predictions",
    "outputs/testA_full71_infer_20260909/all71_seed42_clrnet_r50_36ep/testA/predictions",
    "outputs/testA_full71_infer_20260909/all71_seed43_clrnet_r50_36ep/testA/predictions",
    "outputs/testA_full71_infer_20260909/all71_seed44_clrnet_r50_36ep/testA/predictions",
    "outputs/testA_full71_infer_20260909/clrernet_r50_15ep/testA/predictions",
]

INCUMBENT = "outputs/submit_testA_54ep_trim0.zip"
CONF55 = "outputs/submit_testA_night_conf55_m0.zip"
PACKAGES = {
    "incumbent (reference)": INCUMBENT,
    "soupB novel": "outputs/submit_testA_night_soupB_m0.zip",
    "probe_g4 novel": "outputs/submit_testA_night_probe_g4.zip",
    "gate6 novel": "outputs/submit_testA_night_cons_gate6_v2_m0.zip",
    "swa4 novel": "outputs/submit_testA_night_swa4_54ep_m0.zip",
    "uni novel": "outputs/submit_testA_night_uni_swa4_g6_m0.zip",
}

# true rates recovered from today's scores (probe_g4 is a pure addition; soupB's
# rate depends on what its 51 dropped lanes were worth -- 0.45 is the mild
# assumption and the one used for the fit)
MEASURED = {"probe_g4 novel": 0.3800, "soupB novel": 0.3361}


def load_oracle():
    p = PROJECT_ROOT / "src" / "eval" / "official_oracle" / "score.py"
    spec = importlib.util.spec_from_file_location("official_oracle_score", str(p))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ORACLE = load_oracle()


def read_tree(path: Path):
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
                toks = raw.split()
                pts = [(float(toks[i]), float(toks[i + 1])) for i in range(0, len(toks) - 1, 2)]
                if len(pts) >= 2:
                    lanes.append(pts)
            out[rel] = lanes
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
    WORK.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    roots = [PROJECT_ROOT / p for p in SUPPORTS]
    pseudo = {}
    rels = sorted(str(p.relative_to(roots[0])) for p in roots[0].rglob("*.lines.txt"))
    for idx, rel in enumerate(rels):
        lanes, owners = [], []
        for ti, root in enumerate(roots):
            f = root / rel
            if f.is_file():
                for lane in read_tree(f):
                    lanes.append(lane)
                    owners.append(ti)
        clusters = []
        for oi, lane in enumerate(lanes):
            m = mask_of(lane)
            if m is None:
                continue
            hit = None
            for c in clusters:
                if iou(c["mask"], m) > IOU_THR:
                    hit = c
                    break
            if hit is None:
                clusters.append({"mask": m, "trees": {owners[oi]}})
            else:
                hit["trees"].add(owners[oi])
        pseudo[rel] = [c["mask"] for c in clusters if len(c["trees"]) >= MIN_TREES]
        if (idx + 1) % 300 == 0:
            print(f"  pseudo-GT {idx+1}/{len(rels)} {time.time()-t0:.0f}s", flush=True)
    print(f"pseudo-GT: {sum(len(v) for v in pseudo.values())} lanes")

    def corroboration(masks_by_rel):
        hit = tot = 0
        for rel, masks in masks_by_rel.items():
            g = pseudo.get(rel, [])
            for m in masks:
                tot += 1
                if any(iou(m, x) > IOU_THR for x in g):
                    hit += 1
        return (hit / tot if tot else 0.0), hit, tot

    def novel_of(zp):
        cand = read_zip(PROJECT_ROOT / zp)
        inc = read_zip(PROJECT_ROOT / INCUMBENT)
        out = {}
        for rel, lanes in cand.items():
            bm = [m for m in (mask_of(l) for l in inc.get(rel, [])) if m is not None]
            keep = []
            for lane in lanes:
                m = mask_of(lane)
                if m is None or any(iou(b, m) > IOU_THR for b in bm):
                    continue
                keep.append(m)
            if keep:
                out[rel] = keep
        return out

    def removed_of(zp):
        """incumbent lanes that the candidate no longer reproduces"""
        inc = read_zip(PROJECT_ROOT / INCUMBENT)
        cand = read_zip(PROJECT_ROOT / zp)
        out = {}
        for rel, lanes in inc.items():
            cm = [m for m in (mask_of(l) for l in cand.get(rel, [])) if m is not None]
            keep = []
            for lane in lanes:
                m = mask_of(lane)
                if m is None or any(iou(c, m) > IOU_THR for c in cm):
                    continue
                keep.append(m)
            if keep:
                out[rel] = keep
        return out

    results = {}
    idx_sets = {}
    for name, zp in PACKAGES.items():
        if name == "incumbent (reference)":
            masks = {}
            inc = read_zip(PROJECT_ROOT / INCUMBENT)
            for rel, lanes in inc.items():
                ms = [m for m in (mask_of(l) for l in lanes) if m is not None]
                if ms:
                    masks[rel] = ms
        else:
            masks = novel_of(zp)
        c, hit, tot = corroboration(masks)
        idx_sets[name] = (c, hit, tot)
        results[name] = {"corroboration": round(c, 4), "n": tot, "hit": hit}
        print(f"{name:24s} n={tot:5d} corr={c:.4f}")

    # conf0.55's removed set -- the only set whose fate decides a B slot
    rm = removed_of(CONF55)
    c_rm, hit_rm, tot_rm = corroboration(rm)
    idx_sets["conf55 removed"] = (c_rm, hit_rm, tot_rm)
    results["conf55 removed"] = {"corroboration": round(c_rm, 4), "n": tot_rm, "hit": hit_rm}
    print(f"{'conf55 removed':24s} n={tot_rm:5d} corr={c_rm:.4f}")

    # ---- two-point fit and predictions -------------------------------------
    pts = [(idx_sets[k][0], v) for k, v in MEASURED.items()]
    (x1, y1), (x2, y2) = pts
    b = (y2 - y1) / (x2 - x1)
    a = y1 - b * x1
    print(f"\nfit: r = {a:.4f} + {b:.4f} * corroboration   (anchors {pts})")

    print("\n=== estimates (ESTIMATE, not measurement) ===")
    print(f"{'set':24s}{'corr':>7}{'r_est':>8}{'n':>7}{'dF1 est':>11}")
    est = {}
    for name, (c, hit, tot) in idx_sets.items():
        if name == "incumbent (reference)":
            r = a + b * c
            est[name] = {"r_est": round(r, 4), "corroboration": round(c, 4), "n": tot}
            print(f"{name:24s}{c:>7.3f}{r:>8.4f}{tot:>7}{'—':>11}")
            continue
        r = a + b * c
        if name == "conf55 removed":
            dF1 = U * tot * (THETA - r) * 100      # deletion
        else:
            dF1 = U * tot * (r - THETA) * 100      # addition
        est[name] = {"r_est": round(r, 4), "corroboration": round(c, 4),
                     "n": tot, "dF1_pp_est": round(dF1, 4),
                     "note": "deletion" if name == "conf55 removed" else "addition"}
        print(f"{name:24s}{c:>7.3f}{r:>8.4f}{tot:>7}{dF1:>+11.3f}")

    (WORK / "calibrated.json").write_text(json.dumps({
        "fit": {"a": round(a, 6), "b": round(b, 6), "anchors": MEASURED},
        "theta": round(THETA, 5), "u": U,
        "index": {k: {"corroboration": round(v[0], 4), "hit": v[1], "n": v[2]}
                  for k, v in idx_sets.items()},
        "estimates": est,
        "probe_g4_true_rate": 0.3800,
        "agreement_curve_testA": {"20%": 0.3414, "40%": 0.3800},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {WORK / 'calibrated.json'}")


if __name__ == "__main__":
    main()
