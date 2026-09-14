#!/usr/bin/env python3
"""Price each candidate's ADDED lanes against a calibrated pseudo ground truth.

Why this exists
---------------
testA has no ground truth and the A board is closed, so every 2026-09-15
candidate is unpriced.  The containment ledger says *how many* lanes a
candidate adds or drops but not whether the added ones are real, and the whole
decision hinges on that: adding pays only above theta = F1/2 = 0.3675 true.

Nine independent support trees are complete on testA (900 files each;
``clrernet_36ep`` is truncated to 158 and is excluded).  Lanes that seven or
more of those nine agree on are almost certainly real, so they form a
high-precision (low-recall) pseudo ground truth.  Its recall is calibrated with
the one hard number we do have: the incumbent pack's true rate of 0.803,
recovered earlier by the junk-injection probe.  The incumbent's own lanes are
model output, not support consensus, so using them for calibration is not
circular.

    pseudo_recall = P(incumbent lane matches pseudo-GT) / 0.803
    r_est(added)  = P(added lane matches pseudo-GT) / pseudo_recall

CIRCULARITY WARNING, stated up front: lanes that were *selected by* support
agreement (gate6, and the union through it) match a support-derived pseudo-GT
by construction, so their r_est is an inflated upper bound.  Lanes that come
from a different model's weights (swa*, soup*) are not support-selected and are
priced fairly.  That split is exactly the one that matters, because the
soups are the biggest bet and the one the ledger could not price.
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
MIN_TREES = 7          # of 9 -> 78% agreement
INCUMBENT_TRUE_RATE = 0.803   # from the junk-injection probe, 2026-09-13
P_PLUS_G = 5819.8             # incumbent P + testA G
THETA = 0.3675                # F1/2

WORK = PROJECT_ROOT / "outputs/exp_pseudo_gt_pricing_20260915"

SUPPORTS = [
    ("hires_c40", "outputs/testA_hires/testA_hires_conf0.40/testA/predictions"),
    ("t05_36ep", "outputs/testA_support_trees/t05_36ep/testA/predictions"),
    ("seed202_36ep", "outputs/testA_support_trees/seed202_36ep/testA/predictions"),
    ("seed303_36ep", "outputs/testA_support_trees/seed303_36ep/testA/predictions"),
    ("seed101_36ep", "outputs/testA_support_trees/seed101_36ep/testA/predictions"),
    ("seed42_36ep", "outputs/testA_full71_infer_20260909/all71_seed42_clrnet_r50_36ep/testA/predictions"),
    ("seed43_36ep", "outputs/testA_full71_infer_20260909/all71_seed43_clrnet_r50_36ep/testA/predictions"),
    ("seed44_36ep", "outputs/testA_full71_infer_20260909/all71_seed44_clrnet_r50_36ep/testA/predictions"),
    ("clrernet_15ep", "outputs/testA_full71_infer_20260909/clrernet_r50_15ep/testA/predictions"),
]

INCUMBENT_ZIP = PROJECT_ROOT / "outputs/submit_testA_54ep_trim0.zip"
CANDIDATES = [
    "outputs/submit_testA_night_uni_swa4_g6_m0.zip",
    "outputs/submit_testA_night_cons_gate6_v2_m0.zip",
    "outputs/submit_testA_night_swa4_54ep_m0.zip",
    "outputs/submit_testA_night_swa7_54ep_m0.zip",
    "outputs/submit_testA_night_swa3_54ep_m0.zip",
    "outputs/submit_testA_night_soupA_m0.zip",
    "outputs/submit_testA_night_soupB_m0.zip",
    "outputs/submit_testA_night_soupC_m0.zip",
]


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
    roots = [(n, PROJECT_ROOT / p) for n, p in SUPPORTS]
    missing = [n for n, p in roots if len(list(p.rglob("*.lines.txt"))) != 900]
    if missing:
        print(f"!! incomplete support trees (excluded): {missing}")
        roots = [(n, p) for n, p in roots if n not in missing]
    print(f"using {len(roots)} support trees, pseudo-GT needs >= {MIN_TREES}")

    # ---- build the pseudo ground truth -------------------------------------
    pseudo = {}
    for idx, rel in enumerate(sorted(
            str(p.relative_to(roots[0][1])) for p in roots[0][1].rglob("*.lines.txt"))):
        lanes, owners = [], []
        for ti, (_, root) in enumerate(roots):
            f = root / rel
            if not f.is_file():
                continue
            for lane in read_tree(f):
                lanes.append(lane)
                owners.append(ti)
        if not lanes:
            pseudo[rel] = []
            continue
        masks = [m for m in (mask_of(l) for l in lanes) if m is not None]
        clusters = []          # [{rep_mask, trees:set}]
        order = np.argsort([-max(p[1] for p in l) + min(p[1] for p in l) for l in lanes])
        for oi in order:
            m = mask_of(lanes[oi])
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
        if (idx + 1) % 200 == 0:
            print(f"  pseudo-GT {idx+1}/900 {time.time()-t0:.0f}s", flush=True)

    n_pseudo = sum(len(v) for v in pseudo.values())
    print(f"pseudo-GT lanes: {n_pseudo} ({n_pseudo/900:.2f} per image)")

    incumbent = read_zip(INCUMBENT_ZIP)

    def hit_rate(lanes_by_rel):
        hit = tot = 0
        for rel, lanes in lanes_by_rel.items():
            pm = pseudo.get(rel, [])
            if not lanes:
                continue
            for lane in lanes:
                m = mask_of(lane)
                if m is None:
                    continue
                tot += 1
                for g in pm:
                    if iou(m, g) > IOU_THR:
                        hit += 1
                        break
        return hit, tot

    inc_hit, inc_tot = hit_rate(incumbent)
    inc_rate = inc_hit / inc_tot
    pseudo_recall = inc_rate / INCUMBENT_TRUE_RATE
    print(f"\nincumbent: {inc_hit}/{inc_tot} = {inc_rate:.4f} match pseudo-GT "
          f"(true rate is known {INCUMBENT_TRUE_RATE})")
    print(f"=> pseudo-GT recall calibrated at {pseudo_recall:.4f}\n")

    results = {}
    for zp in CANDIDATES:
        path = PROJECT_ROOT / zp
        if not path.is_file():
            print(f"skip missing {zp}")
            continue
        cand = read_zip(path)
        # novel = candidate lanes that explain no incumbent lane
        novel = {}
        for rel, lanes in cand.items():
            base = incumbent.get(rel, [])
            bm = [m for m in (mask_of(l) for l in base) if m is not None]
            keep = []
            for lane in lanes:
                m = mask_of(lane)
                if m is None:
                    continue
                if any(iou(b, m) > IOU_THR for b in bm):
                    continue
                keep.append(m)
            if keep:
                novel[rel] = keep
        n_novel = sum(len(v) for v in novel.values())
        if n_novel == 0:
            results[Path(zp).name] = {"novel": 0}
            print(f"{Path(zp).name:48s} novel=0")
            continue
        hit = 0
        for rel, masks in novel.items():
            pm = pseudo.get(rel, [])
            for m in masks:
                for g in pm:
                    if iou(m, g) > IOU_THR:
                        hit += 1
                        break
        raw = hit / n_novel
        r_est = min(1.0, raw / pseudo_recall)
        dF1 = 2.0 / P_PLUS_G * n_novel * (r_est - THETA) * 100
        results[Path(zp).name] = {
            "novel": n_novel, "pseudo_hit": hit, "raw_hit_rate": round(raw, 4),
            "r_est": round(r_est, 4), "dF1_pp_est": round(dF1, 4),
        }
        print(f"{Path(zp).name:48s} novel={n_novel:4d} hit={hit:4d} "
              f"raw={raw:.3f} r_est={r_est:.3f} dF1~{dF1:+.3f}pp")

    (WORK / "pricing.json").write_text(json.dumps({
        "pseudo_gt_lanes": n_pseudo,
        "min_trees": MIN_TREES,
        "support_trees": [n for n, _ in roots],
        "incumbent_hit_rate": round(inc_rate, 4),
        "pseudo_recall": round(pseudo_recall, 4),
        "theta": THETA,
        "candidates": results,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {WORK / 'pricing.json'}")


if __name__ == "__main__":
    main()
