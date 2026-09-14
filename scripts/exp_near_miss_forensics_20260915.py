#!/usr/bin/env python3
"""Near-miss forensics on the honest LVO OOF (2026-09-15 night shift).

Why this exists
---------------
The official metric is globally pooled ``F1 = 2*TP/(P+G)``.  Differentiate it
and each elementary edit has a fixed price (testA scale, P+G = 5819.8):

  * DELETE a lane that is false        -> dF1 = +2*theta/(P+G)      = +0.0126pp
  * ADD    a lane that is true         -> dF1 = +2*(1-theta)/(P+G)  = +0.0217pp
  * REPAIR a false lane into a true one-> dF1 = +2/(P+G)            = +0.0344pp

with ``theta = TP/(P+G) = F1/2 = 0.3675``.  So repairing an existing near-miss
is worth 1.58x a fresh true lane and 2.7x a correct deletion -- the same order
of magnitude, not the order-of-magnitude gap a naive reading suggests.  What
makes repair attractive is not the per-lane price but the *volume*: on this OOF
2739 lanes sit in the repairable band versus ~100 that any consensus gate can
find.  Every axis explored so far (consensus union, conf lowering, support
union) has been an *adding* axis and all of them lost.  Before spending any
more of the six B-board shots on adding lanes, this script measures how much
mass is actually sitting in the repairable band.

Concretely, per image it reproduces the frozen official scorer exactly
(spline interpolation -> 30px stroke masks -> Hungarian assignment), then
splits every predicted lane by its best IoU against GT and decomposes each
near-miss into its two orthogonal causes:

  span mismatch    IoU_span    = |y_pred & y_gt| / |y_pred | y_gt|
  lateral mismatch IoU_lateral = (30 - d) / (30 + d), d = mean |dx| in the
                                 common y band

A near-miss whose IoU_span is the binding term is a *bottom/top overshoot*
problem; one whose IoU_lateral binds is a *positioning* problem.  The two want
opposite fixes, so the split decides which one is worth building.

Frozen dependencies: ``src/eval/official_oracle/score.py`` is imported
read-only via importlib (never modified, no .pyc dropped next to it).

Run with the Oracle interpreter so rasterisation matches the official env
(numpy 2.1.3 / cv2 4.12.0)::

    ORACLE_PY scripts/exp_near_miss_forensics_20260915.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.manifest import read_manifest  # noqa: E402

ORACLE_PY = "/Users/seyonmacbook/.workbuddy/binaries/python/envs/lane-oracle-py312/bin/python"

GT_DIR = PROJECT_ROOT / "outputs/a_gt_train_20260905/Lane/anno_txt"
MANIFEST = PROJECT_ROOT / "data/processed/manifest_train.jsonl"
WORK = PROJECT_ROOT / "outputs/exp_near_miss_forensics_20260915"

# honest leave-one-video-out trees; the 36ep one is the closest lineage to the
# 54ep production model, so it is the default.
TREES = {
    "36ep_final": "outputs/lvo_clrnet_r50_36ep_c_export_20260906/final/predictions",
    "36ep_mid": "outputs/lvo_clrnet_r50_36ep_c_export_20260906/midpoint/predictions",
    "15ep": "outputs/lvo_clrnet_r50_15ep_20260904/oof/predictions",
}

IOU_THR = 0.5
NEAR_LO = 0.15  # below this the pair is a plain hallucination, not a near-miss
WIDTH = 30
CX = 683.0   # image horizontal centre (1366 / 2 = 683)
YMID = 360.0  # image vertical centre (720 / 2)

NEAR_BANDS = [(0.15, 0.25), (0.25, 0.35), (0.35, 0.45), (0.45, 0.5)]


def load_oracle():
    oracle_path = PROJECT_ROOT / "src" / "eval" / "official_oracle" / "score.py"
    spec = importlib.util.spec_from_file_location("official_oracle_score", str(oracle_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ORACLE = load_oracle()


def parse_line(text: str):
    toks = text.split()
    return [(float(toks[i]), float(toks[i + 1])) for i in range(0, len(toks) - 1, 2)]


def read_lines(path: Path):
    """Mirror ``parse_lines_txt``: drop consecutive duplicate points first.

    The official scorer calls ``remove_consecutive_duplicates`` before
    splprep, and splprep raises ``Invalid inputs`` on duplicated knots.  Skip
    (rather than raise) any lane that still fails, so one bad annotation
    cannot abort a 7100-image sweep.
    """
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
    """Interpolate, silently dropping lanes splprep refuses."""
    out = []
    for pts in lanes:
        try:
            out.append(ORACLE.interp_lane(pts))
        except Exception:
            out.append(None)
    return out


def x_at_y(pts, yq):
    """Linear interpolation of x(y); pts sorted by y ascending."""
    ys = np.array([p[1] for p in pts], dtype=np.float64)
    xs = np.array([p[0] for p in pts], dtype=np.float64)
    order = np.argsort(ys)
    ys, xs = ys[order], xs[order]
    return float(np.interp(yq, ys, xs))


def span_stats(pred, gt):
    py = [p[1] for p in pred]
    gy = [p[1] for p in gt]
    p0, p1 = min(py), max(py)
    g0, g1 = min(gy), max(gy)
    inter = max(0.0, min(p1, g1) - max(p0, g0))
    union = max(p1, g1) - min(p0, g0)
    span_iou = inter / union if union > 0 else 0.0
    # lateral offset measured only on the common band (where IoU is contested)
    lo, hi = max(p0, g0), min(p1, g1)
    d = None
    if hi - lo > 8:
        ys = np.linspace(lo, hi, 9)
        dx = np.array([x_at_y(pred, y) - x_at_y(gt, y) for y in ys])
        d = float(np.mean(np.abs(dx)))
        d_signed = float(np.mean(dx))
    else:
        d_signed = None
    lat_iou = max(0.0, (WIDTH - d) / (WIDTH + d)) if d is not None else 0.0
    return span_iou, lat_iou, d, d_signed


BUCKETS = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
           (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.01)]


def bucket_of(v):
    for i, (lo, hi) in enumerate(BUCKETS):
        if lo <= v < hi:
            return i
    return len(BUCKETS) - 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tree", default="36ep_final", choices=sorted(TREES))
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--dump-near-miss", type=int, default=0, help="write N examples")
    args = ap.parse_args()

    pred_root = PROJECT_ROOT / TREES[args.tree]
    records = read_manifest(MANIFEST)
    if args.max_images:
        records = records[: args.max_images]

    WORK.mkdir(parents=True, exist_ok=True)

    tp = fp = fn = 0
    pred_best_hist = np.zeros(len(BUCKETS), dtype=np.int64)
    gt_best_hist = np.zeros(len(BUCKETS), dtype=np.int64)
    near = []          # list of dicts, capped
    fragile_tp = 0     # matched pairs with 0.5 < iou <= 0.6
    # sufficient statistics for the global-bias regression
    S1 = S2 = S12 = B1 = B2 = 0.0
    NS = 0
    SD = 0.0
    SD2 = 0.0
    n_img = 0
    n_pred = 0
    n_gt = 0
    dropped_p = 0
    dropped_g = 0
    t0 = time.time()

    for idx, rec in enumerate(records):
        preds = read_lines(pred_root / rec.pred_rel_path)
        gts = read_lines(GT_DIR / rec.pred_rel_path)
        n_img += 1
        n_pred += len(preds)
        n_gt += len(gts)
        if not preds or not gts:
            tp += 0
            fp += len(preds)
            fn += len(gts)
            for _ in preds:
                pred_best_hist[0] += 1
            for _ in gts:
                gt_best_hist[0] += 1
            continue

        ip = safe_interp(preds)
        ig = safe_interp(gts)
        # a lane splprep refuses is dropped from the comparison entirely --
        # it is rare and affects pred and gt sides symmetrically.
        keep_p = [i for i, a in enumerate(ip) if a is not None]
        keep_g = [j for j, a in enumerate(ig) if a is not None]
        if len(keep_p) != len(preds) or len(keep_g) != len(gts):
            dropped_p += len(preds) - len(keep_p)
            dropped_g += len(gts) - len(keep_g)
            preds = [preds[i] for i in keep_p]
            gts = [gts[j] for j in keep_g]
            ip = [ip[i] for i in keep_p]
            ig = [ig[j] for j in keep_g]
        if not preds or not gts:
            fp += len(preds)
            fn += len(gts)
            for _ in preds:
                pred_best_hist[0] += 1
            for _ in gts:
                gt_best_hist[0] += 1
            continue
        pm = [ORACLE.draw_lane_mask(a, WIDTH) for a in ip]
        gm = [ORACLE.draw_lane_mask(a, WIDTH) for a in ig]
        ious = np.zeros((len(preds), len(gts)))
        for i, a in enumerate(pm):
            for j, b in enumerate(gm):
                u = (a | b).sum()
                if u:
                    ious[i, j] = (a & b).sum() / u

        ri, ci = linear_sum_assignment(1 - ious)
        matched = ious[ri, ci] > IOU_THR
        n_tp = int(matched.sum())
        tp += n_tp
        fp += len(preds) - n_tp
        fn += len(gts) - n_tp

        pred_best = ious.max(axis=1) if len(gts) else np.zeros(len(preds))
        gt_best = ious.max(axis=0) if len(preds) else np.zeros(len(gts))
        for v in pred_best:
            pred_best_hist[bucket_of(float(v))] += 1
        for v in gt_best:
            gt_best_hist[bucket_of(float(v))] += 1

        for i, j in zip(ri, ci):
            v = float(ious[i, j])
            if v > IOU_THR:
                if v <= 0.6:
                    fragile_tp += 1
                continue
            if v <= NEAR_LO:
                continue
            s_iou, l_iou, d, d_signed = span_stats(preds[i], gts[j])
            near.append({
                "image": rec.image_id, "iou": round(v, 4),
                "span_iou": round(s_iou, 4), "lat_iou": round(l_iou, 4),
                "d": None if d is None else round(d, 2),
                "d_signed": None if d_signed is None else round(d_signed, 2),
            })
            # ---- residual structure: is there a GLOBAL correctable bias? ----
            # Sample dx = x_gt - x_pred along the common band. Two hypotheses:
            #   scale:  x_gt - cx ~ k * (x_pred - cx)      (k != 1 -> expand/shrink)
            #   drift:  dx ~ c0 + c1 * (y - ymid)          (rotation / slope error)
            # Accumulate the sufficient statistics; ordinary least squares with
            # two regressors is solved once at the end.
            py = [p[1] for p in preds[i]]
            gy = [p[1] for p in gts[j]]
            lo, hi = max(min(py), min(gy)), min(max(py), max(gy))
            if hi - lo > 8:
                for y in np.linspace(lo, hi, 7):
                    xp = x_at_y(preds[i], y)
                    xg = x_at_y(gts[j], y)
                    dx = xg - xp
                    u1 = xp - CX
                    u2 = y - YMID
                    S1 += u1 * u1
                    S2 += u2 * u2
                    S12 += u1 * u2
                    B1 += u1 * dx
                    B2 += u2 * dx
                    NS += 1
                    SD += dx
                    SD2 += dx * dx

        if (idx + 1) % 500 == 0:
            el = time.time() - t0
            print(f"  {idx+1}/{len(records)} tp={tp} fp={fp} fn={fn} "
                  f"near={len(near)} {el:.0f}s", flush=True)

    P = tp + fp
    G = tp + fn
    f1 = 2 * tp / (P + G) if P + G else 0.0
    prec = tp / P if P else 0.0
    rec_ = tp / G if G else 0.0

    # repairable mass, split by what binds the IoU
    n_near = len(near)
    span_bound = sum(1 for r in near if r["span_iou"] < r["lat_iou"])
    lat_bound = n_near - span_bound
    unit = 2.0 / (P + G)

    result = {
        "tree": args.tree,
        "images": n_img,
        "preds": n_pred, "gts": n_gt,
        "TP": tp, "FP": fp, "FN": fn,
        "P": P, "G": G, "F1": f1, "precision": prec, "recall": rec_,
        "pred_best_iou_hist": {f"{lo:.1f}-{hi:.1f}": int(c)
                               for (lo, hi), c in zip(BUCKETS, pred_best_hist)},
        "gt_best_iou_hist": {f"{lo:.1f}-{hi:.1f}": int(c)
                             for (lo, hi), c in zip(BUCKETS, gt_best_hist)},
        "near_miss_pairs": n_near,
        "near_miss_span_bound": span_bound,
        "near_miss_lateral_bound": lat_bound,
        "fragile_tp_05_06": fragile_tp,
        "dropped_pred_lanes": dropped_p,
        "dropped_gt_lanes": dropped_g,
        "unit_dF1_pp_per_repaired_lane": unit * 100,
        "repair_all_near_miss_dF1_pp": n_near * unit * 100,
        "near_miss_span_iou_median": float(np.median([r["span_iou"] for r in near])) if near else None,
        "near_miss_lat_iou_median": float(np.median([r["lat_iou"] for r in near])) if near else None,
        "near_miss_d_abs_median": float(np.median([r["d"] for r in near if r["d"] is not None])) if near else None,
        "near_miss_d_signed_mean": float(np.mean([r["d_signed"] for r in near if r["d_signed"] is not None])) if near else None,
        "elapsed_s": round(time.time() - t0, 1),
    }

    # ---- per-band decomposition: which factor binds at each IoU level? ----
    band_stats = []
    for lo, hi in NEAR_BANDS:
        sel = [r for r in near if lo <= r["iou"] < hi]
        if not sel:
            continue
        sp = [r["span_iou"] for r in sel]
        lt = [r["lat_iou"] for r in sel]
        band_stats.append({
            "band": f"{lo:.2f}-{hi:.2f}",
            "n": len(sel),
            "span_iou_median": round(float(np.median(sp)), 4),
            "lat_iou_median": round(float(np.median(lt)), 4),
            "span_bound_frac": round(sum(1 for r in sel if r["span_iou"] < r["lat_iou"]) / len(sel), 3),
            "d_abs_median": round(float(np.median([r["d"] for r in sel if r["d"] is not None])), 2)
            if any(r["d"] is not None for r in sel) else None,
            # if span were perfect the pair would score ~lat_iou; how many of
            # them would then clear 0.5?  This is the size of the span prize.
            "n_span_fix_would_clear": sum(1 for r in sel if r["lat_iou"] > 0.55),
            "n_lat_fix_would_clear": sum(1 for r in sel if r["span_iou"] > 0.55),
        })
    result["near_bands"] = band_stats

    # ---- global bias regression: dx = c1*(x-cx) + c2*(y-ymid) ----
    if NS > 100:
        A = np.array([[S1, S12], [S12, S2]], dtype=np.float64)
        b = np.array([B1, B2], dtype=np.float64)
        try:
            c1, c2 = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            c1 = c2 = float("nan")
        # marginal (single-regressor) slopes, which is what a one-knob fix needs
        k_scale = B1 / S1 if S1 > 0 else float("nan")
        k_drift = B2 / S2 if S2 > 0 else float("nan")
        mean_dx = SD / NS
        rms_dx = (SD2 / NS) ** 0.5
        result["global_bias"] = {
            "samples": NS,
            "mean_dx_gt_minus_pred": round(mean_dx, 3),
            "rms_dx": round(rms_dx, 3),
            # k_scale > 0 means GT sits further out than the prediction at the
            # same y => predictions are pulled toward the image centre.
            "k_scale_per_px_from_centre": round(float(k_scale), 6),
            "implied_correction_at_400px": round(float(k_scale) * 400.0, 2),
            "k_drift_per_px_down": round(float(k_drift), 6),
            "joint_c1_scale": round(float(c1), 6),
            "joint_c2_drift": round(float(c2), 6),
        }

    (WORK / f"forensics_{args.tree}.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.dump_near_miss:
        (WORK / f"near_miss_samples_{args.tree}.json").write_text(
            json.dumps(near[: args.dump_near_miss], indent=2, ensure_ascii=False),
            encoding="utf-8")

    print()
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
