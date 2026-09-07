"""T12 self-check: prove the metric instrument matches the derived physics.

Run:  python tests/test_metric_selfcheck.py   (no pytest needed)

Expected identities (DECISIONS.md §9):
  * Two identical 30px strokes separated laterally by d px  =>  ~ IoU = (30-d)/(30+d)
      (cv2 thickness=30 paints an effective ~31px stroke, so measured IoU is
       slightly higher than the ideal rectangle model; all within 0.02 tol)
      d=5  -> ~0.7143  (measured 0.722)
      d=10 -> ~0.5000  (measured 0.512; real IoU=0.5 boundary is ~10.3px)
      d=15 -> ~0.3333  (measured 0.348)
  * F1 = 2 * TP / (P + G)
  * GT vs GT  ->  F1 = 1.0, IoU = 1.0
Tolerance: |measured - expected| < 0.02.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np
from common.types import Lane
from eval.matching import compute_f1, pair_iou  # noqa: E402

TOL = 0.02


def hlane(y: float, x0: float = 0.0, x1: float = 1366.0, n: int = 80):
    """A straight horizontal lane at row `y`, spanning the full canvas width."""
    xs = np.linspace(x0, x1, n, dtype=np.float32)
    ys = np.full(n, y, dtype=np.float32)
    return np.stack([xs, ys], axis=1)


def test_gt_vs_gt_is_one():
    lanes = {"img1": [hlane(360.0)]}
    res = compute_f1(lanes, lanes)
    assert abs(res["F1"] - 1.0) < 1e-6, f"F1(GT,GT)={res['F1']}"
    assert abs(pair_iou(hlane(360.0), hlane(360.0)) - 1.0) < 1e-6
    print(f"[ok] GT vs GT: F1={res['F1']:.4f} (expect 1.0000)")


def test_lateral_iou_physics():
    expect = {5: 0.7143, 10: 0.5000, 15: 0.3333}
    for d, exp in expect.items():
        iou = pair_iou(hlane(360.0 + d), hlane(360.0))
        assert abs(iou - exp) < TOL, f"shift {d}px: IoU={iou:.4f} expect~{exp}"
        print(f"[ok] lateral shift {d}px: IoU={iou:.4f} (expect {exp})")


def test_iou_boundary_at_10px():
    # NOTE: the idealized rectangle model gives IoU=0.5 exactly at d=10px, but
    # cv2's thickness=30 stroke is effectively ~31px wide, so the REAL IoU=0.5
    # boundary is ~10.3px. The official metric uses the same cv2 call, so this
    # replication is faithful: a 10px offset still counts as TP, an 11px does not.
    iou10 = pair_iou(hlane(370.0), hlane(360.0))   # 10px
    iou11 = pair_iou(hlane(371.0), hlane(360.0))   # 11px
    gt = {"img1": [hlane(360.0)]}
    r10 = compute_f1({"img1": [hlane(370.0)]}, gt)
    r11 = compute_f1({"img1": [hlane(371.0)]}, gt)
    assert r10["TP"] == 1, f"10px offset should still be TP, got TP={r10['TP']}"
    assert r11["TP"] == 0, f"11px offset should be FN, got TP={r11['TP']}"
    print(f"[ok] TP boundary ~10.3px: 10px->TP(IoU={iou10:.3f}), 11px->FN(IoU={iou11:.3f})")


def test_f1_identity_two_lanes():
    # 2 matched lanes -> F1 = 1.0
    pred = {"img1": [hlane(200.0), hlane(500.0)]}
    gt = {"img1": [hlane(200.0), hlane(500.0)]}
    res = compute_f1(pred, gt)
    assert abs(res["F1"] - 1.0) < 1e-6, res
    print(f"[ok] 2/2 matched: F1={res['F1']:.4f} TP={res['TP']} (expect 1.0)")

    # 1 matched + 1 false positive, 1 GT -> F1 = 2/(2+1) = 0.6667
    pred_fp = {"img1": [hlane(200.0), hlane(600.0)]}   # second is far off (FP)
    gt1 = {"img1": [hlane(200.0)]}
    res2 = compute_f1(pred_fp, gt1)
    assert abs(res2["F1"] - 2.0 / 3.0) < 1e-6, res2
    print(f"[ok] 1 TP + 1 FP / 1 GT: F1={res2['F1']:.4f} (expect 0.6667)")


def test_curved_lane_rasterizes():
    # A curved (parabolic) lane should rasterize without error and match itself.
    ys = np.linspace(100, 700, 60, dtype=np.float32)
    xs = (683.0 + 120.0 * np.sin((ys - 100) / 600.0 * 3.1415)).astype(np.float32)
    curved = np.stack([xs, ys], axis=1)
    iou = pair_iou(curved, curved)
    assert abs(iou - 1.0) < 1e-6, f"curved self-IoU={iou}"
    print(f"[ok] curved lane self-match: IoU={iou:.4f} (expect 1.0)")


if __name__ == "__main__":
    test_gt_vs_gt_is_one()
    test_lateral_iou_physics()
    test_iou_boundary_at_10px()
    test_f1_identity_two_lanes()
    test_curved_lane_rasterizes()
    print("\nALL SELF-CHECKS PASSED  (tolerance = %.2g)" % TOL)
