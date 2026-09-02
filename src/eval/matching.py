"""Lane matching and the F1 aggregate.

Per-image pipeline (aligned with the frozen official Oracle):
  1. rasterize every predicted and ground-truth lane to a binary mask,
  2. build the P x G IoU matrix,
  3. one-to-one assign with cost ``1 - IoU`` (maximizing total IoU),
  4. a pair is a True Positive iff its IoU > `iou_thr` (default 0.5).

Global metric (verified identity, see DECISIONS.md §9):
  F1 = 2 * TP / (P + G)
where P = total predicted lanes, G = total ground-truth lanes over all images.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from .rasterize import (DEFAULT_CANVAS, DEFAULT_INTERP_N, DEFAULT_LINE_TYPE,
                        DEFAULT_LINE_WIDTH, DEFAULT_SPLINE_K, rasterize_lanes)


def pair_iou(pred_points, gt_points, **raster_kwargs) -> float:
    """IoU between a single predicted lane and a single GT lane."""
    from .rasterize import rasterize_lane
    a = rasterize_lane(pred_points, **raster_kwargs)
    b = rasterize_lane(gt_points, **raster_kwargs)
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    return inter / union if union > 0 else 0.0


def iou_matrix(pred_masks: List[np.ndarray],
               gt_masks: List[np.ndarray]) -> np.ndarray:
    """P x G IoU matrix between predicted and GT lane masks."""
    P, G = len(pred_masks), len(gt_masks)
    iou = np.zeros((P, G), dtype=np.float64)
    for i in range(P):
        a = pred_masks[i]
        for j in range(G):
            b = gt_masks[j]
            inter = int(np.logical_and(a, b).sum())
            union = int(np.logical_or(a, b).sum())
            iou[i, j] = inter / union if union > 0 else 0.0
    return iou


def match_image(pred_masks: List[np.ndarray], gt_masks: List[np.ndarray],
                iou_thr: float = 0.5) -> Tuple[int, int, int]:
    """One-to-one Hungarian match. Returns (TP, P, G)."""
    P, G = len(pred_masks), len(gt_masks)
    if P == 0 or G == 0:
        return 0, P, G
    iou = iou_matrix(pred_masks, gt_masks)
    # The Oracle maximizes total IoU first, including below-threshold pairs,
    # and applies the TP threshold only after assignment.
    row, col = linear_sum_assignment(1.0 - iou)
    tp = int((iou[row, col] > iou_thr).sum())
    return tp, P, G


def compute_f1_from_masks(pred_masks_by_img: Dict[str, List[np.ndarray]],
                          gt_masks_by_img: Dict[str, List[np.ndarray]],
                          iou_thr: float = 0.5) -> Dict[str, float]:
    """Aggregate F1 from pre-rasterized masks keyed by image_id."""
    TP = P = G = 0
    # Official evaluation is list/GT driven: a missing prediction is empty,
    # while an extra prediction file not present in the task list is ignored.
    for img_id, gt_m in gt_masks_by_img.items():
        pred_m = pred_masks_by_img.get(img_id, [])
        tp, p, g = match_image(pred_m, gt_m, iou_thr=iou_thr)
        TP += tp
        P += p
        G += g
    f1 = (2.0 * TP / (P + G)) if (P + G) > 0 else 0.0
    return {
        "F1": f1,
        "TP": int(TP),
        "FP": int(P - TP),
        "FN": int(G - TP),
        "P": int(P),
        "G": int(G),
    }


def compute_f1(pred_lanes: Dict[str, List[np.ndarray]],
               gt_lanes: Dict[str, List[np.ndarray]],
               iou_thr: float = 0.5,
               canvas_wh: tuple = DEFAULT_CANVAS,
               line_width: int = DEFAULT_LINE_WIDTH,
               line_type: int = DEFAULT_LINE_TYPE,
               spline_k: int = DEFAULT_SPLINE_K,
               interp_n: int = DEFAULT_INTERP_N,
               densify_step: float | None = None) -> Dict[str, float]:
    """Full metric from in-memory lane point-lists.

    `pred_lanes` / `gt_lanes`: dict image_id -> list of (N,2) float point arrays.

    Final decisions still use the frozen Oracle. ``densify_step`` survives
    only as a compatibility alias for the official integer ``interp_n``.
    """
    raster_kwargs = dict(canvas_wh=canvas_wh, line_width=line_width,
                         line_type=line_type, spline_k=spline_k,
                         interp_n=interp_n)
    if densify_step is not None:
        raster_kwargs["densify_step"] = densify_step
    pred_masks = {k: rasterize_lanes(v, **raster_kwargs) for k, v in pred_lanes.items()}
    gt_masks = {k: rasterize_lanes(v, **raster_kwargs) for k, v in gt_lanes.items()}
    return compute_f1_from_masks(pred_masks, gt_masks, iou_thr=iou_thr)
