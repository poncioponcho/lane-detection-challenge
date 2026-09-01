"""Official F1 metric entry point.

The competition's evaluation is exactly `compute_f1` from matching.py:
rasterize -> per-image Hungarian IoU match (IoU > 0.5 => TP) -> F1 = 2TP/(P+G).

This module re-exports that core and adds a directory-driven wrapper that
accepts a *loader* callable. The loader is competition-format specific and is
implemented in T20 (data parsing); T11/T12 only need the in-memory core, which
is exercised by tests/test_metric_selfcheck.py.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np

from .matching import compute_f1, compute_f1_from_masks, match_image, pair_iou
from .rasterize import (DEFAULT_CANVAS, DEFAULT_DENSIFY_STEP, DEFAULT_LINE_TYPE,
                        DEFAULT_LINE_WIDTH, DEFAULT_SPLINE_K, rasterize_lane,
                        rasterize_lanes)

__all__ = [
    "compute_f1",
    "compute_f1_from_masks",
    "match_image",
    "pair_iou",
    "rasterize_lane",
    "rasterize_lanes",
    "evaluate_dir",
    "DEFAULT_CANVAS",
    "DEFAULT_LINE_WIDTH",
    "DEFAULT_LINE_TYPE",
    "DEFAULT_SPLINE_K",
    "DEFAULT_DENSIFY_STEP",
]

# A loader turns one file path into a list of (N,2) float lane point-arrays.
LoaderFn = Callable[[str], List[np.ndarray]]


def evaluate_dir(pred_dir: str, gt_dir: str,
                 img_list: Optional[List[str]] = None,
                 loader: Optional[LoaderFn] = None,
                 iou_thr: float = 0.5,
                 canvas_wh: tuple = DEFAULT_CANVAS,
                 line_width: int = DEFAULT_LINE_WIDTH,
                 line_type: int = DEFAULT_LINE_TYPE,
                 spline_k: int = DEFAULT_SPLINE_K,
                 densify_step: float = DEFAULT_DENSIFY_STEP) -> Dict[str, float]:
    """Evaluate a prediction directory against a GT directory.

    Parameters
    ----------
    loader : callable(path) -> list[(N,2) ndarray]
        Format-specific parser. MUST be supplied; the competition file schema
        is implemented in T20. Raises if missing.
    img_list : list[str] | None
        Restrict to these image ids (file stems). If None, all GT files.
    """
    if loader is None:
        raise ValueError(
            "loader must be supplied (competition format parser from T20). "
            "T11/T12 validates the in-memory core via tests/test_metric_selfcheck.py."
        )
    import os
    import glob

    gt_files = sorted(glob.glob(os.path.join(gt_dir, "*.json")))
    if img_list is not None:
        want = set(img_list)
        gt_files = [f for f in gt_files if os.path.splitext(os.path.basename(f))[0] in want]

    pred_lanes: Dict[str, List[np.ndarray]] = {}
    gt_lanes: Dict[str, List[np.ndarray]] = {}
    for gf in gt_files:
        stem = os.path.splitext(os.path.basename(gf))[0]
        gt_lanes[stem] = loader(gf)
        pf = os.path.join(pred_dir, os.path.basename(gf))
        pred_lanes[stem] = loader(pf) if os.path.exists(pf) else []

    return compute_f1(
        pred_lanes, gt_lanes,
        iou_thr=iou_thr, canvas_wh=canvas_wh, line_width=line_width,
        line_type=line_type, spline_k=spline_k, densify_step=densify_step,
    )
