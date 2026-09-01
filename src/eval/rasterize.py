"""Lane rasterization: spline densify + thick-stroke bitmap rendering.

This module reproduces the competition's line-rasterization step:
  * fit a cubic B-spline (degree `spline_k`, default 3) through the lane points,
  * resample the smooth curve at `densify_step` px arc-length,
  * paint it as a `line_width`-px wide, `line_type`-connected stroke
    (default 30 px, 8-connected, no anti-aliasing) on a WxH binary canvas.

Physical identity (verified in tests/test_metric_selfcheck.py):
  two identical 30px strokes whose centerlines are laterally separated by d px
  have approximately  IoU = (30 - d) / (30 + d).
  NOTE: cv2's thickness=30 stroke is effectively ~31px wide, so the REAL
  IoU=0.5 boundary is ~10.3px (ideal rectangle model gives exactly 10px).
  This is faithful to the official metric (same cv2 call). Verified in
  tests/test_metric_selfcheck.py; the 0.3px shift is far inside the 2pp
  decision tolerance.
"""
from __future__ import annotations

from typing import Dict

import cv2
import numpy as np

DEFAULT_CANVAS = (1366, 720)
DEFAULT_LINE_WIDTH = 30
DEFAULT_LINE_TYPE = 8          # cv2.LINE_8 (8-connected), no anti-aliasing
DEFAULT_SPLINE_K = 3
DEFAULT_DENSIFY_STEP = 5.0


def densify_linear(pts: np.ndarray, step: float) -> np.ndarray:
    """Insert points so consecutive samples are ~`step` px apart (no smoothing)."""
    if len(pts) < 2:
        return pts.astype(np.float32)
    out = [pts[0]]
    for i in range(1, len(pts)):
        a, b = pts[i - 1], pts[i]
        seg = float(np.linalg.norm(b - a))
        n = max(1, int(np.ceil(seg / step)))
        for j in range(1, n + 1):
            out.append(a + (b - a) * (j / n))
    return np.asarray(out, dtype=np.float32)


def densify_spline(points: np.ndarray, k: int = DEFAULT_SPLINE_K,
                   step: float = DEFAULT_DENSIFY_STEP) -> np.ndarray:
    """Fit a degree-`k` B-spline through `points`, resample at `step` px.

    Falls back to linear densification when there are too few points for a
    spline of the requested degree, or if spline fitting fails for any reason.
    """
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(pts) < 2:
        return pts

    # Drop consecutive duplicates (splprep hates zero-length segments).
    diff = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    if np.any(diff == 0):
        keep = np.concatenate([[True], diff > 0])
        pts = pts[keep]
    if len(pts) < k + 1:
        return densify_linear(pts, step)

    try:
        from scipy.interpolate import splprep, splev
        tck, _u = splprep([pts[:, 0], pts[:, 1]], k=k, s=0)
        # Sample 4x denser than the target step, then re-densify linearly so the
        # final spacing is uniform and the stroke is gap-free.
        n_dense = max(200, int(np.ceil(cv2.arcLength(pts, False) / step) * 4))
        unew = np.linspace(0.0, 1.0, n_dense)
        x, y = splev(unew, tck)
        dense = np.stack([x, y], axis=1).astype(np.float32)
        return densify_linear(dense, step)
    except Exception:
        return densify_linear(pts, step)


def rasterize_lane(points: np.ndarray,
                   canvas_wh: tuple = DEFAULT_CANVAS,
                   line_width: int = DEFAULT_LINE_WIDTH,
                   line_type: int = DEFAULT_LINE_TYPE,
                   spline_k: int = DEFAULT_SPLINE_K,
                   densify_step: float = DEFAULT_DENSIFY_STEP) -> np.ndarray:
    """Render a single lane polyline to a (H, W) uint8 binary mask (0/1)."""
    W, H = canvas_wh
    mask = np.zeros((H, W), dtype=np.uint8)
    if points is None or len(points) < 2:
        return mask
    dense = densify_spline(np.asarray(points, dtype=np.float32).reshape(-1, 2),
                           k=spline_k, step=densify_step)
    if len(dense) < 2:
        return mask
    dense = np.clip(dense, 0, [W - 1, H - 1])
    pts_i = dense.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(mask, [pts_i], isClosed=False, color=1,
                  thickness=line_width, lineType=line_type)
    return mask


def rasterize_lanes(lanes: list, canvas_wh: tuple = DEFAULT_CANVAS,
                    **kwargs: Dict) -> list:
    """Rasterize a list of lane point-arrays; returns a list of masks."""
    return [rasterize_lane(l, canvas_wh=canvas_wh, **kwargs) for l in lanes]
