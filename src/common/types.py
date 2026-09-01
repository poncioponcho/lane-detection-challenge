"""Shared data types and lane geometry helpers.

Coordinate convention: a lane is a polyline of (x, y) points in image space,
x in [0, W) (width), y in [0, H) (height). All geometry helpers operate on
float32 (N, 2) arrays and never touch the filesystem.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List

import numpy as np


@dataclass
class Lane:
    """A single predicted or ground-truth lane polyline."""
    points: np.ndarray            # (N, 2) float32, columns (x, y)
    score: float = 1.0           # confidence; used only by some loaders
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.points = np.asarray(self.points, dtype=np.float32).reshape(-1, 2)
        if self.points.shape[1] != 2:
            raise ValueError(f"Lane.points must be (N,2), got {self.points.shape}")


@dataclass
class ImageAnnotation:
    """All lanes for one image."""
    image_id: str
    lanes: List[Lane]
    img_size: tuple = (1366, 720)   # (W, H)
    clip_id: Optional[str] = None


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------
def resample_lane(points: np.ndarray, step: float = 5.0) -> np.ndarray:
    """Arc-length resample a polyline to ~`step` px between consecutive points.

    Used to normalize lane density before rasterization / error measurement.
    """
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(pts) < 2:
        return pts
    out = [pts[0]]
    for i in range(1, len(pts)):
        a, b = pts[i - 1], pts[i]
        seg = float(np.linalg.norm(b - a))
        n = max(1, int(np.ceil(seg / step)))
        for j in range(1, n + 1):
            out.append(a + (b - a) * (j / n))
    return np.asarray(out, dtype=np.float32)


def clip_lane_to_canvas(points: np.ndarray, canvas_wh: tuple,
                        drop_outside: bool = True) -> np.ndarray:
    """Clip a polyline to the canvas.

    With drop_outside=True, points fully outside the canvas are removed and a
    point on the boundary is kept; consecutive in-bounds points are connected by
    a segment that cv2 clips automatically during rasterization anyway.
    """
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    W, H = canvas_wh
    if drop_outside:
        inside = (pts[:, 0] >= 0) & (pts[:, 0] < W) & (pts[:, 1] >= 0) & (pts[:, 1] < H)
        pts = pts[inside]
    return np.clip(pts, 0, [W - 1, H - 1])


def extrapolate_ends(points: np.ndarray, y_top: float, y_bottom: float) -> np.ndarray:
    """Extend a polyline to span [y_top, y_bottom] by linear extrapolation.

    Lane annotations are often sparse near the horizon / bottom; extending to
    image borders before matching reduces spurious lateral error.
    """
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(pts) < 2:
        return pts
    # fit y = a*x + b  (x here is horizontal position, y is vertical)
    ys, xs = pts[:, 1], pts[:, 0]
    A = np.vstack([ys, np.ones_like(ys)]).T
    (a, b), *_ = np.linalg.lstsq(A, xs, rcond=None)  # x = a*y + b
    new = pts.copy()
    if y_top < ys.min():
        new = np.vstack([np.array([[a * y_top + b, y_top]], dtype=np.float32), new])
    if y_bottom > ys.max():
        new = np.vstack([new, np.array([[a * y_bottom + b, y_bottom]], dtype=np.float32)])
    return new


def mean_lateral_error(lane_a: np.ndarray, lane_b: np.ndarray,
                       y_samples: Optional[np.ndarray] = None) -> float:
    """Mean horizontal offset |x_a - x_b| at shared y samples.

    Samples y at the union range of the two lanes; for each sample we linearly
    interpolate x on both lanes and report the average |dx|. Returns inf if the
    y-ranges do not overlap.
    """
    a = np.asarray(lane_a, dtype=np.float32).reshape(-1, 2)
    b = np.asarray(lane_b, dtype=np.float32).reshape(-1, 2)
    y_lo = max(a[:, 1].min(), b[:, 1].min())
    y_hi = min(a[:, 1].max(), b[:, 1].max())
    if y_hi <= y_lo:
        return float("inf")
    if y_samples is None:
        y_samples = np.linspace(y_lo, y_hi, 20)
    xa = np.interp(y_samples, a[:, 1], a[:, 0])
    xb = np.interp(y_samples, b[:, 1], b[:, 0])
    return float(np.mean(np.abs(xa - xb)))
