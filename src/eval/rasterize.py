"""Fast diagnostic rasterizer aligned with the frozen official Oracle.

The competition implementation is the source of truth. In particular it
removes only consecutive duplicate points, fits an interpolating B-spline in
float64, samples uniformly in spline parameter space, and draws every adjacent
segment with ``cv2.line``. There is deliberately no linear fallback: spline
errors propagate exactly as they do in ``official_oracle/score.py``.

This remains a diagnostic layer. Final decisions must use the frozen Oracle
through ``oracle_runner.py``.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable

import cv2
import numpy as np
from scipy.interpolate import splprep, splev

from common.types import CANVAS_H, CANVAS_W

DEFAULT_CANVAS = (CANVAS_W, CANVAS_H)
DEFAULT_LINE_WIDTH = 30
DEFAULT_LINE_TYPE = cv2.LINE_8
DEFAULT_SPLINE_K = 3
DEFAULT_INTERP_N = 5

# Backward-compatible import name. The value is not a pixel step: score.py
# uses it as the number of parameter-space intervals per input span.
DEFAULT_DENSIFY_STEP = DEFAULT_INTERP_N


def remove_consecutive_duplicates(points: Iterable) -> list[tuple[float, float]]:
    """Return points with adjacent exact duplicates removed."""
    pts = [tuple(point) for point in points]
    if len(pts) < 2:
        return pts
    cleaned = [pts[0]]
    for point in pts[1:]:
        if point != cleaned[-1]:
            cleaned.append(point)
    return cleaned


def parse_lines_txt(path: str | Path) -> list[np.ndarray]:
    """Parse ``.lines.txt`` under the frozen Oracle's strict contract.

    Missing and zero-byte prediction files mean no lanes. Every other format
    error is fatal: commas, NaN/Inf, odd token counts, and lanes with fewer
    than two effective points are rejected.
    """
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return []

    lanes: list[np.ndarray] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_idx, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            coords = line.split()
            if len(coords) % 2:
                raise ValueError(f"file {path} line {line_idx} has an odd token count")
            if len(coords) < 4:
                raise ValueError(f"file {path} line {line_idx} has fewer than 4 values")

            points = []
            for idx in range(0, len(coords), 2):
                x, y = float(coords[idx]), float(coords[idx + 1])
                if not (math.isfinite(x) and math.isfinite(y)):
                    raise ValueError(f"file {path} line {line_idx} contains NaN or Inf")
                points.append((x, y))
            cleaned = remove_consecutive_duplicates(points)
            if len(cleaned) < 2:
                raise ValueError(
                    f"file {path} line {line_idx} has fewer than 2 points after deduplication"
                )
            lanes.append(np.asarray(cleaned, dtype=np.float64))
    return lanes


def interp_lane(points: np.ndarray, interp_n: int = DEFAULT_INTERP_N,
                spline_k: int = DEFAULT_SPLINE_K) -> np.ndarray:
    """Interpolate one lane exactly as the official ``interp_lane`` does."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 2:
        raise ValueError("lane needs at least two points")
    if not np.isfinite(pts).all():
        raise ValueError("lane contains NaN or Inf")

    # Do not deduplicate here. The official parser owns that normalization;
    # direct array/JSON callers must see the same spline failure as score.py.
    k = min(int(spline_k), len(pts) - 1)
    # ``t=interp_n`` is retained for semantic parity. With task=0 scipy
    # ignores t; the frozen Oracle behaves identically.
    tck, u = splprep(
        [pts[:, 0], pts[:, 1]], s=0, t=interp_n, k=k
    )
    u_new = np.linspace(0.0, 1.0, num=(len(u) - 1) * int(interp_n) + 1)
    return np.asarray(splev(u_new, tck), dtype=np.float64).T


def rasterize_interpolated(lane: np.ndarray,
                           canvas_wh: tuple[int, int] = DEFAULT_CANVAS,
                           line_width: int = DEFAULT_LINE_WIDTH,
                           line_type: int = DEFAULT_LINE_TYPE) -> np.ndarray:
    """Draw an already-interpolated lane using the Oracle segment loop."""
    width, height = canvas_wh
    mask = np.zeros((height, width), dtype=np.uint8)
    pts = np.asarray(lane, dtype=np.float64).reshape(-1, 2).copy()
    pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)
    pts_i = pts.astype(np.int32)
    for p1, p2 in zip(pts_i[:-1], pts_i[1:]):
        cv2.line(
            mask,
            tuple(p1),
            tuple(p2),
            color=(1,),
            thickness=int(line_width),
            lineType=int(line_type),
        )
    return mask > 0


def densify_spline(points: np.ndarray, k: int = DEFAULT_SPLINE_K,
                   step: float = DEFAULT_INTERP_N) -> np.ndarray:
    """Compatibility wrapper; ``step`` now has official ``interp_n`` meaning."""
    if int(step) != step or step <= 0:
        raise ValueError("official interp_n must be a positive integer")
    return interp_lane(points, interp_n=int(step), spline_k=k)


def rasterize_lane(points: np.ndarray,
                   canvas_wh: tuple[int, int] = DEFAULT_CANVAS,
                   line_width: int = DEFAULT_LINE_WIDTH,
                   line_type: int = DEFAULT_LINE_TYPE,
                   spline_k: int = DEFAULT_SPLINE_K,
                   interp_n: int = DEFAULT_INTERP_N,
                   densify_step: float | None = None) -> np.ndarray:
    """Render one lane to a boolean mask using official raster semantics."""
    if densify_step is not None:
        if int(densify_step) != densify_step:
            raise ValueError("densify_step is deprecated; official interp_n must be integer")
        interp_n = int(densify_step)
    dense = interp_lane(points, interp_n=interp_n, spline_k=spline_k)
    return rasterize_interpolated(
        dense, canvas_wh=canvas_wh, line_width=line_width, line_type=line_type
    )


def rasterize_lanes(lanes: list, canvas_wh: tuple[int, int] = DEFAULT_CANVAS,
                    **kwargs: Dict) -> list[np.ndarray]:
    """Rasterize a list of lane point arrays."""
    return [rasterize_lane(lane, canvas_wh=canvas_wh, **kwargs) for lane in lanes]
