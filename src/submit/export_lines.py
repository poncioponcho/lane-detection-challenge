"""Delivery-side lane export: list[Lane] -> .lines.txt with STRICT assertions.

This is the OPPOSITE of common/io_utils read (lenient). Here every prediction
line is hard-validated before it can enter a submission (P0-A03, ARCHITECTURE
§3 L361). The submission .lines.txt contract (TASKS.md §0):

  * one lane per line: "x1 y1 x2 y2 ...", 1 decimal digit;
  * an EMPTY file (still written) means "no lane detected" for that image;
  * per-file asserts: even token count, >= 4 values, no NaN/Inf, coordinates
    are ABSOLUTE pixels within [0, 1365] x [0, 719] (never normalized 0..1).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Union

import numpy as np

CANVAS_W, CANVAS_H = 1366, 720


def assert_valid_coords(arr: np.ndarray,
                        w: int = CANVAS_W, h: int = CANVAS_H) -> None:
    """Hard-validate a lane point array; raises ValueError on any violation."""
    arr = np.asarray(arr, dtype=np.float64).reshape(-1, 2)
    if arr.shape[0] < 2:
        raise ValueError(f"lane needs >= 2 points, got {arr.shape[0]}")
    if not np.isfinite(arr).all():
        raise ValueError("lane contains NaN/Inf")
    x, y = arr[:, 0], arr[:, 1]
    if x.min() < 0 or x.max() > w - 1 or y.min() < 0 or y.max() > h - 1:
        raise ValueError(
            f"coord out of bounds [{x.min():.1f},{x.max():.1f}]x"
            f"[{y.min():.1f},{y.max():.1f}] (expect [0,{w - 1}]x[0,{h - 1}])")
    if float(np.abs(arr).max()) < 2.0:
        raise ValueError("coords look normalized (all < 2px); expected "
                         f"absolute {w}x{h} pixels")


def lane_to_line(points, ndigits: int = 1) -> str:
    """Serialize one lane to a .lines.txt line with full validation."""
    arr = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    assert_valid_coords(arr)
    return " ".join(f"{v:.{ndigits}f}" for v in arr.reshape(-1))


def export_lines(lanes: Sequence, path: Union[str, Path],
                 ndigits: int = 1) -> Path:
    """Write a list of lane point-arrays to a .lines.txt file.

    lanes may contain None entries (no lane) — they are skipped. An empty lane
    list still writes an empty file (the file must exist in the submission).
    """
    lines = []
    for l in lanes:
        if l is None or len(l) == 0:
            continue
        lines.append(lane_to_line(l, ndigits))
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return p
