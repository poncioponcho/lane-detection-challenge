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

from common.types import CANVAS_H, CANVAS_W


def clamp_to_canvas(arr: np.ndarray,
                    w: int = CANVAS_W, h: int = CANVAS_H,
                    tolerance: float = 1.0) -> np.ndarray:
    """Snap coordinates that sit on the canvas border back onto the canvas.

    Why this exists (2026-09-17): the organisers' own pipeline CLAMPS, it does
    not reject. ``src/eval/official_oracle/score.py`` is byte-identical to the
    shipped ``score.py`` and does, before rasterising::

        lane[:, 0] = np.clip(lane[:, 0], 0, IMG_SHAPE[1] - 1)   # x -> [0,1365]
        lane[:, 1] = np.clip(lane[:, 1], 0, IMG_SHAPE[0] - 1)   # y -> [0, 719]

    and the shipped ``check_submission.py`` validates no coordinate bounds at
    all (UTF-8, even token count, finiteness, >=2 distinct points, <=64 lanes,
    <=2048 points, exact file set). The rules only promise that out-of-image
    points "will be truncated to the boundary".

    This module used to RAISE on any out-of-canvas point -- stricter than both
    official components. On testB every one of the 13 prediction trees puts 3
    to 21 lanes at x in (1365, 1366], i.e. at most exactly 1.0 px past the
    last valid column (the right image border); a strict guard aborted the
    whole packaging step, and because ``pack()`` sends prepare_submit's output
    to /dev/null it failed silently into a zero-lane zip.

    A bounded tolerance keeps the guard useful: a point a fraction of a pixel
    past the border is a boundary effect that the official scorer clamps away
    with identical geometry, whereas a point far outside means a real decoding
    bug and must still fail loudly.
    """
    a = np.asarray(arr, dtype=np.float64).reshape(-1, 2)
    if not np.isfinite(a).all():
        raise ValueError("lane contains NaN/Inf")
    lo_ok = (a[:, 0].min() >= -tolerance) and (a[:, 1].min() >= -tolerance)
    hi_ok = (a[:, 0].max() <= w - 1 + tolerance) and (a[:, 1].max() <= h - 1 + tolerance)
    if not (lo_ok and hi_ok):
        raise ValueError(
            f"coord out of bounds [{a[:, 0].min():.1f},{a[:, 0].max():.1f}]x"
            f"[{a[:, 1].min():.1f},{a[:, 1].max():.1f}] (canvas "
            f"[0,{w - 1}]x[0,{h - 1}], tolerance {tolerance}px) -- this far "
            "outside means a real bug, not a border effect"
        )
    out = a.copy()
    out[:, 0] = np.clip(out[:, 0], 0, w - 1)
    out[:, 1] = np.clip(out[:, 1], 0, h - 1)
    return out


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
    """Serialize one lane and validate the *serialized* official geometry.

    Distinct float coordinates can collapse after one-decimal formatting. The
    official parser removes consecutive duplicates and rejects a lane with
    fewer than two remaining points, so this guard must run after formatting.
    """
    arr = clamp_to_canvas(points)
    assert_valid_coords(arr)
    tokens = [f"{value:.{ndigits}f}" for value in arr.reshape(-1)]
    serialized = np.asarray(tokens, dtype=np.float64).reshape(-1, 2)
    keep = np.concatenate(([True], np.any(serialized[1:] != serialized[:-1], axis=1)))
    if int(keep.sum()) < 2:
        raise ValueError(
            "lane has fewer than 2 points after serialization and "
            "consecutive-deduplication"
        )
    return " ".join(tokens)


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
