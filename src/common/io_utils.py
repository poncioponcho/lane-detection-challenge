"""Three-format label I/O primitives (ARCHITECTURE §3 L315, DECISIONS §14).

Unified rules for ALL label formats:
  * coordinates are ALWAYS absolute pixels in the original 1366x720 frame;
  * a lane is an (N, 2) float32 array of (x, y) points;
  * .lines.txt output uses 1 decimal digit (submission contract).

Formats (TASKS.md §0):
  .lines.txt  -- one lane per line, "x1 y1 x2 y2 ..." space separated
                 (CULane convention; \\r\\n line endings tolerated);
  .json       -- schema calibrated at data-landing time; parser accepts the
                 common CULane/tuSimple variants and raises with a probe
                 helper otherwise (see probe_json_schema);
  instance .png -- single channel, 0 = background, non-zero = lane instance id.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

DEFAULT_IMG_SIZE = (1366, 720)   # (W, H)


# --------------------------------------------------------------------------
# .lines.txt
# --------------------------------------------------------------------------
def parse_txt_line(s: str) -> Optional[np.ndarray]:
    """Parse one .lines.txt line into (N, 2) float32; empty line -> None.

    Tolerates: leading/trailing whitespace, \r, comma separators, extra spaces.
    Raises ValueError on odd token count or fewer than 2 points.
    """
    s = s.strip().strip("\r")
    if not s:
        return None
    tokens = s.replace(",", " ").split()
    if len(tokens) % 2 != 0:
        raise ValueError(f"odd token count ({len(tokens)}): {s[:80]!r}")
    vals = np.asarray(tokens, dtype=np.float64)
    pts = vals.reshape(-1, 2).astype(np.float32)
    if len(pts) < 2:
        raise ValueError(f"lane needs >= 2 points: {s[:80]!r}")
    return pts


def read_lines_txt(path: str | Path) -> List[np.ndarray]:
    """Read a whole .lines.txt file -> list of lane arrays (may be empty)."""
    lanes: List[np.ndarray] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            pts = parse_txt_line(line)
            if pts is not None:
                lanes.append(pts)
    return lanes


def write_lines_txt(path: str | Path, lanes: List[np.ndarray],
                    ndigits: int = 1) -> None:
    """Write lanes to .lines.txt (1 decimal digit, submission contract).

    An empty lane list produces an empty file (the file must still exist).
    """
    lines = []
    for pts in lanes:
        arr = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
        if len(arr) < 2:
            continue
        lines.append(" ".join(f"{v:.{ndigits}f}" for v in arr.reshape(-1)))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + ("\n" if lines else ""),
                          encoding="utf-8")


# --------------------------------------------------------------------------
# .json
# --------------------------------------------------------------------------
_JSON_LANE_KEYS = ("lanes", "Lines", "lines", "Lanes", "annotations")


def _pts_from_pairs(pairs) -> np.ndarray:
    """[[x, y], ...] or [{x, y}, ...] or flat even-length numeric list."""
    if isinstance(pairs, dict):
        if "x" in pairs and "y" in pairs:
            xs = np.atleast_1d(np.asarray(pairs["x"], dtype=np.float64))
            ys = np.atleast_1d(np.asarray(pairs["y"], dtype=np.float64))
            return np.stack([xs, ys], axis=1).astype(np.float32)
        raise ValueError(f"dict lane without x/y keys: {list(pairs)[:8]}")
    arr = np.asarray(pairs, dtype=np.float64)
    if arr.ndim == 1:                      # flat x1 y1 x2 y2 ...
        if arr.size % 2 != 0 or arr.size < 4:
            raise ValueError(f"flat lane token count {arr.size}")
        return arr.reshape(-1, 2).astype(np.float32)
    if arr.ndim == 2 and arr.shape[1] == 2 and arr.shape[0] >= 2:
        return arr.astype(np.float32)
    raise ValueError(f"unrecognised lane shape {arr.shape}")


def parse_json_lanes(obj) -> List[np.ndarray]:
    """Parse a JSON label object -> list of lane arrays.

    Accepted variants (probe real data first; see probe_json_schema):
      {"lanes": [[[x,y],...], ...]}
      {"Lines": [[{"x":..,"y":..},...], ...]}
      {"lanes": [{"x": [...], "y": [...]}, ...]}
      [[x1,y1,x2,y2,...], ...]              (raw list of flat lanes)
    """
    if isinstance(obj, str):
        obj = json.loads(obj)

    # find the lane container
    container = None
    if isinstance(obj, dict):
        for k in _JSON_LANE_KEYS:
            if k in obj:
                container = obj[k]
                break
        if container is None:
            raise ValueError(
                f"dict label without known lane keys {list(obj)[:10]}")
    elif isinstance(obj, list):
        container = obj
    else:
        raise ValueError(f"unsupported label root type {type(obj).__name__}")

    lanes = []
    for item in container:
        if isinstance(item, dict) and "points" in item:
            pts = _pts_from_pairs(item["points"])
        elif isinstance(item, dict):
            pts = _pts_from_pairs(item)
        else:
            pts = _pts_from_pairs(item)
        if len(pts) >= 2:
            lanes.append(pts)
    return lanes


def read_json_lanes(path: str | Path) -> List[np.ndarray]:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return parse_json_lanes(json.load(f))


def probe_json_schema(path: str | Path, max_depth: int = 3) -> str:
    """Describe the top-level structure of a real .json label file.

    Run this on ONE real sample the moment data lands, then (if needed) extend
    parse_json_lanes. Purely diagnostic; never raises on unknown shapes.
    """
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        obj = json.load(f)

    def describe(o, depth=0) -> str:
        pad = "  " * depth
        if depth > max_depth:
            return f"{pad}…"
        if isinstance(o, dict):
            inner = "; ".join(
                f"{k}: {describe(v, depth + 1).strip()}"
                for k, v in list(o.items())[:6])
            return f"{pad}dict({inner})"
        if isinstance(o, list):
            head = describe(o[0], depth + 1).strip() if o else "empty"
            return f"{pad}list[{len(o)}] of {head}"
        return f"{pad}{type(o).__name__}={o!r}"[:120]

    return describe(obj)


# --------------------------------------------------------------------------
# instance .png
# --------------------------------------------------------------------------
def read_instance_png(path: str | Path, img_size=None,
                      min_rows: int = 4) -> List[np.ndarray]:
    """Extract lane center-polylines from a single-channel instance mask.

    For each instance id > 0: group mask pixels by row y, take the mean x of
    each row -> one point per row -> polyline sorted by y. Rows are dense
    only where the annotation exists (gaps are kept, not interpolated).

    min_rows drops tiny instances (annotation noise). img_size clips points.
    """
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise IOError(f"cannot read instance png: {path}")
    if mask.ndim == 3:                       # tolerate accidentally-saved RGB
        mask = mask[..., 0]
    if mask.ndim != 2:
        raise ValueError(f"instance png must be single channel: {path}")

    W, H = img_size or DEFAULT_IMG_SIZE
    lanes: List[np.ndarray] = []
    ids = [int(v) for v in np.unique(mask) if v != 0]
    for iid in ids:
        ys, xs = np.nonzero(mask == iid)
        rows = np.unique(ys)
        if len(rows) < min_rows:
            continue
        pts = np.empty((len(rows), 2), dtype=np.float32)
        for i, y in enumerate(rows):
            sel = xs[ys == y]
            pts[i] = (float(sel.mean()), float(y))
        # keep strictly inside the frame (mask is already in-frame; belt&braces)
        pts[:, 0] = np.clip(pts[:, 0], 0, W - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, H - 1)
        lanes.append(pts)
    # deterministic order: left-to-right by mean x
    lanes.sort(key=lambda l: float(l[:, 0].mean()))
    return lanes
