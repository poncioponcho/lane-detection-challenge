"""Validate a submit.zip against the full submission contract (P0-A04).

Checks (each is a hard gate; any failure -> ok=False):
  1. single top-level directory named `root` (default "submit");
  2. file set == expected set (no missing, no extra), case-sensitive paths;
  3. per-file .lines.txt: official parse/interpolate/draw smoke, <=64 lanes,
     <=2048 points/lane, 1 decimal, no NaN/Inf, strict frame bounds;
  4. total uncompressed size < max_bytes (default 200 MB).

Returns (ok: bool, report: str). Also writes the report to `report_path` if
given (ARCHITECTURE §3: outputs/reports/verify_<ts>.md).
"""
from __future__ import annotations

import zipfile
import re
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np

from common.types import CANVAS_H, CANVAS_W
from eval.rasterize import (interp_lane, rasterize_interpolated,
                            remove_consecutive_duplicates)
from submit.pack_submit import _normalize_rel

MAX_BYTES = 200 * 1024 * 1024
MAX_LANES_PER_IMAGE = 64
MAX_POINTS_PER_LANE = 2048
_ONE_DECIMAL = re.compile(r"^-?\d+\.\d$")


def validate_line(s: str) -> Optional[str]:
    """Return an error string for one .lines.txt line, or None if valid."""
    s = s.strip()
    if not s:
        return None                          # empty line = no lane, allowed
    tokens = s.split()
    if len(tokens) % 2 != 0:
        return f"odd token count {len(tokens)}"
    if len(tokens) < 4:
        return f"too few values {len(tokens)}"
    try:
        vals = np.asarray(tokens, dtype=np.float64)
    except ValueError:
        return "non-numeric token"
    if not np.isfinite(vals).all():
        return "NaN/Inf"
    if any(_ONE_DECIMAL.fullmatch(token) is None for token in tokens):
        return "not 1-decimal"
    points = vals.reshape(-1, 2)
    if len(points) > MAX_POINTS_PER_LANE:
        return f"too many points {len(points)} > {MAX_POINTS_PER_LANE}"
    cleaned = remove_consecutive_duplicates(points)
    if len(cleaned) < 2:
        return "fewer than 2 points after consecutive-deduplication"
    x, y = vals[0::2], vals[1::2]
    if (x.min() < 0 or x.max() > CANVAS_W - 1 or
            y.min() < 0 or y.max() > CANVAS_H - 1):
        return f"out of bounds x[{x.min():.1f},{x.max():.1f}] y[{y.min():.1f},{y.max():.1f}]"
    if float(vals.max()) < 2.0:
        return "coords look normalized (all < 2px)"
    try:
        # Full-chain smoke on the exact serialized coordinates. This mirrors
        # official parse -> interp_lane -> draw_lane_mask failure semantics.
        dense = interp_lane(np.asarray(cleaned, dtype=np.float64))
        rasterize_interpolated(dense)
    except Exception as exc:
        return f"official geometry smoke failed: {type(exc).__name__}: {exc}"
    return None


def verify_submit(zip_path: Union[str, Path],
                  expected: Iterable[str],
                  root: str = "submit",
                  max_bytes: int = MAX_BYTES,
                  report_path: Optional[Union[str, Path]] = None,
                  ) -> Tuple[bool, str]:
    zip_path = Path(zip_path)
    expected = sorted({_normalize_rel(r) for r in expected if r.strip()})
    errors: List[str] = []
    seen: List[str] = []

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        # 1. single top-level root dir, no absolute paths, no traversal
        for n in names:
            if n.startswith("/") or ".." in n:
                errors.append(f"unsafe path in zip: {n}")
        top = {n.split("/", 1)[0] for n in names}
        if top != {root}:
            errors.append(f"top-level entries {sorted(top)} != {{{root}}}")

        # 2. file set equality (case-sensitive)
        actual = {n[len(root) + 1:] for n in names
                  if n.startswith(root + "/") and not n.endswith("/")}
        want = set(expected)
        missing = want - actual
        extra = actual - want
        if missing:
            errors.append(f"missing {len(missing)} files (e.g. {sorted(missing)[:3]})")
        if extra:
            errors.append(f"extra {len(extra)} files (e.g. {sorted(extra)[:3]})")

        # 3 + 4. per-file content validation + size
        total = 0
        for n in names:
            if n.endswith("/"):
                continue
            data = zf.read(n)
            total += len(data)
            rel = n[len(root) + 1:]
            seen.append(rel)
            if not rel.endswith(".lines.txt"):
                errors.append(f"unexpected non-lines.txt entry: {n}")
                continue
            text = data.decode("utf-8", errors="replace")
            nonempty_lines = sum(bool(line.strip()) for line in text.splitlines())
            if nonempty_lines > MAX_LANES_PER_IMAGE:
                errors.append(
                    f"{rel}: too many lanes {nonempty_lines} > {MAX_LANES_PER_IMAGE}"
                )
            for i, line in enumerate(text.splitlines(), 1):
                err = validate_line(line)
                if err:
                    errors.append(f"{rel}:L{i} {err}")

        if total > max_bytes:
            errors.append(f"size {total / 1e6:.1f} MB > {max_bytes / 1e6:.0f} MB")

    ok = not errors
    report = "\n".join(
        [f"verify {'PASS' if ok else 'FAIL'}  {zip_path.name}",
         f"  root: {root}  files: {len(seen)}  expected: {len(expected)}"]
        + (["  " + e for e in errors] if errors else ["  all checks passed"]))
    if report_path:
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        Path(report_path).write_text(report + "\n", encoding="utf-8")
    return ok, report
