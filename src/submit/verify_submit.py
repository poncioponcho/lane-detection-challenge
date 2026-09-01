"""Validate a submit.zip against the full submission contract (P0-A04).

Checks (each is a hard gate; any failure -> ok=False):
  1. single top-level directory named `root` (default "submit");
  2. file set == expected set (no missing, no extra), case-sensitive paths;
  3. per-file .lines.txt: even tokens, >=4 values, 1 decimal, no NaN/Inf,
     absolute-pixel coords within 1366x720 (never normalized);
  4. total uncompressed size < max_bytes (default 200 MB).

Returns (ok: bool, report: str). Also writes the report to `report_path` if
given (ARCHITECTURE §3: outputs/reports/verify_<ts>.md).
"""
from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np

CANVAS_W, CANVAS_H = 1366, 720
MAX_BYTES = 200 * 1024 * 1024


def validate_line(s: str) -> Optional[str]:
    """Return an error string for one .lines.txt line, or None if valid."""
    s = s.strip()
    if not s:
        return None                          # empty line = no lane, allowed
    tokens = s.replace(",", " ").split()
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
    if any(abs(v - round(v, 1)) > 1e-6 for v in vals):
        return "not 1-decimal"
    x, y = vals[0::2], vals[1::2]
    if x.min() < 0 or x.max() > CANVAS_W or y.min() < 0 or y.max() > CANVAS_H:
        return f"out of bounds x[{x.min():.1f},{x.max():.1f}] y[{y.min():.1f},{y.max():.1f}]"
    if float(vals.max()) < 2.0:
        return "coords look normalized (all < 2px)"
    return None


def verify_submit(zip_path: Union[str, Path],
                  expected: Iterable[str],
                  root: str = "submit",
                  max_bytes: int = MAX_BYTES,
                  report_path: Optional[Union[str, Path]] = None,
                  ) -> Tuple[bool, str]:
    zip_path = Path(zip_path)
    expected = sorted({r.strip().lstrip("./") for r in expected if r.strip()})
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
