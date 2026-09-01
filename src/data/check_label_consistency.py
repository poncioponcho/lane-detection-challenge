"""T21: cross-format label consistency check (ARCHITECTURE §6.2).

The three formats (.lines.txt / .json / instance .png) describe the SAME lanes.
This module verifies they agree per image, producing a badlist of images whose
formats disagree beyond tolerance (target: < 0.1% inconsistency rate on the
full 7100-image train set).

Matching uses the Hungarian algorithm on mean lateral error (same geometric
matching spirit as the official metric, but with a small tolerance because the
instance-PNG -> centerline extraction is lossy by up to half a stroke width).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from common.types import mean_lateral_error

# PNG centerline extraction error <= half the annotation stroke width.
# A 10px-wide stroke gives <= ~5px; a 2px stroke gives <= ~1px. The tolerance
# is configurable and will be calibrated against real data (T22 EDA).
DEFAULT_LAT_ERR_TOL = 6.0
DEFAULT_COUNT_TOL = 1            # allowed lane-count mismatch


@dataclass
class ImageConsistency:
    """Consistency verdict for one image."""
    image_id: str
    ok: bool = True
    n_lines: int = 0
    n_json: int = 0
    n_png: int = 0
    matched: int = 0
    lat_errs: List[float] = field(default_factory=list)   # per matched pair
    missing: List[str] = field(default_factory=list)       # formats absent
    note: str = ""

    @property
    def max_lat_err(self) -> float:
        return max(self.lat_errs) if self.lat_errs else 0.0


def _match_lanes(a: Sequence[np.ndarray], b: Sequence[np.ndarray]):
    """Hungarian one-to-one match by mean lateral error.

    Returns (matched_a_idx, matched_b_idx, costs) for pairs with finite cost
    (overlapping y-ranges). Lanes without a finite-cost match are left unmatched.
    """
    A, B = list(a), list(b)
    if not A or not B:
        return [], [], []
    cost = np.full((len(A), len(B)), 1e6, dtype=np.float64)
    for i in range(len(A)):
        for j in range(len(B)):
            e = mean_lateral_error(A[i], B[j])
            if np.isfinite(e):
                cost[i, j] = e
    finite = cost < 1e6
    if not finite.any():
        return [], [], []
    row, col = linear_sum_assignment(cost)
    matched_a, matched_b, errs = [], [], []
    for r, c in zip(row, col):
        if finite[r, c]:
            matched_a.append(int(r))
            matched_b.append(int(c))
            errs.append(float(cost[r, c]))
    return matched_a, matched_b, errs


def check_image(image_id: str,
                lines_lanes: Optional[Sequence[np.ndarray]],
                json_lanes: Optional[Sequence[np.ndarray]],
                png_lanes: Optional[Sequence[np.ndarray]],
                missing: Optional[Sequence[str]] = None,
                lat_err_tol: float = DEFAULT_LAT_ERR_TOL,
                count_tol: int = DEFAULT_COUNT_TOL) -> ImageConsistency:
    """Compare up to three lane sets for one image; flag disagreement."""
    res = ImageConsistency(image_id=image_id)
    if missing:
        res.missing = list(missing)

    sets = {"lines": lines_lanes, "json": json_lanes, "png": png_lanes}
    res.n_lines = 0 if lines_lanes is None else len(lines_lanes)
    res.n_json = 0 if json_lanes is None else len(json_lanes)
    res.n_png = 0 if png_lanes is None else len(png_lanes)

    present = [k for k, v in sets.items() if v is not None]
    if len(present) < 2:
        # nothing to compare against; not an inconsistency per se
        res.note = "only one format present"
        res.ok = len(present) == 1 or lines_lanes is not None
        return res

    # use lines.txt as the reference where available, else json
    ref_key = "lines" if "lines" in present else "json"
    ref = sets[ref_key]
    other_key = "png" if "png" in present else "json"
    other = sets[other_key]

    if ref_key == "json" and "png" in present and lines_lanes is None:
        # compare json vs png directly
        _, _, errs = _match_lanes(ref, other)
        res.matched = len(errs)
        res.lat_errs = errs
        n_ref, n_oth = len(ref), len(other)
        count_ok = abs(n_ref - n_oth) <= count_tol
        err_ok = all(e <= lat_err_tol for e in errs)
        res.ok = count_ok and err_ok
        if not res.ok:
            res.note = (f"{ref_key}({n_ref}) vs {other_key}({n_oth}): "
                        f"matched={res.matched} max_err={res.max_lat_err:.1f}")
        return res

    # lines.txt is reference; match against the OTHER format(s)
    all_errs: List[float] = []
    ok = True
    notes = []
    for k in ("json", "png"):
        s = sets[k]
        if s is None:
            continue
        _, _, errs = _match_lanes(ref, s)
        all_errs.extend(errs)
        if abs(len(ref) - len(s)) > count_tol:
            ok = False
            notes.append(f"{k} count {len(s)} != lines {len(ref)}")
        if errs and any(e > lat_err_tol for e in errs):
            ok = False
            notes.append(f"{k} max_err {max(errs):.1f} > {lat_err_tol}")
        if len(errs) < min(len(ref), len(s)):
            ok = False
            notes.append(f"{k} only {len(errs)} matched")
    res.matched = len(all_errs)
    res.lat_errs = all_errs
    res.ok = ok
    if notes:
        res.note = "; ".join(notes)
    return res


@dataclass
class ConsistencyReport:
    """Aggregated consistency over a set of images."""
    total: int = 0
    bad: int = 0
    bad_list: List[str] = field(default_factory=list)   # image_ids
    details: List[ImageConsistency] = field(default_factory=list)

    @property
    def bad_rate(self) -> float:
        return self.bad / self.total if self.total else 0.0


def check_many(results: Sequence[ImageConsistency]) -> ConsistencyReport:
    rep = ConsistencyReport()
    rep.details = list(results)
    rep.total = len(results)
    for r in results:
        if not r.ok:
            rep.bad += 1
            rep.bad_list.append(r.image_id)
    return rep


def write_badlist(report: ConsistencyReport, path) -> None:
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(report.bad_list) + "\n", encoding="utf-8")
