"""T21: cross-format label consistency check (ARCHITECTURE §6.2).

The three formats (.lines.txt / .json / palette instance .png) describe the
same lane geometry. The lossless text/JSON pair is checked lane-by-lane; the
lossy raster is checked as a lane-union mask against a 10px redraw. This avoids
mistaking occlusion/color merging during inverse centerline extraction for a
source-label conflict. Target: <0.1% bad images on the full 7100-image set.

Hungarian mean-lateral-error matching remains available for true instance
centerlines and diagnostics; the production manifest audit uses union IoU.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

from common.types import mean_lateral_error

# Optional instance-centerline diagnostic tolerance. Production T21 uses the
# mask-union threshold below because inverse extraction is lossy at occlusions.
DEFAULT_LAT_ERR_TOL = 6.0
DEFAULT_COUNT_TOL = 1            # allowed lane-count mismatch
DEFAULT_MASK_IOU_TOL = 0.75      # lossy palette mask vs 10px line-union render


@dataclass
class ImageConsistency:
    """Consistency verdict for one image."""
    image_id: str
    ok: bool = True
    n_lines: int = 0
    n_json: int = 0
    n_png: Optional[int] = None             # unknown for union-mask comparison
    matched: int = 0
    lat_errs: List[float] = field(default_factory=list)   # per matched pair
    missing: List[str] = field(default_factory=list)       # formats absent
    note: str = ""
    mask_iou: Optional[float] = None

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
                count_tol: int = DEFAULT_COUNT_TOL,
                png_mask: Optional[np.ndarray] = None,
                mask_iou_tol: float = DEFAULT_MASK_IOU_TOL) -> ImageConsistency:
    """Compare up to three lane sets for one image; flag disagreement."""
    res = ImageConsistency(image_id=image_id)
    if missing:
        res.missing = list(missing)

    sets = {"lines": lines_lanes, "json": json_lanes, "png": png_lanes}
    res.n_lines = 0 if lines_lanes is None else len(lines_lanes)
    res.n_json = 0 if json_lanes is None else len(json_lanes)
    res.n_png = None if png_lanes is None else len(png_lanes)

    present = [k for k, v in sets.items() if v is not None]
    if len(present) < 2:
        res.note = "only one format present"
        res.ok = False
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
    ok = not res.missing
    notes = [f"missing: {','.join(res.missing)}"] if res.missing else []
    for k in ("json", "png"):
        if k == "png" and png_mask is not None:
            continue  # semantic union comparison below; no lossy inverse needed
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
    if png_mask is not None:
        rendered = np.zeros(png_mask.shape, dtype=np.uint8)
        for lane in ref:
            points = np.rint(np.asarray(lane)).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(rendered, [points], False, 1, 10, cv2.LINE_8)
        rendered = rendered != 0
        union = int(np.count_nonzero(rendered | png_mask))
        intersection = int(np.count_nonzero(rendered & png_mask))
        res.mask_iou = 1.0 if union == 0 else intersection / union
        if res.mask_iou < mask_iou_tol:
            res.ok = False
            notes.append(f"png union IoU {res.mask_iou:.4f} < {mask_iou_tol}")
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


def lane_sets_exact(a: Sequence[np.ndarray], b: Sequence[np.ndarray]) -> bool:
    """Return whether two ordered, lossless lane sets are pointwise identical."""
    return len(a) == len(b) and all(np.array_equal(x, y) for x, y in zip(a, b))


def write_badlist(report: ConsistencyReport, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(report.bad_list) + "\n", encoding="utf-8")


def run_manifest_consistency(manifest_path: str | Path, lane_root: str | Path,
                             *, lat_err_tol: float = DEFAULT_LAT_ERR_TOL,
                             count_tol: int = 0,
                             mask_iou_tol: float = DEFAULT_MASK_IOU_TOL,
                             ) -> tuple[ConsistencyReport, dict]:
    """Run real-layout three-format consistency in official manifest order."""
    from data.manifest import read_manifest
    from data.parse_labels import parse_labels

    lane_root = Path(lane_root)
    records = read_manifest(manifest_path)
    results = []
    json_exact = 0
    empty_gt = 0
    for record in records:
        bundle = parse_labels(
            lane_root / record.image_path, extract_png_instances=False
        )
        result = check_image(
            record.image_id, bundle.lines_lanes, bundle.json_lanes, None,
            bundle.missing, lat_err_tol, count_tol,
            png_mask=bundle.png_mask, mask_iou_tol=mask_iou_tol,
        )
        results.append(result)
        lines = bundle.lines_lanes or []
        json_lanes = bundle.json_lanes or []
        exact = lane_sets_exact(lines, json_lanes)
        if exact:
            json_exact += 1
        else:
            result.ok = False
            result.note = "; ".join(filter(None, [
                result.note, "lines/json are not pointwise identical"
            ]))
        empty_gt += int(not lines)

    report = check_many(results)
    errors = np.asarray(
        [error for result in results for error in result.lat_errs], dtype=np.float64
    )
    mask_ious = np.asarray(
        [result.mask_iou for result in results if result.mask_iou is not None],
        dtype=np.float64,
    )
    summary = {
        "manifest": str(manifest_path),
        "total_images": report.total,
        "bad_images": report.bad,
        "bad_rate": report.bad_rate,
        "json_lines_exact_images": json_exact,
        "empty_gt_images": empty_gt,
        "lat_err_tol": lat_err_tol,
        "count_tol": count_tol,
        "mask_iou_tol": mask_iou_tol,
        "matched_pairs": int(errors.size),
        "lat_err_px": {
            "max": float(errors.max()) if errors.size else 0.0,
            "p99": float(np.quantile(errors, 0.99)) if errors.size else 0.0,
            "median": float(np.median(errors)) if errors.size else 0.0,
        },
        "mask_union_iou": {
            "min": float(mask_ious.min()) if mask_ious.size else 0.0,
            "p01": float(np.quantile(mask_ious, 0.01)) if mask_ious.size else 0.0,
            "median": float(np.median(mask_ious)) if mask_ious.size else 0.0,
        },
        "bad_details": [asdict(result) for result in results if not result.ok],
    }
    return report, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--lane-root", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--badlist", required=True, type=Path)
    parser.add_argument("--lat-err-tol", type=float, default=DEFAULT_LAT_ERR_TOL)
    parser.add_argument("--count-tol", type=int, default=0)
    parser.add_argument("--mask-iou-tol", type=float, default=DEFAULT_MASK_IOU_TOL)
    args = parser.parse_args()
    report, summary = run_manifest_consistency(
        args.manifest, args.lane_root,
        lat_err_tol=args.lat_err_tol, count_tol=args.count_tol,
        mask_iou_tol=args.mask_iou_tol,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    write_badlist(report, args.badlist)
    print(json.dumps({key: summary[key] for key in (
        "total_images", "bad_images", "bad_rate", "json_lines_exact_images",
        "empty_gt_images", "matched_pairs", "lat_err_px", "mask_union_iou",
    )}))
    if report.bad_rate >= 0.001:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
