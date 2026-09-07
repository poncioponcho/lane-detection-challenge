#!/usr/bin/env python3
"""Check direct CLRNet decoding against an offline-filtered export.

The direct directory must come from an eval-only run with
``conf_threshold=<threshold>``.  The comparison directory must be the result
of filtering a post-NMS export made with a lower threshold (normally 0.0).
Exact line-file equality checks geometry; optional sidecar comparison also
checks that the same probability-ranked lines survived the gate.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eval.score_sidecar import (  # noqa: E402
    probability_score,
    validate_probability_sidecar,
)


def prediction_set(root: Path) -> set[str]:
    if not root.is_dir():
        return set()
    return {
        path.relative_to(root).as_posix()
        for path in root.rglob("*.lines.txt")
        if path.is_file()
    }


def _image_id(relative_path: str) -> str:
    suffix = ".lines.txt"
    if not relative_path.endswith(suffix):
        raise ValueError(f"prediction path is not a .lines.txt file: {relative_path}")
    return relative_path[: -len(suffix)]


def _read_scores(path: Path, label: str) -> tuple[dict[str, Any], dict[str, list[Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_probability_sidecar(payload, name=f"{label}: {path}")
    scores = payload.get("scores_by_image")
    if not isinstance(scores, dict):
        raise ValueError(f"{label}: scores_by_image is not an object: {path}")
    return payload, scores


def compare_decode_outputs(
    direct_root: Path,
    offline_filtered_root: Path,
    threshold: float,
    direct_scores_path: Path | None = None,
    offline_scores_path: Path | None = None,
) -> dict[str, Any]:
    """Return a structured pass/fail report for the two output directories."""
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be a finite probability in [0, 1], got {threshold!r}")
    direct_files = prediction_set(direct_root)
    offline_files = prediction_set(offline_filtered_root)
    missing = sorted(offline_files - direct_files)
    extra = sorted(direct_files - offline_files)
    differing = []
    for relative in sorted(direct_files & offline_files):
        if (direct_root / relative).read_bytes() != (offline_filtered_root / relative).read_bytes():
            differing.append(relative)

    report: dict[str, Any] = {
        "status": "pass" if not missing and not extra and not differing else "mismatch",
        "threshold": threshold,
        "direct_prediction_count": len(direct_files),
        "offline_prediction_count": len(offline_files),
        "missing_from_direct": missing[:20],
        "extra_in_direct": extra[:20],
        "differing_line_files": differing[:20],
        "differing_line_file_count": len(differing),
    }

    if (direct_scores_path is None) != (offline_scores_path is None):
        raise ValueError("provide both --direct-scores and --offline-scores, or neither")
    if direct_scores_path is None:
        return report

    direct_payload, direct_scores = _read_scores(direct_scores_path, "direct")
    offline_payload, offline_scores = _read_scores(offline_scores_path, "offline")
    direct_export_threshold = float(direct_payload["candidate_export_conf_threshold"])
    offline_export_threshold = float(offline_payload["candidate_export_conf_threshold"])
    if not math.isclose(direct_export_threshold, threshold, abs_tol=1e-12):
        raise ValueError(
            f"direct sidecar export threshold {direct_export_threshold} != requested {threshold}"
        )
    if threshold + 1e-12 < offline_export_threshold:
        raise ValueError(
            f"requested threshold {threshold} is below offline export threshold "
            f"{offline_export_threshold}"
        )

    score_mismatches = []
    for relative in sorted(direct_files & offline_files):
        image_id = _image_id(relative)
        direct_values = direct_scores.get(image_id)
        raw_values = offline_scores.get(image_id)
        direct_lines = (direct_root / relative).read_text(encoding="utf-8").splitlines()
        if not isinstance(direct_values, list) or len(direct_values) != len(direct_lines):
            score_mismatches.append({"image_id": image_id, "reason": "direct score/line mismatch"})
            continue
        if not isinstance(raw_values, list):
            score_mismatches.append({"image_id": image_id, "reason": "offline score entry missing"})
            continue
        try:
            expected_values = [
                probability_score(value, context=f"offline score for {image_id}")
                for value in raw_values
                if value is not None and float(value) >= threshold
            ]
            actual_values = [
                probability_score(value, context=f"direct score for {image_id}")
                for value in direct_values
            ]
        except ValueError as exc:
            score_mismatches.append({"image_id": image_id, "reason": str(exc)})
            continue
        if len(expected_values) != len(actual_values) or any(
            not math.isclose(expected, actual, rel_tol=0.0, abs_tol=1e-12)
            for expected, actual in zip(expected_values, actual_values)
        ):
            score_mismatches.append({
                "image_id": image_id,
                "reason": "direct scores differ from offline threshold filter",
                "expected_count": len(expected_values),
                "actual_count": len(actual_values),
            })
    report["score_check"] = {
        "status": "pass" if not score_mismatches else "mismatch",
        "direct_sidecar": str(direct_scores_path),
        "offline_sidecar": str(offline_scores_path),
        "mismatches": score_mismatches[:20],
        "mismatch_count": len(score_mismatches),
    }
    if score_mismatches:
        report["status"] = "mismatch"
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-predictions", required=True, type=Path)
    parser.add_argument("--offline-filtered-predictions", required=True, type=Path)
    parser.add_argument("--threshold", required=True, type=float)
    parser.add_argument("--direct-scores", type=Path)
    parser.add_argument("--offline-scores", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = compare_decode_outputs(
            args.direct_predictions,
            args.offline_filtered_predictions,
            args.threshold,
            args.direct_scores,
            args.offline_scores,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
        return 2
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
