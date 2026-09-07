#!/usr/bin/env python3
"""Merge per-fold LVO score sidecars into one validated OOF sidecar.

``lvo_video_runner.py`` aggregates the exported ``*.lines.txt`` files but
intentionally does not copy ``prediction_scores.json``.  This utility joins
the scores by the authoritative ``oof_evidence.json`` index rather than by
directory order or a glob, and refuses to write a partial or ambiguous OOF
sidecar.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


SCORE_SCHEMA_VERSION = 1
SCORE_SEMANTICS = "positive_class_softmax_probability"
SCORE_RANGE = {"min": 0.0, "max": 1.0, "inclusive": True}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_once(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise SystemExit(f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def line_count(path: Path) -> int:
    if not path.is_file():
        raise SystemExit(f"missing aggregated OOF prediction: {path}")
    return len(path.read_text(encoding="utf-8").splitlines())


def validate_probability(value: Any, *, image_id: str, index: int) -> float:
    if isinstance(value, bool):
        raise SystemExit(f"boolean score for {image_id}[{index}]")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"non-numeric score for {image_id}[{index}]: {value!r}") from exc
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise SystemExit(f"out-of-range score for {image_id}[{index}]: {value!r}")
    return numeric


def safe_rel_path(value: Any, *, image_id: str) -> str:
    if not isinstance(value, str) or not value:
        raise SystemExit(f"missing pred_rel_path for {image_id}")
    rel = PurePosixPath(value)
    if rel.is_absolute() or ".." in rel.parts:
        raise SystemExit(f"unsafe pred_rel_path for {image_id}: {value!r}")
    return rel.as_posix()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"missing JSON artifact: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"expected JSON object: {path}")
    return value


def validate_sidecar_contract(payload: dict[str, Any], path: Path) -> float:
    if payload.get("status") != "pass":
        raise SystemExit(f"score sidecar is not passing: {path}")
    if payload.get("score_schema_version") != SCORE_SCHEMA_VERSION:
        raise SystemExit(f"unsupported score schema: {path}")
    if payload.get("score_semantics") != SCORE_SEMANTICS:
        raise SystemExit(f"score semantics mismatch: {path}")
    if payload.get("score_range") != SCORE_RANGE:
        raise SystemExit(f"score range mismatch: {path}")
    if payload.get("post_nms") is not True:
        raise SystemExit(f"score sidecar is not post-NMS: {path}")
    threshold = payload.get("candidate_export_conf_threshold")
    if isinstance(threshold, bool):
        raise SystemExit(f"invalid export threshold: {path}")
    try:
        threshold = float(threshold)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"invalid export threshold: {path}") from exc
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise SystemExit(f"invalid export threshold: {path}")
    return threshold


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", required=True, type=Path)
    parser.add_argument(
        "--oof-evidence", type=Path,
        help="defaults to EXPERIMENT_ROOT/oof/oof_evidence.json",
    )
    parser.add_argument(
        "--output", type=Path,
        help="defaults to EXPERIMENT_ROOT/oof/prediction_scores.json",
    )
    parser.add_argument(
        "--report", type=Path,
        help="defaults to EXPERIMENT_ROOT/oof/score_merge_evidence.json",
    )
    args = parser.parse_args()

    experiment_root = args.experiment_root.resolve()
    oof_root = experiment_root / "oof"
    evidence_path = (args.oof_evidence or oof_root / "oof_evidence.json").resolve()
    output_path = (args.output or oof_root / "prediction_scores.json").resolve()
    report_path = (
        args.report or oof_root / "score_merge_evidence.json"
    ).resolve()

    evidence = read_json(evidence_path)
    if evidence.get("status") != "pass":
        raise SystemExit(f"OOF evidence is not passing: {evidence_path}")
    index = evidence.get("index")
    if not isinstance(index, dict) or not index:
        raise SystemExit(f"OOF evidence has no image index: {evidence_path}")

    by_fold: dict[str, dict[str, str]] = defaultdict(dict)
    for image_id, entry in index.items():
        if not isinstance(image_id, str) or not image_id:
            raise SystemExit("OOF evidence contains an invalid image_id")
        if not isinstance(entry, dict):
            raise SystemExit(f"invalid OOF index entry: {image_id}")
        fold_name = entry.get("fold_name")
        if not isinstance(fold_name, str) or not fold_name:
            raise SystemExit(f"missing fold_name for {image_id}")
        rel = safe_rel_path(entry.get("pred_rel_path"), image_id=image_id)
        if image_id in by_fold[fold_name]:
            raise SystemExit(f"duplicate OOF image: {image_id}")
        by_fold[fold_name][image_id] = rel

    pred_root = oof_root / "predictions"
    merged: dict[str, list[float]] = {}
    fold_reports = []
    export_threshold: float | None = None
    for fold_name in sorted(by_fold):
        sidecar_path = (
            experiment_root
            / "runs"
            / fold_name
            / "holdout_eval"
            / "val"
            / "prediction_scores.json"
        )
        sidecar = read_json(sidecar_path)
        fold_threshold = validate_sidecar_contract(sidecar, sidecar_path)
        if export_threshold is None:
            export_threshold = fold_threshold
        elif not math.isclose(export_threshold, fold_threshold, rel_tol=0.0, abs_tol=1e-12):
            raise SystemExit(
                f"inconsistent export thresholds: {export_threshold} vs "
                f"{fold_threshold} ({sidecar_path})"
            )
        scores_by_image = sidecar.get("scores_by_image")
        if not isinstance(scores_by_image, dict):
            raise SystemExit(f"missing scores_by_image: {sidecar_path}")
        expected_images = set(by_fold[fold_name])
        if set(scores_by_image) != expected_images:
            missing = expected_images - set(scores_by_image)
            extra = set(scores_by_image) - expected_images
            raise SystemExit(
                f"score image set mismatch for {fold_name}: "
                f"missing={len(missing)} extra={len(extra)}"
            )

        for image_id, rel in by_fold[fold_name].items():
            values = scores_by_image[image_id]
            if not isinstance(values, list):
                raise SystemExit(f"scores for {image_id} are not a list")
            prediction_path = pred_root / rel
            expected_count = line_count(prediction_path)
            if len(values) != expected_count:
                raise SystemExit(
                    f"score/line mismatch for {image_id}: "
                    f"scores={len(values)} lines={expected_count}"
                )
            if image_id in merged:
                raise SystemExit(f"duplicate merged score image: {image_id}")
            merged[image_id] = [
                validate_probability(value, image_id=image_id, index=position)
                for position, value in enumerate(values)
            ]

        fold_reports.append({
            "fold_name": fold_name,
            "images": len(expected_images),
            "sidecar": str(sidecar_path),
            "sidecar_sha256": sha256_file(sidecar_path),
            "candidate_export_conf_threshold": fold_threshold,
        })

    expected_ids = set(index)
    if set(merged) != expected_ids:
        raise SystemExit(
            f"merged score coverage mismatch: expected={len(expected_ids)} "
            f"actual={len(merged)}"
        )
    assert export_threshold is not None

    payload = {
        "status": "pass",
        "protocol": "leave-one-video-out score-sidecar merge",
        "candidate_export_conf_threshold": export_threshold,
        "score_schema_version": SCORE_SCHEMA_VERSION,
        "score_semantics": SCORE_SEMANTICS,
        "score_range": SCORE_RANGE,
        "post_nms": True,
        "images": len(merged),
        "scores_by_image": merged,
    }
    report = {
        "status": "pass",
        "protocol": "leave-one-video-out score-sidecar merge",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_root": str(experiment_root),
        "oof_evidence": str(evidence_path),
        "oof_evidence_sha256": sha256_file(evidence_path),
        "output": str(output_path),
        "image_count": len(merged),
        "fold_count": len(fold_reports),
        "candidate_export_conf_threshold": export_threshold,
        "score_schema_version": SCORE_SCHEMA_VERSION,
        "score_semantics": SCORE_SEMANTICS,
        "post_nms": True,
        "folds": fold_reports,
    }
    if output_path == report_path:
        raise SystemExit("--output and --report must be different paths")
    if output_path.exists():
        raise SystemExit(f"refusing to overwrite existing output: {output_path}")
    if report_path.exists():
        raise SystemExit(f"refusing to overwrite existing output: {report_path}")
    write_json_once(output_path, payload)
    write_json_once(report_path, report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
