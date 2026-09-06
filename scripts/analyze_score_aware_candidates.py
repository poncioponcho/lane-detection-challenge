#!/usr/bin/env python3
"""Build video-cross-fit score-aware candidate diagnostics for LVO OOF.

The input is a post-NMS candidate pool exported with positive-class softmax
probabilities.  Every candidate receives a diagnostic TP/FP label from the
same rasterization and Hungarian assignment used by the frozen official
metric.  Labels are only used to fit models on other videos; all reported
candidate selections must still be checked with the official Oracle because
removing a candidate can change the per-image assignment.

This script deliberately keeps the ranker small and dependency-light:
standardized logistic regression is fitted with SciPy, while Platt and
isotonic mappings are implemented locally.  The output is an OOF score file
that a separate evaluator can use to materialize exact Oracle variants.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment, minimize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.manifest import ManifestRecord, read_manifest  # noqa: E402
from eval.rasterize import parse_lines_txt, rasterize_lanes  # noqa: E402
from eval.score_sidecar import (  # noqa: E402
    POSITIVE_CLASS_SOFTMAX_PROBABILITY,
    validate_probability_sidecar,
)


FEATURE_NAMES = (
    "score",
    "score_rank_fraction",
    "export_rank_fraction",
    "y_span_fraction",
    "line_length_fraction",
    "x_span_fraction",
    "candidate_count_fraction",
    "lower_half_luma_fraction",
)
CANVAS_W = 1366.0
CANVAS_H = 720.0
DIAGONAL = math.hypot(CANVAS_W, CANVAS_H)
MAX_CANDIDATES = 12.0
IOU_THRESHOLD = 0.5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def video_id(clip_id: str) -> str:
    marker = "_1_0_"
    if marker not in clip_id:
        raise ValueError(f"invalid clip id without {marker!r}: {clip_id}")
    return clip_id.split(marker, 1)[0]


def sigmoid(values: np.ndarray | float) -> np.ndarray | float:
    """Numerically stable logistic sigmoid."""
    array = np.asarray(values, dtype=np.float64)
    result = np.empty_like(array)
    positive = array >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-np.clip(array[positive], -60.0, 60.0)))
    exp_values = np.exp(np.clip(array[~positive], -60.0, 60.0))
    result[~positive] = exp_values / (1.0 + exp_values)
    if np.ndim(values) == 0:
        return float(result)
    return result


def fit_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    class_balance: bool,
    l2: float = 1e-3,
) -> dict[str, Any]:
    """Fit a deterministic standardized logistic model without sklearn."""
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if len(x) != len(y) or not len(x):
        raise ValueError("logistic fit requires equally sized non-empty arrays")
    if set(np.unique(y)) - {0.0, 1.0}:
        raise ValueError("logistic labels must be binary")
    if len(np.unique(y)) < 2:
        raise ValueError("logistic fit requires both classes")

    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    z = (x - mean) / scale
    design = np.column_stack((np.ones(len(z)), z))

    if class_balance:
        positive = max(float((y == 1).sum()), 1.0)
        negative = max(float((y == 0).sum()), 1.0)
        weights = np.where(y == 1, len(y) / (2.0 * positive), len(y) / (2.0 * negative))
    else:
        weights = np.ones(len(y), dtype=np.float64)
    weight_sum = float(weights.sum())

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        logits = design @ theta
        loss = np.logaddexp(0.0, logits) - y * logits
        value = float(np.sum(weights * loss) / weight_sum)
        value += float(l2 * np.dot(theta[1:], theta[1:]))
        probabilities = np.asarray(sigmoid(logits))
        gradient = design.T @ (weights * (probabilities - y)) / weight_sum
        gradient[1:] += 2.0 * l2 * theta[1:]
        return value, gradient

    result = minimize(
        lambda theta: objective(theta)[0],
        np.zeros(design.shape[1], dtype=np.float64),
        jac=lambda theta: objective(theta)[1],
        method="L-BFGS-B",
        options={"maxiter": 300, "ftol": 1e-12, "gtol": 1e-8},
    )
    if not result.success:
        raise RuntimeError(f"logistic optimization failed: {result.message}")
    return {
        "intercept": float(result.x[0]),
        "coef": [float(value) for value in result.x[1:]],
        "mean": [float(value) for value in mean],
        "scale": [float(value) for value in scale],
        "class_balance": bool(class_balance),
        "l2": float(l2),
        "iterations": int(getattr(result, "nit", -1)),
    }


def predict_logistic(model: dict[str, Any], features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    mean = np.asarray(model["mean"], dtype=np.float64)
    scale = np.asarray(model["scale"], dtype=np.float64)
    coef = np.asarray(model["coef"], dtype=np.float64)
    logits = float(model["intercept"]) + ((x - mean) / scale) @ coef
    return np.asarray(sigmoid(logits), dtype=np.float64)


def fit_isotonic(values: Iterable[float], labels: Iterable[int]) -> dict[str, list[float]]:
    """Fit a monotone PAVA mapping and return compact interpolation knots."""
    x = np.asarray(list(values), dtype=np.float64)
    y = np.asarray(list(labels), dtype=np.float64)
    if len(x) != len(y) or not len(x):
        raise ValueError("isotonic fit requires equally sized non-empty arrays")
    if set(np.unique(y)) - {0.0, 1.0}:
        raise ValueError("isotonic labels must be binary")
    order = np.argsort(x, kind="mergesort")
    x_sorted, y_sorted = x[order], y[order]
    unique_x, inverse = np.unique(x_sorted, return_inverse=True)
    counts = np.bincount(inverse).astype(np.float64)
    sums = np.bincount(inverse, weights=y_sorted).astype(np.float64)
    means = sums / counts

    block_x: list[float] = []
    block_weight: list[float] = []
    block_mean: list[float] = []
    for value, weight, mean in zip(unique_x, counts, means):
        block_x.append(float(value))
        block_weight.append(float(weight))
        block_mean.append(float(mean))
        while len(block_mean) >= 2 and block_mean[-2] > block_mean[-1]:
            right_x = block_x.pop()
            right_weight = block_weight.pop()
            right_mean = block_mean.pop()
            left_x = block_x.pop()
            left_weight = block_weight.pop()
            left_mean = block_mean.pop()
            merged_weight = left_weight + right_weight
            block_x.append(
                (left_weight * left_x + right_weight * right_x) / merged_weight
            )
            block_weight.append(merged_weight)
            block_mean.append(
                (left_weight * left_mean + right_weight * right_mean) / merged_weight
            )

    return {
        "x": block_x,
        "y": [min(1.0, max(0.0, value)) for value in block_mean],
    }


def predict_isotonic(model: dict[str, list[float]], values: Iterable[float]) -> np.ndarray:
    x = np.asarray(model["x"], dtype=np.float64)
    y = np.asarray(model["y"], dtype=np.float64)
    if not len(x):
        raise ValueError("isotonic model has no knots")
    return np.interp(np.asarray(list(values), dtype=np.float64), x, y, left=y[0], right=y[-1])


def _lane_features(points: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    y_span = float(values[:, 1].max() - values[:, 1].min())
    x_span = float(values[:, 0].max() - values[:, 0].min())
    length = float(np.linalg.norm(np.diff(values, axis=0), axis=1).sum())
    return y_span, x_span, length


def candidate_labels(
    predictions: list[np.ndarray], ground_truth: list[np.ndarray]
) -> tuple[list[int], list[float]]:
    """Return official-style per-prediction TP labels and matched IoUs."""
    labels = [0] * len(predictions)
    matched_ious = [0.0] * len(predictions)
    if not predictions or not ground_truth:
        return labels, matched_ious
    pred_masks = rasterize_lanes(predictions)
    gt_masks = rasterize_lanes(ground_truth)
    ious = np.zeros((len(predictions), len(ground_truth)), dtype=np.float64)
    for pred_index, pred_mask in enumerate(pred_masks):
        for gt_index, gt_mask in enumerate(gt_masks):
            union = np.logical_or(pred_mask, gt_mask).sum()
            if union:
                ious[pred_index, gt_index] = (
                    np.logical_and(pred_mask, gt_mask).sum() / union
                )
    rows, cols = linear_sum_assignment(1.0 - ious)
    for row, col in zip(rows, cols):
        matched_ious[int(row)] = float(ious[row, col])
        labels[int(row)] = int(ious[row, col] > IOU_THRESHOLD)
    return labels, matched_ious


def lower_half_luma(path: Path) -> float:
    with Image.open(path) as image:
        gray = np.asarray(image.convert("L"), dtype=np.float32)
    if gray.ndim != 2 or gray.size == 0:
        raise ValueError(f"image is not a non-empty grayscale-convertible image: {path}")
    return float(gray[gray.shape[0] // 2 :].mean())


def _binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = labels == 1
    negatives = labels == 0
    if not positives.any() or not negatives.any():
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    return float((ranks[positives].sum() - positives.sum() * (positives.sum() + 1) / 2) / (positives.sum() * negatives.sum()))


def _calibration_summary(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    clipped = np.clip(scores, 1e-7, 1.0 - 1e-7)
    return {
        "count": int(len(labels)),
        "positive_rate": float(labels.mean()) if len(labels) else 0.0,
        "auc": _binary_auc(labels, scores),
        "brier": float(np.mean((scores - labels) ** 2)),
        "logloss": float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped))),
    }


def _feature_row(
    record: ManifestRecord,
    candidate_index: int,
    candidate_count: int,
    score: float,
    score_rank: int,
    points: np.ndarray,
    label: int,
    matched_iou: float,
    luma: float,
) -> dict[str, Any]:
    y_span, x_span, length = _lane_features(points)
    denominator = max(candidate_count - 1, 1)
    feature_values = [
        score,
        score_rank / denominator,
        candidate_index / denominator,
        y_span / CANVAS_H,
        length / DIAGONAL,
        x_span / CANVAS_W,
        min(candidate_count, MAX_CANDIDATES) / MAX_CANDIDATES,
        luma / 255.0,
    ]
    return {
        "image_id": record.image_id,
        "clip_id": record.clip_id,
        "video_id": video_id(record.clip_id),
        "candidate_index": candidate_index,
        "score_rank": score_rank + 1,
        "candidate_count": candidate_count,
        "score": float(score),
        "y_span_px": y_span,
        "x_span_px": x_span,
        "line_length_px": length,
        "lower_half_luma": luma,
        "label_tp": int(label),
        "matched_iou": float(matched_iou),
        "features": [float(value) for value in feature_values],
    }


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def analyze(
    manifest_path: Path,
    data_root: Path,
    predictions_root: Path,
    scores_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    records = read_manifest(manifest_path)
    expected = {record.pred_rel_path for record in records}
    actual = {
        path.relative_to(predictions_root).as_posix()
        for path in predictions_root.rglob("*.lines.txt")
        if path.is_file()
    }
    if actual != expected:
        raise ValueError(f"prediction set mismatch: expected {len(expected)}, got {len(actual)}")
    payload = json.loads(scores_path.read_text(encoding="utf-8"))
    export_threshold = validate_probability_sidecar(payload, name=str(scores_path))
    if not math.isclose(export_threshold, 0.0, abs_tol=1e-12):
        raise ValueError(
            "score-aware analysis requires a 0.0 candidate export pool; "
            f"sidecar declares {export_threshold}"
        )
    scores_by_image = payload.get("scores_by_image")
    if not isinstance(scores_by_image, dict) or set(scores_by_image) != {r.image_id for r in records}:
        raise ValueError("score sidecar image coverage does not match manifest")

    rows: list[dict[str, Any]] = []
    by_image: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        pred_path = predictions_root / record.pred_rel_path
        gt_path = data_root / (record.gt_path or "")
        predictions = parse_lines_txt(pred_path)
        ground_truth = parse_lines_txt(gt_path)
        scores = [float(value) for value in scores_by_image[record.image_id]]
        if len(predictions) != len(scores):
            raise ValueError(
                f"{record.image_id}: prediction/score count mismatch "
                f"{len(predictions)} != {len(scores)}"
            )
        labels, matched_ious = candidate_labels(predictions, ground_truth)
        luma = lower_half_luma(data_root / record.image_path)
        score_order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
        score_ranks = {index: rank for rank, index in enumerate(score_order)}
        image_rows = [
            _feature_row(
                record,
                index,
                len(scores),
                scores[index],
                score_ranks[index],
                predictions[index],
                labels[index],
                matched_ious[index],
                luma,
            )
            for index in range(len(scores))
        ]
        rows.extend(image_rows)
        by_image[record.image_id] = image_rows

    if not rows:
        raise ValueError("candidate pool is empty")
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_jsonl(output_dir / "candidate_features.jsonl", rows)

    features = np.asarray([row["features"] for row in rows], dtype=np.float64)
    labels = np.asarray([row["label_tp"] for row in rows], dtype=np.int64)
    videos = np.asarray([row["video_id"] for row in rows])
    scores = np.asarray([row["score"] for row in rows], dtype=np.float64)
    platt_oof = np.full(len(rows), np.nan, dtype=np.float64)
    isotonic_oof = np.full(len(rows), np.nan, dtype=np.float64)
    ranker_oof = np.full(len(rows), np.nan, dtype=np.float64)
    folds = []
    for heldout in sorted(set(videos)):
        train = videos != heldout
        test = ~train
        platt_model = fit_logistic(scores[train], labels[train], class_balance=False, l2=1e-4)
        ranker_model = fit_logistic(features[train], labels[train], class_balance=True, l2=1e-3)
        isotonic_model = fit_isotonic(scores[train], labels[train])
        platt_oof[test] = predict_logistic(platt_model, scores[test])
        ranker_oof[test] = predict_logistic(ranker_model, features[test])
        isotonic_oof[test] = predict_isotonic(isotonic_model, scores[test])
        folds.append({
            "heldout_video": heldout,
            "train_candidates": int(train.sum()),
            "test_candidates": int(test.sum()),
            "train_positive_rate": float(labels[train].mean()),
            "test_positive_rate": float(labels[test].mean()),
            "platt": platt_model,
            "ranker": ranker_model,
            "isotonic": isotonic_model,
            "raw_score_summary": _calibration_summary(labels[test], scores[test]),
            "platt_summary": _calibration_summary(labels[test], platt_oof[test]),
            "isotonic_summary": _calibration_summary(labels[test], isotonic_oof[test]),
            "ranker_summary": _calibration_summary(labels[test], ranker_oof[test]),
        })
    if np.isnan(ranker_oof).any() or np.isnan(platt_oof).any() or np.isnan(isotonic_oof).any():
        raise RuntimeError("cross-fit left candidates without predictions")

    ranker_by_image: dict[str, list[float]] = defaultdict(list)
    platt_by_image: dict[str, list[float]] = defaultdict(list)
    isotonic_by_image: dict[str, list[float]] = defaultdict(list)
    row_index = {id(row): index for index, row in enumerate(rows)}
    for image_id, image_rows in by_image.items():
        for row in image_rows:
            index = row_index[id(row)]
            ranker_by_image[image_id].append(float(ranker_oof[index]))
            platt_by_image[image_id].append(float(platt_oof[index]))
            isotonic_by_image[image_id].append(float(isotonic_oof[index]))

    score_payload = {
        "status": "pass",
        "score_schema_version": 1,
        "score_semantics": "cross_fit_logistic_probability",
        "source_score_semantics": POSITIVE_CLASS_SOFTMAX_PROBABILITY,
        "source_candidate_export_conf_threshold": export_threshold,
        "post_nms": True,
        "feature_names": list(FEATURE_NAMES),
        "videos": sorted(set(videos)),
        "images": len(by_image),
        "candidates": len(rows),
        "ranker_scores_by_image": dict(ranker_by_image),
        "platt_scores_by_image": dict(platt_by_image),
        "isotonic_scores_by_image": dict(isotonic_by_image),
    }
    (output_dir / "score_aware_scores.json").write_text(
        json.dumps(score_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "status": "pass",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "candidate labels from official-style full-pool Hungarian matching; video-level cross-fit",
        "manifest": {"path": str(manifest_path.resolve()), "sha256": sha256_file(manifest_path), "rows": len(records)},
        "data_root": str(data_root.resolve()),
        "predictions_root": str(predictions_root.resolve()),
        "scores_path": str(scores_path.resolve()),
        "scores_sha256": sha256_file(scores_path),
        "candidate_export_conf_threshold": export_threshold,
        "post_nms": True,
        "features": list(FEATURE_NAMES),
        "videos": sorted(set(videos)),
        "images": len(by_image),
        "candidates": len(rows),
        "full_pool_positive_candidates": int(labels.sum()),
        "full_pool_negative_candidates": int((labels == 0).sum()),
        "raw_score_summary": _calibration_summary(labels, scores),
        "platt_oof_summary": _calibration_summary(labels, platt_oof),
        "isotonic_oof_summary": _calibration_summary(labels, isotonic_oof),
        "ranker_oof_summary": _calibration_summary(labels, ranker_oof),
        "folds": folds,
        "score_payload": str(output_dir / "score_aware_scores.json"),
        "candidate_features": str(output_dir / "candidate_features.jsonl"),
    }
    (output_dir / "score_aware_analysis.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Score-aware candidate analysis",
        "",
        f"- videos: {len(summary['videos'])}; images: {summary['images']}; candidates: {summary['candidates']}",
        f"- full-pool diagnostic TP candidates: {summary['full_pool_positive_candidates']}; FP candidates: {summary['full_pool_negative_candidates']}",
        "- labels use full-pool official-style Hungarian assignment; selections require exact Oracle re-evaluation.",
        "",
        "| score/model | AUC | Brier | log loss | positive rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, key in (
        ("raw probability", "raw_score_summary"),
        ("Platt OOF", "platt_oof_summary"),
        ("isotonic OOF", "isotonic_oof_summary"),
        ("feature ranker OOF", "ranker_oof_summary"),
    ):
        metric = summary[key]
        lines.append(
            f"| {name} | {metric['auc']:.6f} | {metric['brier']:.6f} | "
            f"{metric['logloss']:.6f} | {metric['positive_rate']:.6f} |"
        )
    lines.extend([
        "",
        "Platt/isotonic are monotone score calibrations and therefore do not change candidate ordering. "
        "The feature ranker is the only tested component that can change ordering.",
        "",
        f"- JSON: {output_dir / 'score_aware_analysis.json'}",
        f"- OOF scores: {output_dir / 'score_aware_scores.json'}",
    ])
    (output_dir / "score_aware_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--scores", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error(f"refusing to overwrite existing output: {args.output_dir}")
    try:
        summary = analyze(
            args.manifest.resolve(),
            args.data_root.resolve(),
            args.predictions.resolve(),
            args.scores.resolve(),
            args.output_dir.resolve(),
        )
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
        return 2
    print(json.dumps({"status": "pass", "output": str(args.output_dir), "candidates": summary["candidates"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
