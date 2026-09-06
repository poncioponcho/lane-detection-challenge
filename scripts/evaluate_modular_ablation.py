#!/usr/bin/env python3
"""Evaluate small post-processing modules on the corrected 36ep LVO export.

The script deliberately keeps the candidate pool and official scoring separate:
the pool is exported with conf_threshold=0.0 and the frozen official Oracle
scores every materialized variant. This makes confidence, image-conditioned
confidence, and geometry gates directly comparable without re-training.

Selection is fixed before the held-out video is scored. The luma/geometry
grid is a cheap screen, not a license to choose a best point on all eight
videos. stable_gain therefore requires the configured effect-size gate,
a strictly positive paired-bootstrap lower bound, and positive LOCO deltas.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
for import_root in (SRC_ROOT, SCRIPTS_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from data.manifest import ManifestRecord, read_manifest  # noqa: E402
from eval.oracle_runner import run_official_eval  # noqa: E402
from eval.rasterize import parse_lines_txt  # noqa: E402
from eval.score_sidecar import (  # noqa: E402
    probability_score,
    validate_probability_sidecar,
)
from scan_lvo_geometry import paired_bootstrap, prediction_set, video_id  # noqa: E402


REFERENCE_THRESHOLD = 0.40
EFFECT_SIZE_GATE_PP = 0.50


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def threshold_name(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def _f1(tp: int, fp: int, fn: int) -> float:
    denominator = 2 * tp + fp + fn
    return 2.0 * tp / denominator if denominator else 0.0


def _lane_geometry(path: Path) -> tuple[list[float], list[float]]:
    lanes = parse_lines_txt(path)
    spans: list[float] = []
    lengths: list[float] = []
    for lane in lanes:
        values = np.asarray(lane, dtype=np.float64).reshape(-1, 2)
        spans.append(float(values[:, 1].max() - values[:, 1].min()))
        lengths.append(float(np.linalg.norm(np.diff(values, axis=0), axis=1).sum()))
    return spans, lengths


def lower_half_luma(path: Path) -> float:
    with Image.open(path) as image:
        gray = np.asarray(image.convert("L"), dtype=np.float32)
    if gray.ndim != 2 or gray.size == 0:
        raise ValueError(f"image is not a non-empty grayscale-convertible image: {path}")
    return float(gray[gray.shape[0] // 2 :].mean())


def _read_pool(
    records: list[ManifestRecord],
    prediction_root: Path,
    score_path: Path,
    image_root: Path,
) -> dict[str, dict[str, Any]]:
    expected = {record.pred_rel_path for record in records}
    if prediction_set(prediction_root) != expected:
        raise ValueError("prediction tree does not exactly cover the manifest")
    payload = json.loads(score_path.read_text(encoding="utf-8"))
    export_threshold = validate_probability_sidecar(payload, name=str(score_path))
    if not math.isclose(export_threshold, 0.0, abs_tol=1e-12):
        raise ValueError(
            "modular ablation requires a conf_threshold=0.0 candidate pool; "
            f"sidecar declares {export_threshold}"
        )
    scores_by_image = payload.get("scores_by_image")
    if not isinstance(scores_by_image, dict):
        raise ValueError("score sidecar has no scores_by_image object")

    pool: dict[str, dict[str, Any]] = {}
    for record in records:
        path = prediction_root / record.pred_rel_path
        lines = path.read_text(encoding="utf-8").splitlines()
        scores = scores_by_image.get(record.image_id)
        if not isinstance(scores, list) or len(scores) != len(lines):
            raise ValueError(
                f"{record.image_id}: score/line mismatch "
                f"{len(scores) if isinstance(scores, list) else 'missing'} != {len(lines)}"
            )
        numeric_scores = [
            probability_score(value, context=f"{record.image_id} score")
            for value in scores
        ]
        spans, lengths = _lane_geometry(path)
        if len(spans) != len(lines) or len(lengths) != len(lines):
            raise ValueError(f"{record.image_id}: geometry/line mismatch")
        pool[record.image_id] = {
            "record": record,
            "lines": lines,
            "scores": numeric_scores,
            "y_spans": spans,
            "lengths": lengths,
            "luma": lower_half_luma(image_root / record.image_path),
        }
    return pool


def _policy_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for threshold in (0.40, 0.45, 0.50, 0.55, 0.60):
        specs.append({
            "name": f"confidence_t{threshold_name(threshold)}",
            "module": "confidence",
            "kind": "score",
            "threshold": threshold,
        })
    for luma_threshold in (100.0, 110.0, 120.0):
        for high_threshold in (0.50, 0.60):
            specs.append({
                "name": (
                    f"luma_L{int(luma_threshold)}_base040_hi"
                    f"{threshold_name(high_threshold)}"
                ),
                "module": "luma_adaptive_threshold",
                "kind": "luma",
                "base_threshold": REFERENCE_THRESHOLD,
                "luma_threshold": luma_threshold,
                "high_threshold": high_threshold,
            })
    for minimum in (40.0, 60.0, 80.0):
        specs.append({
            "name": f"geometry_yspan_min{int(minimum)}_t040",
            "module": "geometry_yspan_gate",
            "kind": "y_span",
            "threshold": REFERENCE_THRESHOLD,
            "minimum": minimum,
        })
    for minimum in (150.0, 250.0):
        specs.append({
            "name": f"geometry_length_min{int(minimum)}_t040",
            "module": "geometry_length_gate",
            "kind": "length",
            "threshold": REFERENCE_THRESHOLD,
            "minimum": minimum,
        })
    return specs


def _keep(spec: dict[str, Any], pool_row: dict[str, Any], index: int) -> bool:
    score = float(pool_row["scores"][index])
    kind = spec["kind"]
    if kind == "score":
        return score >= float(spec["threshold"])
    if kind == "luma":
        threshold = (
            float(spec["high_threshold"])
            if float(pool_row["luma"]) > float(spec["luma_threshold"])
            else float(spec["base_threshold"])
        )
        return score >= threshold
    if kind == "y_span":
        return (
            score >= float(spec["threshold"])
            and float(pool_row["y_spans"][index]) >= float(spec["minimum"])
        )
    if kind == "length":
        return (
            score >= float(spec["threshold"])
            and float(pool_row["lengths"][index]) >= float(spec["minimum"])
        )
    raise ValueError(f"unknown policy kind: {kind}")


def _write_variant(
    spec: dict[str, Any],
    records: list[ManifestRecord],
    pool: dict[str, dict[str, Any]],
    output_root: Path,
) -> dict[str, Any]:
    target = output_root / spec["name"]
    target.mkdir(parents=True, exist_ok=False)
    before = after = empty = 0
    for record in records:
        row = pool[record.image_id]
        kept = [
            line for index, line in enumerate(row["lines"])
            if _keep(spec, row, index)
        ]
        path = target / record.pred_rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
        before += len(row["lines"])
        after += len(kept)
        empty += not kept
    if prediction_set(target) != {record.pred_rel_path for record in records}:
        raise ValueError(f"{spec['name']}: prediction set mismatch")
    return {
        **spec,
        "path": str(target),
        "lines_before": before,
        "lines_after": after,
        "empty_images": empty,
    }


def _score_variant(
    meta: dict[str, Any],
    records: list[ManifestRecord],
    gt_dir: Path,
    official_python: Path,
    oracle_root: Path,
) -> dict[str, Any]:
    name = meta["name"]
    global_result = run_official_eval(
        meta["path"],
        gt_dir,
        records,
        official_python=official_python,
        per_clip=False,
        output_path=oracle_root / name / "oracle_global.json",
    ).to_dict()
    grouped: dict[str, list[ManifestRecord]] = {}
    for record in records:
        grouped.setdefault(video_id(record.clip_id), []).append(record)
    videos = []
    for video, original_subset in sorted(grouped.items()):
        subset = [
            replace(record, order=index)
            for index, record in enumerate(original_subset)
        ]
        result = run_official_eval(
            meta["path"],
            gt_dir,
            subset,
            official_python=official_python,
            per_clip=False,
            output_path=oracle_root / name / f"oracle_{video}.json",
        ).to_dict()["global"]
        videos.append({
            "video": video,
            "tp": int(result["tp"]),
            "fp": int(result["fp"]),
            "fn": int(result["fn"]),
            "precision": float(result["precision"]),
            "recall": float(result["recall"]),
            "f1": float(result["f1"]),
        })
    counts = global_result["global"]
    summed = tuple(sum(row[key] for row in videos) for key in ("tp", "fp", "fn"))
    if summed != tuple(counts[key] for key in ("tp", "fp", "fn")):
        raise ValueError(f"{name}: global/video Oracle count mismatch")
    return {**meta, "global": counts, "videos": videos}


def _loco_deltas(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    reference_rows = {row["video"]: row for row in reference["videos"]}
    candidate_rows = {row["video"]: row for row in candidate["videos"]}
    if set(reference_rows) != set(candidate_rows):
        raise ValueError("LOCO video sets differ")
    values: dict[str, float] = {}
    for omitted in sorted(reference_rows):
        base = {
            key: sum(
                row[key] for video, row in reference_rows.items() if video != omitted
            )
            for key in ("tp", "fp", "fn")
        }
        cand = {
            key: sum(
                row[key] for video, row in candidate_rows.items() if video != omitted
            )
            for key in ("tp", "fp", "fn")
        }
        values[omitted] = (_f1(cand["tp"], cand["fp"], cand["fn"])
                          - _f1(base["tp"], base["fp"], base["fn"])) * 100.0
    ordered = list(values.values())
    return {
        "delta_pp_by_omitted_video": values,
        "min_delta_pp": float(min(ordered)),
        "max_delta_pp": float(max(ordered)),
        "median_delta_pp": float(np.median(ordered)),
        "all_positive": bool(all(value > 0.0 for value in ordered)),
    }


def _annotate_results(
    reference: dict[str, Any],
    results: list[dict[str, Any]],
    *,
    bootstrap: int,
    seed: int,
) -> None:
    for row in results:
        row["paired_bootstrap"] = paired_bootstrap(
            reference, row, seed=seed, n_bootstrap=bootstrap
        )
        row["loco"] = _loco_deltas(reference, row)
        ci_low = row["paired_bootstrap"]["ci95_delta_pp"][0]
        observed = row["paired_bootstrap"]["observed_delta_pp"]
        row["stable_gain"] = bool(
            observed >= EFFECT_SIZE_GATE_PP
            and ci_low > 0.0
            and row["loco"]["all_positive"]
        )


def _write_report(path: Path, value: dict[str, Any]) -> None:
    lines = [
        "# 36ep corrected sidecar modular ablation",
        "",
        f"- reference: {value['reference']} (corrected positive-class softmax probability)",
        f"- protocol: {value['protocol']}",
        "- all variants use the frozen official Oracle; CI is paired at video level.",
        f"- adoption gate: observed delta pp >= {EFFECT_SIZE_GATE_PP:.2f}, "
        "paired CI lower bound > 0, and all LOCO deltas > 0.",
        "",
        "| variant | module | lines | empty | F1 | delta pp | paired CI(pp) | LOCO min/max(pp) | stable |",
        "|---|---|---:|---:|---:|---:|---|---|---|",
    ]
    for row in value["results"]:
        boot = row["paired_bootstrap"]
        ci = boot["ci95_delta_pp"]
        loco = row["loco"]
        lines.append(
            f"| {row['name']} | {row['module']} | {row['lines_after']} | "
            f"{row['empty_images']} | {row['global']['f1']:.6f} | "
            f"{boot['observed_delta_pp']:+.3f} | "
            f"[{ci[0]:+.3f}, {ci[1]:+.3f}] | "
            f"[{loco['min_delta_pp']:+.3f}, {loco['max_delta_pp']:+.3f}] | "
            f"{'YES' if row['stable_gain'] else 'NO'} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- confidence is the corrected probability threshold module.",
        "- luma_adaptive_threshold changes only the score gate for bright lower-half images.",
        "- geometry_* keeps candidates above the probability threshold and applies one "
        "geometry gate; it does not use GT or candidate labels.",
        "- A positive point estimate without the stated gate is not a submission candidate.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--prediction-root", required=True, type=Path)
    parser.add_argument("--score-sidecar", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--gt-dir", required=True, type=Path)
    parser.add_argument("--official-python", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error(f"refusing to overwrite existing output: {args.output_dir}")
    if args.bootstrap < 1000:
        parser.error("--bootstrap must be at least 1000")

    records = read_manifest(args.manifest)
    pool = _read_pool(records, args.prediction_root, args.score_sidecar, args.image_root)
    args.output_dir.mkdir(parents=True)
    prediction_root = args.output_dir / "predictions"
    oracle_root = args.output_dir / "oracle"
    metas = [
        _write_variant(spec, records, pool, prediction_root)
        for spec in _policy_specs()
    ]
    # The local runner may need PYTHONPATH for Pillow/OpenCV. The official
    # Python must not inherit that path, otherwise user-site packages can
    # shadow its pinned NumPy/SciPy/OpenCV environment.
    runtime_pythonpath = os.environ.pop("PYTHONPATH", None)
    try:
        results = [
            _score_variant(meta, records, args.gt_dir, args.official_python, oracle_root)
            for meta in metas
        ]
    finally:
        if runtime_pythonpath is not None:
            os.environ["PYTHONPATH"] = runtime_pythonpath
    reference = next(
        row for row in results if row["name"] == "confidence_t0p40"
    )
    _annotate_results(reference, results, bootstrap=args.bootstrap, seed=args.seed)
    value = {
        "status": "pass",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": (
            "36ep LVO materialized modular ablation on corrected post-NMS "
            "positive-class softmax probability sidecar"
        ),
        "manifest": {
            "path": str(args.manifest.resolve()),
            "sha256": sha256_file(args.manifest),
            "records": len(records),
            "videos": len({video_id(record.clip_id) for record in records}),
        },
        "prediction_root": str(args.prediction_root.resolve()),
        "score_sidecar": {
            "path": str(args.score_sidecar.resolve()),
            "sha256": sha256_file(args.score_sidecar),
        },
        "image_root": str(args.image_root.resolve()),
        "gt_dir": str(args.gt_dir.resolve()),
        "official_python": str(args.official_python.resolve()),
        "reference": reference["name"],
        "bootstrap": {
            "unit": "video",
            "paired": True,
            "n": args.bootstrap,
            "seed": args.seed,
        },
        "effect_size_gate_pp": EFFECT_SIZE_GATE_PP,
        "results": results,
    }
    json_path = args.output_dir / "modular_ablation.json"
    value["json_path"] = str(json_path)
    json_path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_report(args.output_dir / "modular_ablation.md", value)
    stable = [row["name"] for row in results if row["stable_gain"]]
    print(json.dumps({
        "status": "pass",
        "json": str(json_path),
        "markdown": str(args.output_dir / "modular_ablation.md"),
        "variants": len(results),
        "reference_f1": reference["global"]["f1"],
        "stable_gain_variants": stable,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
