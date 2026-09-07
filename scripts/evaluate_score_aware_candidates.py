#!/usr/bin/env python3
"""Exact Oracle evaluation of cross-fit score-aware candidate selections.

This evaluator consumes the raw post-NMS prediction pool and the OOF feature
ranker scores produced by ``analyze_score_aware_candidates.py``.  It supports
two deliberately small policies: a global ranker-score threshold and a
per-image ranker Top-K cap.  Every output is scored through the frozen
official Oracle, with video-level paired bootstrap against a supplied
reference (normally ``final_conf_0p40``).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from data.manifest import ManifestRecord, read_manifest  # noqa: E402
from eval.oracle_runner import run_official_eval  # noqa: E402
from scan_lvo_geometry import paired_bootstrap, prediction_set, video_id  # noqa: E402


def threshold_name(value: float) -> str:
    return f"{value:.3f}".replace(".", "p")


def _read_scores(path: Path) -> dict[str, list[float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("score_semantics") != "cross_fit_logistic_probability":
        raise ValueError(f"unexpected score-aware semantics: {path}")
    scores = payload.get("ranker_scores_by_image")
    if not isinstance(scores, dict) or not all(isinstance(value, list) for value in scores.values()):
        raise ValueError(f"invalid ranker_scores_by_image: {path}")
    return {str(key): [float(value) for value in values] for key, values in scores.items()}


def _expected_prediction_set(records: list[ManifestRecord]) -> set[str]:
    return {record.pred_rel_path for record in records}


def write_variant(
    name: str,
    records: list[ManifestRecord],
    raw_root: Path,
    ranker_scores: dict[str, list[float]],
    output_root: Path,
    *,
    threshold: float | None = None,
    topk: int | None = None,
) -> dict[str, Any]:
    if (threshold is None) == (topk is None):
        raise ValueError("exactly one of threshold or topk is required")
    if threshold is not None and (not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0):
        raise ValueError(f"ranker threshold must be in [0, 1], got {threshold}")
    if topk is not None and topk < 1:
        raise ValueError(f"topk must be positive, got {topk}")
    target = output_root / name
    target.mkdir(parents=True, exist_ok=False)
    before = after = empty = 0
    for record in records:
        source = raw_root / record.pred_rel_path
        lines = source.read_text(encoding="utf-8").splitlines() if source.exists() else []
        scores = ranker_scores.get(record.image_id)
        if scores is None or len(scores) != len(lines):
            raise ValueError(
                f"{name}: line/ranker score mismatch for {record.image_id}: "
                f"{len(lines)} != {len(scores) if scores is not None else 'missing'}"
            )
        if threshold is not None:
            keep = [float(score) >= threshold for score in scores]
        else:
            order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
            selected = set(order[: min(topk or 0, len(order))])
            keep = [index in selected for index in range(len(scores))]
        kept = [line for line, selected in zip(lines, keep) if selected]
        destination = target / record.pred_rel_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
        before += len(lines)
        after += len(kept)
        empty += not kept
    actual = prediction_set(target)
    expected = _expected_prediction_set(records)
    if actual != expected:
        raise ValueError(f"{name}: prediction set mismatch {len(actual)} != {len(expected)}")
    result: dict[str, Any] = {
        "name": name,
        "policy": "ranker_threshold" if threshold is not None else "ranker_topk",
        "threshold": threshold,
        "topk": topk,
        "path": str(target),
        "lines_before": before,
        "lines_after": after,
        "empty_images": empty,
    }
    return result


def score_variant(
    meta: dict[str, Any],
    records: list[ManifestRecord],
    gt_dir: Path,
    official_python: Path,
    oracle_root: Path,
) -> dict[str, Any]:
    name = meta["name"]
    global_result = run_official_eval(
        Path(meta["path"]), gt_dir, records,
        official_python=official_python, per_clip=False,
        output_path=oracle_root / name / "oracle_global.json",
    ).to_dict()
    grouped: dict[str, list[ManifestRecord]] = {}
    for record in records:
        grouped.setdefault(video_id(record.clip_id), []).append(record)
    videos = []
    for video, original_subset in sorted(grouped.items()):
        subset = [replace(record, order=index) for index, record in enumerate(original_subset)]
        counts = run_official_eval(
            Path(meta["path"]), gt_dir, subset,
            official_python=official_python, per_clip=False,
            output_path=oracle_root / name / f"oracle_{video}.json",
        ).to_dict()["global"]
        videos.append({
            "video": video,
            "tp": int(counts["tp"]),
            "fp": int(counts["fp"]),
            "fn": int(counts["fn"]),
            "precision": float(counts["precision"]),
            "recall": float(counts["recall"]),
            "f1": float(counts["f1"]),
        })
    counts = global_result["global"]
    if tuple(sum(row[key] for row in videos) for key in ("tp", "fp", "fn")) != tuple(
        counts[key] for key in ("tp", "fp", "fn")
    ):
        raise ValueError(f"{name}: global/video Oracle counts disagree")
    return {**meta, "global": counts, "videos": videos}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--gt-dir", required=True, type=Path)
    parser.add_argument("--official-python", required=True, type=Path)
    parser.add_argument("--raw-predictions", required=True, type=Path)
    parser.add_argument("--ranker-scores", required=True, type=Path)
    parser.add_argument("--reference-result", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--ranker-threshold", action="append", type=float, default=[])
    parser.add_argument("--topk", action="append", type=int, default=[])
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error(f"refusing to overwrite existing output: {args.output_dir}")
    records = read_manifest(args.manifest)
    expected = _expected_prediction_set(records)
    if prediction_set(args.raw_predictions) != expected:
        parser.error("raw prediction set does not cover manifest")
    scores = _read_scores(args.ranker_scores)
    if set(scores) != {record.image_id for record in records}:
        parser.error("ranker score image set does not cover manifest")
    reference_payload = json.loads(args.reference_result.read_text(encoding="utf-8"))
    references = {row["name"]: row for row in reference_payload.get("results", [])}
    reference = references.get("final_conf_0p40")
    if reference is None:
        parser.error("reference result has no final_conf_0p40 row")

    args.output_dir.mkdir(parents=True)
    prediction_root = args.output_dir / "predictions"
    oracle_root = args.output_dir / "oracle"
    metas = []
    for threshold in args.ranker_threshold:
        name = f"ranker_thr_{threshold_name(threshold)}"
        metas.append(write_variant(name, records, args.raw_predictions, scores, prediction_root, threshold=threshold))
    for topk in args.topk:
        name = f"ranker_topk_{topk}"
        metas.append(write_variant(name, records, args.raw_predictions, scores, prediction_root, topk=topk))
    if not metas:
        parser.error("provide at least one --ranker-threshold or --topk")

    results = [score_variant(meta, records, args.gt_dir, args.official_python, oracle_root) for meta in metas]
    for row in results:
        row["paired_bootstrap"] = paired_bootstrap(
            reference, row, seed=args.seed, n_bootstrap=args.bootstrap
        )
    value = {
        "status": "pass",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "video-cross-fit feature ranker; exact official Oracle after candidate selection",
        "manifest": str(args.manifest.resolve()),
        "raw_predictions": str(args.raw_predictions.resolve()),
        "ranker_scores": str(args.ranker_scores.resolve()),
        "reference": "final_conf_0p40",
        "bootstrap": {"unit": "video", "paired": True, "n": args.bootstrap, "seed": args.seed},
        "results": results,
    }
    json_path = args.output_dir / "score_aware_eval.json"
    json_path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Score-aware candidate Oracle evaluation",
        "",
        "- reference: `final_conf_0p40` from corrected probability scan.",
        "- ranker predictions are video-cross-fit; all selection results use the frozen official Oracle.",
        "",
        "| variant | policy | threshold | topk | lines | empty | F1 | Δpp vs ref | paired CI(pp) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in results:
        boot = row["paired_bootstrap"]
        ci = boot["ci95_delta_pp"]
        threshold_text = "" if row["threshold"] is None else f"{row['threshold']:.3f}"
        topk_text = "" if row["topk"] is None else str(row["topk"])
        lines.append(
            f"| {row['name']} | {row['policy']} | "
            f"{threshold_text} | {topk_text} | {row['lines_after']} | "
            f"{row['empty_images']} | {row['global']['f1']:.6f} | "
            f"{boot['observed_delta_pp']:+.3f} | [{ci[0]:+.3f}, {ci[1]:+.3f}] |"
        )
    lines.extend([
        "",
        "A candidate policy is not a production change unless the paired CI lower bound is positive and the result is stable across videos.",
        "",
        f"- JSON: {json_path}",
    ])
    (args.output_dir / "score_aware_eval.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "output": str(args.output_dir), "variants": len(results)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
