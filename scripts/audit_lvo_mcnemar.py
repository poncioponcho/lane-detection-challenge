#!/usr/bin/env python3
"""Audit paired GT-lane disagreements between two LVO prediction systems.

For each image, both systems are matched independently to the same GT lanes
using the frozen official Oracle's interpolation, rasterization, Hungarian
assignment, and strict IoU threshold.  A GT lane is a paired unit:

* ``b``: baseline correct, candidate incorrect;
* ``c``: candidate correct, baseline incorrect.

Thus ``c - b`` is exactly the candidate-minus-baseline delta TP.  Prediction
FPs are not paired by this construction and are reported separately.  This
is an audit of paired disagreements, not a replacement for the official
video-cluster bootstrap or a claim that GT lanes are independent samples.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


def load_official_score(path: Path):
    spec = importlib.util.spec_from_file_location("frozen_official_score", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load official Oracle: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"expected JSON object: {path}")
    return value


def read_manifest(path: Path) -> list[dict[str, Any]]:
    records = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            record = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"invalid manifest JSON at {path}:{line_no}: {exc}") from exc
        if not isinstance(record, dict):
            raise SystemExit(f"invalid manifest row at {path}:{line_no}")
        required = {"image_id", "pred_rel_path", "gt_path", "clip_id"}
        if not required.issubset(record):
            raise SystemExit(f"manifest row missing fields at {path}:{line_no}")
        records.append(record)
    if not records:
        raise SystemExit(f"empty manifest: {path}")
    return records


def official_matches(score: Any, pred_path: Path, gt_path: Path) -> tuple[list[bool], int, int]:
    """Return per-GT correctness plus official TP and prediction count."""
    pred = score.parse_lines_txt(str(pred_path))
    gt = score.parse_lines_txt(str(gt_path))
    if not gt:
        return [], 0, len(pred)
    if not pred:
        return [False] * len(gt), 0, 0

    interp_pred = [score.interp_lane(lane) for lane in pred]
    interp_gt = [score.interp_lane(lane) for lane in gt]
    pred_masks = [score.draw_lane_mask(lane, score.LINE_WIDTH) for lane in interp_pred]
    gt_masks = [score.draw_lane_mask(lane, score.LINE_WIDTH) for lane in interp_gt]
    ious = np.zeros((len(pred_masks), len(gt_masks)), dtype=np.float64)
    for pred_index, pred_mask in enumerate(pred_masks):
        for gt_index, gt_mask in enumerate(gt_masks):
            union = np.logical_or(pred_mask, gt_mask).sum()
            if union:
                ious[pred_index, gt_index] = (
                    np.logical_and(pred_mask, gt_mask).sum() / union
                )

    # This is intentionally the same assignment as score.py: maximize total
    # IoU first, then apply the strict > 0.5 decision to assigned pairs.
    rows, cols = linear_sum_assignment(1.0 - ious)
    correct = [False] * len(gt)
    for row, col in zip(rows, cols):
        correct[int(col)] = bool(ious[row, col] > score.IOU_THRESHOLDS[0])
    return correct, int(sum(correct)), len(pred)


def exact_mcnemar_pvalue(b: int, c: int) -> float | None:
    """Two-sided exact conditional McNemar p-value without scipy.stats."""
    n = b + c
    if n == 0:
        return None
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * tail)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    base_tp = sum(row["baseline_tp"] for row in rows)
    cand_tp = sum(row["candidate_tp"] for row in rows)
    base_p = sum(row["baseline_p"] for row in rows)
    cand_p = sum(row["candidate_p"] for row in rows)
    gt = sum(row["gt"] for row in rows)
    b = sum(row["b"] for row in rows)
    c = sum(row["c"] for row in rows)
    both_correct = sum(row["both_correct"] for row in rows)
    both_wrong = sum(row["both_wrong"] for row in rows)
    result = {
        "images": len(rows),
        "gt": gt,
        "baseline_tp": base_tp,
        "candidate_tp": cand_tp,
        "baseline_fp": base_p - base_tp,
        "candidate_fp": cand_p - cand_tp,
        "delta_tp": cand_tp - base_tp,
        "delta_fp": (cand_p - cand_tp) - (base_p - base_tp),
        "b_baseline_only_correct": b,
        "c_candidate_only_correct": c,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "disagreements": b + c,
        "delta_tp_from_c_minus_b": c - b,
        "sqrt_b_plus_c": math.sqrt(b + c),
        "mcnemar_z_signed": ((c - b) / math.sqrt(b + c)) if b + c else None,
        "mcnemar_exact_two_sided_p": exact_mcnemar_pvalue(b, c),
    }
    if result["delta_tp"] != result["delta_tp_from_c_minus_b"]:
        raise SystemExit(f"internal paired TP mismatch: {result}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-pred-root", required=True, type=Path)
    parser.add_argument("--candidate-experiment-root", required=True, type=Path)
    parser.add_argument("--manifests-root", required=True, type=Path)
    parser.add_argument("--lane-root", required=True, type=Path)
    parser.add_argument("--official-score", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--fold", action="append",
        help="fold directory name; repeat to select folds (default: all passing folds)",
    )
    args = parser.parse_args()

    score_path = args.official_score.resolve()
    score = load_official_score(score_path)
    baseline_root = args.baseline_pred_root.resolve()
    candidate_root = args.candidate_experiment_root.resolve()
    manifests_root = args.manifests_root.resolve()
    lane_root = args.lane_root.resolve()

    if args.fold:
        fold_names = list(dict.fromkeys(args.fold))
    else:
        fold_names = []
        for state_path in sorted((candidate_root / "runs").glob("fold_*/fold_state.json")):
            state = read_json(state_path)
            if state.get("status") == "pass":
                fold_names.append(state_path.parent.name)
    if not fold_names:
        raise SystemExit("no folds selected")

    image_rows: list[dict[str, Any]] = []
    fold_results = []
    video_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for fold_name in fold_names:
        manifest_path = manifests_root / fold_name / "manifest_holdout.jsonl"
        records = read_manifest(manifest_path)
        state_path = candidate_root / "runs" / fold_name / "fold_state.json"
        state = read_json(state_path)
        if state.get("status") != "pass":
            raise SystemExit(f"selected fold is not pass: {fold_name}")
        candidate_pred_root = Path(
            state.get("holdout_eval", {}).get(
                "prediction_root",
                str(candidate_root / "runs" / fold_name / "holdout_eval" / "val" / "predictions"),
            )
        )
        if not candidate_pred_root.is_absolute():
            candidate_pred_root = candidate_root / candidate_pred_root
        fold_image_rows = []
        for record in records:
            image_id = str(record["image_id"])
            rel = Path(record["pred_rel_path"])
            baseline_path = baseline_root / rel
            candidate_path = candidate_pred_root / rel
            gt_rel = Path(record["gt_path"])
            gt_path = lane_root / gt_rel
            for path in (baseline_path, candidate_path, gt_path):
                if not path.is_file():
                    raise SystemExit(f"missing audit input for {image_id}: {path}")
            base_correct, base_tp, base_p = official_matches(score, baseline_path, gt_path)
            cand_correct, cand_tp, cand_p = official_matches(score, candidate_path, gt_path)
            if len(base_correct) != len(cand_correct):
                raise SystemExit(f"GT count mismatch for {image_id}")
            b = sum(x and not y for x, y in zip(base_correct, cand_correct))
            c = sum(y and not x for x, y in zip(base_correct, cand_correct))
            row = {
                "fold_name": fold_name,
                "video": str(record["clip_id"]).split("_1_0_", 1)[0],
                "image_id": image_id,
                "gt": len(base_correct),
                "baseline_tp": base_tp,
                "candidate_tp": cand_tp,
                "baseline_p": base_p,
                "candidate_p": cand_p,
                "b": int(b),
                "c": int(c),
                "both_correct": int(sum(x and y for x, y in zip(base_correct, cand_correct))),
                "both_wrong": int(sum(not x and not y for x, y in zip(base_correct, cand_correct))),
            }
            fold_image_rows.append(row)
            image_rows.append(row)
            video_rows[row["video"]].append(row)

        summary = summarize(fold_image_rows)
        # fold_state.diagnostic is the runner's fast diagnostic metric, not
        # the frozen official Oracle.  Keep the comparison as evidence, but
        # do not make it a gate: the purpose of this audit is to establish
        # the exact official-Hungarian paired units used below.
        diagnostic = state.get("holdout_eval", {}).get("diagnostic", {})
        diagnostic_triplet = {
            "tp": diagnostic.get("TP"),
            "fp": diagnostic.get("FP"),
            "fn": diagnostic.get("FN"),
        }
        official_triplet = {
            "tp": summary["candidate_tp"],
            "fp": summary["candidate_fp"],
            "fn": summary["gt"] - summary["candidate_tp"],
        }
        fold_results.append({
            "fold_name": fold_name,
            "summary": summary,
            "runner_diagnostic": diagnostic_triplet,
            "runner_diagnostic_matches_frozen_oracle": (
                diagnostic_triplet == official_triplet
            ),
            "frozen_oracle_candidate": official_triplet,
        })

    video_results = [
        {"video": video, "summary": summarize(rows)}
        for video, rows in sorted(video_rows.items())
    ]
    total = summarize(image_rows)
    value = {
        "status": "pass",
        "protocol": {
            "name": "paired GT-lane official-Hungarian McNemar audit",
            "iou_threshold": float(score.IOU_THRESHOLDS[0]),
            "line_width": int(score.LINE_WIDTH),
            "unit": "ground-truth lane within image",
            "b_definition": "baseline correct, candidate incorrect",
            "c_definition": "candidate correct, baseline incorrect",
            "warning": (
                "GT lanes and frames are correlated; sqrt(b+c) and the exact "
                "McNemar p-value are paired-disagreement diagnostics, not a "
                "video-cluster confidence interval."
            ),
        },
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "official_score": str(score_path),
        "official_score_sha256": sha256_file(score_path),
        "baseline_prediction_root": str(baseline_root),
        "candidate_experiment_root": str(candidate_root),
        "lane_root": str(lane_root),
        "folds": fold_results,
        "videos": video_results,
        "total": total,
        "image_rows": image_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite output: {args.output}")
    args.output.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "folds": fold_names, "total": total,
                      "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
