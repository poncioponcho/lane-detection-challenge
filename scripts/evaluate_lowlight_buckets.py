#!/usr/bin/env python3
"""Evaluate a low-light candidate by scene bucket with the frozen Oracle.

The scene labels are training-only diagnostics.  This script does not use
them to decide test-time inference; it only answers whether the same
conditional preprocessing helped the labeled low-light or normal buckets in
video-disjoint OOF.  Each bucket is scored globally and per video so the
paired bootstrap remains at the video-cluster level.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.manifest import ManifestRecord, read_manifest  # noqa: E402
from eval.oracle_runner import run_official_eval  # noqa: E402

from scan_lvo_geometry import paired_bootstrap, prediction_set, video_id  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _counts(value: dict[str, Any]) -> dict[str, Any]:
    global_counts = value["global"]
    return {
        "tp": int(global_counts["tp"]),
        "fp": int(global_counts["fp"]),
        "fn": int(global_counts["fn"]),
        "precision": float(global_counts["precision"]),
        "recall": float(global_counts["recall"]),
        "f1": float(global_counts["f1"]),
    }


def _score(
    name: str,
    pred_dir: Path,
    records: list[ManifestRecord],
    gt_dir: Path,
    official_python: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if not records:
        raise ValueError(f"empty bucket: {name}")
    records = [replace(record, order=index) for index, record in enumerate(records)]
    result = run_official_eval(
        pred_dir,
        gt_dir,
        records,
        official_python=official_python,
        per_clip=False,
        output_path=output_dir / name / "oracle_global.json",
    ).to_dict()
    rows = []
    for video in sorted({video_id(record.clip_id) for record in records}):
        subset = [
            replace(record, order=index)
            for index, record in enumerate(
                record for record in records if video_id(record.clip_id) == video
            )
        ]
        value = run_official_eval(
            pred_dir,
            gt_dir,
            subset,
            official_python=official_python,
            per_clip=False,
            output_path=output_dir / name / f"oracle_{video}.json",
        ).to_dict()
        counts = _counts(value)
        rows.append({"video": video, **counts, "n_images": len(subset)})
    global_counts = _counts(result)
    summed = tuple(sum(row[key] for row in rows) for key in ("tp", "fp", "fn"))
    if summed != tuple(global_counts[key] for key in ("tp", "fp", "fn")):
        raise RuntimeError(f"{name}: global and video Oracle counts disagree")
    return {
        "name": name,
        "images": len(records),
        "clips": len({record.clip_id for record in records}),
        "videos": len(rows),
        "global": {**global_counts, "n_images": len(records)},
        "videos_detail": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--scene-labels", required=True, type=Path)
    parser.add_argument("--gt-dir", required=True, type=Path)
    parser.add_argument("--official-python", required=True, type=Path)
    parser.add_argument("--base-predictions", required=True, type=Path)
    parser.add_argument("--candidate-predictions", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error(f"refusing to overwrite existing output: {args.output_dir}")

    records = read_manifest(args.manifest)
    expected = {record.pred_rel_path for record in records}
    for label, root in (
        ("base", args.base_predictions),
        ("candidate", args.candidate_predictions),
    ):
        if prediction_set(root) != expected:
            parser.error(f"{label} prediction set does not cover manifest")
    labels_payload = json.loads(args.scene_labels.read_text(encoding="utf-8"))
    labels = labels_payload.get("labels")
    if not isinstance(labels, dict):
        parser.error("scene labels has no labels object")

    bucket_records: dict[str, list[ManifestRecord]] = {
        "low_light": [],
        "normal": [],
    }
    for record in records:
        illumination = labels.get(record.clip_id, {}).get("illumination")
        if illumination in bucket_records:
            bucket_records[illumination].append(record)
    if any(not value for value in bucket_records.values()):
        parser.error("scene label buckets must both be non-empty")

    args.output_dir.mkdir(parents=True)
    oracle_dir = args.output_dir / "oracle"
    results = []
    for bucket, subset in bucket_records.items():
        base = _score(
            f"base_{bucket}", args.base_predictions, subset, args.gt_dir,
            args.official_python, oracle_dir,
        )
        candidate = _score(
            f"candidate_{bucket}", args.candidate_predictions, subset, args.gt_dir,
            args.official_python, oracle_dir,
        )
        base_for_bootstrap = {
            "global": base["global"],
            "videos": base["videos_detail"],
        }
        candidate_for_bootstrap = {
            "global": candidate["global"],
            "videos": candidate["videos_detail"],
        }
        bootstrap = paired_bootstrap(
            base_for_bootstrap,
            candidate_for_bootstrap,
            seed=args.seed,
            n_bootstrap=args.bootstrap,
        )
        results.append({
            "bucket": bucket,
            "base": base,
            "candidate": candidate,
            "paired_bootstrap": bootstrap,
        })

    value = {
        "status": "pass",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "video-disjoint low-light scene bucket Oracle comparison",
        "manifest": {"path": str(args.manifest.resolve()), "sha256": sha256_file(args.manifest)},
        "scene_labels": {"path": str(args.scene_labels.resolve()), "sha256": sha256_file(args.scene_labels)},
        "gt_dir": str(args.gt_dir.resolve()),
        "official_python": str(args.official_python.resolve()),
        "base_predictions": str(args.base_predictions.resolve()),
        "candidate_predictions": str(args.candidate_predictions.resolve()),
        "bootstrap": {"unit": "video", "paired": True, "n": args.bootstrap, "seed": args.seed},
        "results": results,
    }
    json_path = args.output_dir / "lowlight_bucket_eval.json"
    json_path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Conditional low-light bucket evaluation",
        "",
        "- Scene labels are used for training-only diagnosis, not test-time routing.",
        "- Both candidates are scored by the frozen official Oracle; CI resamples video clusters.",
        "",
        "| bucket | images | videos | base F1 | candidate F1 | Δpp | paired CI(pp) |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in results:
        base = row["base"]
        candidate = row["candidate"]
        boot = row["paired_bootstrap"]
        ci = boot["ci95_delta_pp"]
        lines.append(
            f"| `{row['bucket']}` | {base['images']} | {base['videos']} | "
            f"{base['global']['f1']:.6f} | {candidate['global']['f1']:.6f} | "
            f"{boot['observed_delta_pp']:+.3f} | [{ci[0]:+.3f}, {ci[1]:+.3f}] |"
        )
    lines.extend([
        "",
        "The low-light bucket contains only three video clusters in this dataset; its interval is diagnostic, not a broad causal claim.",
        "",
        f"- JSON: `{json_path}`",
    ])
    (args.output_dir / "lowlight_bucket_eval.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "output": str(args.output_dir), "buckets": len(results)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
