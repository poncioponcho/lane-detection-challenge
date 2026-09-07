#!/usr/bin/env python3
"""Derive and audit deterministic leave-one-video-out manifests.

The frozen ``v1_seed42`` manifests are never edited by this utility.  LVO
manifests are derived from the complete 7100-image training manifest and are
written below an experiment-specific output directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from data.manifest import ManifestRecord, read_manifest, write_manifest


VIDEO_MARKER = "_1_0_"
VIDEO_NAME = re.compile(r"^v[0-9]+$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def video_id(clip_id: str) -> str:
    if VIDEO_MARKER not in clip_id:
        raise ValueError(f"clip id has no expected video marker: {clip_id}")
    value = clip_id.split(VIDEO_MARKER, 1)[0]
    if not VIDEO_NAME.fullmatch(value):
        raise ValueError(f"invalid video id {value!r} from clip {clip_id!r}")
    return value


def relabel(records: list[ManifestRecord], split: str) -> list[ManifestRecord]:
    return [replace(record, split=split, order=index)
            for index, record in enumerate(records)]


def clip_count(records: list[ManifestRecord]) -> int:
    return len({record.clip_id for record in records})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path,
                        help="complete 7100-image train manifest")
    parser.add_argument("--frozen-train", required=True, type=Path)
    parser.add_argument("--frozen-val", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    source = args.source.resolve()
    frozen_train = args.frozen_train.resolve()
    frozen_val = args.frozen_val.resolve()
    output = args.output.resolve()
    for path in (source, frozen_train, frozen_val):
        if not path.is_file():
            raise SystemExit(f"manifest missing: {path}")

    source_records = read_manifest(source)
    if any(record.split != "train" for record in source_records):
        raise SystemExit("source manifest must contain only split=train records")
    source_images = {record.image_id for record in source_records}
    source_clips = {record.clip_id for record in source_records}
    source_videos = {video_id(record.clip_id) for record in source_records}
    if len(source_records) != 7100:
        raise SystemExit(f"expected 7100 source images, got {len(source_records)}")
    if len(source_clips) != 71:
        raise SystemExit(f"expected 71 source clips, got {len(source_clips)}")
    if len(source_videos) != 8:
        raise SystemExit(f"expected 8 source videos, got {len(source_videos)}")

    frozen_before = {
        "train": {"path": str(frozen_train), "sha256": sha256_file(frozen_train)},
        "val": {"path": str(frozen_val), "sha256": sha256_file(frozen_val)},
    }
    output.mkdir(parents=True, exist_ok=True)
    folds = []
    # Sorting is intentional: fold numbering is independent of source order.
    for fold_index, heldout_video in enumerate(sorted(source_videos)):
        heldout = [record for record in source_records
                   if video_id(record.clip_id) == heldout_video]
        train = [record for record in source_records
                 if video_id(record.clip_id) != heldout_video]
        train_ids = {record.image_id for record in train}
        holdout_ids = {record.image_id for record in heldout}
        if train_ids & holdout_ids:
            raise SystemExit(f"fold {heldout_video} has train/holdout overlap")
        if train_ids | holdout_ids != source_images:
            raise SystemExit(f"fold {heldout_video} does not cover source images")

        fold_dir = output / f"fold_{fold_index:02d}_{heldout_video}"
        train_path = fold_dir / "manifest_train.jsonl"
        holdout_path = fold_dir / "manifest_holdout.jsonl"
        write_manifest(train_path, relabel(train, "train"))
        write_manifest(holdout_path, relabel(heldout, "val"))
        # Re-read each artifact so the evidence covers the serialized files,
        # not merely the in-memory lists used to write them.
        train_check = read_manifest(train_path)
        holdout_check = read_manifest(holdout_path)
        if len(train_check) != len(train) or len(holdout_check) != len(heldout):
            raise SystemExit(f"fold {heldout_video} round-trip row count mismatch")
        folds.append({
            "fold": fold_index,
            "video": heldout_video,
            "train": {
                "path": str(train_path),
                "sha256": sha256_file(train_path),
                "rows": len(train_check),
                "clips": clip_count(train_check),
            },
            "holdout": {
                "path": str(holdout_path),
                "sha256": sha256_file(holdout_path),
                "rows": len(holdout_check),
                "clips": clip_count(holdout_check),
            },
            "train_videos": sorted({video_id(r.clip_id) for r in train_check}),
            "holdout_videos": sorted({video_id(r.clip_id) for r in holdout_check}),
        })

    holdout_ids_all = []
    for fold in folds:
        holdout_ids_all.extend(
            record.image_id
            for record in read_manifest(Path(fold["holdout"]["path"]))
        )
    if len(holdout_ids_all) != len(source_records):
        raise SystemExit("LVO holdout coverage has the wrong number of rows")
    if len(set(holdout_ids_all)) != len(holdout_ids_all):
        raise SystemExit("LVO holdout coverage contains duplicate images")
    if set(holdout_ids_all) != source_images:
        raise SystemExit("LVO holdout coverage differs from source images")

    evidence = {
        "status": "pass",
        "protocol": "leave-one-video-out",
        "source": {
            "path": str(source),
            "sha256": sha256_file(source),
            "rows": len(source_records),
            "clips": len(source_clips),
            "videos": sorted(source_videos),
        },
        "frozen_inputs_before": frozen_before,
        "folds": folds,
        "coverage": {
            "source_rows": len(source_records),
            "holdout_rows": len(holdout_ids_all),
            "unique_holdout_rows": len(set(holdout_ids_all)),
            "exact_source_image_set": True,
            "each_image_held_out_once": True,
        },
        "project_git_head": git_head(args.project_root.resolve()),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    evidence_path = output / "folds_evidence.json"
    evidence_path.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": "pass",
        "output": str(output),
        "evidence": str(evidence_path),
        "videos": sorted(source_videos),
        "folds": len(folds),
        "source_rows": len(source_records),
        "holdout_rows": len(holdout_ids_all),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
