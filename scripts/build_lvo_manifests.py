#!/usr/bin/env python3
"""Build auditable leave-one-video-out manifests from a weighted train list.

The source manifest defines the unique image universe.  The weighted manifest
may repeat source rows for hard-example oversampling, but repetitions are
allowed only in a fold's training side; each holdout is emitted once from the
unique source manifest.  A fixed optimizer budget can then be supplied to the
LVO runner independently of the number of repeated rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise SystemExit(f"manifest is empty: {path}")
    return rows


def video_id(clip_id: str) -> str:
    marker = "_1_0_"
    if marker not in clip_id:
        raise SystemExit(f"invalid clip id without {marker!r}: {clip_id}")
    value = clip_id.split(marker, 1)[0]
    if not re.fullmatch(r"v[0-9]+", value):
        raise SystemExit(f"invalid video id {value!r} from {clip_id!r}")
    return value


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # A derived manifest is its own ordered task list.  The source ``order``
    # is not valid after filtering, and repeated training rows need distinct
    # positions even though they intentionally share image_id.
    normalized = []
    for order, row in enumerate(rows):
        value = dict(row)
        value["order"] = order
        normalized.append(value)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in normalized),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-manifest", required=True, type=Path,
        help="unique source image universe, normally the 63-clip v1 manifest",
    )
    parser.add_argument(
        "--weighted-manifest", required=True, type=Path,
        help="same universe with optional repeated hard-example rows",
    )
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output_root}")

    source = read_rows(args.source_manifest)
    weighted = read_rows(args.weighted_manifest)
    source_ids = [row["image_id"] for row in source]
    weighted_ids = [row["image_id"] for row in weighted]
    if len(source_ids) != len(set(source_ids)):
        raise SystemExit("source manifest contains duplicate image_id values")
    if set(weighted_ids) != set(source_ids):
        raise SystemExit("weighted manifest image universe differs from source manifest")
    if any(row.get("split") != "train" for row in source + weighted):
        raise SystemExit("LVO manifests must contain train rows only")

    source_by_video: dict[str, list[dict]] = {}
    for row in source:
        source_by_video.setdefault(video_id(row["clip_id"]), []).append(row)
    videos = sorted(source_by_video)
    if len(videos) < 2:
        raise SystemExit("need at least two video groups for LVO")

    args.output_root.mkdir(parents=True)
    manifests = args.output_root / "manifests"
    write_rows(manifests / "source_manifest_train.jsonl", source)
    folds = []
    for index, heldout in enumerate(videos):
        fold = manifests / f"fold_{index:02d}_{heldout}"
        holdout = source_by_video[heldout]
        train = [row for row in weighted if video_id(row["clip_id"]) != heldout]
        expected_train = set(source_ids) - {row["image_id"] for row in holdout}
        actual_train = {row["image_id"] for row in train}
        if actual_train != expected_train:
            raise SystemExit(f"{heldout}: weighted train image universe mismatch")
        write_rows(fold / "manifest_train.jsonl", train)
        write_rows(fold / "manifest_holdout.jsonl", holdout)
        folds.append({
            "index": index,
            "video": heldout,
            "train_rows": len(train),
            "train_unique_rows": len(expected_train),
            "holdout_rows": len(holdout),
            "train_repetition_factor": len(train) / len(expected_train),
            "holdout_repetitions": dict(
                Counter(row["image_id"] for row in holdout)
            ),
        })

    metadata = {
        "status": "pass",
        "protocol": "leave-one-video-out weighted training manifests",
        "source_manifest": str(args.source_manifest.resolve()),
        "source_manifest_sha256": sha256_file(args.source_manifest),
        "source_rows": len(source),
        "weighted_manifest": str(args.weighted_manifest.resolve()),
        "weighted_manifest_sha256": sha256_file(args.weighted_manifest),
        "weighted_rows": len(weighted),
        "unique_rows": len(source),
        "videos": videos,
        "folds": folds,
    }
    (args.output_root / "manifest_build.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "pass",
        "output_root": str(args.output_root),
        "source_rows": len(source),
        "weighted_rows": len(weighted),
        "folds": len(folds),
        "videos": videos,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
