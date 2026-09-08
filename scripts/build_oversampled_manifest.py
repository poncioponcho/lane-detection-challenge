#!/usr/bin/env python3
"""Build a hard-clip oversampled training manifest (T4-B).

Ranks clips by per-clip OOF F1 (frozen-Oracle evaluation of the 15ep LVO
OOF run), selects the bottom ``--hard-fraction`` of the clips that appear in
``--base-manifest``, and emits their rows ``--ratio`` times while every
other row is emitted once.  Training then runs through run_training.py with
``--train-manifest <output>`` and a matching ``--iters-per-epoch``; no
UnLanedet-side sampler code is touched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", required=True, type=Path)
    parser.add_argument("--per-clip-oracle", required=True, type=Path,
                        help="oracle_global_per_clip.json from the 15ep LVO OOF run")
    parser.add_argument("--hard-fraction", type=float, default=0.25)
    parser.add_argument("--ratio", type=int, default=3)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    args = parser.parse_args()
    if not 0.0 < args.hard_fraction < 1.0:
        raise SystemExit("--hard-fraction must be in (0, 1)")
    if args.ratio < 2:
        raise SystemExit("--ratio must be >= 2")

    base_rows = [
        json.loads(line)
        for line in args.base_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    oracle = json.loads(args.per_clip_oracle.read_text(encoding="utf-8"))
    per_clip = oracle["per_clip"]

    base_clips = sorted({row["clip_id"] for row in base_rows})
    missing = [clip for clip in base_clips if clip not in per_clip]
    if missing:
        raise SystemExit(f"per-clip oracle is missing clips present in the base manifest: {missing}")

    ranked = sorted(base_clips, key=lambda clip: per_clip[clip]["f1"])
    hard_count = max(1, round(len(ranked) * args.hard_fraction))
    hard_clips = set(ranked[:hard_count])

    with args.output.open("w", encoding="utf-8") as handle:
        for row in base_rows:
            repeats = args.ratio if row["clip_id"] in hard_clips else 1
            for _ in range(repeats):
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    output_rows = sum(
        args.ratio if row["clip_id"] in hard_clips else 1 for row in base_rows
    )
    metadata = {
        "base_manifest": str(args.base_manifest),
        "base_manifest_sha256": sha256_file(args.base_manifest),
        "base_manifest_rows": len(base_rows),
        "output_manifest": str(args.output),
        "output_manifest_sha256": sha256_file(args.output),
        "output_manifest_rows": output_rows,
        "per_clip_oracle": str(args.per_clip_oracle),
        "per_clip_oracle_global_f1": oracle["global"]["f1"],
        "hard_fraction": args.hard_fraction,
        "ratio": args.ratio,
        "hard_clip_count": hard_count,
        "hard_clips": [
            {
                "clip_id": clip,
                "oof_f1": per_clip[clip]["f1"],
                "tp": per_clip[clip]["tp"],
                "fp": per_clip[clip]["fp"],
                "fn": per_clip[clip]["fn"],
            }
            for clip in ranked[:hard_count]
        ],
    }
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output),
        "rows": output_rows,
        "hard_clips": hard_count,
        "hard_clip_ids": sorted(hard_clips),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
