#!/usr/bin/env python3
"""Build data/processed/manifest_testB.jsonl on B-board release day (9/16).

Critical-path item: `package_testA_submit.py` and `infer_testA.py` are already
split-parameterised (`--split testB`), and `src/data/manifest.py` already knows
testB is an unlabeled split with a verified 100-frames-per-clip rule. The only
missing piece on release day is the ordered list file, so this wrapper:

1. takes the official testB list if the platform ships one, otherwise derives
   it from the extracted JPEGImages tree (sorted by clip, then frame id);
2. runs the frozen manifest builder (duplicate / path-safety / existence /
   100-frames-per-clip checks all stay on);
3. writes data/processed/manifest_testB.jsonl.

Usage on 9/16:
    python scripts/build_testB_manifest.py \
      --lane-root /hy-tmp/datasets/HardLane/Lane \
      --output data/processed/manifest_testB.jsonl
    # or, when the platform ships an explicit list:
    python scripts/build_testB_manifest.py --lane-root ... --official-list testB.txt
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.manifest import build_manifest, write_manifest  # noqa: E402


def derive_list(lane_root: Path, list_path: Path) -> int:
    images = sorted(
        (p for p in (lane_root / "JPEGImages").rglob("*.jpg")),
        key=lambda p: (p.parent.name, p.stem),
    )
    if not images:
        raise SystemExit(f"no JPEGImages found under {lane_root / 'JPEGImages'}")
    list_path.parent.mkdir(parents=True, exist_ok=True)
    list_path.write_text(
        "".join(f"JPEGImages/{p.parent.name}/{p.name}\n" for p in images),
        encoding="utf-8",
    )
    return len(images)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lane-root", required=True, type=Path)
    ap.add_argument("--official-list", type=Path, default=None)
    ap.add_argument("--split", default="testB")
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    lane_root = args.lane_root.resolve()
    list_path = args.official_list
    if list_path is None:
        list_path = args.output.with_suffix(".list.txt")
        n = derive_list(lane_root, list_path)
        print(f"derived list from JPEGImages: {n} rows -> {list_path}")
    else:
        list_path = list_path.resolve()
        print(f"using official list: {list_path}")

    records = build_manifest(list_path, lane_root, args.split)
    write_manifest(args.output, records)
    counts = Counter(r.clip_id for r in records)
    print(f"manifest rows: {len(records)}  clips: {len(counts)}")
    for clip, count in sorted(counts.items()):
        print(f"  {clip}: {count}")
    print(f"written: {args.output}")


if __name__ == "__main__":
    main()
