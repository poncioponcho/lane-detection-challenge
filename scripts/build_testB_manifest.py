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
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.manifest import build_manifest, write_manifest  # noqa: E402


def known_clip_ids(paths: list[Path]) -> set[str]:
    """Clip ids already accounted for by an earlier manifest."""
    known: set[str] = set()
    for p in paths:
        if not p.is_file():
            continue
        for raw in p.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            rec = json.loads(raw)
            if rec.get("clip_id"):
                known.add(rec["clip_id"])
    return known


def derive_list(lane_root: Path, list_path: Path, exclude: set[str],
                expected_clips: int) -> int:
    """Derive the testB list as *only* the clips nobody has seen before.

    Do NOT glob every JPEGImages file.  The instance tree also holds the 71
    training clips, the 9 testA clips and the `_hflip` augmentation dirs, so a
    blind glob yields ~9700 rows mislabelled as testB and would silently send
    the whole training set through inference.  testB is defined as the
    complement: clip directories that are absent from the train/testA manifests
    and are not augmentation folders.
    """
    root = lane_root / "JPEGImages"
    clips = sorted(
        d for d in root.iterdir()
        if d.is_dir() and not d.name.endswith("_hflip") and d.name not in exclude
    )
    if not clips:
        raise SystemExit(
            f"no unseen clip dirs under {root} (excluded {len(exclude)} known "
            "clips). Are the testB images extracted?"
        )
    if expected_clips and len(clips) != expected_clips:
        raise SystemExit(
            f"expected {expected_clips} unseen clips, found {len(clips)}: "
            f"{[c.name for c in clips]}. Refusing to guess -- pass "
            "--expected-clips 0 to override after checking by hand."
        )
    images = sorted(
        (p for c in clips for p in c.glob("*.jpg")),
        key=lambda p: (p.parent.name, p.stem),
    )
    list_path.parent.mkdir(parents=True, exist_ok=True)
    list_path.write_text(
        "".join(f"JPEGImages/{p.parent.name}/{p.name}\n" for p in images),
        encoding="utf-8",
    )
    print(f"unseen clips: {len(clips)} -> {[c.name for c in clips]}")
    return len(images)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lane-root", required=True, type=Path)
    ap.add_argument("--official-list", type=Path, default=None)
    ap.add_argument("--split", default="testB")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--expect-clips", type=int, default=10)
    ap.add_argument("--exclude", type=Path, nargs="*", default=None,
                    help="manifests whose clips are already known "
                         "(default: train + testA)")
    args = ap.parse_args()

    lane_root = args.lane_root.resolve()
    list_path = args.official_list
    if list_path is None:
        excl = args.exclude or [
            PROJECT_ROOT / "data/processed/manifest_train.jsonl",
            PROJECT_ROOT / "data/processed/manifest_testA.jsonl",
        ]
        exclude = known_clip_ids([Path(p) for p in excl])
        print(f"excluding {len(exclude)} known clips from {[Path(p).name for p in excl]}")
        list_path = args.output.with_suffix(".list.txt")
        n = derive_list(lane_root, list_path, exclude, args.expect_clips)
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
