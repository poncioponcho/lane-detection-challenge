"""Ordered manifest construction from official train/test task lists.

The official list order is part of the evaluation contract. This module never
uses a set or directory scan to construct tasks; each non-empty official line
becomes exactly one :class:`ManifestRecord` with a stable ``order``.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Iterator

LABELED_SPLITS = frozenset({"train", "val"})
UNLABELED_SPLITS = frozenset({"testA", "testB"})
VERIFIED_100_FRAME_SPLITS = frozenset({"train", "testA"})
KNOWN_SPLITS = LABELED_SPLITS | UNLABELED_SPLITS


@dataclass(frozen=True)
class ManifestRecord:
    """One official-list line represented as an ordered task."""

    image_id: str
    image_path: str
    pred_rel_path: str
    gt_path: str | None
    clip_id: str
    frame_id: str
    split: str
    order: int

    @classmethod
    def from_dict(cls, value: dict) -> "ManifestRecord":
        return cls(**value)


def _parse_official_image_path(raw: str) -> tuple[str, str, str]:
    """Return ``(relative image path, clip_id, frame_id)`` for one list row."""
    raw = raw.strip()
    if not raw:
        raise ValueError("official task row is empty")
    # Lists use POSIX paths even when tools run on another platform.
    path = PurePosixPath(raw.lstrip("/"))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe official task path: {raw!r}")
    if len(path.parts) != 3 or path.parts[0] != "JPEGImages":
        raise ValueError(
            f"expected JPEGImages/<clip>/<frame>.jpg, got {raw!r}"
        )
    if path.suffix.lower() not in {".jpg", ".jpeg"}:
        raise ValueError(f"official task is not a JPEG image: {raw!r}")
    clip_id, frame_id = path.parts[-2], path.stem
    if not clip_id or not frame_id:
        raise ValueError(f"invalid clip/frame id: {raw!r}")
    return path.as_posix(), clip_id, frame_id


def build_manifest(list_path: str | Path, lane_root: str | Path, split: str,
                   *, validate_exists: bool = True,
                   enforce_verified_frame_count: bool = True) -> list[ManifestRecord]:
    """Build and validate an ordered manifest from an official list file.

    ``lane_root`` is the directory containing ``JPEGImages/``, ``anno_txt/``
    (for labeled splits), and ``data/``. Paths stored in records are relative
    to this root and use POSIX separators.
    """
    if split not in KNOWN_SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {sorted(KNOWN_SPLITS)}")
    list_path, lane_root = Path(list_path), Path(lane_root)
    if not list_path.is_file():
        raise FileNotFoundError(f"official list does not exist: {list_path}")

    records: list[ManifestRecord] = []
    seen: dict[str, int] = {}
    for source_line, raw in enumerate(list_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        image_path, clip_id, frame_id = _parse_official_image_path(raw)
        image_id = f"{clip_id}/{frame_id}"
        if image_id in seen:
            raise ValueError(
                f"duplicate image_id {image_id!r} at source lines "
                f"{seen[image_id]} and {source_line}"
            )
        seen[image_id] = source_line
        pred_rel_path = f"{clip_id}/{frame_id}.lines.txt"
        gt_path = f"anno_txt/{pred_rel_path}" if split in LABELED_SPLITS else None
        record = ManifestRecord(
            image_id=image_id,
            image_path=image_path,
            pred_rel_path=pred_rel_path,
            gt_path=gt_path,
            clip_id=clip_id,
            frame_id=frame_id,
            split=split,
            order=len(records),
        )
        if validate_exists:
            image_file = lane_root / Path(record.image_path)
            if not image_file.is_file():
                raise FileNotFoundError(f"manifest image missing: {image_file}")
            if record.gt_path is not None:
                gt_file = lane_root / Path(record.gt_path)
                if not gt_file.is_file():
                    raise FileNotFoundError(f"manifest GT missing: {gt_file}")
        records.append(record)

    if not records:
        raise ValueError(f"official list has no tasks: {list_path}")
    if enforce_verified_frame_count and split in VERIFIED_100_FRAME_SPLITS:
        counts = Counter(record.clip_id for record in records)
        wrong = {clip: count for clip, count in counts.items() if count != 100}
        if wrong:
            preview = dict(list(sorted(wrong.items()))[:10])
            raise ValueError(
                f"verified split {split} requires exactly 100 frames per clip; "
                f"violations={preview}"
            )
    return records


def write_manifest(path: str | Path, records: Iterable[ManifestRecord]) -> None:
    """Write records as deterministic UTF-8 JSONL without changing order."""
    path = Path(path)
    records = list(records)
    validate_manifest(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def read_manifest(path: str | Path) -> list[ManifestRecord]:
    """Read JSONL records and enforce their ordering/identity contract."""
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                records.append(ManifestRecord.from_dict(json.loads(raw)))
            except (TypeError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid manifest JSON at line {line_no}") from exc
    validate_manifest(records)
    return records


def validate_manifest(records: Iterable[ManifestRecord]) -> None:
    """Reject empty, reordered, duplicate, or internally inconsistent records."""
    records = list(records)
    if not records:
        raise ValueError("manifest is empty")
    seen = set()
    split = records[0].split
    for expected_order, record in enumerate(records):
        if record.order != expected_order:
            raise ValueError(
                f"manifest order mismatch at index {expected_order}: {record.order}"
            )
        if record.image_id in seen:
            raise ValueError(f"duplicate manifest image_id: {record.image_id}")
        seen.add(record.image_id)
        if record.split != split or record.split not in KNOWN_SPLITS:
            raise ValueError("manifest must contain one known split")
        if record.image_id != f"{record.clip_id}/{record.frame_id}":
            raise ValueError(f"inconsistent image_id: {record.image_id}")
        image_path = PurePosixPath(record.image_path)
        if (image_path.is_absolute() or ".." in image_path.parts or
                len(image_path.parts) != 3 or image_path.parts[0] != "JPEGImages" or
                image_path.parts[1] != record.clip_id or
                image_path.stem != record.frame_id or
                image_path.suffix.lower() not in {".jpg", ".jpeg"}):
            raise ValueError(f"inconsistent image_path: {record.image_path}")
        expected_pred = f"{record.clip_id}/{record.frame_id}.lines.txt"
        if record.pred_rel_path != expected_pred:
            raise ValueError(f"inconsistent pred_rel_path: {record.pred_rel_path}")
        expected_gt = f"anno_txt/{expected_pred}"
        if record.split in LABELED_SPLITS and record.gt_path != expected_gt:
            raise ValueError(
                f"labeled record has inconsistent gt_path: {record.gt_path}"
            )
        if record.split in UNLABELED_SPLITS and record.gt_path is not None:
            raise ValueError(f"unlabeled record has gt_path: {record.image_id}")


def iter_clip_groups(records: Iterable[ManifestRecord]) -> Iterator[tuple[str, list[ManifestRecord]]]:
    """Yield consecutive clip groups without sorting or losing manifest order."""
    current_clip = None
    group: list[ManifestRecord] = []
    closed = set()
    for record in records:
        if current_clip is None:
            current_clip = record.clip_id
        if record.clip_id != current_clip:
            closed.add(current_clip)
            yield current_clip, group
            if record.clip_id in closed:
                raise ValueError(f"clip is non-contiguous in manifest: {record.clip_id}")
            current_clip, group = record.clip_id, []
        group.append(record)
    if group:
        yield current_clip, group


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-path", required=True, type=Path)
    parser.add_argument("--lane-root", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=sorted(KNOWN_SPLITS))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    records = build_manifest(args.list_path, args.lane_root, args.split)
    write_manifest(args.output, records)
    clips = len({record.clip_id for record in records})
    print(json.dumps({"output": str(args.output), "split": args.split,
                      "records": len(records), "clips": clips}))


if __name__ == "__main__":
    main()
