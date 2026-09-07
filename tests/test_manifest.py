from __future__ import annotations

import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data.manifest import (build_manifest, iter_clip_groups, read_manifest,
                           validate_manifest, write_manifest)  # noqa: E402


def _make_lane_root(tmp_path: Path, *, clips=1, frames=2, labeled=True):
    lane_root = tmp_path / "Lane"
    rows = []
    for clip_idx in range(clips):
        clip = f"clip{clip_idx}"
        for frame_idx in range(frames):
            frame = f"{frame_idx:05d}"
            image = lane_root / "JPEGImages" / clip / f"{frame}.jpg"
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(b"jpeg")
            if labeled:
                gt = lane_root / "anno_txt" / clip / f"{frame}.lines.txt"
                gt.parent.mkdir(parents=True, exist_ok=True)
                gt.write_text("0 0 1 1\n", encoding="utf-8")
            rows.append(f"/JPEGImages/{clip}/{frame}.jpg")
    task_list = lane_root / "data" / "tasks.txt"
    task_list.parent.mkdir(parents=True, exist_ok=True)
    task_list.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return lane_root, task_list


def test_labeled_manifest_preserves_order_and_roundtrips(tmp_path):
    root, task_list = _make_lane_root(tmp_path, clips=2, frames=2)
    records = build_manifest(
        task_list, root, "train", enforce_verified_frame_count=False
    )
    assert [record.image_id for record in records] == [
        "clip0/00000", "clip0/00001", "clip1/00000", "clip1/00001"
    ]
    assert [record.order for record in records] == list(range(4))
    assert records[0].image_path == "JPEGImages/clip0/00000.jpg"
    assert records[0].pred_rel_path == "clip0/00000.lines.txt"
    assert records[0].gt_path == "anno_txt/clip0/00000.lines.txt"

    output = tmp_path / "manifest.jsonl"
    write_manifest(output, records)
    assert read_manifest(output) == records
    assert [clip for clip, _group in iter_clip_groups(records)] == ["clip0", "clip1"]


def test_unlabeled_manifest_has_no_gt(tmp_path):
    root, task_list = _make_lane_root(tmp_path, labeled=False)
    records = build_manifest(
        task_list, root, "testA", enforce_verified_frame_count=False
    )
    assert all(record.gt_path is None for record in records)


def test_duplicate_rows_are_rejected_before_construction(tmp_path):
    root, task_list = _make_lane_root(tmp_path)
    first = task_list.read_text(encoding="utf-8").splitlines()[0]
    task_list.write_text(f"{first}\n{first}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate image_id"):
        build_manifest(task_list, root, "train", enforce_verified_frame_count=False)


@pytest.mark.parametrize("missing", ["image", "gt"])
def test_labeled_existence_checks(tmp_path, missing):
    root, task_list = _make_lane_root(tmp_path)
    target = next((root / ("JPEGImages" if missing == "image" else "anno_txt")).rglob("*.*"))
    target.unlink()
    with pytest.raises(FileNotFoundError, match="missing"):
        build_manifest(task_list, root, "train", enforce_verified_frame_count=False)


def test_verified_splits_require_100_frames_but_testb_does_not(tmp_path):
    root, task_list = _make_lane_root(tmp_path, frames=2, labeled=False)
    with pytest.raises(ValueError, match="exactly 100 frames"):
        build_manifest(task_list, root, "testA")
    assert len(build_manifest(task_list, root, "testB")) == 2


def test_manifest_contract_rejects_reorder_and_wrong_gt_semantics(tmp_path):
    root, task_list = _make_lane_root(tmp_path)
    records = build_manifest(
        task_list, root, "train", enforce_verified_frame_count=False
    )
    with pytest.raises(ValueError, match="order mismatch"):
        validate_manifest([replace(records[0], order=1), records[1]])
    with pytest.raises(ValueError, match="inconsistent gt_path"):
        validate_manifest([replace(records[0], gt_path=None), records[1]])
    with pytest.raises(ValueError, match="inconsistent gt_path"):
        validate_manifest([
            replace(records[0], gt_path="anno_txt/other/99999.lines.txt"),
            records[1],
        ])


@pytest.mark.parametrize("image_path", [
    "JPEGImages/other/00000.jpg",
    "JPEGImages/clip0/99999.jpg",
    "JPEGImages/clip0/00000.png",
    "../JPEGImages/clip0/00000.jpg",
])
def test_manifest_contract_rejects_inconsistent_image_path(tmp_path, image_path):
    root, task_list = _make_lane_root(tmp_path)
    records = build_manifest(
        task_list, root, "train", enforce_verified_frame_count=False
    )
    with pytest.raises(ValueError, match="inconsistent image_path"):
        validate_manifest([replace(records[0], image_path=image_path), records[1]])


def test_non_contiguous_clip_is_rejected(tmp_path):
    root, task_list = _make_lane_root(tmp_path, clips=2, frames=2)
    records = build_manifest(
        task_list, root, "train", enforce_verified_frame_count=False
    )
    scrambled = [records[0], replace(records[2], order=1),
                 replace(records[1], order=2), replace(records[3], order=3)]
    with pytest.raises(ValueError, match="non-contiguous"):
        list(iter_clip_groups(scrambled))
