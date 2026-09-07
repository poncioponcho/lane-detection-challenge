from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from common.io_utils import write_lines_txt  # noqa: E402
from data.eda import analyze_manifest, select_overlay_rows  # noqa: E402
from data.manifest import ManifestRecord  # noqa: E402


def _record(order: int, clip: str, frame: str) -> ManifestRecord:
    return ManifestRecord(
        image_id=f"{clip}/{frame}",
        image_path=f"JPEGImages/{clip}/{frame}.jpg",
        pred_rel_path=f"{clip}/{frame}.lines.txt",
        gt_path=f"anno_txt/{clip}/{frame}.lines.txt",
        clip_id=clip,
        frame_id=frame,
        split="train",
        order=order,
    )


def test_eda_counts_health_and_cut_exposure(tmp_path):
    records = [_record(0, "c0", "00000"), _record(1, "c1", "00000")]
    root = tmp_path / "Lane"
    write_lines_txt(root / records[0].gt_path, [])
    duplicated = np.asarray([[1, 100], [1, 100], [3, 400]], dtype=np.float32)
    regular = np.asarray([[10, 200], [12, 700]], dtype=np.float32)
    write_lines_txt(root / records[1].gt_path, [duplicated, regular])
    scenes = {
        "c0": {"weather": "clear", "illumination": "normal", "artifact": [], "geometry": []},
        "c1": {"weather": "mixed", "illumination": "normal", "artifact": ["glare"], "geometry": ["curve"]},
    }
    report, rows = analyze_manifest(
        records, root, scene_labels=scenes, split_by_clip={"c0": "train", "c1": "val"}
    )

    assert report["dataset"] == {
        "images": 2, "clips": 2, "lanes": 2, "points": 5,
        "empty_images": 1, "empty_rate": 0.5,
    }
    assert report["lanes_per_image"]["histogram"] == {"0": 1, "2": 1}
    health = report["annotation_health"]
    assert health["lanes_with_consecutive_duplicates"] == 1
    assert health["consecutive_duplicate_points_removed"] == 1
    assert health["lanes_with_fewer_than_two_points_after_consecutive_dedup"] == 0
    assert report["geometry"]["cut_height_exposure"]["180"]["affected_lanes"] == 1
    assert report["scenes"]["clip_counts"]["artifact"] == {"glare": 1, "none": 1}
    assert report["split_comparison"]["val"]["lanes"] == 2
    selected = select_overlay_rows(rows, scenes)
    assert selected[0]["category"] == "empty_gt"
    assert len({row["image_id"] for row in selected}) == len(selected)


def test_eda_rejects_scene_or_split_clip_drift(tmp_path):
    record = _record(0, "c0", "00000")
    write_lines_txt(tmp_path / record.gt_path, [])
    try:
        analyze_manifest([record], tmp_path, scene_labels={"wrong": {}})
        assert False, "scene/manifest mismatch must fail"
    except ValueError as error:
        assert "scene labels do not match" in str(error)
    try:
        analyze_manifest([record], tmp_path, split_by_clip={"wrong": "train"})
        assert False, "split/manifest mismatch must fail"
    except ValueError as error:
        assert "split does not match" in str(error)
