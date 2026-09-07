from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data.manifest import ManifestRecord  # noqa: E402
from data.split_by_clip import (  # noqa: E402
    SceneLabels, Split, binary_features, choose_val_clips, load_scene_labels,
    split_manifest, validate_split_against_labels,
    validate_spot_frames_against_manifest,
)


def _labels(count=12):
    result = {}
    for index in range(count):
        result[f"clip{index:02d}"] = SceneLabels(
            weather=("rain" if index % 3 == 0 else "fog" if index % 3 == 1 else "clear"),
            illumination=("low_light" if index % 2 else "normal"),
            artifact=("glare",) if index in {1, 4, 7} else (),
            geometry=("curve",) if index % 2 else ("crossroad",),
            spot_frames=("00000", "00050"),
        )
    return result


def test_binary_features_include_none_buckets():
    label = SceneLabels("clear", "normal", spot_frames=("00000",))
    assert binary_features(label) == {
        "weather:clear", "illumination:normal", "artifact:none", "geometry:none"
    }


def test_seeded_split_is_deterministic_and_protects_singleton():
    labels = _labels()
    labels["clip11"] = SceneLabels(
        "mixed", "normal", geometry=("curve",), spot_frames=("00000",)
    )
    first, audit = choose_val_clips(labels, val_size=6, seed=42, trials=2_000)
    second, _ = choose_val_clips(labels, val_size=6, seed=42, trials=2_000)
    assert first == second
    assert "clip11" not in first
    assert audit["feature_counts"]["weather:mixed"]["val_status"] == "N/A"


def test_lane_histogram_is_secondary_tiebreaker_without_scene_tradeoff():
    labels = _labels()
    # clip00/06 and clip01/07 are scene-identical pairs.  Make the latter two
    # the empty-GT sources; a six-of-twelve validation set should select one
    # when the primary scene score is tied.
    histograms = {clip: {3: 100} for clip in labels}
    histograms["clip06"] = {0: 40, 3: 60}
    histograms["clip07"] = {0: 40, 3: 60}
    first, audit = choose_val_clips(
        labels, val_size=6, seed=42, trials=2_000,
        lane_count_histograms=histograms,
    )
    second, audit_again = choose_val_clips(
        labels, val_size=6, seed=42, trials=2_000,
        lane_count_histograms=histograms,
    )
    assert first == second
    assert audit == audit_again
    assert set(first) & {"clip06", "clip07"}
    assert audit["lane_count_histograms"]["val"]["0"] == 40


def test_lane_histogram_clip_set_must_match_labels():
    with pytest.raises(ValueError, match="histograms"):
        choose_val_clips(_labels(), val_size=6, lane_count_histograms={"wrong": {0: 1}})


def test_split_rejects_overlap_and_invalid_val_size():
    with pytest.raises(ValueError, match="overlap"):
        Split("bad", tuple(f"c{i}" for i in range(6)), tuple(f"c{i}" for i in range(6)))
    with pytest.raises(ValueError, match=r"\[6, 10\]"):
        choose_val_clips(_labels(), val_size=5)


def test_scene_qc_requires_second_review_for_singletons(tmp_path):
    labels = _labels()
    labels["clip11"] = SceneLabels("mixed", "normal", spot_frames=("00000",))
    bundle = {
        "schema_version": 1,
        "quality_control": {"second_reviewed_clips": []},
        "labels": {
            clip: {
                "weather": label.weather,
                "illumination": label.illumination,
                "artifact": list(label.artifact),
                "geometry": list(label.geometry),
                "confidence": label.confidence,
                "spot_frames": list(label.spot_frames),
            }
            for clip, label in labels.items()
        },
    }
    path = tmp_path / "labels.json"
    path.write_text(json.dumps(bundle), encoding="utf-8")
    with pytest.raises(ValueError, match="rare-feature"):
        load_scene_labels(path)
    bundle["quality_control"]["second_reviewed_clips"] = ["clip11"]
    path.write_text(json.dumps(bundle), encoding="utf-8")
    loaded, _ = load_scene_labels(path)
    assert loaded == labels


def test_scene_qc_threshold_covers_all_rare_feature_holders(tmp_path):
    labels = _labels()
    glare_clips = sorted(clip for clip, label in labels.items() if label.artifact)
    bundle = {
        "schema_version": 1,
        "quality_control": {
            "rare_feature_max_holders": 3,
            "second_reviewed_clips": glare_clips[:-1],
        },
        "labels": {
            clip: {
                "weather": label.weather, "illumination": label.illumination,
                "artifact": list(label.artifact), "geometry": list(label.geometry),
                "confidence": label.confidence, "spot_frames": list(label.spot_frames),
            }
            for clip, label in labels.items()
        },
    }
    path = tmp_path / "labels.json"
    path.write_text(json.dumps(bundle), encoding="utf-8")
    with pytest.raises(ValueError, match=glare_clips[-1]):
        load_scene_labels(path)
    bundle["quality_control"]["second_reviewed_clips"] = glare_clips
    path.write_text(json.dumps(bundle), encoding="utf-8")
    load_scene_labels(path)


def test_split_roundtrip_and_manifest_order(tmp_path):
    labels = _labels()
    val, audit = choose_val_clips(labels, val_size=6, trials=1_000)
    val_set = set(val)
    split = Split(
        "test", tuple(clip for clip in labels if clip not in val_set),
        tuple(clip for clip in labels if clip in val_set), audit=audit,
    )
    validate_split_against_labels(split, labels)
    path = tmp_path / "split.yaml"
    split.save(path)
    assert Split.load(path) == split

    records = []
    for clip in labels:
        for frame in ("00000", "00001"):
            records.append(ManifestRecord(
                image_id=f"{clip}/{frame}", image_path=f"JPEGImages/{clip}/{frame}.jpg",
                pred_rel_path=f"{clip}/{frame}.lines.txt",
                gt_path=f"anno_txt/{clip}/{frame}.lines.txt", clip_id=clip,
                frame_id=frame, split="train", order=len(records),
            ))
    train, validation = split_manifest(records, split)
    assert [record.order for record in train] == list(range(len(train)))
    assert [record.order for record in validation] == list(range(len(validation)))
    assert not ({record.clip_id for record in train} & {record.clip_id for record in validation})
    assert [record.image_id for record in train] == [
        record.image_id for record in records if record.clip_id not in val_set
    ]


def test_split_label_contract_rejects_singleton_in_validation():
    labels = _labels()
    labels["clip11"] = SceneLabels("mixed", "normal", spot_frames=("00000",))
    invalid = Split(
        "invalid",
        tuple(f"clip{i:02d}" for i in range(6)),
        tuple(f"clip{i:02d}" for i in range(6, 12)),
    )
    with pytest.raises(ValueError, match="singleton features"):
        validate_split_against_labels(invalid, labels)


def test_spot_frames_must_exist_in_official_manifest():
    labels = {
        "clip00": SceneLabels("clear", "normal", spot_frames=("00000", "00003"))
    }
    records = [ManifestRecord(
        image_id="clip00/00000", image_path="JPEGImages/clip00/00000.jpg",
        pred_rel_path="clip00/00000.lines.txt",
        gt_path="anno_txt/clip00/00000.lines.txt", clip_id="clip00",
        frame_id="00000", split="train", order=0,
    )]
    with pytest.raises(ValueError, match="00003"):
        validate_spot_frames_against_manifest(labels, records)
    validate_spot_frames_against_manifest(
        {"clip00": SceneLabels("clear", "normal", spot_frames=("00000",))}, records
    )
