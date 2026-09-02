from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data.manifest import build_manifest, write_manifest  # noqa: E402
from eval.oracle_runner import run_official_eval  # noqa: E402


def _fixture(tmp_path):
    lane_root = tmp_path / "Lane"
    pred_root = tmp_path / "pred"
    rows = []
    for clip in ("clip_a", "clip_b"):
        for frame in ("00000", "00003"):
            image = lane_root / "JPEGImages" / clip / f"{frame}.jpg"
            gt = lane_root / "anno_txt" / clip / f"{frame}.lines.txt"
            pred = pred_root / clip / f"{frame}.lines.txt"
            image.parent.mkdir(parents=True, exist_ok=True)
            gt.parent.mkdir(parents=True, exist_ok=True)
            pred.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(b"jpeg")
            lane = "100 700 200 500 300 300 400 100\n"
            gt.write_text(lane, encoding="utf-8")
            pred.write_text(lane, encoding="utf-8")
            rows.append(f"/JPEGImages/{clip}/{frame}.jpg")
    task_list = lane_root / "data" / "train.txt"
    task_list.parent.mkdir(parents=True)
    task_list.write_text("\n".join(rows) + "\n", encoding="utf-8")
    records = build_manifest(
        task_list, lane_root, "train", enforce_verified_frame_count=False
    )
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(manifest, records)
    return lane_root, pred_root, manifest


def test_runner_returns_global_and_per_clip_counts(tmp_path):
    lane_root, pred_root, manifest = _fixture(tmp_path)
    output = tmp_path / "oracle.json"
    result = run_official_eval(
        pred_root, lane_root / "anno_txt", manifest,
        official_python=sys.executable, enforce_official_env=False,
        output_path=output,
    )
    assert result.global_.tp == 4
    assert result.global_.fp == result.global_.fn == 0
    assert result.global_.f1 == 1.0
    assert result.global_.n_images == 4
    assert list(result.per_clip) == ["clip_a", "clip_b"]
    assert all(value.tp == 2 and value.f1 == 1.0 for value in result.per_clip.values())
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["global"]["tp"] == 4
    assert "must never be averaged" in saved["aggregation_note"]
    assert saved["oracle_sha256"]["score.py"].startswith("b2f4c9b2")


def test_runner_missing_prediction_counts_as_fn(tmp_path):
    lane_root, pred_root, manifest = _fixture(tmp_path)
    (pred_root / "clip_a" / "00000.lines.txt").unlink()
    result = run_official_eval(
        pred_root, lane_root / "anno_txt", manifest,
        official_python=sys.executable, enforce_official_env=False,
    )
    assert (result.global_.tp, result.global_.fp, result.global_.fn) == (3, 0, 1)
    mean_clip_f1 = sum(value.f1 for value in result.per_clip.values()) / len(result.per_clip)
    assert result.global_.f1 == pytest.approx(6 / 7)
    assert mean_clip_f1 == pytest.approx((2 / 3 + 1.0) / 2)
    assert mean_clip_f1 != pytest.approx(result.global_.f1)


def test_runner_rejects_unlabeled_manifest(tmp_path):
    lane_root, pred_root, train_manifest = _fixture(tmp_path)
    records = build_manifest(
        lane_root / "data" / "train.txt", lane_root, "testA",
        enforce_verified_frame_count=False,
    )
    manifest = tmp_path / "testA.jsonl"
    write_manifest(manifest, records)
    with pytest.raises(ValueError, match="labeled manifest"):
        run_official_eval(
            pred_root, lane_root / "anno_txt", manifest,
            official_python=sys.executable, enforce_official_env=False,
        )


def test_production_mode_rejects_nonofficial_environment(tmp_path):
    lane_root, pred_root, manifest = _fixture(tmp_path)
    with pytest.raises(RuntimeError, match="Python 3.12|package pins mismatch"):
        run_official_eval(
            pred_root, lane_root / "anno_txt", manifest,
            official_python=sys.executable,
        )
