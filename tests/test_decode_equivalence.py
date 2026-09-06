from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
SPEC = importlib.util.spec_from_file_location(
    "check_decode_equivalence", ROOT / "scripts/check_decode_equivalence.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

from eval.score_sidecar import (  # noqa: E402
    POSITIVE_CLASS_SOFTMAX_PROBABILITY,
    make_score_sidecar_contract,
)


def _write_sidecar(path: Path, scores: list[float], threshold: float) -> None:
    payload = {
        "status": "pass",
        "images": 1,
        "scores_by_image": {"clip/00003": scores},
        **make_score_sidecar_contract(
            POSITIVE_CLASS_SOFTMAX_PROBABILITY, threshold
        ),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_direct_decode_matches_offline_probability_filter(tmp_path):
    rel = "clip/00003.lines.txt"
    direct = tmp_path / "direct"
    offline = tmp_path / "offline_filtered"
    raw = tmp_path / "raw"
    line_a = "100.00000 700.00000 200.00000 500.00000\n"
    line_b = "900.00000 700.00000 1000.00000 500.00000\n"
    (direct / rel).parent.mkdir(parents=True)
    (offline / rel).parent.mkdir(parents=True)
    (raw / rel).parent.mkdir(parents=True)
    (direct / rel).write_text(line_a, encoding="utf-8")
    (offline / rel).write_text(line_a, encoding="utf-8")
    (raw / rel).write_text(line_a + line_b, encoding="utf-8")

    direct_scores = tmp_path / "direct_scores.json"
    raw_scores = tmp_path / "raw_scores.json"
    _write_sidecar(direct_scores, [0.8], 0.5)
    _write_sidecar(raw_scores, [0.8, 0.2], 0.0)

    report = MODULE.compare_decode_outputs(
        direct,
        offline,
        0.5,
        direct_scores,
        raw_scores,
    )
    assert report["status"] == "pass"
    assert report["score_check"]["status"] == "pass"


def test_direct_decode_reports_geometry_mismatch(tmp_path):
    rel = "clip/00003.lines.txt"
    direct = tmp_path / "direct"
    offline = tmp_path / "offline"
    for root, line in (
        (direct, "100.00000 700.00000 200.00000 500.00000\n"),
        (offline, "101.00000 700.00000 200.00000 500.00000\n"),
    ):
        path = root / rel
        path.parent.mkdir(parents=True)
        path.write_text(line, encoding="utf-8")
    report = MODULE.compare_decode_outputs(direct, offline, 0.5)
    assert report["status"] == "mismatch"
    assert report["differing_line_file_count"] == 1


def test_direct_score_sidecar_must_match_requested_threshold(tmp_path):
    rel = "clip/00003.lines.txt"
    for name in ("direct", "offline"):
        path = tmp_path / name / rel
        path.parent.mkdir(parents=True)
        path.write_text("100.00000 700.00000 200.00000 500.00000\n", encoding="utf-8")
    direct_scores = tmp_path / "direct_scores.json"
    raw_scores = tmp_path / "raw_scores.json"
    _write_sidecar(direct_scores, [0.8], 0.4)
    _write_sidecar(raw_scores, [0.8], 0.0)
    with pytest.raises(ValueError, match="requested 0.5"):
        MODULE.compare_decode_outputs(
            tmp_path / "direct",
            tmp_path / "offline",
            0.5,
            direct_scores,
            raw_scores,
        )
