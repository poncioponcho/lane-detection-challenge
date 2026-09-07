from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    "evaluate_modular_ablation",
    ROOT / "scripts/evaluate_modular_ablation.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_policy_grid_contains_one_reference_and_expected_modules():
    specs = MODULE._policy_specs()
    names = {spec["name"] for spec in specs}
    assert "confidence_t0p40" in names
    assert any(spec["module"] == "luma_adaptive_threshold" for spec in specs)
    assert any(spec["module"] == "geometry_yspan_gate" for spec in specs)
    assert any(spec["module"] == "geometry_length_gate" for spec in specs)


def test_keep_uses_luma_only_for_adaptive_policy():
    spec = {
        "kind": "luma",
        "base_threshold": 0.40,
        "luma_threshold": 100.0,
        "high_threshold": 0.60,
    }
    dark = {"scores": [0.50], "luma": 99.0, "y_spans": [100.0], "lengths": [200.0]}
    bright = {"scores": [0.50], "luma": 101.0, "y_spans": [100.0], "lengths": [200.0]}
    assert MODULE._keep(spec, dark, 0)
    assert not MODULE._keep(spec, bright, 0)


def test_loco_deltas_require_same_video_set():
    base = {"videos": [{"video": "a", "tp": 1, "fp": 0, "fn": 1}]}
    candidate = {"videos": [{"video": "b", "tp": 1, "fp": 0, "fn": 1}]}
    with pytest.raises(ValueError, match="LOCO video sets differ"):
        MODULE._loco_deltas(base, candidate)


def test_f1_matches_official_count_identity():
    assert MODULE._f1(10, 5, 5) == pytest.approx(10 / 15)
