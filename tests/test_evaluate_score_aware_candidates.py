from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    "evaluate_score_aware_candidates",
    ROOT / "scripts/evaluate_score_aware_candidates.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_threshold_name_is_stable():
    assert MODULE.threshold_name(0.5) == "0p500"
    assert MODULE.threshold_name(0.475) == "0p475"


def test_read_scores_rejects_non_ranker_semantics(tmp_path):
    path = tmp_path / "scores.json"
    path.write_text('{"score_semantics":"positive_class_softmax_probability"}', encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected score-aware semantics"):
        MODULE._read_scores(path)
