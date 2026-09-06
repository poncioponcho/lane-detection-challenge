from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from eval.score_sidecar import (  # noqa: E402
    POSITIVE_CLASS_SOFTMAX_PROBABILITY,
    PROBABILITY_SCORE_RANGE,
    SCORE_SCHEMA_VERSION,
    make_score_sidecar_contract,
    probability_score,
    validate_probability_sidecar,
)


def test_probability_sidecar_contract_is_explicit():
    payload = make_score_sidecar_contract(
        POSITIVE_CLASS_SOFTMAX_PROBABILITY, 0.0
    )
    assert payload == {
        "score_schema_version": SCORE_SCHEMA_VERSION,
        "score_semantics": POSITIVE_CLASS_SOFTMAX_PROBABILITY,
        "score_range": PROBABILITY_SCORE_RANGE,
        "post_nms": True,
        "candidate_export_conf_threshold": 0.0,
    }
    assert validate_probability_sidecar(payload) == 0.0


def test_probability_sidecar_rejects_legacy_raw_logit_metadata():
    payload = make_score_sidecar_contract("unknown", 0.0)
    with pytest.raises(ValueError, match="refusing raw-logit"):
        validate_probability_sidecar(payload)

    with pytest.raises(ValueError, match="refusing raw-logit"):
        validate_probability_sidecar({"scores_by_image": {}})


@pytest.mark.parametrize("value", [-0.01, 1.01, float("nan"), float("inf")])
def test_probability_score_rejects_out_of_range_values(value):
    with pytest.raises(ValueError, match="probability"):
        probability_score(value, context="test")
