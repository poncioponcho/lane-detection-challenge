from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    "analyze_score_aware_candidates",
    ROOT / "scripts/analyze_score_aware_candidates.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_isotonic_fit_is_monotone_and_predicts_bounds():
    model = MODULE.fit_isotonic([0.1, 0.2, 0.3, 0.4], [1, 0, 1, 1])
    assert model["x"] == sorted(model["x"])
    assert model["y"] == sorted(model["y"])
    values = MODULE.predict_isotonic(model, [0.0, 0.25, 0.5])
    assert np.all((values >= 0.0) & (values <= 1.0))
    assert np.all(np.diff(values) >= -1e-12)


def test_fit_logistic_requires_both_classes():
    with pytest.raises(ValueError, match="both classes"):
        MODULE.fit_logistic(np.zeros((3, 1)), np.ones(3), class_balance=False)


def test_candidate_labels_mark_unmatched_and_low_iou_as_fp():
    pred = [
        np.asarray([[100.0, 700.0], [100.0, 600.0]]),
        np.asarray([[900.0, 700.0], [900.0, 600.0]]),
    ]
    gt = [np.asarray([[100.0, 700.0], [100.0, 600.0]])]
    labels, ious = MODULE.candidate_labels(pred, gt)
    assert labels == [1, 0]
    assert ious[0] > 0.5
    assert ious[1] == 0.0


def test_sigmoid_is_finite_at_large_logits():
    values = MODULE.sigmoid(np.asarray([-1000.0, 0.0, 1000.0]))
    assert np.all(np.isfinite(values))
    assert values[0] < 1e-20
    assert values[1] == pytest.approx(0.5)
    assert values[2] > 0.999999
