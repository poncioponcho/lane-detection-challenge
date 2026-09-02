"""Differential contract tests: local diagnostic metric vs frozen Oracle."""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from eval import matching  # noqa: E402
from eval.rasterize import (interp_lane, parse_lines_txt, rasterize_lane,
                            remove_consecutive_duplicates)  # noqa: E402


def _load_oracle():
    path = ROOT / "src" / "eval" / "official_oracle" / "score.py"
    spec = importlib.util.spec_from_file_location("frozen_score_for_diff_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


ORACLE = _load_oracle()


def _lane(x_shift=0.0):
    return np.asarray([
        [120.25 + x_shift, 710.0],
        [220.5 + x_shift, 540.0],
        [390.75 + x_shift, 365.0],
        [610.125 + x_shift, 190.0],
    ], dtype=np.float64)


@pytest.mark.parametrize("points", [
    [[10.0, 20.0], [40.0, 80.0]],
    [[10.0, 20.0], [40.0, 80.0], [20.0, 140.0]],  # fold-back
    _lane(),
    [[-30.0, 800.0], [300.0, 500.0], [900.0, 200.0], [1500.0, -20.0]],
])
def test_interpolation_and_mask_are_oracle_exact(points):
    points = np.asarray(points, dtype=np.float64)
    local_dense = interp_lane(points)
    oracle_dense = ORACLE.interp_lane([tuple(point) for point in points])
    np.testing.assert_array_equal(local_dense, oracle_dense)
    np.testing.assert_array_equal(
        rasterize_lane(points),
        ORACLE.draw_lane_mask(oracle_dense, ORACLE.LINE_WIDTH),
    )


@pytest.mark.parametrize("shift", [0.0, 5.0, 9.0, 10.0, 10.5, 11.0, 15.0, 25.0])
def test_iou_boundary_scan_is_oracle_exact(shift):
    pred, gt = _lane(shift), _lane()
    local_iou = matching.pair_iou(pred, gt)
    pred_mask = ORACLE.draw_lane_mask(ORACLE.interp_lane(pred), ORACLE.LINE_WIDTH)
    gt_mask = ORACLE.draw_lane_mask(ORACLE.interp_lane(gt), ORACLE.LINE_WIDTH)
    oracle_iou = (pred_mask & gt_mask).sum() / (pred_mask | gt_mask).sum()
    assert local_iou == oracle_iou

    local = matching.compute_f1({"image": [pred]}, {"image": [gt]})
    oracle = ORACLE.culane_metric_single([pred], [gt])[0.5]
    assert [local["TP"], local["FP"], local["FN"]] == oracle


@pytest.mark.parametrize("lane_count", [0, 1, 3, 7])
def test_lane_counts_and_empty_cases_are_oracle_exact(lane_count):
    lanes = [_lane(index * 80.0) for index in range(lane_count)]
    local = matching.compute_f1({"image": lanes}, {"image": lanes})
    oracle = ORACLE.culane_metric_single(lanes, lanes)[0.5]
    assert [local["TP"], local["FP"], local["FN"]] == oracle

    missing_pred = matching.compute_f1({}, {"image": lanes})
    oracle_missing = ORACLE.culane_metric_single([], lanes)[0.5]
    assert [missing_pred["TP"], missing_pred["FP"], missing_pred["FN"]] == oracle_missing


def test_parser_missing_empty_duplicate_and_nonconsecutive_duplicate(tmp_path):
    missing = tmp_path / "missing.lines.txt"
    assert parse_lines_txt(missing) == ORACLE.parse_lines_txt(str(missing)) == []

    empty = tmp_path / "empty.lines.txt"
    empty.write_bytes(b"")
    assert parse_lines_txt(empty) == ORACLE.parse_lines_txt(str(empty)) == []

    label = tmp_path / "duplicates.lines.txt"
    label.write_text("10 20 10 20 30 40 10 20\n", encoding="utf-8")
    local, oracle = parse_lines_txt(label), ORACLE.parse_lines_txt(str(label))
    assert len(local) == len(oracle) == 1
    np.testing.assert_array_equal(local[0], np.asarray(oracle[0], dtype=np.float64))
    assert local[0].tolist() == [[10.0, 20.0], [30.0, 40.0], [10.0, 20.0]]


@pytest.mark.parametrize("body", [
    "1 2 3\n",            # odd token count
    "1 2\n",              # one point
    "1 2 nan 4\n",        # non-finite
    "1 2 1 2\n",          # too short after adjacent dedup
    "1,2 3,4\n",          # commas are not official syntax
])
def test_parser_failures_match_oracle(tmp_path, body):
    path = tmp_path / "bad.lines.txt"
    path.write_text(body, encoding="utf-8")
    with pytest.raises((ValueError, TypeError)):
        parse_lines_txt(path)
    with pytest.raises((ValueError, TypeError)):
        ORACLE.parse_lines_txt(str(path))


def test_hungarian_assignment_applies_threshold_after_global_assignment(monkeypatch):
    # Official counterexample from DECISIONS §17.1. Global max-IoU assignment
    # chooses 0.9 + 0.4, then thresholding yields one TP. Pre-masking would
    # choose 0.6 + 0.6 and incorrectly yield two.
    matrix = np.asarray([[0.9, 0.6], [0.6, 0.4]], dtype=np.float64)
    monkeypatch.setattr(matching, "iou_matrix", lambda _pred, _gt: matrix)
    dummy = [np.zeros((1, 1), dtype=bool)] * 2
    assert matching.match_image(dummy, dummy, iou_thr=0.5) == (1, 2, 2)


def test_spline_failure_propagates_without_linear_fallback(monkeypatch):
    # Force scipy's fitter to fail and prove the diagnostic layer does not
    # silently replace the official geometry with linear interpolation.
    import eval.rasterize as rasterize

    def fail(*_args, **_kwargs):
        raise ValueError("synthetic scipy failure")

    monkeypatch.setattr(rasterize, "splprep", fail)
    with pytest.raises(ValueError, match="synthetic scipy failure"):
        rasterize.interp_lane(_lane())


def test_duplicate_helper_matches_oracle():
    points = [(1, 2), (1, 2), (3, 4), (1, 2)]
    assert remove_consecutive_duplicates(points) == ORACLE.remove_consecutive_duplicates(points)


def test_direct_array_duplicate_failure_matches_oracle():
    # Deduplication belongs to parse_lines_txt, not interp_lane. JSON/array
    # callers must not get a locally-smoothed score when the Oracle would fail.
    points = np.asarray([[10.0, 10.0], [10.0, 10.0], [20.0, 20.0]])
    with pytest.raises(Exception):
        interp_lane(points)
    with pytest.raises(Exception):
        ORACLE.interp_lane(points)
