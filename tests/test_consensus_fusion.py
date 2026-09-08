from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.autodl.consensus_fusion import fuse_image  # noqa: E402


def test_fuse_image_preserves_three_value_shape_for_all_empty_inputs():
    lanes, scores, subquorum = fuse_image(
        [[], [], []],
        [[], [], []],
        quorum=2,
        max_dx=15.0,
    )

    assert lanes == []
    assert scores == []
    assert subquorum == 0


def test_fuse_image_median_preserves_best_endpoint_grid():
    y = [719.0, 714.0, 709.0]
    lanes = [
        [np.array([[100.0, y[0]], [110.0, y[1]], [120.0, y[2]]])],
        [np.array([[102.0, y[0]], [112.0, y[1]], [122.0, y[2]]])],
    ]
    fused, scores, subquorum = fuse_image(
        lanes, [[0.8], [0.7]], quorum=2, max_dx=15.0, geometry="median"
    )

    assert subquorum == 0
    assert scores == [0.75]
    assert fused[0][:, 0].tolist() == [101.0, 111.0, 121.0]
    assert fused[0][:, 1].tolist() == y
