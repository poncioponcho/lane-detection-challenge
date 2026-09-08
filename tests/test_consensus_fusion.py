from pathlib import Path
import sys


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
