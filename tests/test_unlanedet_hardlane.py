import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest


MODULE_PATH = (
    Path(__file__).parents[1] / "src/integrations/unlanedet_hardlane.py"
)
SPEC = importlib.util.spec_from_file_location("unlanedet_hardlane", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _row(order=0, split="train"):
    return {
        "image_id": "clip/00003",
        "image_path": "JPEGImages/clip/00003.jpg",
        "pred_rel_path": "clip/00003.lines.txt",
        "gt_path": "anno_txt/clip/00003.lines.txt",
        "clip_id": "clip",
        "frame_id": "00003",
        "split": split,
        "order": order,
    }


def test_stable_consecutive_dedup_preserves_nonconsecutive_points():
    points = [(1, 3), (1, 3), (2, 2), (1, 3)]
    assert MODULE.stable_consecutive_dedup(points) == [(1.0, 3.0), (2.0, 2.0), (1.0, 3.0)]


def test_canonicalize_lane_retains_two_points_and_sorts_bottom_to_top():
    assert MODULE.canonicalize_lane([(4, 1), (2, 7)]) == [(2.0, 7.0), (4.0, 1.0)]


def test_canonicalize_lane_rejects_collapsed_or_nonfinite():
    with pytest.raises(ValueError, match="fewer than 2"):
        MODULE.canonicalize_lane([(1, 2), (1, 2)])
    with pytest.raises(ValueError, match="NaN/Inf"):
        MODULE.canonicalize_lane([(1, 2), (np.nan, 3)])


def test_read_lines_lanes_keeps_short_lines(tmp_path):
    label = tmp_path / "sample.lines.txt"
    label.write_text("1 2 3 4\n9 8 9 8 7 6\n", encoding="utf-8")
    lanes = MODULE.read_lines_lanes(label)
    assert lanes == [[(3.0, 4.0), (1.0, 2.0)], [(9.0, 8.0), (7.0, 6.0)]]


class _FakePillowImage:
    def __init__(self, mode, array):
        self.mode = mode
        self.array = array

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def __array__(self, dtype=None, copy=None):
        return np.array(self.array, dtype=dtype, copy=True if copy is None else copy)


def _install_fake_pillow(monkeypatch, mode, array):
    image_api = types.SimpleNamespace(open=lambda _path: _FakePillowImage(mode, array))
    monkeypatch.setitem(sys.modules, "PIL", types.SimpleNamespace(Image=image_api))


def test_read_palette_indices_preserves_palette_ids(tmp_path, monkeypatch):
    _install_fake_pillow(monkeypatch, "P", np.array([[0, 1], [2, 3]], dtype=np.uint8))
    assert MODULE.read_palette_indices(tmp_path / "mask.png").tolist() == [[0, 1], [2, 3]]


def test_read_palette_indices_rejects_rgb(tmp_path, monkeypatch):
    path = tmp_path / "mask.png"
    _install_fake_pillow(monkeypatch, "RGB", np.zeros((2, 2, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="indexed/grayscale"):
        MODULE.read_palette_indices(path)


def test_read_manifest_rows_checks_order_split_and_paths(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(_row()) + "\n", encoding="utf-8")
    assert MODULE.read_manifest_rows(manifest, "train")[0]["image_id"] == "clip/00003"

    broken = _row(order=1)
    manifest.write_text(json.dumps(broken) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="order mismatch"):
        MODULE.read_manifest_rows(manifest, "train")

    broken = _row()
    broken["image_path"] = "JPEGImages/other/00003.jpg"
    manifest.write_text(json.dumps(broken) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="inconsistent image_path"):
        MODULE.read_manifest_rows(manifest, "train")

    broken = _row()
    broken["pred_rel_path"] = "other/00003.lines.txt"
    manifest.write_text(json.dumps(broken) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="inconsistent pred_rel_path"):
        MODULE.read_manifest_rows(manifest, "train")


def test_repository_manifests_match_autodl_contract():
    root = Path(__file__).parents[1]
    train = MODULE.read_manifest_rows(
        root / "data/processed/manifest_train_v1_seed42.jsonl", "train"
    )
    val = MODULE.read_manifest_rows(
        root / "data/processed/manifest_val_v1_seed42.jsonl", "val"
    )
    assert (len(train), len(val)) == (6300, 800)
    assert not {row["clip_id"] for row in train}.intersection(
        row["clip_id"] for row in val
    )


def test_autodl_configs_and_scripts_encode_execution_contract():
    root = Path(__file__).parents[1]
    clr = (root / "configs/unlanedet/clrnet_r50_hardlane.py").read_text()
    ad = (root / "configs/unlanedet/adnet_r34_hardlane.py").read_text()
    probe = (root / "scripts/autodl/probe_weights.py").read_text()
    smoke = (root / "scripts/autodl/smoke_dataloader_and_loss.py").read_text()
    for source in (clr, ad):
        assert 'cut_height = 180' in source
        assert 'max_gt_lanes = 8' in source
        assert 'candidate_topk = 12' in source
        assert '"init_checkpoint"' in source
        assert '"seed": 42' in source
        assert '"cudnn_benchmark": False' in source
        assert '"max_to_keep": 40' in source
        assert "MODEL.WEIGHTS" not in source
    assert 'num_priors = 192' in clr
    assert 'num_classes = max_gt_lanes + 1' in clr
    assert 'anchors_num = 300' in ad
    assert "263_994_164" in probe and "292_961_772" in probe
    assert 'EXPECTED = {"train": (6300, 227), "val": (800, 35)}' in smoke
