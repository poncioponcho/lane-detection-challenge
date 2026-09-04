import ast
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


def test_split_registry_separates_labeled_from_prediction_only():
    assert MODULE.SUPPORTED_SPLITS == {"train", "val", "testA", "testB"}
    assert MODULE.is_labeled_split("train") and MODULE.is_labeled_split("val")
    assert not MODULE.is_labeled_split("testA")
    assert not MODULE.is_labeled_split("testB")


def test_read_manifest_rows_accepts_unlabeled_split(tmp_path):
    manifest = tmp_path / "testA.jsonl"
    row = _row(split="testA")
    row["gt_path"] = None
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    rows = MODULE.read_manifest_rows(manifest, "testA")
    assert len(rows) == 1
    assert rows[0]["gt_path"] is None


def test_read_manifest_rows_rejects_gt_leak_into_unlabeled_split(tmp_path):
    manifest = tmp_path / "testA.jsonl"
    manifest.write_text(json.dumps(_row(split="testA")) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="carries gt_path"):
        MODULE.read_manifest_rows(manifest, "testA")


def test_read_manifest_rows_rejects_null_gt_for_labeled_split(tmp_path):
    manifest = tmp_path / "train.jsonl"
    row = _row(split="train")
    row["gt_path"] = None
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="null gt_path"):
        MODULE.read_manifest_rows(manifest, "train")


def _make_dataset(tmp_path, labeled, create_image=True):
    dataset = MODULE.HardLaneDataset.__new__(MODULE.HardLaneDataset)
    dataset.data_root = str(tmp_path)
    dataset.split = "train" if labeled else "testA"
    dataset.labeled = labeled
    dataset.cfg = None
    image_dir = tmp_path / "JPEGImages" / "clip"
    image_dir.mkdir(parents=True, exist_ok=True)
    if create_image:
        (image_dir / "00003.jpg").write_bytes(b"")
    manifest = tmp_path / ("train.jsonl" if labeled else "testA.jsonl")
    row = _row(split=dataset.split)
    if not labeled:
        row["gt_path"] = None
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    dataset.manifest_path = str(manifest)
    return dataset


def test_dataset_load_annotations_skips_gt_and_mask_for_unlabeled(tmp_path):
    dataset = _make_dataset(tmp_path, labeled=False)
    dataset.load_annotations()
    info = dataset.data_infos[0]
    assert info["labeled"] is False
    # Must be [] not None: GenerateLaneLine iterates lanes even when not training.
    assert info["lanes"] == []
    assert info["anno_path"] is None and info["mask_path"] is None


def test_dataset_load_annotations_still_requires_gt_for_labeled(tmp_path):
    dataset = _make_dataset(tmp_path, labeled=True)
    with pytest.raises(FileNotFoundError, match="manifest GT missing"):
        dataset.load_annotations()


def test_repository_testA_manifest_is_unlabeled_and_ordered():
    root = Path(__file__).parents[1]
    rows = MODULE.read_manifest_rows(root / "data/processed/manifest_testA.jsonl", "testA")
    assert len(rows) == 900
    assert {row["split"] for row in rows} == {"testA"}
    assert all(row["gt_path"] is None for row in rows)


class _FakeLane:
    """CLRNet lane object: callable over normalized y, returns x fractions."""

    def __init__(self, x):
        self.x = float(x)

    def __call__(self, ys):
        return np.full(len(ys), self.x, dtype=np.float64)


def _make_evaluator(tmp_path, labeled, lane_lists=None):
    evaluator = MODULE.HardLaneEvaluator.__new__(MODULE.HardLaneEvaluator)
    evaluator.output_basedir = str(tmp_path / "out")
    evaluator.cfg = types.SimpleNamespace(
        sample_y=[300.0, 400.0, 500.0], ori_img_h=720.0, ori_img_w=1366.0
    )
    evaluator.metric = "F1"
    evaluator.data_infos = [
        {
            "image_id": "clip/00003",
            "labeled": labeled,
            # Unlabeled rows still carry [] (not None) so GenerateLaneLine's
            # non-training path can iterate them; "labeled" is the discriminator.
            "lanes": [[(683.0, 300.0), (683.0, 500.0)]] if labeled else [],
        }
    ]
    return evaluator


def test_evaluator_exports_predictions_without_f1_for_unlabeled(tmp_path):
    evaluator = _make_evaluator(tmp_path, labeled=False)
    result = evaluator.evaluate([[_FakeLane(0.5)]])
    assert result == {}
    prediction = tmp_path / "out" / "predictions" / "clip" / "00003.lines.txt"
    assert prediction.is_file()
    summary = json.loads((tmp_path / "out" / "unlabeled_summary.json").read_text())
    assert summary["status"] == "predictions_only"
    assert summary["images"] == 1
    assert not (tmp_path / "out" / "diagnostic_metric.json").exists()


def test_evaluator_still_scores_labeled_after_unlabeled_support(tmp_path):
    evaluator = _make_evaluator(tmp_path, labeled=True)
    result = evaluator.evaluate([[_FakeLane(0.5)]])
    assert "F1" in result
    assert (tmp_path / "out" / "diagnostic_metric.json").is_file()
    assert not (tmp_path / "out" / "unlabeled_summary.json").exists()


def test_evaluator_rejects_mixed_labeled_and_unlabeled_rows(tmp_path):
    evaluator = _make_evaluator(tmp_path, labeled=False)
    evaluator.data_infos.append({"image_id": "clip/00004", "labeled": True, "lanes": []})
    with pytest.raises(ValueError, match="mixed labeled/unlabeled"):
        evaluator.evaluate([[_FakeLane(0.5)], [_FakeLane(0.5)]])


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


def _literal_module_constant(source: str, name: str):
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found as a module-level literal")


def test_infer_testA_script_matches_unlabeled_split_contract():
    root = Path(__file__).parents[1]
    source = (root / "scripts/autodl/infer_testA.py").read_text(encoding="utf-8")
    assert _literal_module_constant(source, "UNLABELED_SPLITS") == MODULE.UNLABELED_SPLITS
    # The eval must actually be redirected at the unlabeled split...
    for key in (
        "dataloader.test.dataset.manifest_path",
        "dataloader.test.dataset.split",
        "dataloader.evaluator.output_basedir",
        "train.init_checkpoint",
    ):
        assert f'override("{key}"' in source
    # ...and must refuse to believe a score computed without ground truth.
    assert "diagnostic_metric.json" in source
    assert "predictions_only" in source


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
