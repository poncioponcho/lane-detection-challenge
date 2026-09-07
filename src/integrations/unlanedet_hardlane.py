"""HardLane manifest dataset and diagnostic evaluator for UnLanedet.

This module is imported by the AutoDL-only LazyConfig files.  The small pure
helpers deliberately remain importable without PyTorch/UnLanedet so their data
contract can be tested on the local CPU environment.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path, PurePosixPath
from typing import Iterable, Sequence

import numpy as np

from src.eval.score_sidecar import (
    POSITIVE_CLASS_SOFTMAX_PROBABILITY,
    UNKNOWN_SCORE_SEMANTICS,
    make_score_sidecar_contract,
    probability_score,
)

# imgaug==0.4.0 (pinned by UnLanedet's requirements) still references numpy
# aliases that were removed in numpy>=1.24 (np.bool etc.) at call time -- e.g.
# dtype=np.bool in augmenters/meta.py. The project pins numpy==1.26.4, so
# restore the removed alias family here. This module is imported by every
# AutoDL LazyConfig, so the shim executes in each train/eval process before
# DataLoader workers fork and hit the augmentation path.
for _alias, _target in (
    ("bool", bool),
    ("int", int),
    ("float", float),
    ("complex", complex),
    ("object", object),
    ("str", str),
):
    if not hasattr(np, _alias):
        setattr(np, _alias, _target)

try:  # AutoDL dependency; intentionally absent from the local CPU venv.
    import cv2
    from unlanedet.data.base_dataset import BaseDataset
    from unlanedet.data.transform import DataContainer as DC
    from unlanedet.evaluation import DatasetEvaluator

    _UNLANEDET_IMPORT_ERROR: Exception | None = None
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover - AutoDL path
    cv2 = None
    BaseDataset = object
    DatasetEvaluator = object
    DC = None
    _UNLANEDET_IMPORT_ERROR = exc


LOGGER = logging.getLogger(__name__)
CANVAS_W = 1366
CANVAS_H = 720

# Splits whose manifest rows carry a GT path (train/val) versus the competition
# prediction-only splits (testA/testB). Unlabeled rows have gt_path = None and
# must not be required to ship anno_txt/Annotations files: the released
# testA set (900 images / 9 clips) ships JPEGImages only.
LABELED_SPLITS = {"train", "val"}
UNLABELED_SPLITS = {"testA", "testB"}
SUPPORTED_SPLITS = LABELED_SPLITS | UNLABELED_SPLITS


def lower_half_luma(image: np.ndarray, cut_height: int = 0) -> float:
    """Return mean BT.601 luma in the visible lower-half road region.

    The input is the raw OpenCV BGR image before the dataset crop.  Using the
    lower half avoids letting sky/ceiling pixels decide whether a road image
    needs the optional low-light correction.  ``cut_height`` is also honored
    so the region never includes pixels removed by the model crop.
    """
    array = np.asarray(image)
    if array.ndim == 2:
        gray = array.astype(np.float32, copy=False)
    elif array.ndim == 3 and array.shape[2] >= 3:
        blue, green, red = (array[..., index].astype(np.float32, copy=False)
                            for index in range(3))
        gray = 0.114 * blue + 0.587 * green + 0.299 * red
    else:
        raise ValueError(f"expected grayscale or BGR image, got shape={array.shape}")
    start = max(gray.shape[0] // 2, int(cut_height))
    region = gray[start:]
    if region.size == 0:
        raise ValueError(f"image has no lower-half ROI: shape={array.shape}")
    return float(region.mean())


def apply_conditional_gamma(
    image: np.ndarray,
    *,
    enabled: bool,
    luma_threshold: float,
    gamma: float,
    cut_height: int = 0,
) -> np.ndarray:
    """Brighten only dark images with a gentle, deterministic gamma curve.

    This function is deliberately independent of imgaug and works for both
    train and eval/test paths.  A gamma below one brightens shadows.  The
    decision is made from the pristine image before random training
    augmentations, so inference uses the same trigger and curve.
    """
    array = np.asarray(image)
    if not enabled:
        return array
    if not np.isfinite(luma_threshold) or not 0.0 <= luma_threshold <= 255.0:
        raise ValueError(f"luma_threshold must be in [0, 255], got {luma_threshold!r}")
    if not np.isfinite(gamma) or gamma <= 0.0:
        raise ValueError(f"gamma must be positive and finite, got {gamma!r}")
    if lower_half_luma(array, cut_height=cut_height) > luma_threshold:
        return array
    normalized = array.astype(np.float32) / 255.0
    corrected = np.power(np.clip(normalized, 0.0, 1.0), gamma) * 255.0
    return np.clip(corrected, 0.0, 255.0).astype(array.dtype, copy=False)


def conditional_gamma_settings(cfg) -> tuple[bool, float, float]:
    """Read and validate the optional preprocessing contract from ``cfg``."""
    raw = cfg.get("conditional_gamma", {}) if cfg is not None else {}
    if raw is None:
        raw = {}
    enabled = bool(raw.get("enabled", False))
    luma_threshold = float(raw.get("luma_threshold", 42.0))
    gamma = float(raw.get("gamma", 0.85))
    if not np.isfinite(luma_threshold) or not 0.0 <= luma_threshold <= 255.0:
        raise ValueError(
            f"conditional_gamma.luma_threshold must be in [0, 255], got {luma_threshold!r}"
        )
    if not np.isfinite(gamma) or gamma <= 0.0:
        raise ValueError(
            f"conditional_gamma.gamma must be positive and finite, got {gamma!r}"
        )
    return enabled, luma_threshold, gamma


def is_labeled_split(split: str) -> bool:
    """Whether a split name carries lane-line ground truth."""
    return split in LABELED_SPLITS


def _sample_tensors_finite(sample: dict) -> bool:
    """True when every transformed tensor in the sample is finite (no NaN/Inf).

    ToTensor outputs are CPU tensors with no autograd, so ``.numpy()`` is safe
    and keeps this module importable without PyTorch (local CPU contract).
    """
    for key in ("img", "lane_line", "seg"):
        tensor = sample.get(key)
        if tensor is None:
            continue
        values = tensor.numpy() if hasattr(tensor, "numpy") else np.asarray(tensor)
        if not np.isfinite(values).all():
            return False
    return True


def _require_unlanedet() -> None:
    if _UNLANEDET_IMPORT_ERROR is not None:
        raise RuntimeError(
            "HardLaneDataset requires the pinned UnLanedet AutoDL environment"
        ) from _UNLANEDET_IMPORT_ERROR


def stable_consecutive_dedup(points: Sequence[Sequence[float]]) -> list[tuple[float, float]]:
    """Remove only adjacent duplicate points while preserving source order."""
    result: list[tuple[float, float]] = []
    for raw_point in points:
        point = (float(raw_point[0]), float(raw_point[1]))
        if not result or point != result[-1]:
            result.append(point)
    return result


def canonicalize_lane(points: Sequence[Sequence[float]]) -> list[tuple[float, float]]:
    """Apply the competition-safe lane normalization used before transforms.

    Unlike upstream CULane, this never converts through ``set`` and never drops
    valid two- or three-point lanes. Equal-y ordering is stable.
    """
    lane = stable_consecutive_dedup(points)
    if len(lane) < 2:
        raise ValueError("lane has fewer than 2 points after consecutive deduplication")
    if not np.isfinite(np.asarray(lane, dtype=np.float64)).all():
        raise ValueError("lane contains NaN/Inf")
    return sorted(lane, key=lambda point: -point[1])


def read_lines_lanes(path: str | Path) -> list[list[tuple[float, float]]]:
    """Read one official ``.lines.txt`` without upstream's lossy filtering."""
    lanes: list[list[tuple[float, float]]] = []
    for line_number, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            values = [float(token) for token in raw.split()]
        except ValueError as exc:
            raise ValueError(f"non-numeric lane at {path}:{line_number}") from exc
        if len(values) < 4 or len(values) % 2:
            raise ValueError(f"invalid coordinate count at {path}:{line_number}")
        points = list(zip(values[0::2], values[1::2]))
        lanes.append(canonicalize_lane(points))
    return lanes


def read_palette_indices(path: str | Path) -> np.ndarray:
    """Read original palette indices; never collapse a color PNG to B/G/R."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required to preserve palette PNG indices") from exc
    with Image.open(path) as image:
        if image.mode not in {"P", "L", "I", "I;16"}:
            raise ValueError(
                f"expected indexed/grayscale annotation PNG, got mode={image.mode!r}: {path}"
            )
        mask = np.array(image, copy=True)
    if mask.ndim != 2:
        raise ValueError(f"expected 2-D annotation mask, got {mask.shape}: {path}")
    if mask.min(initial=0) < 0 or mask.max(initial=0) > 255:
        raise ValueError(f"annotation ids must fit uint8: {path}")
    return mask.astype(np.uint8, copy=False)


def _safe_relative_path(raw: str, expected_prefix: str) -> PurePosixPath:
    path = PurePosixPath(raw)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"unsafe manifest path: {raw!r}")
    if path.parts[0] != expected_prefix:
        raise ValueError(f"expected {expected_prefix}/..., got {raw!r}")
    return path


def read_manifest_rows(path: str | Path, expected_split: str) -> list[dict]:
    """Read the frozen JSONL order and recheck identities used by training."""
    rows: list[dict] = []
    seen: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid manifest JSON at line {line_number}") from exc
            required = {
                "image_id", "image_path", "pred_rel_path", "gt_path", "clip_id",
                "frame_id", "split", "order",
            }
            missing = required.difference(row)
            if missing:
                raise ValueError(f"manifest line {line_number} missing {sorted(missing)}")
            if row["order"] != len(rows):
                raise ValueError(f"manifest order mismatch at line {line_number}")
            if row["split"] != expected_split:
                raise ValueError(
                    f"manifest split mismatch at line {line_number}: {row['split']!r}"
                )
            expected_id = f"{row['clip_id']}/{row['frame_id']}"
            if row["image_id"] != expected_id or expected_id in seen:
                raise ValueError(f"duplicate/inconsistent image_id at line {line_number}")
            image_path = _safe_relative_path(row["image_path"], "JPEGImages")
            if image_path.parts[1] != row["clip_id"] or image_path.stem != row["frame_id"]:
                raise ValueError(f"inconsistent image_path at line {line_number}")
            if row["pred_rel_path"] != f"{row['clip_id']}/{row['frame_id']}.lines.txt":
                raise ValueError(f"inconsistent pred_rel_path at line {line_number}")
            if is_labeled_split(expected_split):
                if row["gt_path"] is None:
                    raise ValueError(f"labeled split {expected_split!r} has null gt_path at line {line_number}")
                gt_path = _safe_relative_path(row["gt_path"], "anno_txt")
                if gt_path.parts[1] != row["clip_id"] or gt_path.name != f"{row['frame_id']}.lines.txt":
                    raise ValueError(f"inconsistent gt_path at line {line_number}")
            elif row["gt_path"] is not None:
                # Leaking GT into a prediction-only split would silently turn a
                # submission rehearsal into a self-scoring run.
                raise ValueError(f"unlabeled split {expected_split!r} carries gt_path at line {line_number}")
            seen.add(expected_id)
            rows.append(row)
    if not rows:
        raise ValueError(f"manifest is empty: {path}")
    return rows


class HardLaneDataset(BaseDataset):
    """Manifest-backed training/validation dataset for HardLane-F100."""

    def __init__(self, data_root, manifest_path, split, cut_height, processes=None, cfg=None):
        _require_unlanedet()
        if split not in SUPPORTED_SPLITS:
            raise ValueError(
                f"HardLaneDataset supports {sorted(SUPPORTED_SPLITS)}, got {split!r}"
            )
        self.split = split
        self.labeled = is_labeled_split(split)
        super().__init__(data_root, split, cut_height, processes=processes, cfg=cfg)
        (
            self.conditional_gamma_enabled,
            self.conditional_gamma_luma_threshold,
            self.conditional_gamma_value,
        ) = conditional_gamma_settings(cfg)
        LOGGER.info(
            "Conditional gamma: enabled=%s threshold=%.3f gamma=%.3f",
            self.conditional_gamma_enabled,
            self.conditional_gamma_luma_threshold,
            self.conditional_gamma_value,
        )
        self.manifest_path = str(manifest_path)
        self.load_annotations()

    def load_annotations(self):
        self.data_infos = []
        rows = read_manifest_rows(self.manifest_path, self.split)
        data_root = Path(self.data_root)
        for row in rows:
            image_rel = PurePosixPath(row["image_path"])
            image_path = data_root.joinpath(*image_rel.parts)
            if not image_path.is_file():
                raise FileNotFoundError(f"manifest image missing: {image_path}")
            info = {
                "image_id": row["image_id"],
                "img_name": image_rel.as_posix(),
                "img_path": str(image_path),
                "manifest_order": row["order"],
                "labeled": self.labeled,
            }
            if self.labeled:
                gt_rel = PurePosixPath(row["gt_path"])
                mask_rel = PurePosixPath("Annotations", row["clip_id"], f"{row['frame_id']}.png")
                gt_path = data_root.joinpath(*gt_rel.parts)
                mask_path = data_root.joinpath(*mask_rel.parts)
                for kind, file_path in (("GT", gt_path), ("mask", mask_path)):
                    if not file_path.is_file():
                        raise FileNotFoundError(f"manifest {kind} missing: {file_path}")
                lanes = read_lines_lanes(gt_path)
                if self.cfg is not None and len(lanes) > int(self.cfg.max_lanes):
                    raise ValueError(
                        f"{row['image_id']} has {len(lanes)} lanes, exceeds target capacity "
                        f"{self.cfg.max_lanes}"
                    )
                info.update(
                    {
                        "anno_path": str(gt_path),
                        "mask_path": str(mask_path),
                        "lanes": lanes,
                        "is_empty": not lanes,
                    }
                )
            else:
                # Prediction-only split: keep the key set identical so consumers
                # never hit a KeyError. lanes must be [] rather than None --
                # GenerateLaneLine.__call__ iterates sample["lanes"] even with
                # training=False (cut_height offset loop + lane_to_linestrings),
                # so None would raise TypeError. [] is the same state the 35
                # empty-GT val images already exercise. "labeled" is the
                # discriminator; is_empty=True only reflects the empty list and
                # must not be read as "this clip has no lanes".
                info.update(
                    {
                        "anno_path": None,
                        "mask_path": None,
                        "lanes": [],
                        "is_empty": True,
                    }
                )
            self.data_infos.append(info)
        if self.labeled:
            LOGGER.info(
                "Loaded HardLane %s manifest: %d images, %d empty",
                self.split,
                len(self.data_infos),
                sum(info["is_empty"] for info in self.data_infos),
            )
        else:
            LOGGER.info(
                "Loaded HardLane %s manifest: %d images (unlabeled, predictions only)",
                self.split,
                len(self.data_infos),
            )

    def _build_raw_sample(self, data_info: dict, image, mask):
        """Pristine pre-transforms sample; processes() mutates its input in
        place (sample["img"] is replaced by the normalized tensor), so every
        augmentation attempt must start from a freshly built dict."""
        sample = data_info.copy()
        sample["img"] = image[self.cut_height :, :, :]
        if self.training:
            sample["mask"] = mask[self.cut_height :, :]
        return sample

    def __getitem__(self, idx):
        data_info = self.data_infos[idx]
        image = cv2.imread(data_info["img_path"], cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"cannot decode image: {data_info['img_path']}")
        if tuple(image.shape[:2]) != (CANVAS_H, CANVAS_W):
            raise ValueError(f"unexpected image shape {image.shape}: {data_info['img_path']}")
        image = apply_conditional_gamma(
            image,
            enabled=self.conditional_gamma_enabled,
            luma_threshold=self.conditional_gamma_luma_threshold,
            gamma=self.conditional_gamma_value,
            cut_height=self.cut_height,
        )

        mask = None
        if self.training:
            mask = read_palette_indices(data_info["mask_path"])
            if tuple(mask.shape) != (CANVAS_H, CANVAS_W):
                raise ValueError(f"unexpected mask shape {mask.shape}: {data_info['mask_path']}")
            # HardLane palette ids can exceed the seg head's class budget: the
            # pinned GenerateLaneLine feeds raw mask values into the seg NLL loss
            # (clr_head.py), whose class count is num_classes = max_gt_lanes + 1
            # = 9. Ids 9/10 appear in 51/7100 images and made nll_loss trigger a
            # device-side assert once shuffling sampled one of them (observed as
            # "CUDA error: device-side assert triggered" at gate iter 5). Collapse
            # overflowing ids into background: seg is auxiliary supervision and
            # the primary lane-line GT comes from anno_txt, so no line-level
            # label is lost.
            # adnet's param_config has no num_classes (no seg head); fall back
            # to max_lanes + 1, which equals the clrnet value (9).
            num_seg_classes = int(
                self.cfg.get("num_classes", int(self.cfg.max_lanes) + 1)
            )
            if mask.max(initial=0) >= num_seg_classes:
                LOGGER.warning(
                    "HardLane mask %s has seg ids >= num_classes(%d); collapsing to background",
                    data_info["image_id"],
                    num_seg_classes,
                )
                mask = np.where(mask >= num_seg_classes, 0, mask).astype(np.uint8, copy=False)

        sample = self.processes(self._build_raw_sample(data_info, image, mask))
        if self.training and not _sample_tensors_finite(sample):
            # Rarely, a degenerate random-augmentation draw (e.g. Affine pushing
            # a short lane fully out of the image) makes GenerateLaneLine emit
            # NaN targets; that NaN then poisons the loss (observed as an
            # AddmmBackward0 NaN on gate iter 0). Re-draw with fresh RNG state;
            # each attempt is effectively an independent draw.
            exhausted = True
            for _ in range(5):
                sample = self.processes(self._build_raw_sample(data_info, image, mask))
                if _sample_tensors_finite(sample):
                    exhausted = False
                    break
            if exhausted:
                LOGGER.warning(
                    "HardLane sample %s stayed non-finite after 6 augmentation draws",
                    data_info["image_id"],
                )
        sample["meta"] = DC(
            {
                "full_img_path": data_info["img_path"],
                "img_name": data_info["img_name"],
                "image_id": data_info["image_id"],
                "manifest_order": data_info["manifest_order"],
            },
            cpu_only=True,
        )
        return sample


class HardLaneEvaluator(DatasetEvaluator):
    """Training-time diagnostic F1; final decisions still use frozen Oracle."""

    def __init__(self, output_basedir, cfg, metric="F1"):
        _require_unlanedet()
        self.output_basedir = str(output_basedir)
        self.cfg = cfg
        self.metric = metric
        self.data_infos = []

    def reset(self):
        return None

    def _prediction_points(self, prediction: Iterable) -> list[np.ndarray]:
        points, _, _ = self._prediction_points_and_scores(prediction)
        return points

    def _prediction_points_and_scores(
        self, prediction: Iterable
    ) -> tuple[list[np.ndarray], list[float | None], list[str | None]]:
        sample_ys = np.asarray(list(self.cfg.sample_y), dtype=np.float64)
        normalized_ys = sample_ys / float(self.cfg.ori_img_h)
        result: list[np.ndarray] = []
        scores: list[float | None] = []
        score_semantics: list[str | None] = []
        for lane in prediction:
            xs = np.asarray(lane(normalized_ys), dtype=np.float64)
            valid = np.isfinite(xs) & (xs >= 0.0) & (xs < 1.0)
            points = np.column_stack(
                (xs[valid] * float(self.cfg.ori_img_w), sample_ys[valid])
            )
            if len(points) >= 2:
                result.append(points)
                metadata = getattr(lane, "metadata", {}) or {}
                confidence = metadata.get("conf")
                semantics = metadata.get("score_semantics")
                if semantics is not None and not isinstance(semantics, str):
                    raise ValueError(
                        f"lane score_semantics must be a string, got {semantics!r}"
                    )
                if confidence is None:
                    scores.append(None)
                elif semantics == POSITIVE_CLASS_SOFTMAX_PROBABILITY:
                    scores.append(
                        probability_score(
                            confidence,
                            context="positive-class softmax confidence",
                        )
                    )
                else:
                    try:
                        value = float(confidence)
                    except (TypeError, ValueError) as exc:
                        raise ValueError(
                            f"lane confidence is not numeric: {confidence!r}"
                        ) from exc
                    if not np.isfinite(value):
                        raise ValueError(f"lane confidence is not finite: {value!r}")
                    scores.append(value)
                score_semantics.append(semantics)
        return result, scores, score_semantics

    @staticmethod
    def _candidate_export_conf_threshold(cfg) -> float:
        test_parameters = getattr(cfg, "test_parameters", None)
        value = getattr(test_parameters, "conf_threshold", 0.0)
        return 0.0 if value is None else float(value)

    @staticmethod
    def _write_prediction(path: Path, points: Sequence[np.ndarray]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            " ".join(f"{value:.5f}" for value in lane.reshape(-1)) for lane in points
        ]
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    def evaluate(self, predictions):
        if len(predictions) != len(self.data_infos):
            raise ValueError(f"prediction count {len(predictions)} != manifest count {len(self.data_infos)}")
        # "labeled" (not lane emptiness) is the discriminator: unlabeled rows
        # legitimately carry lanes == [] to keep the transform chain working.
        labeled_flags = {bool(info["labeled"]) for info in self.data_infos}
        if len(labeled_flags) > 1:
            raise ValueError("mixed labeled/unlabeled rows in one evaluator")
        labeled = labeled_flags.pop() if labeled_flags else False

        prediction_root = Path(self.output_basedir) / "predictions"
        prediction_by_image = {}
        score_by_image = {}
        semantics_seen: set[str] = set()
        for info, prediction in zip(self.data_infos, predictions):
            points, scores, line_semantics = self._prediction_points_and_scores(
                prediction
            )
            prediction_by_image[info["image_id"]] = points
            score_by_image[info["image_id"]] = scores
            semantics_seen.update(
                semantics for semantics in line_semantics if semantics is not None
            )
            self._write_prediction(prediction_root / f"{info['image_id']}.lines.txt", points)

        if len(semantics_seen) > 1:
            raise ValueError(
                "mixed score semantics in one evaluator: "
                f"{sorted(semantics_seen)}"
            )
        score_semantics = next(iter(semantics_seen), UNKNOWN_SCORE_SEMANTICS)

        output_root = Path(self.output_basedir)
        output_root.mkdir(parents=True, exist_ok=True)
        score_path = output_root / "prediction_scores.json"
        score_contract = make_score_sidecar_contract(
            score_semantics,
            self._candidate_export_conf_threshold(self.cfg),
        )
        score_path.write_text(
            json.dumps(
                {
                    "status": "pass",
                    "images": len(score_by_image),
                    "scores_by_image": score_by_image,
                    **score_contract,
                    "note": (
                        "scores correspond to exported lines after model NMS; "
                        "export eval must use conf_threshold=0.0 for offline threshold scans; "
                        "unknown semantics are intentionally rejected by scan tools"
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        if not labeled:
            # testA/testB: no GT exists, so any F1 here would be fabricated.
            # Export the predictions and leave scoring to the frozen Oracle or
            # the competition server.
            summary = {
                "status": "predictions_only",
                "images": len(prediction_by_image),
                "prediction_root": str(prediction_root),
            }
            (output_root / "unlabeled_summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            LOGGER.info(
                "Exported %d unlabeled predictions to %s (no diagnostic F1)",
                len(prediction_by_image),
                prediction_root,
            )
            return {}

        # The evaluator runs inside train_net's process whose CWD (and thus
        # sys.path seed) is the UnLanedet repo root, so neither the project
        # root ("src" package) nor src/ itself ("common" package) is importable
        # by default (observed as ModuleNotFoundError: No module named
        # 'common' in the gate final eval). Mirror the probe/smoke pattern and
        # add both roots up front.
        integration_root = Path(__file__).resolve().parent
        src_root = integration_root.parent
        project_root = src_root.parent
        for path in (str(project_root), str(src_root)):
            if path not in sys.path:
                sys.path.insert(0, path)
        from src.eval.matching import compute_f1

        gt_by_image = {
            info["image_id"]: [np.asarray(lane, dtype=np.float64) for lane in info["lanes"]]
            for info in self.data_infos
        }
        result = compute_f1(prediction_by_image, gt_by_image)
        (output_root / "diagnostic_metric.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return result
