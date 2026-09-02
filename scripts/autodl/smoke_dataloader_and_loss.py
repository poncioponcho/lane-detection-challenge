#!/usr/bin/env python3
"""AutoDL-only HardLane dataset, loss/backward, and demo inference smoke."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np


PINNED_UNLANEDET_COMMIT = "03921844220adb2e65c840de2d9759478d5c3d4c"
EXPECTED = {"train": (6300, 227), "val": (800, 35)}


def required_absolute_env(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"{name} must be set")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise SystemExit(f"{name} must be absolute, got {value!r}")
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def disable_random_augmentation(dataset_cfg) -> None:
    transforms = dataset_cfg.processes[0].transforms
    for transform in transforms:
        transform.p = 1.0 if transform.name == "Resize" else 0.0


def load_adapted_state(torch, path: Path) -> tuple[dict, dict]:
    if not path.is_file():
        raise FileNotFoundError(f"run probe_weights.py first; missing {path}")
    payload = torch.load(str(path), map_location="cpu")
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise ValueError(f"not a HardLane adapted checkpoint: {path}")
    metadata = payload.get("_hardlane_adaptation")
    if not isinstance(metadata, dict):
        raise ValueError(f"checkpoint lacks adaptation evidence: {path}")
    return payload["model"], metadata


def assert_dataset_contract(dataset, split: str) -> tuple[int, int, int, int]:
    expected_count, expected_empty = EXPECTED[split]
    count = len(dataset)
    empty_indices = [idx for idx, info in enumerate(dataset.data_infos) if info["is_empty"]]
    nonempty_indices = [idx for idx, info in enumerate(dataset.data_infos) if not info["is_empty"]]
    if count != expected_count or len(empty_indices) != expected_empty:
        raise AssertionError(
            f"{split} contract mismatch: count={count}, empty={len(empty_indices)}, "
            f"expected={EXPECTED[split]}"
        )
    if [info["manifest_order"] for info in dataset.data_infos] != list(range(count)):
        raise AssertionError(f"{split} manifest order changed")
    return count, len(empty_indices), empty_indices[0], nonempty_indices[0]


def collate_one(collate, sample):
    return collate([sample], samples_per_gpu=1)


def assert_sampler_coverage(loader, empty_indices: list[int], expected_count: int) -> dict:
    sampled = [int(index) for batch in loader.batch_sampler for index in batch]
    if len(sampled) != expected_count or len(set(sampled)) != expected_count:
        raise AssertionError(
            f"train sampler coverage changed: sampled={len(sampled)}, "
            f"unique={len(set(sampled))}, expected={expected_count}"
        )
    missing_empty = sorted(set(empty_indices).difference(sampled))
    if missing_empty:
        raise AssertionError(f"train sampler dropped empty-GT indices: {missing_empty[:10]}")
    return {
        "sampled": len(sampled),
        "unique": len(set(sampled)),
        "empty_gt_sampled": len(empty_indices),
    }


def finite_backward(torch, model, batch, label: str) -> dict:
    model.train()
    model.zero_grad(set_to_none=True)
    losses = model(batch)
    if not isinstance(losses, dict) or not losses:
        raise AssertionError(f"{label}: model did not return a loss dict")
    values = {}
    for name, value in losses.items():
        scalar = float(value.detach().cpu())
        if not np.isfinite(scalar):
            raise FloatingPointError(f"{label}: non-finite {name}={scalar}")
        values[name] = scalar
    total = sum(losses.values())
    if not bool(torch.isfinite(total).item()):
        raise FloatingPointError(f"{label}: non-finite total loss")
    total.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if not gradients:
        raise AssertionError(f"{label}: backward produced no gradients")
    if not all(bool(torch.isfinite(gradient).all().item()) for gradient in gradients):
        raise FloatingPointError(f"{label}: backward produced non-finite gradients")
    values["total"] = float(total.detach().cpu())
    values["parameters_with_grad"] = len(gradients)
    return values


def render_preprocess_overlay(cv2, info: dict, cfg, output_path: Path) -> dict:
    image = cv2.imread(info["img_path"], cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(info["img_path"])
    source = image.copy()
    cropped = image[int(cfg.cut_height):]
    network = cv2.resize(cropped, (int(cfg.img_w), int(cfg.img_h)))
    sx = float(cfg.img_w) / float(cfg.ori_img_w)
    sy = float(cfg.img_h) / float(cfg.ori_img_h - cfg.cut_height)
    max_roundtrip_error = 0.0
    for lane in info["lanes"]:
        original = np.asarray(lane, dtype=np.float64)
        visible = original[original[:, 1] >= float(cfg.cut_height)]
        if len(visible) < 2:
            continue
        transformed = np.column_stack(
            (visible[:, 0] * sx, (visible[:, 1] - float(cfg.cut_height)) * sy)
        )
        restored = np.column_stack(
            (transformed[:, 0] / sx, transformed[:, 1] / sy + float(cfg.cut_height))
        )
        max_roundtrip_error = max(
            max_roundtrip_error, float(np.max(np.abs(restored - visible)))
        )
        cv2.polylines(source, [np.rint(visible).astype(np.int32)], False, (0, 255, 255), 4)
        cv2.polylines(network, [np.rint(transformed).astype(np.int32)], False, (0, 255, 255), 3)
    source_preview = cv2.resize(source, (607, 320))
    contact = np.concatenate((source_preview, network), axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), contact):
        raise IOError(f"failed to write {output_path}")
    if max_roundtrip_error > 1e-9:
        raise AssertionError(f"preprocess coordinate round trip error={max_roundtrip_error}")
    return {
        "path": str(output_path),
        "sha256": sha256_file(output_path),
        "max_roundtrip_error": max_roundtrip_error,
    }


def render_demo(cv2, info: dict, lanes, output_path: Path) -> dict:
    image = cv2.imread(info["img_path"], cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(info["img_path"])
    rendered = 0
    for lane in lanes:
        points = np.asarray(lane.points, dtype=np.float64) * np.array([1366.0, 720.0])
        valid = np.isfinite(points).all(axis=1)
        points = points[valid]
        if len(points) < 2:
            continue
        cv2.polylines(image, [np.rint(points).astype(np.int32)], False, (0, 255, 0), 4)
        rendered += 1
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise IOError(f"failed to write {output_path}")
    return {
        "path": str(output_path),
        "sha256": sha256_file(output_path),
        "decoded_lanes": len(lanes),
        "rendered_lanes": rendered,
    }


def smoke_model(torch, cv2, LazyConfig, instantiate, collate, name, config_path, checkpoint_path, output_root):
    cfg = LazyConfig.load(str(config_path))
    disable_random_augmentation(cfg.dataloader.train.dataset)
    target_capacity = int(cfg.model.head.cfg.max_lanes)
    candidate_topk = int(cfg.model.head.cfg.test_parameters.nms_topk)
    if target_capacity != 8 or candidate_topk != 12:
        raise AssertionError(
            f"{name}: max_gt_lanes/candidate_topk={target_capacity}/{candidate_topk}"
        )
    if name == "clrnet_r50":
        if int(cfg.model.head.cfg.num_classes) != 9 or int(cfg.model.head.num_priors) != 192:
            raise AssertionError("CLRNet field mapping changed")
    elif int(cfg.model.head.start_points_num) != 300:
        raise AssertionError("ADNet field mapping changed")

    train_dataset = instantiate(cfg.dataloader.train.dataset)
    val_dataset = instantiate(cfg.dataloader.test.dataset)
    train_contract = assert_dataset_contract(train_dataset, "train")
    val_contract = assert_dataset_contract(val_dataset, "val")
    batch_size = int(cfg.dataloader.train.total_batch_size)
    if len(train_dataset) % batch_size or not bool(cfg.dataloader.train.drop_last):
        raise AssertionError(f"{name}: train sampler could drop a partial batch")
    train_loader = instantiate(cfg.dataloader.train)
    sampler = assert_sampler_coverage(
        train_loader,
        [idx for idx, info in enumerate(train_loader.dataset.data_infos) if info["is_empty"]],
        train_contract[0],
    )

    model = instantiate(cfg.model).to("cuda")
    state, checkpoint_metadata = load_adapted_state(torch, checkpoint_path)
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.unexpected_keys:
        raise AssertionError(f"{name}: adapted checkpoint has unexpected keys")
    expected_reinitialized = sorted(checkpoint_metadata["reinitialized_parameters"])
    if sorted(incompatible.missing_keys) != expected_reinitialized:
        raise AssertionError(f"{name}: adapted checkpoint missing-key evidence changed")

    empty_train_idx = train_contract[2]
    nonempty_train_idx = train_contract[3]
    empty_batch = collate_one(collate, train_dataset[empty_train_idx])
    nonempty_batch = collate_one(collate, train_dataset[nonempty_train_idx])
    empty_valid_targets = int((empty_batch["lane_line"][:, :, 1] == 1).sum().item())
    nonempty_valid_targets = int((nonempty_batch["lane_line"][:, :, 1] == 1).sum().item())
    if empty_valid_targets != 0 or nonempty_valid_targets < 1:
        raise AssertionError(
            f"{name}: transformed target counts empty/nonempty="
            f"{empty_valid_targets}/{nonempty_valid_targets}"
        )
    losses = {
        "empty": finite_backward(torch, model, empty_batch, f"{name}/empty"),
        "nonempty": finite_backward(torch, model, nonempty_batch, f"{name}/nonempty"),
    }

    demo_index = val_contract[3]
    demo_batch = collate_one(collate, val_dataset[demo_index])
    model.eval()
    with torch.no_grad():
        output = model(demo_batch)
        lanes = model.get_lanes(output)[0]
    if len(lanes) > candidate_topk:
        raise AssertionError(f"{name}: decoded {len(lanes)} lanes > candidate_topk")
    demo = render_demo(
        cv2,
        val_dataset.data_infos[demo_index],
        lanes,
        output_root / f"{name}_demo.jpg",
    )
    overlay = render_preprocess_overlay(
        cv2,
        val_dataset.data_infos[demo_index],
        cfg.model.head.cfg,
        output_root / f"{name}_preprocess_overlay.jpg",
    )
    result = {
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_metadata": checkpoint_metadata,
        "reinitialized_on_load": sorted(incompatible.missing_keys),
        "train": {"images": train_contract[0], "empty_gt": train_contract[1]},
        "val": {"images": val_contract[0], "empty_gt": val_contract[1]},
        "max_gt_lanes": target_capacity,
        "candidate_topk": candidate_topk,
        "train_sampler": sampler,
        "losses": losses,
        "demo": demo,
        "preprocess_overlay": overlay,
    }
    del model, state, train_dataset, val_dataset, train_loader
    del empty_batch, nonempty_batch, demo_batch
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    project_root = required_absolute_env("HARDLANE_PROJECT_ROOT")
    required_absolute_env("HARDLANE_DATA_ROOT")
    unlanedet_root = required_absolute_env("UNLANEDET_ROOT")
    weights_root = required_absolute_env("HARDLANE_WEIGHTS_ROOT")
    if git_head(unlanedet_root) != PINNED_UNLANEDET_COMMIT:
        raise SystemExit("UnLanedet HEAD does not match pinned commit")

    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(unlanedet_root))
    import cv2
    import torch
    from unlanedet.config import LazyConfig, instantiate
    from unlanedet.data.transform.collate import collate

    if not torch.cuda.is_available():
        raise SystemExit("this smoke must run on the AutoDL CUDA instance")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    output_root = Path(
        os.environ.get("HARDLANE_OUTPUT_ROOT", project_root / "outputs" / "autodl")
    ) / "smoke"
    specs = {
        "clrnet_r50": (
            project_root / "configs/unlanedet/clrnet_r50_hardlane.py",
            weights_root / "adapted_clrnet_r50_hardlane.pth",
        ),
        "adnet_r34": (
            project_root / "configs/unlanedet/adnet_r34_hardlane.py",
            weights_root / "adapted_adnet_r34_hardlane.pth",
        ),
    }
    models = {
        name: smoke_model(
            torch, cv2, LazyConfig, instantiate, collate, name, config, checkpoint, output_root
        )
        for name, (config, checkpoint) in specs.items()
    }
    report = {
        "status": "pass",
        "execution_environment": "AutoDL CUDA",
        "pinned_unlanedet_commit": PINNED_UNLANEDET_COMMIT,
        "project_git_head": git_head(project_root),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "models": models,
    }
    output = args.output or output_root / "dataloader_loss_smoke.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "output": str(output), "models": list(models)}))


if __name__ == "__main__":
    main()
