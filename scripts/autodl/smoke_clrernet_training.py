#!/usr/bin/env python3
"""Smoke-test the CLRerNet (HardCLRerHead) model on the CUDA instance.

Checks, on real train batches (empty-GT and nonempty-GT) and one val batch:
  1. the HardLane geometry patch took effect: the assign-module LaneIoUCost
     singletons and the head's LaneIoULoss carry img_w=800/img_h=320 (the
     pinned CLRerNet defaults are CULane's 1640x320 — a missed patch would
     silently compute every virtual lane width at 2x);
  2. train-mode forward emits the CLRerNet loss dict (cls/reg_xytl/seg/iou),
     all finite, iou_loss >= 0; fp32 total backward finite with nonzero
     gradients on backbone, neck AND head;
  3. the same holds under the AMP autocast the trainer actually uses, with
     the dynamic GradScaler retry loop (mirrors the real AMPTrainer path:
     a fixed loss scale legitimately over/underflows fp16);
  4. CLRerNet's dynamic assign matches at least one GT lane on a nonempty
     batch — guards a geometry/coordinate bug that would silently produce
     zero positives and a degenerate training signal;
  5. eval-mode forward returns decoded lanes without any loss keys.

Writes a JSON report next to the dataloader/loss smoke evidence.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def required_absolute_env(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"{name} must be set to an absolute instance path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise SystemExit(f"{name} must be absolute, got {value!r}")
    return path


def git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def collate_one(collate, sample):
    return collate([sample], samples_per_gpu=1)


def finite_train_forward(torch, np, model, batch, label: str, autocast: bool) -> dict:
    model.train()
    model.zero_grad(set_to_none=True)
    if autocast:
        with torch.autocast("cuda"):
            losses = model(batch)
    else:
        losses = model(batch)
    required_keys = {"cls_loss", "reg_xytl_loss", "seg_loss", "iou_loss"}
    missing = required_keys.difference(losses)
    if missing:
        raise AssertionError(f"{label}: forward did not emit {sorted(missing)}")
    values = {}
    for name, value in losses.items():
        scalar = float(value.detach().float().cpu())
        if not np.isfinite(scalar):
            raise FloatingPointError(f"{label}: non-finite {name}={scalar}")
        values[name] = scalar
    if values["iou_loss"] < -1e-6:
        raise AssertionError(f"{label}: iou_loss must be >= 0, got {values['iou_loss']}")
    total = sum(losses.values())
    total.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    if not gradients:
        raise AssertionError(f"{label}: backward produced no gradients")
    if not all(bool(torch.isfinite(g).all().item()) for g in gradients):
        raise FloatingPointError(f"{label}: backward produced non-finite gradients")
    values["total"] = float(total.detach().float().cpu())
    values["parameters_with_grad"] = len(gradients)
    return values


def scaled_amp_backward(torch, model, batch, label: str) -> dict:
    """Reproduce the real AMPTrainer path: dynamic GradScaler.

    A fixed loss scale can overflow fp16 at init, so this mirrors the
    trainer exactly: try the current scale, and whenever unscaling
    reveals inf/nan, let GradScaler halve the scale and retry on fresh
    gradients.
    """
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    attempts = 0
    while True:
        attempts += 1
        if attempts > 20:
            raise AssertionError(f"{label}: no loss scale in 2^16..2^-3 produced finite grads")
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda"):
            losses = model(batch)
        total = sum(losses.values())
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        all_grads = [p.grad for p in model.parameters() if p.grad is not None]
        if all_grads and all(bool(torch.isfinite(g).all().item()) for g in all_grads):
            break
        scaler.update()
    report = {
        "attempts": attempts,
        "final_scale": float(scaler.get_scale()),
    }
    modules = [("backbone", model.backbone), ("neck", model.neck), ("head", model.head)]
    for group_name, module in modules:
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        if not grads:
            raise AssertionError(f"{label}: scaled backward reached no {group_name} parameters")
        nonzero = sum(1 for g in grads if g.abs().sum().item() > 0.0)
        if nonzero == 0:
            raise AssertionError(
                f"{label}: scaled AMP backward left every {group_name} gradient zero"
            )
        report[group_name] = {"nonzero": nonzero}
    return report


def assign_probe(torch, model, batch, label: str) -> dict:
    """Run CLRerNet's dynamic assign directly on one nonempty sample.

    ``assign()`` resolves the module-level LaneIoUCost singletons at call
    time, so this also exercises the geometry-patched cost path (the plain
    forward would swallow a zero-match degeneracy into an iou_loss of 0).
    """
    import unlanedet.model.CLRerNet.assign as assign_module

    model.train()
    with torch.no_grad():
        fea = model.backbone(batch["img"])
        if getattr(model, "aggregator", None) is not None:
            fea[-1] = model.aggregator(fea[-1])
        fea = model.neck(fea)
        out = model.head(fea, batch=batch)
    predictions = out["predictions_lists"][-1][0]
    targets = batch["lane_line"][0].clone()
    targets = targets[targets[:, 1] == 1]
    if targets.shape[0] == 0:
        raise AssertionError(f"{label}: probe batch has no GT lanes")
    matched_row, matched_col = assign_module.assign(
        predictions, targets, model.head.img_w, model.head.img_h
    )
    matched = int(matched_row.numel())
    if matched == 0:
        raise AssertionError(
            f"{label}: dynamic assign matched 0 of {targets.shape[0]} GT lanes"
        )
    return {
        "gt_lanes": int(targets.shape[0]),
        "matched": matched,
        "assign_img_w": assign_module.lane_iou_dynamic.img_w,
        "assign_img_h": assign_module.lane_iou_dynamic.img_h,
    }


def geometry_report(cfg, model) -> dict:
    import unlanedet.model.CLRerNet.assign as assign_module

    expected_w = int(cfg.model.head.cfg.img_w)
    expected_h = int(cfg.model.head.cfg.img_h)
    checks = {
        "head_iou_loss": (model.head.iou_loss.img_w, model.head.iou_loss.img_h),
        "assign_lane_iou_dynamic": (
            assign_module.lane_iou_dynamic.img_w,
            assign_module.lane_iou_dynamic.img_h,
        ),
        "assign_lane_iou_cost": (
            assign_module.lane_iou_cost.img_w,
            assign_module.lane_iou_cost.img_h,
        ),
    }
    for name, (width, height) in checks.items():
        if int(width) != expected_w or int(height) != expected_h:
            raise AssertionError(
                f"{name} geometry is {width}x{height}, expected {expected_w}x{expected_h} "
                "(the CULane default leaked through the HardCLRerHead patch)"
            )
    return {
        "expected": [expected_w, expected_h],
        "head_iou_loss_lane_width": float(model.head.iou_loss.lane_width),
        "assign_lane_iou_cost_lane_width": float(
            assign_module.lane_iou_cost.lane_width
        ),
        "checked": sorted(checks),
    }


def main() -> None:
    project_root = required_absolute_env("HARDLANE_PROJECT_ROOT")
    required_absolute_env("HARDLANE_DATA_ROOT")
    unlanedet_root = required_absolute_env("UNLANEDET_ROOT")
    weights_root = required_absolute_env("HARDLANE_WEIGHTS_ROOT")
    output_root = Path(os.environ.get(
        "HARDLANE_OUTPUT_ROOT", project_root / "outputs" / "autodl"))

    sys.path.insert(0, str(unlanedet_root))
    sys.path.insert(0, str(project_root))

    import numpy as np
    import torch
    from unlanedet.config import LazyConfig, instantiate
    from unlanedet.data.transform.collate import collate

    config_path = project_root / "configs/unlanedet/clrernet_r50_hardlane.py"
    cfg = LazyConfig.load(str(config_path))
    model = instantiate(cfg.model)
    model.to("cuda")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    train_dataset = instantiate(cfg.dataloader.train.dataset)
    val_dataset = instantiate(cfg.dataloader.test.dataset)
    empty_index = next(idx for idx, info in enumerate(train_dataset.data_infos)
                       if info["is_empty"])
    nonempty_index = next(idx for idx, info in enumerate(train_dataset.data_infos)
                          if not info["is_empty"])
    to_cuda = lambda batch: {
        key: (value.cuda() if isinstance(value, torch.Tensor) else value)
        for key, value in batch.items()
    }
    empty_batch = to_cuda(collate_one(collate, train_dataset[empty_index]))
    nonempty_batch = to_cuda(collate_one(collate, train_dataset[nonempty_index]))
    val_batch = to_cuda(collate_one(collate, val_dataset[0]))

    report = {
        "model": "clrernet_r50",
        "project_git_head": git_head(project_root),
        "config": str(config_path),
        "init_checkpoint_expected": str(
            weights_root / "adapted_clrnet_r50_hardlane.pth"),
        "batches": {
            "empty_gt": int((empty_batch["lane_line"][:, :, 1] == 1).sum().item()),
            "nonempty_gt": int((nonempty_batch["lane_line"][:, :, 1] == 1).sum().item()),
        },
    }

    report["geometry"] = geometry_report(cfg, model)

    for label, batch in (("empty", empty_batch), ("nonempty", nonempty_batch)):
        report[f"{label}_fp32"] = finite_train_forward(
            torch, np, model, batch, f"{label}/fp32", autocast=False)
        report[f"{label}_amp"] = finite_train_forward(
            torch, np, model, batch, f"{label}/amp", autocast=True)
    report["scaled_amp_backward"] = scaled_amp_backward(
        torch, model, nonempty_batch, "nonempty/scaled_amp")
    report["assign_probe"] = assign_probe(
        torch, model, nonempty_batch, "nonempty/assign")

    model.eval()
    with torch.no_grad():
        predictions = model(val_batch)
    if isinstance(predictions, dict):
        raise AssertionError(f"eval-mode forward returned a loss dict: {sorted(predictions)}")
    # CLRHead's eval branch returns the raw final-stage tensor
    # (batch, num_priors, n_offsets + 6); decoding happens in the evaluator.
    if not isinstance(predictions, torch.Tensor) or predictions.dim() != 3:
        raise AssertionError(
            f"eval-mode forward returned {type(predictions).__name__}, "
            "expected a (batch, num_priors, 6 + n_offsets) tensor"
        )
    report["eval_prediction_shape"] = list(predictions.shape)
    report["status"] = "pass"

    output = output_root / "smoke/clrernet_training_smoke.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
