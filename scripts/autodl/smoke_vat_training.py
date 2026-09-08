#!/usr/bin/env python3
"""Smoke-test the VATCLRNet adversarial-consistency model on the CUDA instance.

Checks, on real train batches (empty-GT and nonempty-GT) and one val batch:
  1. train-mode forward emits ``loss_vat`` alongside the CLRNet losses,
     all finite, loss_vat strictly positive, total backward finite;
  2. the same holds under the AMP autocast the trainer actually uses;
  3. ``vat_weight=0`` degrades to the plain CLRNet loss dict (no loss_vat);
  4. eval-mode forward returns plain predictions without any loss keys;
  5. backward on ``loss_vat`` alone (fp32 structural probe) reaches nonzero
     finite gradients on backbone, neck AND head — guards against a detached
     consistency loss that would silently regularize only part of the model;
  6. the GradScaler-scaled summed-loss backward (real AMPTrainer path) also
     produces finite nonzero gradients after unscaling.

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
    if "loss_vat" not in losses:
        raise AssertionError(f"{label}: forward did not emit loss_vat")
    values = {}
    for name, value in losses.items():
        scalar = float(value.detach().float().cpu())
        if not np.isfinite(scalar):
            raise FloatingPointError(f"{label}: non-finite {name}={scalar}")
        values[name] = scalar
    if values["loss_vat"] <= 0.0:
        raise AssertionError(f"{label}: loss_vat must be positive, got {values['loss_vat']}")
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


def vat_only_backward(torch, model, batch, label: str) -> dict:
    """Backward ``loss_vat`` alone; it must reach backbone, neck and head.

    Runs in pure fp32: this is a structural graph-connectivity probe, and an
    unscaled AMP backward legitimately underflows such a small loss to exact
    zero in fp16 (the real trainer's GradScaler protects that path, which the
    following scaled check reproduces).
    """
    model.train()
    model.zero_grad(set_to_none=True)
    losses = model(batch)
    losses["loss_vat"].backward()
    groups = []
    modules = [("backbone", model.backbone), ("neck", model.neck), ("head", model.head)]
    if getattr(model, "aggregator", None) is not None:
        modules.append(("aggregator", model.aggregator))
    report = {"loss_vat": float(losses["loss_vat"].detach())}
    for group_name, module in modules:
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        if not grads:
            raise AssertionError(f"{label}: loss_vat reached no {group_name} parameters")
        if not all(bool(torch.isfinite(g).all().item()) for g in grads):
            raise FloatingPointError(f"{label}: non-finite {group_name} gradients from loss_vat")
        nonzero = sum(1 for g in grads if g.abs().sum().item() > 0.0)
        if nonzero == 0:
            raise AssertionError(
                f"{label}: loss_vat gradient is zero on every {group_name} parameter "
                "(the consistency loss got detached from the model graph)"
            )
        report[group_name] = {
            "with_grad": len(grads),
            "nonzero": nonzero,
            "abs_sum": sum(g.abs().sum().item() for g in grads),
        }
        groups.append(group_name)
    report["groups_reached"] = groups
    return report


def vat_scaled_amp_backward(torch, model, batch, label: str) -> dict:
    """Reproduce the real AMPTrainer path: GradScaler-scaled summed-loss backward."""
    model.train()
    model.zero_grad(set_to_none=True)
    scaler = torch.cuda.amp.GradScaler(init_scale=2.0 ** 16, growth_interval=10 ** 9)
    with torch.autocast("cuda"):
        losses = model(batch)
    total = sum(losses.values())
    scaler.scale(total).backward()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    scaler.unscale_(optimizer)
    report = {"scaled_loss_vat_share": float(
        losses["loss_vat"].detach() / total.detach())}
    modules = [("backbone", model.backbone), ("neck", model.neck), ("head", model.head)]
    for group_name, module in modules:
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        if not grads:
            raise AssertionError(f"{label}: scaled backward reached no {group_name} parameters")
        if not all(bool(torch.isfinite(g).all().item()) for g in grads):
            raise FloatingPointError(f"{label}: non-finite {group_name} gradients after unscale")
        nonzero = sum(1 for g in grads if g.abs().sum().item() > 0.0)
        if nonzero == 0:
            raise AssertionError(
                f"{label}: scaled AMP backward left every {group_name} gradient zero"
            )
        report[group_name] = {"nonzero": nonzero}
    return report


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

    config_path = project_root / "configs/unlanedet/clrnet_r50_hardlane_vat.py"
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
        "model": "clrnet_r50_vat",
        "project_git_head": git_head(project_root),
        "config": str(config_path),
        "init_checkpoint_expected": str(
            weights_root / "adapted_clrnet_r50_hardlane.pth"),
        "batches": {
            "empty_gt": int((empty_batch["lane_line"][:, :, 1] == 1).sum().item()),
            "nonempty_gt": int((nonempty_batch["lane_line"][:, :, 1] == 1).sum().item()),
        },
    }
    for label, batch in (("empty", empty_batch), ("nonempty", nonempty_batch)):
        report[f"{label}_fp32"] = finite_train_forward(
            torch, np, model, batch, f"{label}/fp32", autocast=False)
        report[f"{label}_amp"] = finite_train_forward(
            torch, np, model, batch, f"{label}/amp", autocast=True)
    report["vat_only_backward"] = vat_only_backward(
        torch, model, nonempty_batch, "nonempty/vat_only")
    report["vat_scaled_amp_backward"] = vat_scaled_amp_backward(
        torch, model, nonempty_batch, "nonempty/vat_scaled_amp")

    model.vat_weight = 0.0
    model.train()
    losses_off = model(nonempty_batch)
    if "loss_vat" in losses_off:
        raise AssertionError("vat_weight=0 must disable the loss_vat term")
    report["vat_disabled_loss_keys"] = sorted(losses_off.keys())
    model.vat_weight = float(cfg.model.vat_weight)

    model.eval()
    with torch.no_grad():
        predictions = model(val_batch)
    if isinstance(predictions, dict) and "predictions_lists" in predictions:
        shape = tuple(predictions["predictions_lists"][-1].shape)
    else:
        shape = tuple(predictions[-1].shape)
    report["eval_prediction_shape"] = list(shape)
    report["status"] = "pass"

    output = output_root / "smoke/vat_training_smoke.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
