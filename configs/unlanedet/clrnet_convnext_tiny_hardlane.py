"""AutoDL ConvNeXt-Tiny 15-epoch screen on the frozen HardLane split.

The data, augmentation, head, and evaluation contract are inherited from the
production CLRNet-R50 config. Only the backbone, its FPN channel contract,
and the checkpoint/output paths differ. This keeps the screen a backbone
comparison instead of silently changing preprocessing or scoring.
"""
import os
from pathlib import Path

import torch
from fvcore.common.param_scheduler import CosineParamScheduler
from omegaconf import OmegaConf

from unlanedet.config import LazyCall as L
from unlanedet.model import ConvNeXt, FPN
from unlanedet.model.CLRNet import CLRHead, CLRNet
from unlanedet.solver.build import get_default_optimizer_params

from .clrnet_r50_hardlane import dataloader, param_config


def _required_env(name):
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} must be set to an absolute AutoDL path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise RuntimeError(f"{name} must be absolute, got {value!r}")
    return path


project_root = _required_env("HARDLANE_PROJECT_ROOT")
weights_root = _required_env("HARDLANE_WEIGHTS_ROOT")
output_root = Path(
    os.environ.get("HARDLANE_OUTPUT_ROOT", project_root / "outputs" / "autodl")
)

num_priors = 192
batch_size = 12
train_images = 6300
epochs = 15
iterations_per_epoch = train_images // batch_size

model = L(CLRNet)(
    backbone=L(ConvNeXt)(
        in_chans=3,
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        out_indices=[0, 1, 2, 3],
    ),
    neck=L(FPN)(
        in_channels=[192, 384, 768],
        out_channels=64,
        num_outs=3,
        attention=False,
    ),
    head=L(CLRHead)(
        num_priors=num_priors,
        refine_layers=3,
        fc_hidden_dim=64,
        sample_points=36,
        cfg=param_config,
    ),
)

# Keep the production preprocessing and evaluation split identical.
dataloader.train.total_batch_size = batch_size
dataloader.test.total_batch_size = batch_size
dataloader.evaluator.output_basedir = str(
    output_root / "runs/convnext_tiny_15ep/val"
)

optimizer = L(torch.optim.AdamW)(
    params=L(get_default_optimizer_params)(
        base_lr="${..lr}",
        weight_decay_norm=0.0,
    ),
    lr=0.0003,
    betas=(0.9, 0.999),
    weight_decay=0.01,
)
lr_multiplier = L(CosineParamScheduler)(start_value=1.0, end_value=0.001)

train = OmegaConf.create(
    {
        "output_dir": str(output_root / "runs/convnext_tiny_15ep"),
        "init_checkpoint": str(
            weights_root / "adapted_clrnet_convnext_tiny_hardlane.pth"
        ),
        "max_iter": iterations_per_epoch * epochs,
        "amp": {"enabled": True},
        "ddp": {
            "broadcast_buffers": False,
            "find_unused_parameters": False,
            "fp16_compression": False,
        },
        "checkpointer": {"period": iterations_per_epoch, "max_to_keep": 3},
        "eval_period": iterations_per_epoch,
        "log_period": 20,
        "device": "cuda",
        "seed": 42,
        "cudnn_benchmark": False,
    }
)
