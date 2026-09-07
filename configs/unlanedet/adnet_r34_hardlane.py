"""AutoDL ADNet-R34 baseline for the frozen HardLane clip split."""
import os
import sys
from pathlib import Path

project_root_env = os.environ.get("HARDLANE_PROJECT_ROOT")
if project_root_env:
    sys.path.insert(0, project_root_env)

import torch
from fvcore.common.param_scheduler import CosineParamScheduler
from omegaconf import OmegaConf

from unlanedet.config import LazyCall as L
from unlanedet.data.build import build_batch_data_loader
from unlanedet.data.transform import CollectHm, GenerateLanePts, ToTensor
from unlanedet.model import Detector, ResNetWrapper
from unlanedet.model.ADNet import SA_FPN, SPGHead
from unlanedet.solver.build import get_default_optimizer_params

def _required_env(name):
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} must be set to an absolute AutoDL path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise RuntimeError(f"{name} must be absolute, got {value!r}")
    return path


project_root = _required_env("HARDLANE_PROJECT_ROOT")
data_root = _required_env("HARDLANE_DATA_ROOT")
weights_root = _required_env("HARDLANE_WEIGHTS_ROOT")
output_root = Path(os.environ.get("HARDLANE_OUTPUT_ROOT", project_root / "outputs" / "autodl"))
from src.integrations.unlanedet_hardlane import HardLaneDataset, HardLaneEvaluator

ori_img_w = 1366
ori_img_h = 720
img_w = 800
img_h = 320
cut_height = 180
num_points = 72
max_gt_lanes = 8
candidate_topk = 12
anchors_num = 300
sample_y = list(range(719, 179, -5)) + [180]
batch_size = 12
train_images = 6300
epochs = 15
iterations_per_epoch = train_images // batch_size

param_config = OmegaConf.create(
    {
        "in_channels": [128, 256, 512],
        "anchor_feat_channels": 64,
        "num_points": num_points,
        "img_w": img_w,
        "img_h": img_h,
        "fpn_down_scale": [8, 16, 32],
        "anchors_num": anchors_num,
        "regw": 6,
        "hmw": 2,
        "thetalossw": 3,
        "cls_loss_w": 6,
        "dynamic_after": 10,
        "do_mask": False,
        "max_lanes": max_gt_lanes,
        "train_parameters": {
            "conf_threshold": None,
            "nms_thres": 45.0,
            "nms_topk": candidate_topk,
        },
        "test_parameters": {
            "conf_threshold": 0.3,
            "nms_thres": 45.0,
            "nms_topk": candidate_topk,
        },
        "sample_y": sample_y,
        "ori_img_w": ori_img_w,
        "ori_img_h": ori_img_h,
        "cut_height": cut_height,
        "hm_down_scale": 8,
        "dataset_type": "HardLane",
    }
)

model = L(Detector)(
    backbone=L(ResNetWrapper)(
        resnet="resnet34",
        pretrained=False,
        replace_stride_with_dilation=[False, False, False],
        out_conv=False,
    ),
    neck=L(SA_FPN)(
        in_channels=[128, 256, 512],
        out_channels=64,
        num_outs=3,
    ),
    head=L(SPGHead)(
        S=num_points,
        anchor_feat_channels=64,
        img_width=img_w,
        img_height=img_h,
        start_points_num=anchors_num,
        cfg=param_config,
    ),
)

augmentation = [
    dict(name="Resize", parameters=dict(size=dict(height=img_h, width=img_w)), p=1.0),
    dict(name="HorizontalFlip", parameters=dict(p=1.0), p=0.5),
    dict(name="ChannelShuffle", parameters=dict(p=1.0), p=0.1),
    dict(
        name="MultiplyAndAddToBrightness",
        parameters=dict(mul=(0.85, 1.15), add=(-10, 10)),
        p=0.6,
    ),
    dict(name="AddToHueAndSaturation", parameters=dict(value=(-10, 10)), p=0.7),
    dict(
        name="OneOf",
        transforms=[
            dict(name="MotionBlur", parameters=dict(k=(3, 5))),
            dict(name="MedianBlur", parameters=dict(k=(3, 5))),
        ],
        p=0.2,
    ),
    dict(
        name="Affine",
        parameters=dict(
            translate_percent=dict(x=(-0.1, 0.1), y=(-0.1, 0.1)),
            rotate=(-10, 10),
            scale=(0.8, 1.2),
        ),
        p=0.7,
    ),
    dict(name="Resize", parameters=dict(size=dict(height=img_h, width=img_w)), p=1.0),
]
train_keys = ["img", "lane_line", "gt_hm", "shape_hm", "shape_hm_mask"]
train_process = [
    L(GenerateLanePts)(transforms=augmentation, cfg=param_config),
    L(CollectHm)(
        down_scale=8,
        hm_down_scale=8,
        max_mask_sample=5,
        line_width=3,
        radius=12,
        theta_thr=0.5,
        keys=train_keys,
        meta_keys=["gt_points"],
        cfg=param_config,
    ),
    L(ToTensor)(keys=train_keys),
]
val_process = [
    L(GenerateLanePts)(
        training=False,
        transforms=[
            dict(name="Resize", parameters=dict(size=dict(height=img_h, width=img_w)), p=1.0)
        ],
        cfg=param_config,
    ),
    L(ToTensor)(keys=["img", "lane_line"]),
]

dataloader = OmegaConf.create()
dataloader.train = L(build_batch_data_loader)(
    dataset=L(HardLaneDataset)(
        data_root=str(data_root),
        manifest_path=str(project_root / "data/processed/manifest_train_v1_seed42.jsonl"),
        split="train",
        cut_height=cut_height,
        processes=train_process,
        cfg=param_config,
    ),
    total_batch_size=batch_size,
    num_workers=4,
    drop_last=True,
    shuffle=True,
    pin_memory=True,
    persistent_workers=True,
    seed=42,
)
dataloader.test = L(build_batch_data_loader)(
    dataset=L(HardLaneDataset)(
        data_root=str(data_root),
        manifest_path=str(project_root / "data/processed/manifest_val_v1_seed42.jsonl"),
        split="val",
        cut_height=cut_height,
        processes=val_process,
        cfg=param_config,
    ),
    total_batch_size=batch_size,
    num_workers=4,
    drop_last=False,
    shuffle=False,
    pin_memory=True,
    persistent_workers=True,
    seed=42,
)
dataloader.evaluator = L(HardLaneEvaluator)(
    output_basedir=str(output_root / "adnet_r34" / "val"),
    cfg=param_config,
    metric="F1",
)

optimizer = L(torch.optim.AdamW)(
    params=L(get_default_optimizer_params)(
        base_lr="${..lr}",
        weight_decay_norm=0.0,
    ),
    lr=0.0007,
    betas=(0.9, 0.999),
    weight_decay=0.01,
)
lr_multiplier = L(CosineParamScheduler)(start_value=1.0, end_value=0.001)

train = OmegaConf.create(
    {
        "output_dir": str(output_root / "adnet_r34"),
        "init_checkpoint": str(weights_root / "adapted_adnet_r34_hardlane.pth"),
        "max_iter": iterations_per_epoch * epochs,
        "amp": {"enabled": True},
        "ddp": {
            "broadcast_buffers": False,
            "find_unused_parameters": False,
            "fp16_compression": False,
        },
        "checkpointer": {"period": iterations_per_epoch, "max_to_keep": 40},
        "eval_period": iterations_per_epoch,
        "log_period": 20,
        "device": "cuda",
        "seed": 42,
        "cudnn_benchmark": False,
    }
)
