"""CLRerNet head (LaneIoU + SimOTA-style dynamic assign) for HardLane.

Pinned UnLanedet's CLRerNet carries CULane geometry in three places that do
not match our 800x320 HardLane input (all default to img_w=1640):

* ``unlanedet.model.CLRerNet.assign`` builds two module-level
  ``LaneIoUCost`` singletons that ``assign()`` resolves from module globals
  at call time;
* ``CLRerHead.__init__`` constructs ``self.iou_loss`` with the CULane
  defaults (the comment in assign.py says other datasets must change these).

``HardCLRerHead`` rebuilds the loss with HardLane geometry and rebinds the
assign singletons, so every process that builds the model (trainer,
eval-only recovery) gets consistent geometry without touching the pinned
repo.  The parameter layout is identical to CLRHead (the replaced modules
hold no parameters), so checkpoints stay interchangeable with the CLRNet
baseline and the adapted pretrained checkpoint loads unchanged.
"""
from __future__ import annotations

import unlanedet.model.CLRerNet.assign as clrernet_assign
from unlanedet.model.CLRerNet import CLRerHead
from unlanedet.model.CLRerNet.assign import LaneIoUCost
from unlanedet.model.CLRerNet.lane_iou import LaneIoULoss


def patch_assign_geometry(img_w: float, img_h: float) -> None:
    """Rebind the assign-module LaneIoUCost singletons to our geometry.

    ``assign()`` looks ``lane_iou_dynamic``/``lane_iou_cost`` up in module
    globals at call time and nothing else imports them by value, so the
    rebinding applies to every later ``assign()`` call in this process.
    """
    clrernet_assign.lane_iou_dynamic = LaneIoUCost(
        use_pred_start_end=False,
        use_giou=True,
        img_h=img_h,
        img_w=img_w,
    )
    clrernet_assign.lane_iou_cost = LaneIoUCost(
        lane_width=30.0 / img_w,
        use_pred_start_end=True,
        use_giou=True,
        img_h=img_h,
        img_w=img_w,
    )


class HardCLRerHead(CLRerHead):
    """CLRerHead with HardLane geometry injected into every IoU consumer."""

    def __init__(
        self,
        num_points=72,
        prior_feat_channels=64,
        fc_hidden_dim=64,
        num_priors=192,
        num_fc=2,
        refine_layers=3,
        sample_points=36,
        cfg=None,
    ):
        super().__init__(
            num_points=num_points,
            prior_feat_channels=prior_feat_channels,
            fc_hidden_dim=fc_hidden_dim,
            num_priors=num_priors,
            num_fc=num_fc,
            refine_layers=refine_layers,
            sample_points=sample_points,
            cfg=cfg,
        )
        # CLRerHead's own construction uses the CULane defaults (img_w=1640);
        # rebuild with the geometry this config actually trains at.  The
        # official lane_width is expressed as a fraction of img_w (7.5/800),
        # so dividing by cfg.img_w keeps the pixel width at 7.5.
        self.iou_loss = LaneIoULoss(
            loss_weight=cfg.iou_loss_weight,
            lane_width=7.5 / cfg.img_w,
            img_h=cfg.img_h,
            img_w=cfg.img_w,
        )
        patch_assign_geometry(cfg.img_w, cfg.img_h)
