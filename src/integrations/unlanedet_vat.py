"""Feature-space virtual adversarial training (VAT) for CLRNet on HardLane.

T4 "adversarial consistency" track (docs/top3_sprint_plan_20260908.md):
during training a small perturbation on the multi-scale backbone features is
refined for one power iteration so that it maximizes the CLRNet detection
loss, then the KL divergence between the clean and adversarially perturbed
per-anchor class distributions is emitted as ``loss_vat`` so the stock
SimpleTrainer/AMPTrainer picks it up via ``sum(loss_dict.values())``.

VATCLRNet adds no parameters or buffers on purpose: its state_dict layout is
identical to CLRNet, so checkpoints are interchangeable with the plain
baseline and existing inference/export tooling keeps working unchanged.

The consistency loss is computed on features that stay attached to the model
graph (the perturbation direction is a detached constant), so its gradient
reaches backbone, neck and head — not just the head.  smoke_vat_training.py
guards this property explicitly.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from unlanedet.model.CLRNet import CLRNet


class VATCLRNet(CLRNet):
    """CLRNet plus a feature-space virtual adversarial consistency loss.

    vat_weight  : multiplier on the emitted ``loss_vat`` (0 disables VAT).
    vat_eps     : adversarial perturbation size, relative to each feature
                  level's standard deviation.
    vat_xi      : perturbation size used while probing for the adversarial
                  direction (the power-iteration step size).
    vat_power_iters : number of power iterations refining the direction.
    """

    def __init__(self, backbone=None, aggregator=None, neck=None, head=None,
                 vat_weight=1.0, vat_eps=0.05, vat_xi=0.01, vat_power_iters=1):
        super().__init__(backbone=backbone, aggregator=aggregator,
                         neck=neck, head=head)
        self.vat_weight = float(vat_weight)
        self.vat_eps = float(vat_eps)
        self.vat_xi = float(vat_xi)
        self.vat_power_iters = int(vat_power_iters)

    def forward(self, batch):
        if not self.training or self.vat_weight <= 0.0:
            return super().forward(batch)
        batch = self.to_cuda(batch)
        fea = self.backbone(batch["img"])
        if self.aggregator:
            fea[-1] = self.aggregator(fea[-1])
        if self.neck:
            fea = self.neck(fea)
        out = self.head(fea, batch=batch)
        output = self.head.loss(out, batch)
        output["loss_vat"] = self.vat_weight * self._vat_consistency(fea, batch, out)
        return output

    def _perturb(self, fea, direction, scale):
        """Return fea perturbed by ``scale`` feature-stds along ``direction``."""
        perturbed = []
        for feat, direction_level in zip(fea, direction):
            std = feat.detach().float().std()
            normalized = direction_level.float() / (
                direction_level.float().std() + 1e-12)
            perturbed.append(feat + (scale * std).to(feat.dtype) * normalized.to(feat.dtype))
        return perturbed

    def _final_class_probs(self, out):
        # fp32 + clamp: under AMP the softmax can underflow to exact 0 in fp16
        # and F.kl_div would then compute 0 * log(0) = NaN on the target side.
        logits = out["predictions_lists"][-1][..., :2]
        return F.softmax(logits.float(), dim=-1).clamp_min(1e-8)

    def _kl_to(self, p_hat, out_adv):
        logits = out_adv["predictions_lists"][-1][..., :2]
        log_p = F.log_softmax(logits.float(), dim=-1)
        kl = F.kl_div(log_p, p_hat, reduction="none").sum(dim=(1, 2))
        return kl.mean()

    def _vat_consistency(self, fea, batch, out):
        with torch.no_grad():
            p_hat = self._final_class_probs(out)
        direction = [torch.randn_like(feat) for feat in fea]
        for _ in range(self.vat_power_iters):
            direction = [d.detach().requires_grad_(True) for d in direction]
            with torch.enable_grad():
                perturbed = self._perturb(fea, direction, self.vat_xi)
                out_d = self.head(perturbed, batch=batch)
                loss_d = sum(self.head.loss(out_d, batch).values())
                grads = torch.autograd.grad(loss_d, direction)
            direction = [g.detach() for g in grads]
        # fea stays attached so the consistency gradient reaches the whole
        # model; the detached direction keeps the perturbation a constant
        # offset, which is exactly VAT semantics.
        perturbed = self._perturb(fea, direction, self.vat_eps)
        out_adv = self.head(perturbed, batch=batch)
        return self._kl_to(p_hat, out_adv)
