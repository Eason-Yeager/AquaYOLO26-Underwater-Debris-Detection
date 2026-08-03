from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .metrics import box_iou_xyxy


def box_loss(pred_xyxy: torch.Tensor, target_xyxy: torch.Tensor) -> torch.Tensor:
    iou = box_iou_xyxy(pred_xyxy.detach().cpu().numpy(), target_xyxy.detach().cpu().numpy())
    return torch.tensor(1.0 - float(iou.mean()), device=pred_xyxy.device)


def cls_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, targets.float())


def obj_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, targets.float())


@dataclass
class LossWeights:
    box: float = 7.5
    cls: float = 0.5
    obj: float = 1.5
    turbidity: float = 0.01
    domain: float = 1.0


def total_loss(box: torch.Tensor, cls: torch.Tensor, obj: torch.Tensor, turb: torch.Tensor | None = None, dom: torch.Tensor | None = None, weights: LossWeights | None = None) -> torch.Tensor:
    w = weights or LossWeights()
    out = w.box * box + w.cls * cls + w.obj * obj
    if turb is not None:
        out = out + w.turbidity * turb
    if dom is not None:
        out = out + w.domain * dom
    return out
