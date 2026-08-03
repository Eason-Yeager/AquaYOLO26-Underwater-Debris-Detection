from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .da_progloss import DomainClassifier
from .turb_stal import TurbidityEstimator
from .uwbn import UnderwaterAwareBatchNorm2d


@dataclass
class AquaYOLO26Outputs:
    detections: torch.Tensor
    turbidity: torch.Tensor | None = None
    domain_logits: torch.Tensor | None = None


class TinyBackbone(nn.Module):
    def __init__(self, in_ch=3, ch=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, ch, 3, 2, 1),
            nn.SiLU(),
            UnderwaterAwareBatchNorm2d(ch),
            nn.Conv2d(ch, ch * 2, 3, 2, 1),
            nn.SiLU(),
            nn.BatchNorm2d(ch * 2),
            nn.Conv2d(ch * 2, ch * 4, 3, 2, 1),
            nn.SiLU(),
            nn.BatchNorm2d(ch * 4),
        )

    def forward(self, x):
        return self.net(x)


class AquaYOLO26(nn.Module):
    def __init__(self, num_classes=5, with_turbidity=True, with_domain=True):
        super().__init__()
        self.backbone = TinyBackbone()
        self.neck = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.SiLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.SiLU(),
        )
        self.head = nn.Conv2d(128, num_classes + 5, 1)
        self.turbidity = TurbidityEstimator(256) if with_turbidity else None
        self.domain = DomainClassifier(128) if with_domain else None

    def forward(self, x, domain_lambda: float = 0.0):
        feat = self.backbone(x)
        neck = self.neck(feat)
        det = self.head(neck)
        turb = self.turbidity(feat) if self.turbidity is not None else None
        dom = self.domain(neck, domain_lambda) if self.domain is not None else None
        return AquaYOLO26Outputs(detections=det, turbidity=turb, domain_logits=dom)
