from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class UnderwaterAwareBatchNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-5, momentum: float = 0.1,
                 alpha_phys=(0.80, 0.04, 0.01)):
        super().__init__()
        self.eps = eps
        self.bn = nn.BatchNorm2d(channels, eps=eps, momentum=momentum)
        prior = torch.tensor(alpha_phys, dtype=torch.float32)
        groups = torch.arange(channels) * 3 // channels
        alpha0 = prior[groups].clamp_min(1e-6)
        self.alpha_raw = nn.Parameter(torch.log(torch.expm1(alpha0)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        bands = torch.chunk(x.abs().mean(dim=(2, 3)), 3, dim=1)
        red_like = bands[0].mean(dim=1, keepdim=True)
        blue_like = bands[-1].mean(dim=1, keepdim=True)
        depth_proxy = torch.sigmoid(1.0 - red_like / (blue_like + self.eps))
        alpha = F.softplus(self.alpha_raw).view(1, c, 1, 1)
        x_comp = x * torch.exp(alpha * depth_proxy.view(b, 1, 1, 1))
        return self.bn(x_comp)
