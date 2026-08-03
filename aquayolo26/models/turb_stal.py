from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TurbidityEstimator(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.net(feature).squeeze(1)


def turbidity_proxy_loss(tau: torch.Tensor, images: torch.Tensor, beta_proxy: float = 5.0, eps: float = 1e-6) -> torch.Tensor:
    red = images[:, 0].mean(dim=(1, 2))
    blue = images[:, 2].mean(dim=(1, 2))
    proxy = torch.sigmoid(beta_proxy * (1.0 - red / (blue + eps)))
    return F.mse_loss(tau, proxy.detach())


def apply_turb_stal(
    stal_score: torch.Tensor,
    gt_area: torch.Tensor,
    tau: torch.Tensor,
    k0: int = 10,
    beta_k: float = 1.0,
    lambda_tau: float = 0.5,
    small_area_threshold: float = 0.01,
):
    small = (gt_area < small_area_threshold).float()
    weight = 1.0 + lambda_tau * tau[:, None] * small[None, :]
    score = stal_score * weight
    k = torch.clamp((k0 * (1.0 + beta_k * tau)).round().long(), min=k0, max=int(k0 * (1.0 + beta_k)))
    return score, k
