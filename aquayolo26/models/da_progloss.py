from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        ctx.weight = weight
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.weight * grad_output, None


def grl(x, weight):
    return GradientReverse.apply(x, weight)


def da_lambda(step: int, total_steps: int, lambda_max: float = 1.0, gamma: float = 10.0) -> float:
    p = min(max(float(step) / max(total_steps, 1), 0.0), 1.0)
    return lambda_max * (2.0 / (1.0 + math.exp(-gamma * p)) - 1.0)


class DomainClassifier(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, feature, lam):
        return self.net(grl(feature, lam)).squeeze(1)


def domain_loss(logits, domain_labels):
    return F.binary_cross_entropy_with_logits(logits, domain_labels.float())
