from __future__ import annotations

import math


def cosine_warmup(step: int, total_steps: int, start: float, end: float) -> float:
    if total_steps <= 0:
        return end
    t = min(max(step / total_steps, 0.0), 1.0)
    return start + (end - start) * 0.5 * (1.0 - math.cos(math.pi * t))


def progressive_weight(step: int, total_steps: int, start: float, end: float) -> float:
    return cosine_warmup(step, total_steps, start, end)


def grl_schedule(step: int, total_steps: int, lambda_max: float = 1.0, gamma: float = 10.0) -> float:
    if total_steps <= 0:
        return lambda_max
    p = min(max(step / total_steps, 0.0), 1.0)
    return lambda_max * (2.0 / (1.0 + math.exp(-gamma * p)) - 1.0)
