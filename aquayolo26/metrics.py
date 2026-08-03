from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def box_iou_xyxy(box1: np.ndarray, box2: np.ndarray) -> np.ndarray:
    box1 = np.asarray(box1, dtype=float)
    box2 = np.asarray(box2, dtype=float)
    x1 = np.maximum(box1[..., 0], box2[..., 0])
    y1 = np.maximum(box1[..., 1], box2[..., 1])
    x2 = np.minimum(box1[..., 2], box2[..., 2])
    y2 = np.minimum(box1[..., 3], box2[..., 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area1 = np.maximum(0.0, box1[..., 2] - box1[..., 0]) * np.maximum(0.0, box1[..., 3] - box1[..., 1])
    area2 = np.maximum(0.0, box2[..., 2] - box2[..., 0]) * np.maximum(0.0, box2[..., 3] - box2[..., 1])
    union = area1 + area2 - inter + 1e-9
    return inter / union


def mean_average_precision(scores: list[float]) -> float:
    return float(np.mean(scores)) if scores else 0.0


@dataclass
class MetricSummary:
    map50: float = 0.0
    map5095: float = 0.0
    small_map: float = 0.0
    recall: float = 0.0
    fps: float = 0.0


def format_metric_summary(summary: MetricSummary) -> str:
    return (
        f"mAP@0.5={summary.map50:.3f}, "
        f"mAP@0.5:0.95={summary.map5095:.3f}, "
        f"mAPS={summary.small_map:.3f}, "
        f"Recall={summary.recall:.3f}, "
        f"FPS={summary.fps:.2f}"
    )
