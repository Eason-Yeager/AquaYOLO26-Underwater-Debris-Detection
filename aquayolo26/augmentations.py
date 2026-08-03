from __future__ import annotations

import cv2
import numpy as np
import torch


def resize_letterbox(image: np.ndarray, size: int = 640, color=(114, 114, 114)) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), color, dtype=np.uint8)
    top = (size - nh) // 2
    left = (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas


def to_tensor(image: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(image).permute(2, 0, 1).float() / 255.0


def basic_train_transform(image: np.ndarray, size: int = 640) -> torch.Tensor:
    return to_tensor(resize_letterbox(image, size=size))
