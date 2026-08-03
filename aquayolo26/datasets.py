from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def load_image_paths(split_file: str | Path) -> list[Path]:
    return [Path(line.strip()) for line in Path(split_file).read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_yolo_label_file(label_path: Path) -> np.ndarray:
    if not label_path.exists():
        return np.zeros((0, 5), dtype=np.float32)
    rows = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        rows.append([float(x) for x in parts])
    return np.asarray(rows, dtype=np.float32)


@dataclass
class Sample:
    image_path: Path
    label_path: Path


class YoloSplitDataset(Dataset):
    def __init__(self, split_file: str | Path, label_root: str | Path | None = None, transform: Callable | None = None):
        self.images = load_image_paths(split_file)
        self.label_root = Path(label_root) if label_root else None
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        image_path = self.images[idx]
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label_path = image_path.with_suffix(".txt") if self.label_root is None else self.label_root / f"{image_path.stem}.txt"
        labels = parse_yolo_label_file(label_path)
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        return image, torch.from_numpy(labels)


def collate_yolo(batch):
    images, labels = zip(*batch)
    return torch.stack(list(images)), list(labels)
