from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ExperimentConfig:
    model: str = "aquayolo26n"
    imgsz: int = 640
    epochs: int = 200
    batch: int = 16
    optimizer: str = "MuSGD"
    lr0: float = 0.01
    lrf: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 5e-4
    seed: int = 42
    deterministic: bool = True
    data: Path | None = None
    target_data: Path | None = None
    project: str = "runs/train"
    name: str = "aquayolo26"
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class InferenceResult:
    image_path: Path
    boxes: list[tuple[float, float, float, float, float, int]]
    latency_ms: float | None = None


@dataclass
class BenchmarkRow:
    engine: str
    iters: int
    fps: float
    latency_ms: float
