from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import yaml

from .metrics import MetricSummary, format_metric_summary
from .types import ExperimentConfig
from .utils import ensure_dir


def save_experiment_config(cfg: ExperimentConfig, out_dir: str | Path) -> Path:
    out = ensure_dir(out_dir)
    path = out / "experiment_config.yaml"
    path.write_text(yaml.safe_dump(asdict(cfg), sort_keys=False), encoding="utf-8")
    return path


def save_summary(summary: MetricSummary, out_dir: str | Path) -> Path:
    out = ensure_dir(out_dir)
    path = out / "summary.txt"
    path.write_text(format_metric_summary(summary) + "\n", encoding="utf-8")
    return path
