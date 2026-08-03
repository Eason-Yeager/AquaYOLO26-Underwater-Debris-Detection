from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

from aquayolo26.models import AquaYOLO26
from aquayolo26.utils import set_seed, ensure_dir


def load_cfg(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def build_model(cfg: dict) -> AquaYOLO26:
    return AquaYOLO26(
        num_classes=int(cfg.get("num_classes", 5)),
        with_turbidity=cfg.get("turb_stal", {}).get("enabled", False),
        with_domain=cfg.get("da_progloss", {}).get("enabled", False),
    )


def train_main(args: argparse.Namespace) -> None:
    cfg = load_cfg(args.cfg)
    set_seed(int(cfg.get("seed", args.seed or 42)), bool(cfg.get("deterministic", True)))
    ensure_dir(args.project)

    model = build_model(cfg)
    if args.weights:
        try:
            model.load_state_dict(torch.load(args.weights, map_location="cpu"), strict=False)
        except Exception:
            pass

    yolo = YOLO(args.weights or "yolo26n.pt")
    yolo.train(
        data=args.data,
        epochs=int(cfg.get("epochs", 200)),
        imgsz=int(cfg.get("imgsz", 640)),
        batch=int(cfg.get("batch", 16)),
        device=args.device,
        project=args.project,
        name=args.name,
        seed=int(cfg.get("seed", 42)),
        deterministic=bool(cfg.get("deterministic", True)),
    )


def train(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--target-data")
    parser.add_argument("--weights")
    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--name", default="aquayolo26")
    parser.add_argument("--device", default=0)
    parser.add_argument("--seed", type=int, default=42)
    train_main(parser.parse_args(argv))


if __name__ == "__main__":
    train()
