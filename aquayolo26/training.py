from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import DataLoader

from .augmentations import basic_train_transform, resize_letterbox, to_tensor
from .checkpointing import save_checkpoint
from .datasets import YoloSplitDataset, collate_yolo
from .logging_utils import append_text, write_csv
from .losses import LossWeights, total_loss
from .metrics import MetricSummary
from .models import AquaYOLO26
from .schedules import cosine_warmup, grl_schedule, progressive_weight
from .utils import ensure_dir, set_seed


@dataclass
class TrainerState:
    epoch: int = 0
    step: int = 0
    best_score: float = -1.0


@dataclass
class TrainingArtifacts:
    checkpoint_dir: Path
    log_dir: Path
    csv_path: Path


class AquaYOLO26Trainer:
    def __init__(self, cfg: dict, data_cfg: dict, device: str | int = 0):
        self.cfg = cfg
        self.data_cfg = data_cfg
        self.device = torch.device(f"cuda:{device}" if torch.cuda.is_available() and str(device) != "cpu" else "cpu")
        self.model = AquaYOLO26(
            num_classes=len(data_cfg.get("names", {0: "obj"})),
            with_turbidity=cfg.get("turb_stal", {}).get("enabled", True),
            with_domain=cfg.get("da_progloss", {}).get("enabled", True),
        ).to(self.device)
        self.loss_weights = LossWeights(
            box=7.5,
            cls=0.5,
            obj=1.5,
            turbidity=float(cfg.get("turb_stal", {}).get("lambda_turb", 0.01)),
            domain=1.0,
        )
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=float(cfg.get("lr0", 0.01)),
            momentum=float(cfg.get("momentum", 0.937)),
            weight_decay=float(cfg.get("weight_decay", 5e-4)),
            nesterov=True,
        )

    def build_loader(self, split_file: str, batch_size: int, shuffle: bool) -> DataLoader:
        ds = YoloSplitDataset(split_file, transform=lambda img: basic_train_transform(img, self.cfg.get("imgsz", 640)))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=2, pin_memory=True, collate_fn=collate_yolo)

    def train_one_epoch(self, loader: DataLoader, epoch: int, total_steps: int, state: TrainerState) -> dict[str, float]:
        self.model.train()
        loss_sum = 0.0
        for images, labels in loader:
            state.step += 1
            images = images.to(self.device)
            out = self.model(images, domain_lambda=grl_schedule(state.step, total_steps, lambda_max=float(self.cfg.get("da_progloss", {}).get("lambda_max", 1.0)), gamma=float(self.cfg.get("da_progloss", {}).get("gamma", 10.0))))
            base = out.detections.mean()
            box_loss = base * 0.0
            cls_loss = base * 0.0
            obj_loss = base * 0.0
            turb_loss = out.turbidity.mean() * 0.0 if out.turbidity is not None else None
            dom_loss = out.domain_logits.mean() * 0.0 if out.domain_logits is not None else None
            loss = total_loss(box_loss, cls_loss, obj_loss, turb=turb_loss, dom=dom_loss, weights=self.loss_weights)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            loss_sum += float(loss.detach().cpu())
        return {"train_loss": loss_sum / max(len(loader), 1)}

    def validate(self, loader: DataLoader) -> MetricSummary:
        self.model.eval()
        with torch.no_grad():
            seen = 0
            for images, _ in loader:
                _ = self.model(images.to(self.device), domain_lambda=0.0)
                seen += images.size(0)
        return MetricSummary(map50=0.0, map5095=0.0, small_map=0.0, recall=0.0, fps=0.0)

    def fit(self, train_split: str, val_split: str, out_dir: str | Path) -> TrainingArtifacts:
        out = ensure_dir(out_dir)
        ckpt_dir = ensure_dir(out / "checkpoints")
        log_dir = ensure_dir(out / "logs")
        csv_path = out / "metrics.csv"
        set_seed(int(self.cfg.get("seed", 42)), bool(self.cfg.get("deterministic", True)))
        train_loader = self.build_loader(train_split, int(self.cfg.get("batch", 16)), True)
        val_loader = self.build_loader(val_split, int(self.cfg.get("batch", 16)), False)
        state = TrainerState()
        rows = []
        total_steps = max(len(train_loader) * int(self.cfg.get("epochs", 200)), 1)
        for epoch in range(int(self.cfg.get("epochs", 200))):
            state.epoch = epoch
            train_stats = self.train_one_epoch(train_loader, epoch, total_steps, state)
            metrics = self.validate(val_loader)
            rows.append({"epoch": epoch, **train_stats, **asdict(metrics)})
            append_text(log_dir / "train.log", f"epoch={epoch} train_loss={train_stats['train_loss']:.4f}")
            score = metrics.map50
            if score > state.best_score:
                state.best_score = score
                save_checkpoint(ckpt_dir / "best.pt", {"epoch": epoch, "model": self.model.state_dict(), "cfg": self.cfg})
            save_checkpoint(ckpt_dir / f"epoch_{epoch:03d}.pt", {"epoch": epoch, "model": self.model.state_dict(), "cfg": self.cfg})
        write_csv(csv_path, rows)
        return TrainingArtifacts(checkpoint_dir=ckpt_dir, log_dir=log_dir, csv_path=csv_path)
