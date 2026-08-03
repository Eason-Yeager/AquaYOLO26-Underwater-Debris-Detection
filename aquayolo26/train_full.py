from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .training import AquaYOLO26Trainer


def load_yaml(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Full AquaYOLO26 training loop")
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out-dir", default="runs/full_train")
    parser.add_argument("--device", default=0)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg = load_yaml(args.cfg)
    data_cfg = load_yaml(args.data)
    trainer = AquaYOLO26Trainer(cfg, data_cfg, device=args.device)
    trainer.fit(
        train_split=data_cfg["train"],
        val_split=data_cfg["val"],
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
