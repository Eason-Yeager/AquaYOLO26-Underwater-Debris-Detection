from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def validate(weights: str, data: str, split: str = "test", imgsz: int = 640, device=0):
    model = YOLO(weights)
    return model.val(data=data, split=split, imgsz=imgsz, device=device)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=0)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    validate(args.weights, args.data, args.split, args.imgsz, args.device)


if __name__ == "__main__":
    main()
