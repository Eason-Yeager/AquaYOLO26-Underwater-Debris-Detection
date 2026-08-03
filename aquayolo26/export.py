from __future__ import annotations

import argparse

from ultralytics import YOLO


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--format", default="engine")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--device", default=0)
    parser.add_argument("--output")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    model = YOLO(args.weights)
    model.export(format=args.format, imgsz=args.imgsz, half=args.half, device=args.device)


if __name__ == "__main__":
    main()
