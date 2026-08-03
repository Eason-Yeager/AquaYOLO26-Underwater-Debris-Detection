from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from .types import InferenceResult


def run_inference(weights: str, source: str, imgsz: int = 640, conf: float = 0.25) -> list[InferenceResult]:
    model = YOLO(weights)
    results = []
    for item in model.predict(source=source, imgsz=imgsz, conf=conf, stream=True, verbose=False):
        boxes = []
        if item.boxes is not None:
            xyxy = item.boxes.xyxy.cpu().numpy()
            cls = item.boxes.cls.cpu().numpy().astype(int)
            score = item.boxes.conf.cpu().numpy()
            for box, s, c in zip(xyxy, score, cls):
                boxes.append((float(box[0]), float(box[1]), float(box[2]), float(box[3]), float(s), int(c)))
        results.append(InferenceResult(image_path=Path(getattr(item, "path", source)), boxes=boxes))
    return results


def draw_boxes(image: np.ndarray, detections: list[tuple[float, float, float, float, float, int]]) -> np.ndarray:
    out = image.copy()
    for x1, y1, x2, y2, score, cls in detections:
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(out, f"{cls}:{score:.2f}", (int(x1), max(0, int(y1) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return out


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run_inference(args.weights, args.source, args.imgsz, args.conf)


if __name__ == "__main__":
    main()
