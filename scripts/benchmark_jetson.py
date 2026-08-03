from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


def load_images(list_file, imgsz):
    paths = [p.strip() for p in Path(list_file).read_text().splitlines() if p.strip()]
    batch = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        batch.append(cv2.resize(img, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR))
    return batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--csv", required=True)
    args = parser.parse_args()

    model = YOLO(args.engine, task="detect")
    images = load_images(args.images, args.imgsz)
    if not images:
        raise SystemExit("no images loaded")
    sample = images[: min(len(images), args.iters)]

    for i in range(args.warmup):
        model.predict(sample[i % len(sample)], imgsz=args.imgsz, verbose=False)

    start = time.perf_counter()
    for i in range(args.iters):
        model.predict(sample[i % len(sample)], imgsz=args.imgsz, verbose=False)
    elapsed = time.perf_counter() - start

    fps = args.iters / elapsed
    latency_ms = 1000.0 / fps
    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["engine", "iters", "fps", "latency_ms"])
        writer.writeheader()
        writer.writerow({
            "engine": args.engine,
            "iters": args.iters,
            "fps": f"{fps:.2f}",
            "latency_ms": f"{latency_ms:.2f}",
        })
    print(f"FPS={fps:.2f}, latency={latency_ms:.2f} ms")


if __name__ == "__main__":
    main()
