"""
Synthetic Underwater Dataset Generator.

Generates photorealistic synthetic underwater images with ground-truth
bounding boxes for all three dataset splits. Useful for:
  - CI/CD pipeline testing (no real data download required)
  - Debugging the training pipeline end-to-end
  - Verifying AquaYOLO26 module behavior

Simulates three degradation levels (Section 4.1.4 / Figure 5):
  (a) Clear shallow water, natural illumination
  (b) Moderate turbidity, suspended sediment
  (c) Severe attenuation, ROV artificial lighting

Usage:
    python tools/generate_synthetic_data.py \\
        --seaclear-root data/seaclear \\
        --trashcan-root data/trashcan \\
        --icra19-root data/trash_icra19 \\
        --n-train 200 --n-val 40 --n-test 40
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import cv2
import random
import numpy as np
from pathlib import Path
from typing import List, Tuple

from aquayolo26.utils.dataset import SUPER_CAT_NAMES


def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic underwater dataset")
    parser.add_argument("--seaclear-root", type=str, default="data/seaclear")
    parser.add_argument("--trashcan-root", type=str, default="data/trashcan")
    parser.add_argument("--icra19-root", type=str, default="data/trash_icra19")
    parser.add_argument("--n-train", type=int, default=200)
    parser.add_argument("--n-val", type=int, default=40)
    parser.add_argument("--n-test", type=int, default=40)
    parser.add_argument("--img-w", type=int, default=640)
    parser.add_argument("--img-h", type=int, default=480)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-classes", type=int, default=5,
                        help="5 for SeaClear consolidated, 3 for TrashCan/ICRA19")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Image synthesis
# ---------------------------------------------------------------------------

def make_underwater_background(h: int, w: int, degradation: str = "clear") -> np.ndarray:
    """
    Generate a synthetic underwater background image.

    degradation: 'clear' | 'moderate' | 'severe'
    """
    img = np.zeros((h, w, 3), dtype=np.float32)

    if degradation == "clear":
        # Shallow water: moderate blue-green, good contrast
        img[:, :, 0] = np.random.uniform(20, 60)   # R
        img[:, :, 1] = np.random.uniform(90, 160)  # G
        img[:, :, 2] = np.random.uniform(120, 200) # B
        # Caustic light patterns
        for _ in range(random.randint(3, 8)):
            cx, cy = random.randint(0, w), random.randint(0, h)
            r = random.randint(20, 80)
            brightness = random.uniform(20, 60)
            cv2.circle(img, (cx, cy), r, (brightness, brightness, brightness), -1)

    elif degradation == "moderate":
        # Turbid: hazy, lower contrast, green-dominant
        img[:, :, 0] = np.random.uniform(30, 70)
        img[:, :, 1] = np.random.uniform(100, 160)
        img[:, :, 2] = np.random.uniform(100, 170)
        # Sediment haze
        haze = np.random.uniform(0, 30, (h, w)).astype(np.float32)
        img[:, :, 0] += haze * 0.3
        img[:, :, 1] += haze * 0.6
        img[:, :, 2] += haze * 0.5

    else:  # severe
        # Deep water / ROV: very dark, strong blue cast, spotlight
        img[:, :, 0] = np.random.uniform(5, 25)
        img[:, :, 1] = np.random.uniform(20, 60)
        img[:, :, 2] = np.random.uniform(40, 100)
        # ROV spotlight
        cx, cy = w // 2 + random.randint(-100, 100), h // 2 + random.randint(-80, 80)
        for ch, strength in enumerate([50, 100, 120]):
            spot = np.zeros((h, w), dtype=np.float32)
            cv2.circle(spot, (cx, cy), random.randint(100, 200), strength, -1)
            spot = cv2.GaussianBlur(spot, (51, 51), 30)
            img[:, :, ch] += spot

    # Add noise
    noise = np.random.normal(0, random.uniform(3, 12), (h, w, 3)).astype(np.float32)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)

    # Slight blur to simulate scattering
    if degradation != "clear":
        blur_k = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (blur_k, blur_k), 0)

    return img


def draw_debris_object(
    img: np.ndarray,
    cls_id: int,
    cx_norm: float,
    cy_norm: float,
    w_norm: float,
    h_norm: float,
) -> np.ndarray:
    """Draw a synthetic debris object on the image."""
    H, W = img.shape[:2]
    x1 = int((cx_norm - w_norm / 2) * W)
    y1 = int((cy_norm - h_norm / 2) * H)
    x2 = int((cx_norm + w_norm / 2) * W)
    y2 = int((cy_norm + h_norm / 2) * H)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    if x2 <= x1 or y2 <= y1:
        return img

    # Different visual appearance per class
    debris_colors = [
        (200, 220, 240),  # Plastic — pale
        (120, 130, 140),  # Metal — gray
        (180, 220, 200),  # Glass — greenish
        (160, 140, 100),  # Fishing gear — brown
        (100, 110, 105),  # Other — dark
    ]
    color = debris_colors[cls_id % len(debris_colors)]
    shape = random.choice(["rect", "ellipse", "contour"])
    roi = img[y1:y2, x1:x2].copy()
    if shape == "rect":
        cv2.rectangle(roi, (0, 0), (roi.shape[1], roi.shape[0]), color, -1)
    elif shape == "ellipse":
        cx = roi.shape[1] // 2
        cy = roi.shape[0] // 2
        cv2.ellipse(roi, (cx, cy), (cx, cy), 0, 0, 360, color, -1)
    else:
        pts = np.array([[random.randint(0, roi.shape[1]),
                         random.randint(0, roi.shape[0])]
                        for _ in range(random.randint(4, 8))])
        cv2.fillConvexPoly(roi, pts, color)
    # Blend with background
    alpha = random.uniform(0.4, 0.85)
    img[y1:y2, x1:x2] = cv2.addWeighted(img[y1:y2, x1:x2], 1 - alpha, roi, alpha, 0)
    # Edge blur
    if random.random() < 0.5:
        k = random.choice([3, 5])
        img[max(0, y1-k):min(H, y2+k), max(0, x1-k):min(W, x2+k)] = \
            cv2.GaussianBlur(img[max(0, y1-k):min(H, y2+k), max(0, x1-k):min(W, x2+k)], (k, k), 0)
    return img


def generate_image_with_labels(
    h: int, w: int,
    num_classes: int,
    degradation: str = "clear",
    n_objects: Tuple[int, int] = (1, 5),
) -> Tuple[np.ndarray, List[str]]:
    """Generate one synthetic underwater image with YOLO labels."""
    img = make_underwater_background(h, w, degradation)

    n = random.randint(*n_objects)
    label_lines = []
    for _ in range(n):
        cls = random.randint(0, num_classes - 1)
        # Vary box sizes: include small targets (δ_s < 0.01 area)
        size_mode = random.choices(["small", "medium", "large"], weights=[0.3, 0.5, 0.2])[0]
        if size_mode == "small":
            bw = random.uniform(0.02, 0.08)
            bh = random.uniform(0.02, 0.08)
        elif size_mode == "medium":
            bw = random.uniform(0.08, 0.25)
            bh = random.uniform(0.08, 0.25)
        else:
            bw = random.uniform(0.2, 0.45)
            bh = random.uniform(0.15, 0.35)
        cx = random.uniform(bw / 2 + 0.01, 1 - bw / 2 - 0.01)
        cy = random.uniform(bh / 2 + 0.01, 1 - bh / 2 - 0.01)
        img = draw_debris_object(img, cls, cx, cy, bw, bh)
        label_lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    return img, label_lines


def generate_dataset(
    root: str,
    splits: dict,
    img_h: int,
    img_w: int,
    num_classes: int,
    prefix: str = "img",
    seed: int = 42,
):
    """Generate a full dataset with train/val/test splits."""
    random.seed(seed)
    np.random.seed(seed)
    root = Path(root)

    degradation_dist = {"clear": 0.4, "moderate": 0.4, "severe": 0.2}
    degradations = list(degradation_dist.keys())
    weights = list(degradation_dist.values())

    total = 0
    for split, n in splits.items():
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for i in range(n):
            deg = random.choices(degradations, weights=weights)[0]
            img, labels = generate_image_with_labels(
                img_h, img_w, num_classes, degradation=deg
            )
            fname = f"{prefix}_{split}_{i:05d}.jpg"
            cv2.imwrite(str(img_dir / fname), img)
            (lbl_dir / f"{prefix}_{split}_{i:05d}.txt").write_text("\n".join(labels))
            total += 1

        print(f"  {split}: {n} images ({img_w}×{img_h})")

    print(f"  Total: {total} images generated in {root}")


def main():
    args = parse_args()

    print("Generating synthetic SeaClear dataset...")
    generate_dataset(
        root=args.seaclear_root,
        splits={"train": args.n_train, "val": args.n_val, "test": args.n_test},
        img_h=args.img_h, img_w=args.img_w,
        num_classes=5,  # SeaClear consolidated
        prefix="seaclear",
        seed=args.seed,
    )

    print("Generating synthetic TrashCan dataset...")
    generate_dataset(
        root=args.trashcan_root,
        splits={"train": args.n_train // 5, "test": args.n_test},
        img_h=args.img_h, img_w=args.img_w,
        num_classes=3,  # trash/bio/rov
        prefix="trashcan",
        seed=args.seed + 1,
    )

    print("Generating synthetic Trash-ICRA19 dataset...")
    generate_dataset(
        root=args.icra19_root,
        splits={"test": args.n_test},
        img_h=args.img_h, img_w=args.img_w,
        num_classes=3,
        prefix="icra19",
        seed=args.seed + 2,
    )

    print("\nAll synthetic datasets generated.")
    print("Run verification: python tools/verify_dataset.py --all")
    print("\nNOTE: Replace synthetic data with real datasets for research use.")
    print("  SeaClear:      https://www.nature.com/articles/s41597-024-03759-2")
    print("  TrashCan 1.0:  https://conservancy.umn.edu/handle/11299/214366")
    print("  Trash-ICRA19:  https://doi.org/10.1109/ICRA.2019.8793975")


if __name__ == "__main__":
    main()
