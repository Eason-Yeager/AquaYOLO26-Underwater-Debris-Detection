"""
Dataset Verification Tool.

Checks that all three datasets are correctly prepared in YOLO format
and reports statistics: image counts, label counts, class distributions,
small-object ratios, and sample visualizations.

Usage:
    python tools/verify_dataset.py --config configs/seaclear.yaml
    python tools/verify_dataset.py --all
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import yaml
import cv2
import numpy as np
from pathlib import Path
from collections import Counter
from typing import Dict, List

from aquayolo26.utils.dataset import SUPER_CAT_NAMES, NUM_CLASSES


def parse_args():
    parser = argparse.ArgumentParser(description="Verify dataset preparation")
    parser.add_argument("--config", type=str, default=None,
                        help="Dataset config YAML to verify")
    parser.add_argument("--all", action="store_true",
                        help="Verify all three datasets")
    parser.add_argument("--visualize", action="store_true",
                        help="Save sample annotated images")
    parser.add_argument("--vis-dir", type=str, default="runs/vis/dataset")
    parser.add_argument("--n-samples", type=int, default=5)
    return parser.parse_args()


def verify_split(root: Path, split: str, class_names: List[str]) -> Dict:
    """Verify one split and return statistics."""
    img_dir = root / "images" / split
    lbl_dir = root / "labels" / split

    if not img_dir.exists():
        return {"error": f"Missing: {img_dir}"}

    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    img_paths = [p for p in img_dir.glob("*") if p.suffix.lower() in exts]
    lbl_paths = list(lbl_dir.glob("*.txt")) if lbl_dir.exists() else []

    cls_counts = Counter()
    box_areas = []
    img_sizes = []
    missing_labels = 0
    empty_labels = 0

    for img_path in img_paths[:200]:  # sample up to 200 for speed
        # Check label
        lbl_path = lbl_dir / (img_path.stem + ".txt") if lbl_dir.exists() else None
        if lbl_path is None or not lbl_path.exists():
            missing_labels += 1
            continue
        lines = lbl_path.read_text().strip().split("\n")
        lines = [l for l in lines if l.strip()]
        if not lines:
            empty_labels += 1
            continue
        # Image size
        img = cv2.imread(str(img_path))
        if img is not None:
            h, w = img.shape[:2]
            img_sizes.append((h, w))
        else:
            h, w = 640, 640
        for line in lines:
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            bw, bh = float(parts[3]), float(parts[4])
            cls_counts[cls] += 1
            box_areas.append(bw * w * bh * h)  # area in pixels²

    box_areas = np.array(box_areas) if box_areas else np.array([0])
    small_ratio = (box_areas < 32 * 32).mean() if len(box_areas) else 0

    return {
        "images": len(img_paths),
        "labels": len(lbl_paths),
        "missing_labels": missing_labels,
        "empty_labels": empty_labels,
        "class_counts": dict(cls_counts),
        "small_object_ratio": float(small_ratio),
        "mean_box_area_px2": float(box_areas.mean()),
        "median_box_area_px2": float(np.median(box_areas)),
        "avg_img_size": (
            int(np.mean([s[0] for s in img_sizes])),
            int(np.mean([s[1] for s in img_sizes]))
        ) if img_sizes else (0, 0),
    }


def visualize_samples(root: Path, split: str, class_names: List[str],
                       vis_dir: Path, n: int = 5):
    """Save n sample images with bounding box annotations."""
    img_dir = root / "images" / split
    lbl_dir = root / "labels" / split
    vis_dir.mkdir(parents=True, exist_ok=True)
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    img_paths = sorted(p for p in img_dir.glob("*") if p.suffix.lower() in exts)[:n]
    colors = [(255, 82, 82), (255, 168, 0), (0, 200, 255), (60, 220, 60), (160, 80, 255)]
    for img_path in img_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if lbl_path.exists():
            for line in lbl_path.read_text().strip().split("\n"):
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)
                color = colors[cls % len(colors)]
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                name = class_names[cls] if cls < len(class_names) else f"cls{cls}"
                cv2.putText(img, name, (x1, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        out = vis_dir / f"{root.name}_{split}_{img_path.name}"
        cv2.imwrite(str(out), img)


def verify_dataset_config(config_path: str, visualize: bool = False,
                           vis_dir: Path = None, n_samples: int = 5):
    """Verify one dataset from its config YAML."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    dataset_name = cfg["dataset"]["name"]
    root = Path(cfg["dataset"]["root"])
    class_names_map = cfg.get("names", {})
    class_names = [class_names_map.get(i, f"cls{i}")
                   for i in range(cfg["dataset"].get("num_classes", 5))]

    print(f"\n{'='*60}")
    print(f"  Dataset: {dataset_name}")
    print(f"  Root:    {root}")
    print(f"{'='*60}")

    if not root.exists():
        print(f"  ✗ Root directory not found: {root}")
        print(f"    Run: python scripts/prepare_{dataset_name.lower().split()[0]}.py")
        return

    for split in ("train", "val", "test"):
        if not (root / "images" / split).exists():
            continue
        stats = verify_split(root, split, class_names)
        if "error" in stats:
            print(f"  {split}: {stats['error']}")
            continue
        print(f"\n  [{split}]")
        print(f"    Images:          {stats['images']}")
        print(f"    Labels:          {stats['labels']}")
        if stats["missing_labels"] > 0:
            print(f"    Missing labels:  {stats['missing_labels']} ⚠️")
        if stats["empty_labels"] > 0:
            print(f"    Empty labels:    {stats['empty_labels']}")
        print(f"    Avg image size:  {stats['avg_img_size'][1]}×{stats['avg_img_size'][0]}")
        print(f"    Small obj ratio: {stats['small_object_ratio']*100:.1f}% "
              f"(area < 32²px)")
        print(f"    Median box area: {stats['median_box_area_px2']:.0f} px²")
        print(f"    Class distribution:")
        for cls_id, count in sorted(stats["class_counts"].items()):
            name = class_names[cls_id] if cls_id < len(class_names) else f"cls{cls_id}"
            bar = "█" * min(int(count / max(stats["class_counts"].values()) * 30), 30)
            print(f"      [{cls_id}] {name:<20} {count:>5}  {bar}")

        if visualize and vis_dir and split == "test":
            vis_split_dir = vis_dir / dataset_name / split
            visualize_samples(root, split, class_names, vis_split_dir, n_samples)
            print(f"    Samples saved: {vis_split_dir}")

    print()


def main():
    args = parse_args()
    vis_dir = Path(args.vis_dir) if args.visualize else None

    if args.all:
        configs = [
            "configs/seaclear.yaml",
            "configs/trashcan.yaml",
            "configs/trash_icra19.yaml",
        ]
        for cfg in configs:
            if Path(cfg).exists():
                verify_dataset_config(cfg, args.visualize, vis_dir, args.n_samples)
            else:
                print(f"Config not found: {cfg}")
    elif args.config:
        verify_dataset_config(args.config, args.visualize, vis_dir, args.n_samples)
    else:
        print("Provide --config <path> or --all")
        print("Example: python tools/verify_dataset.py --config configs/seaclear.yaml")


if __name__ == "__main__":
    main()
