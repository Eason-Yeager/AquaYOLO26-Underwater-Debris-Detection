"""
SeaClear Marine Debris Dataset Preparation Script.

Downloads (or validates) the SeaClear dataset and converts annotations
to YOLO label format with site-stratified train/val/test split.

Reference: Đuraš et al., Scientific Data 2024.
DOI: 10.1038/s41597-024-03759-2

Usage:
    python scripts/prepare_seaclear.py --root data/seaclear
    python scripts/prepare_seaclear.py --root data/seaclear --verify-only
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import json
import shutil
import random
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple

from aquayolo26.utils.dataset import SEACLEAR_SUPER_CAT


# ---------------------------------------------------------------------------
# SeaClear 40-class names (fine-grained)
# ---------------------------------------------------------------------------

SEACLEAR_CLASS_NAMES = [
    # Plastic (0-6)
    "plastic_bag", "plastic_bottle", "plastic_packaging", "plastic_cup",
    "plastic_sheet", "plastic_container", "plastic_other",
    # Metal (7-11)
    "metal_can", "metal_pipe", "metal_wire", "metal_fragment", "metal_other",
    # Glass (12-14)
    "glass_bottle", "glass_fragment", "glass_other",
    # Fishing gear (15-20)
    "fishing_net", "fishing_rope", "fishing_hook", "fishing_line",
    "fishing_trap", "fishing_other",
    # Other debris (21-39)
    "tire", "fabric", "rubber", "wood", "paper", "ceramic", "brick",
    "paint", "foam", "clothing", "shoe", "bag_other", "electronic",
    "battery", "chemical_container", "unknown_debris", "biological",
    "structural", "unclassified",
]

SUPER_CAT_NAMES = ["Plastic", "Metal", "Glass", "Fishing_gear", "Other_debris"]


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare SeaClear dataset")
    parser.add_argument("--root", type=str, default="data/seaclear",
                        help="Output root directory for SeaClear")
    parser.add_argument("--coco-json", type=str, default=None,
                        help="Path to existing SeaClear COCO annotation JSON "
                             "(if already downloaded)")
    parser.add_argument("--images-dir", type=str, default=None,
                        help="Path to existing SeaClear images directory")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--consolidate", action="store_true", default=True,
                        help="Map 40 classes → 5 super-categories")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verify-only", action="store_true",
                        help="Only verify existing dataset structure")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# COCO annotation conversion
# ---------------------------------------------------------------------------

def coco_to_yolo(
    ann_json_path: str,
    images_dir: str,
    output_root: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    consolidate: bool = True,
    seed: int = 42,
):
    """
    Convert SeaClear COCO-format annotations to YOLO format.

    COCO bbox format: [x_min, y_min, width, height] (absolute pixels)
    YOLO format:  class_id  cx  cy  w  h  (normalized 0-1)

    Applies site-stratified train/val/test split.
    """
    random.seed(seed)
    output_root = Path(output_root)

    # Load COCO annotations
    print(f"Loading COCO annotations from: {ann_json_path}")
    with open(ann_json_path) as f:
        coco = json.load(f)

    # Build image info map
    img_info = {img["id"]: img for img in coco["images"]}
    # Build per-image annotations
    img_anns: Dict[int, List] = defaultdict(list)
    for ann in coco["annotations"]:
        img_anns[ann["image_id"]].append(ann)

    # Extract site metadata from file_name (SeaClear uses site-coded filenames)
    # Filename format example: croatia_site1_frame00123.jpg
    def extract_site(filename: str) -> str:
        parts = Path(filename).stem.split("_")
        if len(parts) >= 2:
            return f"{parts[0]}_{parts[1]}"
        return parts[0]

    # Group images by site
    site_images: Dict[str, List[int]] = defaultdict(list)
    for img_id, info in img_info.items():
        site = extract_site(info["file_name"])
        site_images[site].append(img_id)

    print(f"Found {len(img_info)} images across {len(site_images)} sites.")

    # Site-stratified split
    train_ids, val_ids, test_ids = [], [], []
    for site, ids in site_images.items():
        random.shuffle(ids)
        n = len(ids)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_ids.extend(ids[:n_train])
        val_ids.extend(ids[n_train:n_train + n_val])
        test_ids.extend(ids[n_train + n_val:])

    print(f"Split: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")

    split_map = {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
    }

    # Create output directories
    for split in ("train", "val", "test"):
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Write YOLO labels and copy images
    converted = 0
    skipped = 0
    for split, ids in split_map.items():
        for img_id in ids:
            info = img_info[img_id]
            filename = info["file_name"]
            width = info["width"]
            height = info["height"]

            # Source image
            src_img = Path(images_dir) / filename
            if not src_img.exists():
                skipped += 1
                continue

            # Copy image
            dst_img = output_root / "images" / split / filename
            shutil.copy2(src_img, dst_img)

            # Write label file
            lines = []
            for ann in img_anns.get(img_id, []):
                cat_id = ann["category_id"] - 1  # 0-indexed
                if consolidate:
                    cat_id = SEACLEAR_SUPER_CAT.get(cat_id, 4)
                x_min, y_min, bw, bh = ann["bbox"]
                cx = (x_min + bw / 2) / width
                cy = (y_min + bh / 2) / height
                nw = bw / width
                nh = bh / height
                # Clamp to [0, 1]
                cx = max(0.0, min(1.0, cx))
                cy = max(0.0, min(1.0, cy))
                nw = max(0.0, min(1.0, nw))
                nh = max(0.0, min(1.0, nh))
                if nw > 0 and nh > 0:
                    lines.append(f"{cat_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

            lbl_stem = Path(filename).stem
            lbl_path = output_root / "labels" / split / f"{lbl_stem}.txt"
            with open(lbl_path, "w") as f:
                f.write("\n".join(lines))
            converted += 1

    print(f"Converted: {converted} images | Skipped (missing): {skipped}")
    print(f"Output: {output_root}")


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_dataset(root: str):
    """Check expected directory layout and file counts."""
    root = Path(root)
    expected = {
        "train": 6027,
        "val": 1292,
        "test": 1291,
    }
    print(f"\nVerifying SeaClear dataset at: {root}")
    all_ok = True
    for split, expected_count in expected.items():
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split
        n_imgs = len(list(img_dir.glob("*"))) if img_dir.exists() else 0
        n_lbls = len(list(lbl_dir.glob("*.txt"))) if lbl_dir.exists() else 0
        status = "✓" if n_imgs >= expected_count * 0.95 else "✗"
        all_ok = all_ok and (n_imgs >= expected_count * 0.95)
        print(f"  {status} {split}: {n_imgs} images, {n_lbls} labels "
              f"(expected ~{expected_count})")
    if all_ok:
        print("Dataset verified successfully.\n")
    else:
        print("WARNING: some splits appear incomplete.\n")
    return all_ok


# ---------------------------------------------------------------------------
# Synthetic demo data generation (for CI/testing without actual download)
# ---------------------------------------------------------------------------

def generate_demo_data(root: str, n_per_split: int = 10):
    """
    Generate minimal synthetic YOLO-format data for testing the pipeline
    without downloading the full SeaClear dataset.

    Creates n_per_split placeholder images and label files per split.
    """
    import numpy as np
    root = Path(root)
    print(f"Generating {n_per_split} synthetic demo images per split in {root}")
    for split in ("train", "val", "test"):
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n_per_split):
            # Random 640×480 "underwater" image (green-blue dominant)
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            img[:, :, 0] = np.random.randint(10, 60)    # R — attenuated
            img[:, :, 1] = np.random.randint(80, 160)   # G
            img[:, :, 2] = np.random.randint(100, 200)  # B — dominant
            # Add noise
            img = np.clip(img + np.random.randint(-20, 20, img.shape, dtype=np.int16), 0, 255).astype(np.uint8)
            import cv2
            fname = f"demo_{split}_{i:04d}.jpg"
            cv2.imwrite(str(img_dir / fname), img)
            # Random label (1-3 boxes)
            n_boxes = random.randint(1, 3)
            lines = []
            for _ in range(n_boxes):
                cls = random.randint(0, 4)
                cx = random.uniform(0.1, 0.9)
                cy = random.uniform(0.1, 0.9)
                w = random.uniform(0.05, 0.3)
                h = random.uniform(0.05, 0.3)
                lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            lbl_path = lbl_dir / f"demo_{split}_{i:04d}.txt"
            lbl_path.write_text("\n".join(lines))
    print("Demo data generated. Use --coco-json for real SeaClear conversion.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.verify_only:
        verify_dataset(args.root)
        return

    if args.coco_json and args.images_dir:
        coco_to_yolo(
            ann_json_path=args.coco_json,
            images_dir=args.images_dir,
            output_root=args.root,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            consolidate=args.consolidate,
            seed=args.seed,
        )
        verify_dataset(args.root)
    else:
        print(
            "No --coco-json / --images-dir provided.\n"
            "Generating synthetic demo data for pipeline testing...\n"
            "Download SeaClear from: https://www.nature.com/articles/s41597-024-03759-2\n"
            "Then rerun: python scripts/prepare_seaclear.py "
            "--root data/seaclear --coco-json <path> --images-dir <path>"
        )
        generate_demo_data(args.root, n_per_split=20)
        verify_dataset(args.root)


if __name__ == "__main__":
    main()
