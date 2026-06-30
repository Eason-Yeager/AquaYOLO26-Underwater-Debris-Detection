"""
TrashCan 1.0 Dataset Preparation Script.

Converts TrashCan COCO-format annotations to YOLO format.
Samples unlabeled target-domain images for DA-ProgLoss training.

Reference: Hong et al., arXiv 2020. DOI: 10.48550/arXiv.2007.08097
Download: https://conservancy.umn.edu/handle/11299/214366

Usage:
    python scripts/prepare_trashcan.py --root data/trashcan \\
        --coco-json path/to/instances_train.json \\
        --images-dir path/to/train
    # or: generate demo data for pipeline testing
    python scripts/prepare_trashcan.py --root data/trashcan
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

TRASHCAN_CLASS_MAP = {
    "trash": 0,
    "bio": 1,
    "rov": 2,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare TrashCan 1.0 dataset")
    parser.add_argument("--root", type=str, default="data/trashcan")
    parser.add_argument("--coco-json", type=str, default=None)
    parser.add_argument("--images-dir", type=str, default=None)
    parser.add_argument("--da-unlabeled", type=int, default=1000,
                        help="Number of images to copy to train/ for DA sampling")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def coco_to_yolo_trashcan(coco_json, images_dir, output_root, da_unlabeled=1000, seed=42):
    random.seed(seed)
    output_root = Path(output_root)
    with open(coco_json) as f:
        coco = json.load(f)

    # Build category id → class index map
    cat_map = {}
    for cat in coco.get("categories", []):
        name = cat["name"].lower()
        for key in TRASHCAN_CLASS_MAP:
            if key in name:
                cat_map[cat["id"]] = TRASHCAN_CLASS_MAP[key]
                break
        if cat["id"] not in cat_map:
            cat_map[cat["id"]] = 0  # default to trash

    img_info = {img["id"]: img for img in coco["images"]}
    img_anns = defaultdict(list)
    for ann in coco["annotations"]:
        img_anns[ann["image_id"]].append(ann)

    all_ids = list(img_info.keys())
    random.shuffle(all_ids)

    # DA unlabeled: first da_unlabeled images → train/ (labels withheld)
    # Test: remainder → test/ (labels provided)
    da_ids = all_ids[:da_unlabeled]
    test_ids = all_ids[da_unlabeled:]

    for split, ids in [("train", da_ids), ("test", test_ids)]:
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)
        for img_id in ids:
            info = img_info[img_id]
            src = Path(images_dir) / info["file_name"]
            if not src.exists():
                continue
            dst = output_root / "images" / split / info["file_name"]
            shutil.copy2(src, dst)
            if split == "test":
                lines = []
                for ann in img_anns.get(img_id, []):
                    cls = cat_map.get(ann["category_id"], 0)
                    x, y, bw, bh = ann["bbox"]
                    W, H = info["width"], info["height"]
                    cx = (x + bw / 2) / W
                    cy = (y + bh / 2) / H
                    nw = bw / W
                    nh = bh / H
                    if nw > 0 and nh > 0:
                        lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                lbl = output_root / "labels" / split / (Path(info["file_name"]).stem + ".txt")
                lbl.write_text("\n".join(lines))
    print(f"TrashCan: {len(da_ids)} DA-unlabeled images → train/, "
          f"{len(test_ids)} test images → test/")
    print(f"Output: {output_root}")


def generate_demo_trashcan(root, n=50):
    import numpy as np, cv2
    root = Path(root)
    for split in ("train", "test"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            img = np.random.randint(0, 80, (480, 640, 3), dtype=np.uint8)
            img[:, :, 2] += 120  # deep water: very blue
            fname = f"trashcan_{split}_{i:04d}.jpg"
            cv2.imwrite(str(root / "images" / split / fname), img)
            if split == "test":
                cls = random.randint(0, 2)
                cx, cy, w, h = (random.uniform(0.2, 0.8), random.uniform(0.2, 0.8),
                                random.uniform(0.1, 0.4), random.uniform(0.1, 0.4))
                (root / "labels" / split / f"trashcan_{split}_{i:04d}.txt").write_text(
                    f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                )
    print(f"Demo TrashCan data generated in {root}.")


def main():
    args = parse_args()
    if args.verify_only:
        print("TrashCan verification not yet implemented.")
        return
    if args.coco_json and args.images_dir:
        coco_to_yolo_trashcan(args.coco_json, args.images_dir, args.root,
                              args.da_unlabeled, args.seed)
    else:
        print("No --coco-json provided. Generating synthetic demo data...")
        generate_demo_trashcan(args.root, n=50)


if __name__ == "__main__":
    main()
