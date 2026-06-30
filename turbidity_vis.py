"""
Turbidity Estimator Visualization Tool.

Visualizes the per-image turbidity index τ ∈ [0, 1] estimated by the
Turb-STAL turbidity estimator across a set of images.

Shows:
  - Turbidity score bar overlay on each image
  - Ranked gallery from lowest (clear) to highest (turbid)
  - Comparison of estimated τ vs physical red/blue ratio proxy

Usage:
    python tools/turbidity_vis.py \\
        --weights runs/train/aquayolo26n/weights/best.pt \\
        --source data/seaclear/test/images \\
        --save-dir runs/vis/turbidity
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import cv2
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple

from aquayolo26 import AquaYOLO26, NUM_CLASSES
from aquayolo26.modules import TurbidityEstimator
from aquayolo26.utils.dataset import letterbox


def parse_args():
    parser = argparse.ArgumentParser(description="Turbidity estimator visualization")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--save-dir", type=str, default="runs/vis/turbidity")
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--max-images", type=int, default=50)
    return parser.parse_args()


def compute_rb_proxy(img_rgb: np.ndarray) -> float:
    """Compute red/blue ratio proxy from Eq. 8."""
    r = img_rgb[:, :, 0].mean()
    b = img_rgb[:, :, 2].mean()
    return float(r / (b + 1e-6))


def overlay_turbidity_bar(img_bgr: np.ndarray, tau: float) -> np.ndarray:
    """Draw a colored turbidity score bar on the image."""
    h, w = img_bgr.shape[:2]
    bar_h = 30
    bar_w = int(w * tau)
    # Color: green (clear) → red (turbid)
    g = int(255 * (1 - tau))
    r = int(255 * tau)
    color = (0, g, r)  # BGR
    result = img_bgr.copy()
    cv2.rectangle(result, (0, h - bar_h), (bar_w, h), color, -1)
    cv2.rectangle(result, (0, h - bar_h), (w, h), (200, 200, 200), 1)
    label = f"tau={tau:.3f}"
    cv2.putText(result, label, (5, h - 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return result


def main():
    args = parse_args()

    device = torch.device(
        f"cuda:{args.device}" if torch.cuda.is_available() and args.device != "cpu"
        else "cpu"
    )

    # Load model
    ckpt = torch.load(args.weights, map_location="cpu")
    cfg = ckpt.get("cfg", {})
    model_cfg = cfg.get("model", {})
    bb_ch = 256 if model_cfg.get("variant", "n") == "n" else 512

    model = AquaYOLO26(
        variant=model_cfg.get("variant", "n"),
        num_classes=model_cfg.get("num_classes", NUM_CLASSES),
        use_uwbn=model_cfg.get("use_uwbn", True),
        use_turb_stal=True,
        use_da=False,
    )
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.to(device).eval()

    if model.turb_estimator is None:
        print("This model was trained without Turb-STAL. No turbidity estimator available.")
        return

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Collect images
    p = Path(args.source)
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    if p.is_dir():
        img_paths = sorted(str(x) for x in p.glob("*") if x.suffix.lower() in exts)
    else:
        img_paths = [str(p)]
    img_paths = img_paths[:args.max_images]
    print(f"Processing {len(img_paths)} images...")

    results: List[Tuple[str, float, float]] = []  # (path, tau, rb_ratio)

    with torch.no_grad():
        for img_path in tqdm(img_paths):
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_lb, _, _ = letterbox(img_rgb, new_shape=args.img_size)
            tensor = torch.from_numpy(img_lb.transpose(2, 0, 1)).float() / 255.0
            tensor = tensor.unsqueeze(0).to(device)

            # Get turbidity estimate via backbone hook
            feat_holder = [None]

            def hook(m, i, o):
                feat_holder[0] = o

            h = list(model.yolo26.model.model.children())[9].register_forward_hook(hook)
            try:
                _ = model(tensor)
            finally:
                h.remove()

            tau = 0.0
            if feat_holder[0] is not None:
                tau = float(model.turb_estimator(feat_holder[0]).item())

            rb = compute_rb_proxy(img_lb)
            results.append((img_path, tau, rb))

            # Save individual overlay
            annotated = overlay_turbidity_bar(img_bgr, tau)
            stem = Path(img_path).stem
            cv2.imwrite(str(save_dir / f"{stem}_turb.jpg"), annotated)

    if not results:
        print("No results.")
        return

    # Sort by turbidity
    results.sort(key=lambda x: x[1])

    # --- Ranked gallery plot ---
    n = min(len(results), 16)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 4))
    axes = np.array(axes).flatten()
    for i, (path, tau, rb) in enumerate(results[:n]):
        img_bgr = cv2.imread(path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_sm = cv2.resize(img_rgb, (200, 150))
        axes[i].imshow(img_sm)
        axes[i].set_title(f"τ={tau:.3f} | R/B={rb:.2f}", fontsize=9)
        axes[i].axis("off")
    for i in range(n, len(axes)):
        axes[i].axis("off")
    plt.suptitle("Turbidity Gallery (sorted: clear → turbid)", fontsize=13)
    plt.tight_layout()
    gallery_path = save_dir / "turbidity_gallery.png"
    plt.savefig(gallery_path, dpi=120, bbox_inches="tight")
    plt.close()

    # --- Scatter: estimated τ vs R/B proxy ---
    taus = [r[1] for r in results]
    rbs  = [r[2] for r in results]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(rbs, taus, alpha=0.7, s=40, c=taus, cmap="RdYlGn_r")
    ax.set_xlabel("Red/Blue Ratio (physical proxy, Eq. 8)", fontsize=11)
    ax.set_ylabel("Estimated τ (TurbidityEstimator, Eq. 7)", fontsize=11)
    ax.set_title("Estimated Turbidity vs Physical Proxy", fontsize=12)
    ax.grid(True, alpha=0.3)
    scatter_path = save_dir / "turbidity_scatter.png"
    plt.tight_layout()
    plt.savefig(scatter_path, dpi=120, bbox_inches="tight")
    plt.close()

    print(f"\nResults saved to: {save_dir}")
    print(f"  Turbidity gallery: {gallery_path}")
    print(f"  Scatter plot:      {scatter_path}")
    print(f"\nStatistics (n={len(results)}):")
    print(f"  Mean τ: {np.mean(taus):.3f}")
    print(f"  Min  τ: {np.min(taus):.3f}")
    print(f"  Max  τ: {np.max(taus):.3f}")
    print(f"  Std  τ: {np.std(taus):.3f}")


if __name__ == "__main__":
    main()
