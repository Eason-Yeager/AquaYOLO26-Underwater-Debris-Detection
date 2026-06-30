"""
Grad-CAM Visualization Tool for AquaYOLO26.

Generates Grad-CAM activation maps (Selvaraju et al., ICCV 2017) to
visualize which spatial regions the backbone attends to. Reproduces
the analysis in Section 5.5 / Figure 7:
  - YOLO26n (baseline): diffuse activations across sediment and haze
  - AquaYOLO26n:       concentrated activations on debris boundaries

Usage:
    python tools/gradcam_vis.py \\
        --weights runs/train/aquayolo26n/weights/best.pt \\
        --source data/seaclear/test/images \\
        --layer backbone.20 \\
        --save-dir runs/vis/gradcam
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import cv2
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from aquayolo26 import AquaYOLO26, NUM_CLASSES
from aquayolo26.utils.dataset import letterbox


def parse_args():
    parser = argparse.ArgumentParser(description="Grad-CAM visualization")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--source", type=str, required=True,
                        help="Image file or directory")
    parser.add_argument("--layer", type=str, default="backbone.20",
                        help="Target layer name for Grad-CAM (e.g. 'backbone.20')")
    parser.add_argument("--save-dir", type=str, default="runs/vis/gradcam")
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Blend alpha for heatmap overlay (0-1)")
    parser.add_argument("--max-images", type=int, default=20)
    return parser.parse_args()


class GradCAM:
    """
    Grad-CAM for a specified layer in an AquaYOLO26 model.

    Hooks the target layer's forward output and backward gradient,
    then computes the weighted feature map activation.
    """

    def __init__(self, model: torch.nn.Module, target_layer_name: str):
        self.model = model
        self.target_layer = self._find_layer(model, target_layer_name)
        if self.target_layer is None:
            raise ValueError(f"Layer '{target_layer_name}' not found in model.")
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _find_layer(self, model, name):
        """Find a layer by dot-separated name."""
        parts = name.split(".")
        m = model
        for p in parts:
            if p.isdigit():
                try:
                    m = list(m.children())[int(p)]
                except (IndexError, TypeError):
                    return None
            else:
                m = getattr(m, p, None)
                if m is None:
                    return None
        return m

    def _register_hooks(self):
        def fwd_hook(module, input, output):
            self.activations = output.detach()

        def bwd_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(fwd_hook)
        self.target_layer.register_backward_hook(bwd_hook)

    def generate(self, input_tensor: torch.Tensor) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for input_tensor.

        Args:
            input_tensor: (1, 3, H, W) preprocessed image tensor.

        Returns:
            cam: (H, W) normalized heatmap, values in [0, 1].
        """
        self.model.zero_grad()
        output = self.model(input_tensor)

        # Use the sum of all detection scores as the target scalar
        if hasattr(output, "__iter__"):
            r = output[0]
            if hasattr(r, "boxes") and r.boxes is not None and len(r.boxes):
                target = r.boxes.conf.sum()
            else:
                target = output[0].sum() if hasattr(output[0], "sum") else torch.tensor(1.0)
        else:
            target = output.sum()

        target.backward()

        # Grad-CAM: global average pool gradients → channel weights
        grads = self.gradients   # (1, C, H', W')
        acts  = self.activations # (1, C, H', W')
        weights = grads.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * acts).sum(dim=1).squeeze(0)    # (H', W')
        cam = torch.relu(cam).cpu().numpy()

        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam


def overlay_heatmap(img_bgr: np.ndarray, cam: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Overlay Grad-CAM heatmap on original image."""
    h, w = img_bgr.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    heatmap = cv2.applyColorMap(
        (cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    blended = cv2.addWeighted(img_bgr, 1 - alpha, heatmap, alpha, 0)
    return blended


def collect_images(source, max_images):
    p = Path(source)
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    if p.is_dir():
        paths = sorted(str(x) for x in p.glob("*") if x.suffix.lower() in exts)
    elif p.suffix.lower() in exts:
        paths = [str(p)]
    else:
        paths = []
    return paths[:max_images]


def main():
    args = parse_args()

    device = torch.device(
        f"cuda:{args.device}" if torch.cuda.is_available() and args.device != "cpu"
        else "cpu"
    )

    # Load model (requires gradients for Grad-CAM)
    ckpt = torch.load(args.weights, map_location="cpu")
    cfg = ckpt.get("cfg", {})
    model_cfg = cfg.get("model", {})
    model = AquaYOLO26(
        variant=model_cfg.get("variant", "n"),
        num_classes=model_cfg.get("num_classes", NUM_CLASSES),
        use_uwbn=model_cfg.get("use_uwbn", True),
        use_turb_stal=False,
        use_da=False,
    )
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.to(device).eval()
    # Enable gradients for Grad-CAM
    for p in model.parameters():
        p.requires_grad_(True)

    # Grad-CAM
    try:
        cam_gen = GradCAM(model, args.layer)
    except ValueError as e:
        print(f"Error: {e}")
        print("Available layers (first 30):")
        for i, (name, _) in enumerate(model.named_modules()):
            print(f"  {name}")
            if i > 30:
                break
        return

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    img_paths = collect_images(args.source, args.max_images)
    if not img_paths:
        print(f"No images found in: {args.source}")
        return
    print(f"Processing {len(img_paths)} images. Saving to: {save_dir}")

    for img_path in tqdm(img_paths, desc="Grad-CAM"):
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_lb, _, _ = letterbox(img_rgb, new_shape=args.img_size)
        tensor = torch.from_numpy(img_lb.transpose(2, 0, 1)).float() / 255.0
        tensor = tensor.unsqueeze(0).to(device)

        try:
            cam = cam_gen.generate(tensor)
        except Exception as e:
            print(f"Grad-CAM failed for {img_path}: {e}")
            continue

        # Resize back to original and overlay
        overlay = overlay_heatmap(img_bgr, cam, alpha=args.alpha)

        stem = Path(img_path).stem
        out_path = save_dir / f"{stem}_gradcam.jpg"
        cv2.imwrite(str(out_path), overlay)

    print(f"Grad-CAM visualizations saved to: {save_dir}")
    print("\nReference (Figure 7 in paper):")
    print("  YOLO26n:      diffuse activations across sediment and haze")
    print("  AquaYOLO26n:  concentrated on debris boundaries (UWBN effect)")


if __name__ == "__main__":
    main()
