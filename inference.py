"""
AquaYOLO26 Inference Script.

Runs inference on images/video and saves visualized detection results.

Usage:
    python scripts/inference.py \\
        --weights runs/train/aquayolo26n/weights/best.pt \\
        --source path/to/images \\
        --conf 0.25 --save
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

from aquayolo26 import AquaYOLO26, SUPER_CAT_NAMES, NUM_CLASSES
from aquayolo26.utils.dataset import letterbox


COLORS = [
    (255,  82,  82),   # Plastic    — red
    (255, 168,   0),   # Metal      — orange
    (0,   200, 255),   # Glass      — cyan
    (60,  220,  60),   # Fishing    — green
    (160,  80, 255),   # Other      — purple
]


def parse_args():
    parser = argparse.ArgumentParser(description="AquaYOLO26 inference")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--source", type=str, required=True,
                        help="Image/folder/video path or 'webcam'")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--save", action="store_true",
                        help="Save annotated output images")
    parser.add_argument("--save-dir", type=str, default="runs/inference")
    parser.add_argument("--show", action="store_true",
                        help="Display results in a window")
    parser.add_argument("--class-names", type=str, default=None,
                        help="Comma-separated class names (default: SeaClear super-cats)")
    return parser.parse_args()


def load_model(weights_path, device):
    ckpt = torch.load(weights_path, map_location="cpu")
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
    return model


def collect_sources(source_path):
    """Collect image paths from file, directory, or video."""
    p = Path(source_path)
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    vid_exts = {".mp4", ".avi", ".mov", ".mkv"}
    if p.is_dir():
        return sorted(str(x) for x in p.glob("*") if x.suffix.lower() in img_exts), "images"
    elif p.suffix.lower() in img_exts:
        return [str(p)], "images"
    elif p.suffix.lower() in vid_exts:
        return [str(p)], "video"
    else:
        raise ValueError(f"Unsupported source: {source_path}")


def preprocess(img_bgr, img_size):
    """BGR numpy → normalized RGB tensor."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_lb, ratio, pad = letterbox(img_rgb, new_shape=img_size)
    tensor = torch.from_numpy(img_lb.transpose(2, 0, 1)).float() / 255.0
    return tensor.unsqueeze(0), ratio, pad, img_rgb.shape[:2]


def draw_detections(img_bgr, boxes, scores, classes, class_names, conf_thr=0.25):
    """Draw bounding boxes and labels on a BGR image."""
    for box, score, cls in zip(boxes, scores, classes):
        if score < conf_thr:
            continue
        x1, y1, x2, y2 = map(int, box)
        color = COLORS[cls % len(COLORS)]
        name = class_names[cls] if cls < len(class_names) else f"cls{cls}"
        label = f"{name} {score:.2f}"
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img_bgr, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img_bgr, label, (x1 + 2, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img_bgr


def main():
    args = parse_args()

    device = torch.device(
        f"cuda:{args.device}" if torch.cuda.is_available() and args.device != "cpu"
        else "cpu"
    )
    class_names = args.class_names.split(",") if args.class_names else SUPER_CAT_NAMES

    print(f"Loading model: {args.weights}")
    model = load_model(args.weights, device)
    print(f"Classes: {class_names}")

    sources, src_type = collect_sources(args.source)

    save_dir = Path(args.save_dir)
    if args.save:
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving results to: {save_dir}")

    if src_type == "video":
        cap = cv2.VideoCapture(sources[0])
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if args.save:
            out_path = save_dir / (Path(sources[0]).stem + "_det.mp4")
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            tensor, ratio, pad, (h0, w0) = preprocess(frame, args.img_size)
            tensor = tensor.to(device)
            with torch.no_grad():
                results = model(tensor)
            if hasattr(results, "__iter__"):
                r = results[0]
                boxes = r.boxes.xyxy.cpu().numpy() if r.boxes else np.empty((0, 4))
                scores = r.boxes.conf.cpu().numpy() if r.boxes else np.empty(0)
                classes = r.boxes.cls.cpu().numpy().astype(int) if r.boxes else np.empty(0, int)
            else:
                boxes, scores, classes = np.empty((0, 4)), np.empty(0), np.empty(0, int)
            annotated = draw_detections(frame.copy(), boxes, scores, classes, class_names, args.conf)
            if args.save:
                writer.write(annotated)
            if args.show:
                cv2.imshow("AquaYOLO26", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            frame_idx += 1
        cap.release()
        if args.save:
            writer.release()
        cv2.destroyAllWindows()
        print(f"Processed {frame_idx} frames.")
    else:
        # Image inference
        for img_path in tqdm(sources, desc="Inference"):
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                print(f"Warning: cannot read {img_path}, skipping.")
                continue
            tensor, ratio, pad, (h0, w0) = preprocess(img_bgr, args.img_size)
            tensor = tensor.to(device)
            with torch.no_grad():
                results = model(tensor)
            if hasattr(results, "__iter__"):
                r = results[0]
                boxes = r.boxes.xyxy.cpu().numpy() if r.boxes else np.empty((0, 4))
                scores = r.boxes.conf.cpu().numpy() if r.boxes else np.empty(0)
                classes = r.boxes.cls.cpu().numpy().astype(int) if r.boxes else np.empty(0, int)
            else:
                boxes, scores, classes = np.empty((0, 4)), np.empty(0), np.empty(0, int)

            annotated = draw_detections(img_bgr.copy(), boxes, scores, classes, class_names, args.conf)

            if args.save:
                out_path = save_dir / Path(img_path).name
                cv2.imwrite(str(out_path), annotated)

            if args.show:
                cv2.imshow("AquaYOLO26", annotated)
                key = cv2.waitKey(0)
                if key == ord("q"):
                    break

        cv2.destroyAllWindows()
        n = len(sources)
        print(f"Done. Processed {n} image(s).")
        if args.save:
            print(f"Results saved to: {save_dir}")


if __name__ == "__main__":
    main()
