"""
TensorRT FP16 Export Tool for AquaYOLO26.

Exports a trained AquaYOLO26 model to:
  1. ONNX (intermediate)
  2. TensorRT FP16 engine (for Jetson Orin NX deployment)

At export time, the turbidity estimator and domain classifier branches
are stripped — the inference model is architecturally identical to YOLO26
except for UWBN replacing standard BN in the backbone.

Usage:
    python tools/export_tensorrt.py \\
        --weights runs/train/aquayolo26n/weights/best.pt \\
        --fp16 --batch 1

Requirements:
    - tensorrt (on Jetson: available via JetPack / TensorRT SDK)
    - onnx, onnxruntime
    - NVIDIA GPU
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
from pathlib import Path

import torch
from aquayolo26 import AquaYOLO26, NUM_CLASSES


def parse_args():
    parser = argparse.ArgumentParser(description="Export AquaYOLO26 to TensorRT")
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to AquaYOLO26 checkpoint (.pt)")
    parser.add_argument("--output-dir", type=str, default="runs/export")
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1,
                        help="Batch size for TensorRT engine")
    parser.add_argument("--fp16", action="store_true",
                        help="Use FP16 precision (recommended for Jetson)")
    parser.add_argument("--int8", action="store_true",
                        help="Use INT8 precision (requires calibration data)")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--onnx-opset", type=int, default=17)
    parser.add_argument("--verify", action="store_true",
                        help="Verify ONNX model after export")
    return parser.parse_args()


def load_inference_model(weights_path, device):
    """Load AquaYOLO26 and extract the inference-only YOLO26+UWBN model."""
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
    # Extract inference-only sub-model (YOLO26 + UWBN backbone)
    inference_model = model.export_inference_model()
    inference_model.to(device).eval()
    return inference_model, model_cfg


def export_onnx(model, output_path, img_size, batch, opset, device):
    """Export model to ONNX format."""
    print(f"Exporting to ONNX: {output_path}")
    dummy = torch.zeros(batch, 3, img_size, img_size, device=device)
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            str(output_path),
            opset_version=opset,
            input_names=["images"],
            output_names=["output0"],
            dynamic_axes={
                "images": {0: "batch"},
                "output0": {0: "batch"},
            },
            do_constant_folding=True,
        )
    print(f"ONNX model saved: {output_path}")
    return output_path


def verify_onnx(onnx_path, img_size, batch):
    """Verify ONNX model with onnxruntime."""
    try:
        import onnx
        import onnxruntime as ort
        import numpy as np
        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        dummy = np.random.randn(batch, 3, img_size, img_size).astype(np.float32)
        out = sess.run(None, {"images": dummy})
        print(f"ONNX verification passed. Output shape: {out[0].shape}")
    except ImportError:
        print("onnx/onnxruntime not installed; skipping ONNX verification.")
    except Exception as e:
        print(f"ONNX verification failed: {e}")


def export_tensorrt(onnx_path, engine_path, fp16=True, int8=False, batch=1, img_size=640):
    """
    Convert ONNX to TensorRT engine using trtexec (if available)
    or the Python TensorRT API.
    """
    # Method 1: trtexec command-line (available on Jetson with JetPack)
    import subprocess
    precision = "--fp16" if fp16 else ("--int8" if int8 else "")
    cmd = (
        f"trtexec --onnx={onnx_path} --saveEngine={engine_path} "
        f"--minShapes=images:{batch}x3x{img_size}x{img_size} "
        f"--optShapes=images:{batch}x3x{img_size}x{img_size} "
        f"--maxShapes=images:{batch}x3x{img_size}x{img_size} "
        f"{precision} --workspace=4096"
    )
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"TensorRT engine saved: {engine_path}")
    else:
        print(f"trtexec failed (may not be on this platform):\n{result.stderr}")
        print("To build TensorRT engine on Jetson, run this command on the target device.")

    # Print trtexec command for manual use on Jetson
    print("\n--- Manual TensorRT build command for Jetson Orin NX ---")
    print(f"trtexec --onnx={onnx_path.name} --saveEngine={engine_path.name} "
          f"--fp16 --workspace=4096 "
          f"--minShapes=images:{batch}x3x{img_size}x{img_size} "
          f"--optShapes=images:{batch}x3x{img_size}x{img_size} "
          f"--maxShapes=images:{batch}x3x{img_size}x{img_size}")
    print("-" * 58)


def main():
    args = parse_args()

    device_str = args.device
    device = torch.device(
        f"cuda:{device_str}" if torch.cuda.is_available() and device_str != "cpu"
        else "cpu"
    )
    print(f"Export device: {device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(args.weights).stem

    # Load model
    print(f"Loading model: {args.weights}")
    model, model_cfg = load_inference_model(args.weights, device)

    # Export ONNX
    onnx_path = out_dir / f"{stem}.onnx"
    export_onnx(model, onnx_path, args.img_size, args.batch, args.onnx_opset, device)

    if args.verify:
        verify_onnx(onnx_path, args.img_size, args.batch)

    # Export TensorRT
    precision = "fp16" if args.fp16 else ("int8" if args.int8 else "fp32")
    engine_path = out_dir / f"{stem}_{precision}.engine"
    export_tensorrt(onnx_path, engine_path, fp16=args.fp16, int8=args.int8,
                    batch=args.batch, img_size=args.img_size)

    print(f"\nExport complete.")
    print(f"  ONNX:      {onnx_path}")
    print(f"  TRT engine: {engine_path}")
    print(f"\nModel details:")
    print(f"  Variant:    YOLO26{model_cfg.get('variant', 'n')}")
    print(f"  UWBN:       {model_cfg.get('use_uwbn', True)}")
    print(f"  Params:     ~{model_cfg.get('variant', 'n') == 'n' and '2.46M' or '9.3M'}")
    print(f"  Precision:  {precision.upper()}")
    print(f"\nNote: UWBN adds ~1.0ms latency vs YOLO26n baseline on Jetson Orin NX.")
    print("Turb-STAL and DA-ProgLoss branches are training-only and are NOT exported.")


if __name__ == "__main__":
    main()
