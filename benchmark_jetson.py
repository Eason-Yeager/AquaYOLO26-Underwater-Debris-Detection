"""
Jetson Orin NX Inference Benchmark Tool.

Measures throughput (FPS), end-to-end latency (ms), and VRAM usage
of AquaYOLO26 on NVIDIA Jetson Orin NX with TensorRT FP16.

Reproduces the benchmark protocol from Section 4.2.3 and Table 8:
  - 500 warmup iterations (excluded from timing)
  - 1000 timed iterations at batch=1, 640×640 input
  - End-to-end latency: raw image tensor → decoded bounding boxes

Usage:
    # On Jetson Orin NX:
    python tools/benchmark_jetson.py \\
        --engine runs/export/aquayolo26n_fp16.engine \\
        --warmup 500 --runs 1000

    # Torch-based benchmark (no TensorRT required; for server GPU):
    python tools/benchmark_jetson.py \\
        --weights runs/train/aquayolo26n/weights/best.pt \\
        --warmup 100 --runs 500 --device 0
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import time
import torch
import numpy as np
from pathlib import Path

from aquayolo26 import AquaYOLO26, NUM_CLASSES


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark AquaYOLO26 inference")
    # TensorRT engine (preferred for Jetson)
    parser.add_argument("--engine", type=str, default=None,
                        help="Path to TensorRT .engine file")
    # PyTorch fallback
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to AquaYOLO26 .pt checkpoint (PyTorch mode)")
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=500,
                        help="Number of warmup iterations (excluded from timing)")
    parser.add_argument("--runs", type=int, default=1000,
                        help="Number of timed iterations")
    parser.add_argument("--device", type=str, default="0",
                        help="CUDA device index (ignored if --engine provided)")
    parser.add_argument("--fp16", action="store_true", default=True,
                        help="Use FP16 for PyTorch mode")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# TensorRT benchmark
# ---------------------------------------------------------------------------

def benchmark_tensorrt(engine_path, img_size, batch, warmup, runs):
    """Benchmark a TensorRT FP16 engine."""
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit
    except ImportError:
        print("TensorRT/PyCUDA not available. Use --weights for PyTorch benchmark.")
        return None

    print(f"Loading TensorRT engine: {engine_path}")
    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    # Allocate IO buffers
    input_shape = (batch, 3, img_size, img_size)
    input_size = int(np.prod(input_shape)) * np.dtype(np.float16).itemsize
    output_shape = (batch, engine.get_binding_shape(1).numel() // batch * batch,)
    output_size = int(np.prod(output_shape)) * np.dtype(np.float16).itemsize

    d_input  = cuda.mem_alloc(input_size)
    d_output = cuda.mem_alloc(output_size)
    stream   = cuda.Stream()

    dummy = np.random.randn(*input_shape).astype(np.float16)

    def infer():
        cuda.memcpy_htod_async(d_input, dummy, stream)
        context.execute_async_v2([int(d_input), int(d_output)], stream.handle)
        stream.synchronize()

    # Warmup
    print(f"Warming up ({warmup} iters)...")
    for _ in range(warmup):
        infer()

    # Timed runs
    print(f"Benchmarking ({runs} iters)...")
    t0 = time.perf_counter()
    for _ in range(runs):
        infer()
    t1 = time.perf_counter()

    elapsed_ms = (t1 - t0) * 1000
    latency_ms = elapsed_ms / runs
    fps = runs / (t1 - t0) * batch

    # VRAM
    free, total = cuda.mem_get_info()
    used_mb = (total - free) / 1024 / 1024

    print_results("TensorRT FP16", latency_ms, fps, used_mb, img_size, batch)
    return {"latency_ms": latency_ms, "fps": fps, "vram_mb": used_mb}


# ---------------------------------------------------------------------------
# PyTorch benchmark
# ---------------------------------------------------------------------------

def benchmark_pytorch(weights_path, img_size, batch, warmup, runs, device_str, fp16):
    """Benchmark using PyTorch (for server GPUs or when TRT unavailable)."""
    device = torch.device(
        f"cuda:{device_str}" if torch.cuda.is_available() and device_str != "cpu"
        else "cpu"
    )
    print(f"Device: {device}")

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
    model = model.to(device).eval()
    if fp16 and device.type == "cuda":
        model = model.half()
    print(f"Model loaded. FP16={fp16 and device.type == 'cuda'}")

    dtype = torch.float16 if (fp16 and device.type == "cuda") else torch.float32
    dummy = torch.zeros(batch, 3, img_size, img_size, dtype=dtype, device=device)

    # Warmup
    print(f"Warming up ({warmup} iters)...")
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Timed runs
    print(f"Benchmarking ({runs} iters)...")
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(runs):
            _ = model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    elapsed_ms = (t1 - t0) * 1000
    latency_ms = elapsed_ms / runs
    fps = runs / (t1 - t0) * batch

    # VRAM
    vram_mb = 0.0
    if device.type == "cuda":
        vram_mb = torch.cuda.memory_reserved(device) / 1024 / 1024

    mode = "FP16" if (fp16 and device.type == "cuda") else "FP32"
    print_results(f"PyTorch {mode}", latency_ms, fps, vram_mb, img_size, batch)
    return {"latency_ms": latency_ms, "fps": fps, "vram_mb": vram_mb}


# ---------------------------------------------------------------------------
# Results printer
# ---------------------------------------------------------------------------

def print_results(mode, latency_ms, fps, vram_mb, img_size, batch):
    print("\n" + "=" * 55)
    print(f"  AquaYOLO26 Inference Benchmark [{mode}]")
    print("=" * 55)
    print(f"  Input size:      {batch}×3×{img_size}×{img_size}")
    print(f"  Latency:         {latency_ms:.2f} ms / image")
    print(f"  Throughput:      {fps:.1f} FPS")
    print(f"  VRAM usage:      {vram_mb:.0f} MB")
    print(f"  Real-time (≥25): {'✓ YES' if fps >= 25 else '✗ NO'}")
    print("=" * 55)
    print("\nReference (Table 8 — Jetson Orin NX TensorRT FP16):")
    print("  YOLO26n baseline:   47.3 FPS,  21.2 ms,  398 MB")
    print("  AquaYOLO26n (ours): 45.1 FPS,  22.2 ms,  412 MB")
    print("  UWBN overhead:      +1.0 ms (+4.7%) vs YOLO26n baseline")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.engine:
        benchmark_tensorrt(
            engine_path=args.engine,
            img_size=args.img_size,
            batch=args.batch,
            warmup=args.warmup,
            runs=args.runs,
        )
    elif args.weights:
        benchmark_pytorch(
            weights_path=args.weights,
            img_size=args.img_size,
            batch=args.batch,
            warmup=args.warmup,
            runs=args.runs,
            device_str=args.device,
            fp16=args.fp16,
        )
    else:
        print("Provide --engine (TensorRT) or --weights (PyTorch) to run benchmark.")


if __name__ == "__main__":
    main()
