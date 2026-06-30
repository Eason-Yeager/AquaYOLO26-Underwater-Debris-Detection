"""
Ablation Study Runner.

Trains and evaluates all five ablation variants from Table 3 / Section 4.5.3:

  V0: YOLO26 baseline        (no UWBN, no Turb-STAL, no DA)
  V1: + UWBN only
  V2: + Turb-STAL only
  V3: + DA-ProgLoss only
  V4: AquaYOLO26 (full)

Reports mAP@0.5, mAP_small@0.5, SR@0.5 on SeaClear and ICRA19.
Reproduces Table 7 results.

Usage:
    python scripts/run_ablation.py \\
        --config configs/aquayolo26n.yaml \\
        --data configs/seaclear.yaml \\
        --da-data configs/trashcan.yaml \\
        --transfer-data configs/trash_icra19.yaml \\
        --epochs 200 --device 0
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import json
import subprocess
from pathlib import Path
from datetime import datetime


ABLATION_VARIANTS = [
    {
        "name": "V0_baseline",
        "label": "V0: YOLO26 (baseline)",
        "flags": ["--no-uwbn", "--no-turb-stal", "--no-da"],
    },
    {
        "name": "V1_uwbn",
        "label": "V1: +UWBN",
        "flags": ["--no-turb-stal", "--no-da"],
    },
    {
        "name": "V2_turbstal",
        "label": "V2: +Turb-STAL",
        "flags": ["--no-uwbn", "--no-da"],
    },
    {
        "name": "V3_da",
        "label": "V3: +DA-ProgLoss",
        "flags": ["--no-uwbn", "--no-turb-stal"],
    },
    {
        "name": "V4_full",
        "label": "V4: AquaYOLO26 (full)",
        "flags": [],
    },
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run AquaYOLO26 ablation study")
    parser.add_argument("--config", type=str, default="configs/aquayolo26n.yaml")
    parser.add_argument("--data", type=str, default="configs/seaclear.yaml")
    parser.add_argument("--da-data", type=str, default="configs/trashcan.yaml")
    parser.add_argument("--transfer-data", type=str, default="configs/trash_icra19.yaml")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--ablation-dir", type=str, default="runs/ablation")
    parser.add_argument("--eval-only", action="store_true",
                        help="Only evaluate existing checkpoints (skip training)")
    parser.add_argument("--source-map", type=float, default=None,
                        help="Within-domain mAP for cross-domain ΔmAP calc")
    return parser.parse_args()


def run_variant(variant, args, ablation_dir):
    """Train one ablation variant."""
    v_dir = Path(ablation_dir) / variant["name"]
    v_dir.mkdir(parents=True, exist_ok=True)

    # Build train command
    cmd = [
        sys.executable, "scripts/train.py",
        "--config", args.config,
        "--data", args.data,
        "--epochs", str(args.epochs),
        "--batch", str(args.batch),
        "--device", args.device,
        "--run-dir", str(v_dir),
    ] + variant["flags"]

    # Add DA data only if DA is not disabled
    if "--no-da" not in variant["flags"]:
        cmd += ["--da-data", args.da_data, "--da-unlabeled", "1000"]

    print(f"\n{'='*60}")
    print(f"  Training: {variant['label']}")
    print(f"  Command:  {' '.join(cmd[2:])}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, cwd=os.getcwd())
    return result.returncode == 0


def evaluate_variant(variant, args, ablation_dir):
    """Evaluate one variant on SeaClear test and Trash-ICRA19."""
    v_dir = Path(ablation_dir) / variant["name"]
    weights = v_dir / "weights" / "best.pt"

    if not weights.exists():
        print(f"  ✗ Weights not found: {weights}")
        return None

    results = {}

    # SeaClear test
    seaclear_json = v_dir / "seaclear_results.json"
    cmd_sc = [
        sys.executable, "scripts/evaluate.py",
        "--weights", str(weights),
        "--data", args.data,
        "--split", "test",
        "--device", args.device,
        "--save-json", str(seaclear_json),
    ]
    subprocess.run(cmd_sc)
    if seaclear_json.exists():
        with open(seaclear_json) as f:
            results["seaclear"] = json.load(f)

    # Trash-ICRA19 zero-shot
    icra_json = v_dir / "icra19_results.json"
    cmd_icra = [
        sys.executable, "scripts/evaluate.py",
        "--weights", str(weights),
        "--data", args.transfer_data,
        "--split", "test",
        "--device", args.device,
        "--save-json", str(icra_json),
        "--zero-shot",
    ]
    if args.source_map:
        cmd_icra += ["--source-map", str(args.source_map)]
    subprocess.run(cmd_icra)
    if icra_json.exists():
        with open(icra_json) as f:
            results["icra19"] = json.load(f)

    return results


def print_ablation_table(all_results):
    """Print Table 7 style ablation results."""
    print("\n" + "=" * 90)
    print("  ABLATION STUDY RESULTS (cf. Table 7 in paper)")
    print("=" * 90)
    header = f"{'Variant':<30} {'UWBN':<6} {'Turb-STAL':<11} {'DA':<5} "
    header += f"{'SeaClear mAP@0.5':>17} {'mAP_s@0.5':>10} {'SR@0.5':>7} {'ICRA19 mAP':>11}"
    print(header)
    print("-" * 90)

    for v, res in all_results.items():
        has_uwbn = "V1" in v or "V4" in v
        has_turb = "V2" in v or "V4" in v
        has_da   = "V3" in v or "V4" in v
        sc = res.get("seaclear", {})
        ic = res.get("icra19", {})
        row = f"  {v:<28} {'Y' if has_uwbn else '-':<6} {'Y' if has_turb else '-':<11} "
        row += f"{'Y' if has_da else '-':<5} "
        row += f"{sc.get('mAP@0.5', '--'):>17} {sc.get('mAP_small@0.5', '--'):>10} "
        row += f"{sc.get('SR@0.5', '--'):>7} {ic.get('mAP@0.5', '--'):>11}"
        print(row)

    print("=" * 90)
    print("\nReference values (paper Table 7):")
    print("  V0 YOLO26n baseline: SeaClear 64.8 | mAP_s 31.4 | SR 59.6 | ICRA19 38.7")
    print("  V1 +UWBN:            SeaClear 67.1 | mAP_s 33.5 | SR 61.8 | ICRA19 40.9")
    print("  V2 +Turb-STAL:       SeaClear 67.8 | mAP_s 36.0 | SR 67.4 | ICRA19 39.8")
    print("  V3 +DA-ProgLoss:     SeaClear 66.2 | mAP_s 32.0 | SR 60.5 | ICRA19 44.5")
    print("  V4 AquaYOLO26n:      SeaClear 72.3 | mAP_s 38.4 | SR 67.9 | ICRA19 47.6")


def main():
    args = parse_args()
    ablation_dir = args.ablation_dir
    Path(ablation_dir).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"AquaYOLO26 Ablation Study — {timestamp}")
    print(f"Output: {ablation_dir}")

    all_results = {}
    for variant in ABLATION_VARIANTS:
        v_name = variant["name"]
        if not args.eval_only:
            success = run_variant(variant, args, ablation_dir)
            if not success:
                print(f"  ✗ Training failed for {v_name}")
        res = evaluate_variant(variant, args, ablation_dir)
        if res:
            all_results[f"{v_name} [{variant['label']}]"] = res

    if all_results:
        print_ablation_table(all_results)
        # Save summary
        summary_path = Path(ablation_dir) / "ablation_summary.json"
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nFull results saved to: {summary_path}")


if __name__ == "__main__":
    main()
