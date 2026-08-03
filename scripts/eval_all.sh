#!/usr/bin/env bash
set -euo pipefail

WEIGHTS="${1:-weights/aquayolo26n_seaclear_seed42.pt}"

python aquayolo26/val.py --weights "$WEIGHTS" --data data/seaclear.yaml --split test --imgsz 640
python aquayolo26/val.py --weights "$WEIGHTS" --data data/trashcan_uda.yaml --split test --imgsz 640
python aquayolo26/val.py --weights "$WEIGHTS" --data data/trash_icra19_zero_shot.yaml --split test --imgsz 640
