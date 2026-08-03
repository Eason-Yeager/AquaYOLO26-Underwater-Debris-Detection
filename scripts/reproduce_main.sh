#!/usr/bin/env bash
set -euo pipefail

python scripts/prepare_splits.py \
  --seaclear-root /path/to/SeaClear \
  --trashcan-root /path/to/TrashCan-Instance \
  --trash-icra19-root /path/to/Trash-ICRA19 \
  --out-dir data/splits \
  --seed 42

python -m aquayolo26 train \
  --cfg configs/aquayolo26n_seaclear.yaml \
  --data data/seaclear.yaml \
  --target-data data/trashcan_uda.yaml \
  --weights yolo26n.pt

python -m aquayolo26 val --weights weights/aquayolo26n_seaclear_seed42.pt --data data/seaclear.yaml
python -m aquayolo26 export --weights weights/aquayolo26n_seaclear_seed42.pt --format engine --half
