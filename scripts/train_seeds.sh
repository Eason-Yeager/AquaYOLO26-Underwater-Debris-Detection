#!/usr/bin/env bash
set -euo pipefail

CFG="${1:-configs/aquayolo26n_seaclear.yaml}"
shift || true
SEEDS=("$@")
if [ "${#SEEDS[@]}" -eq 0 ]; then
  SEEDS=(42 123 2024)
fi

for SEED in "${SEEDS[@]}"; do
  python aquayolo26/train.py \
    --cfg "$CFG" \
    --data data/seaclear.yaml \
    --target-data data/trashcan_uda.yaml \
    --weights yolo26n.pt \
    --seed "$SEED" \
    --project runs/seaclear_main \
    --name "aquayolo26_seed${SEED}"
done
