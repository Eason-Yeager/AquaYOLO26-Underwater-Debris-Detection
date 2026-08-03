#!/usr/bin/env bash
set -euo pipefail

for SEED in 42 123 2024; do
  python -m aquayolo26.train_full \
    --cfg configs/full_training.yaml \
    --data data/seaclear.yaml \
    --out-dir runs/table4/seed_${SEED}
done
