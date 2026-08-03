#!/usr/bin/env bash
set -euo pipefail

python -m aquayolo26.train_full \
  --cfg configs/aquayolo26n_seaclear.yaml \
  --data data/seaclear.yaml \
  --out-dir runs/full_train
