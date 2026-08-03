#!/usr/bin/env bash
set -euo pipefail

python -m aquayolo26.train_full --cfg configs/ablation_baseline_yolo26n.yaml --data data/seaclear.yaml --out-dir runs/ablation/baseline
python -m aquayolo26.train_full --cfg configs/ablation_uwbn.yaml --data data/seaclear.yaml --out-dir runs/ablation/uwbn
python -m aquayolo26.train_full --cfg configs/ablation_turb_stal.yaml --data data/seaclear.yaml --out-dir runs/ablation/turb_stal
python -m aquayolo26.train_full --cfg configs/ablation_da_progloss.yaml --data data/seaclear.yaml --out-dir runs/ablation/da_progloss
