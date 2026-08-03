# AquaYOLO26

AquaYOLO26 is a YOLO26-based underwater debris detector with three additions:
Underwater-aware Batch Normalization (UWBN), Turbidity-conditioned STAL, and
Domain-Adversarial Progressive Loss.

The repository is organized around the SeaClear main experiment, TrashCan
target-unlabeled adaptation, and Trash-ICRA19 zero-shot evaluation.

## Release Status

This repository currently provides selected code, annotations, and
dataset-related resources for review and early reproducibility. The complete
implementation, training and evaluation scripts, model checkpoints,
configuration files, and releasable dataset materials will be made publicly
available upon final acceptance of the paper. Third-party datasets remain
subject to their original licenses.

## Layout

- `aquayolo26/`: model, training, validation, and export code
- `configs/`: main, repeated-run, and ablation settings
- `data/`: dataset YAML files and split lists
- `scripts/`: split generation, repeated training, export, and Jetson tests
- `weights/`: checkpoints and TensorRT engines
- `logs/`: training logs and benchmark CSV files
- `requirements.txt` / `environment.yml`: runtime setup

## Quick start

```bash
conda env create -f environment.yml
conda activate aquayolo26

python scripts/prepare_splits.py \
  --seaclear-root /path/to/SeaClear \
  --trashcan-root /path/to/TrashCan-Instance \
  --trash-icra19-root /path/to/Trash-ICRA19 \
  --out-dir data/splits

python aquayolo26/train.py \
  --cfg configs/aquayolo26n_seaclear.yaml \
  --data data/seaclear.yaml \
  --target-data data/trashcan_uda.yaml \
  --weights yolo26n.pt

python -m aquayolo26.train_full \
  --cfg configs/full_training.yaml \
  --data data/seaclear.yaml \
  --out-dir runs/full_train

bash scripts/reproduce_table4.sh
bash scripts/ablation_suite.sh

python -m aquayolo26 val --weights weights/aquayolo26n_seaclear_seed42.pt --data data/seaclear.yaml
python -m aquayolo26 export --weights weights/aquayolo26n_seaclear_seed42.pt --format engine --half
python -m aquayolo26 infer --weights weights/aquayolo26n_seaclear_seed42.pt --source /path/to/image_or_dir
```

## Training Recipe

Main runs use 640 x 640 letterboxed inputs, 200 epochs, batch size 16, MuSGD
settings inherited from YOLO26, and the random seeds `42`, `123`, and `2024`.
SeaClear is used as the labeled source domain. TrashCan contributes 1,000
unlabeled target-domain frames for DA-ProgLoss alignment. Trash-ICRA19 is held
out for zero-shot evaluation.

Jetson Orin NX measurements should be made from the exported FP16 TensorRT
engine with `scripts/benchmark_jetson.py`.
