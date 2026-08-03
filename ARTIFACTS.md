# Artifacts

The code expects the following experiment artifacts when reproducing the paper
tables end to end.

## Checkpoints

- `weights/aquayolo26n_seaclear_seed42.pt`
- `weights/aquayolo26n_seaclear_seed123.pt`
- `weights/aquayolo26n_seaclear_seed2024.pt`
- `weights/aquayolo26s_seaclear_seed42.pt`
- `weights/aquayolo26n_fp16.engine`

## Logs

- `logs/seaclear_main_seed42/`
- `logs/seaclear_main_seed123/`
- `logs/seaclear_main_seed2024/`
- `logs/jetson_orin_nx_fp16.csv`

Run `scripts/prepare_splits.py` to write the split files into `data/splits/`.
