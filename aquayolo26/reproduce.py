from __future__ import annotations

from pathlib import Path


def default_commands() -> list[str]:
    return [
        "python scripts/prepare_splits.py --seaclear-root /path/to/SeaClear --trashcan-root /path/to/TrashCan-Instance --trash-icra19-root /path/to/Trash-ICRA19",
        "python -m aquayolo26 train --cfg configs/aquayolo26n_seaclear.yaml --data data/seaclear.yaml --target-data data/trashcan_uda.yaml --weights yolo26n.pt",
        "python -m aquayolo26 val --weights weights/aquayolo26n_seaclear_seed42.pt --data data/seaclear.yaml",
        "python -m aquayolo26 export --weights weights/aquayolo26n_seaclear_seed42.pt --format engine --half",
        "python scripts/benchmark_jetson.py --engine weights/aquayolo26n_fp16.engine --images data/splits/seaclear_test.txt --csv logs/jetson_orin_nx_fp16.csv",
    ]


def render_commands(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(default_commands()) + "\n", encoding="utf-8")
    return p
