from __future__ import annotations

import argparse
import random
from pathlib import Path


def list_images(root: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(str(p) for p in root.rglob("*") if p.suffix.lower() in exts)


def write_list(path: Path, items):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(items) + "\n", encoding="utf-8")


def split(items, train_ratio=0.7, val_ratio=0.15, seed=42):
    rng = random.Random(seed)
    items = list(items)
    rng.shuffle(items)
    n = len(items)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return items[:n_train], items[n_train:n_train + n_val], items[n_train + n_val:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seaclear-root", required=True)
    parser.add_argument("--trashcan-root", required=True)
    parser.add_argument("--trash-icra19-root", required=True)
    parser.add_argument("--out-dir", default="data/splits")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.out_dir)
    seaclear = list_images(Path(args.seaclear_root))
    train, val, test = split(seaclear, seed=args.seed)
    write_list(out / "seaclear_train.txt", train)
    write_list(out / "seaclear_val.txt", val)
    write_list(out / "seaclear_test.txt", test)

    trashcan = list_images(Path(args.trashcan_root))
    write_list(out / "trashcan_uda_unlabeled_1000.txt", trashcan[:1000])
    write_list(out / "trashcan_test.txt", trashcan[1000:])

    icra19 = list_images(Path(args.trash_icra19_root))
    write_list(out / "trash_icra19_test.txt", icra19)

    (out / "seeds.json").write_text(
        '{"main_and_ablation_runs":[42,123,2024],"split_generation_seed":42,"torch_deterministic":true,"cudnn_benchmark":false}\n',
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
