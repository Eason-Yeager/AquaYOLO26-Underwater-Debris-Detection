from __future__ import annotations

import argparse

from .export import main as export_main
from .infer import main as infer_main
from .train_full import main as full_train_main
from .train import train
from .val import main as val_main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aquayolo26")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("train")
    sub.add_parser("fit")
    sub.add_parser("val")
    sub.add_parser("export")
    sub.add_parser("infer")
    return parser


def main():
    parser = build_parser()
    args, remaining = parser.parse_known_args()
    if args.cmd == "train":
        train(remaining)
    elif args.cmd == "fit":
        full_train_main(remaining)
    elif args.cmd == "val":
        val_main(remaining)
    elif args.cmd == "export":
        export_main(remaining)
    elif args.cmd == "infer":
        infer_main(remaining)


if __name__ == "__main__":
    main()
