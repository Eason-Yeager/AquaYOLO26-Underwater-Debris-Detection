from __future__ import annotations

from pathlib import Path
from typing import Iterable


def read_split_list(path: str | Path) -> list[str]:
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_split_list(path: str | Path, items: Iterable[str]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(items) + "\n", encoding="utf-8")


def build_yaml_text(path: str, train: str | None = None, val: str | None = None,
                    test: str | None = None, names: dict[int, str] | None = None) -> str:
    lines = [f"path: {path}"]
    if train:
        lines.append(f"train: {train}")
    if val:
        lines.append(f"val: {val}")
    if test:
        lines.append(f"test: {test}")
    if names:
        lines.append("names:")
        for k, v in names.items():
            lines.append(f"  {k}: {v}")
    return "\n".join(lines) + "\n"
