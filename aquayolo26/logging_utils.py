from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable


def write_csv(path: str | Path, rows: Iterable[dict[str, object]]) -> Path:
    rows = list(rows)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        p.write_text("", encoding="utf-8")
        return p
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return p


def append_text(path: str | Path, text: str) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")
    return p
