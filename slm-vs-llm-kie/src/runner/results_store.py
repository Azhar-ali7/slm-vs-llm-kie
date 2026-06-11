"""Incremental, resumable results storage (JSON Lines)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

CELL_KEYS = ("dataset", "doc_id", "model_id", "shot_mode", "input_variant", "sample_idx")


def cell_key(row: dict[str, Any]) -> tuple:
    return tuple(row[k] for k in CELL_KEYS)


def append_row(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_rows(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def completed_keys(path: str | Path) -> set[tuple]:
    """Cells already written successfully (no hard error) — for resumability."""
    done = set()
    for row in load_rows(path):
        if not row.get("error"):
            done.add(cell_key(row))
    return done


def write_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
