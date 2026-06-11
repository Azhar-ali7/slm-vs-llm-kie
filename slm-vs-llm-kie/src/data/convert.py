"""Convert downloaded SROIE and Kleister-Charity files into the framework's
normalised JSONL records. Pure functions (no network) — `scripts/convert_data.py`
drives them on files you have already downloaded.

SROIE (zzzDavid/ICDAR-2019-SROIE):
  data/box/<id>.txt  lines: x1,y1,x2,y2,x3,y3,x4,y4,transcript
  data/key/<id>.txt  JSON: {"company","date","address","total"}

Kleister-Charity (applicaai/kleister-charity):
  <split>/in.tsv.xz   tab-separated; OCR text columns (no coordinates)
  <split>/expected.tsv  space-separated key=value, spaces encoded as '_'
"""
from __future__ import annotations

import json
import lzma
import re
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# SROIE
# ---------------------------------------------------------------------------


def parse_sroie_box_line(line: str) -> tuple[str, list[float]] | None:
    """`x1,y1,...,x4,y4,transcript` -> (text, axis-aligned [minx,miny,maxx,maxy]).

    Splits at most 8 times so commas inside the transcript are preserved.
    """
    parts = line.rstrip("\n").split(",", 8)
    if len(parts) < 9:
        return None
    try:
        coords = [float(p) for p in parts[:8]]
    except ValueError:
        return None
    text = parts[8].strip()
    if not text:
        return None
    xs = coords[0::2]
    ys = coords[1::2]
    return text, [min(xs), min(ys), max(xs), max(ys)]


def _read_gold_json(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8", errors="ignore").strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    fields = ("company", "date", "address", "total")
    return {f: data.get(f) for f in fields}


def sroie_records_from_dirs(box_dir: str | Path, key_dir: str | Path,
                            split: str) -> list[dict[str, Any]]:
    box_dir, key_dir = Path(box_dir), Path(key_dir)
    records = []
    for box_file in sorted(box_dir.glob("*.txt")):
        stem = box_file.stem
        key_file = key_dir / f"{stem}.txt"
        if not key_file.exists():
            key_file = key_dir / f"{stem}.json"
        if not key_file.exists():
            continue  # only keep labelled receipts
        words = []
        for line in box_file.read_text(encoding="utf-8", errors="ignore").splitlines():
            parsed = parse_sroie_box_line(line)
            if parsed:
                text, bbox = parsed
                words.append({"text": text, "bbox": bbox})
        if not words:
            continue
        records.append({
            "doc_id": stem,
            "dataset": "sroie",
            "schema_name": "sroie",
            "split": split,
            "page": 1,
            "words": words,
            "key_values": {},
            "tables": [],
            "gold": _read_gold_json(key_file),
        })
    return records


# ---------------------------------------------------------------------------
# Kleister-Charity
# ---------------------------------------------------------------------------

_CUES = re.compile(
    r"income|expenditure|spending|spent|charity|registered|trustees|"
    r"year\s+end|annual|postcode|£|total\s+funds",
    re.IGNORECASE,
)


def parse_kleister_expected(line: str) -> dict[str, Any]:
    """`key=value key=value ...` with '_' standing in for spaces inside values."""
    out: dict[str, Any] = {}
    for tok in line.strip().split():
        if "=" not in tok:
            continue
        key, _, value = tok.partition("=")
        out[key] = value.replace("_", " ").strip()
    return out


def select_lines(text: str, head: int = 60, max_lines: int = 250,
                 tokens_per_line: int = 16) -> list[str]:
    """Turn a document's OCR text into selected pseudo-lines.

    Keeps the first `head` lines plus any line matching financial cues, up to
    `max_lines`. If the text has no newlines (flattened in the TSV), it is first
    chunked into pseudo-lines of `tokens_per_line` tokens.
    """
    if "\n" in text:
        raw_lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    else:
        toks = text.split()
        raw_lines = [" ".join(toks[i:i + tokens_per_line])
                     for i in range(0, len(toks), tokens_per_line)]

    selected: list[str] = []
    seen: set[int] = set()
    for i, ln in enumerate(raw_lines):
        if i < head or _CUES.search(ln):
            if i not in seen:
                selected.append(ln)
                seen.add(i)
        if len(selected) >= max_lines:
            break
    return selected


def _choose_text_column(cols: list[str]) -> str:
    """in.tsv columns: filename, keys, djvu, tesseract, textract, combined.
    Prefer the richest text column that is non-empty."""
    # indices 5 (combined), 4 (textract), 3 (tesseract), 2 (djvu)
    for idx in (5, 4, 3, 2):
        if idx < len(cols) and cols[idx].strip():
            return cols[idx]
    return cols[-1] if cols else ""


def kleister_records_from_tsv(in_tsv_xz: str | Path, expected_tsv: str | Path,
                              split: str, head: int = 60, max_lines: int = 250,
                              ) -> list[dict[str, Any]]:
    in_tsv_xz, expected_tsv = Path(in_tsv_xz), Path(expected_tsv)
    with lzma.open(in_tsv_xz, "rt", encoding="utf-8", errors="ignore") as fh:
        in_rows = [line.rstrip("\n").split("\t") for line in fh if line.strip()]

    gold_rows = []
    if expected_tsv.exists():
        gold_rows = [parse_kleister_expected(l)
                     for l in expected_tsv.read_text(encoding="utf-8", errors="ignore").splitlines()]

    records = []
    for i, cols in enumerate(in_rows):
        doc_id = cols[0] if cols and cols[0] else f"kleister_{split}_{i:04d}"
        text = _choose_text_column(cols)
        lines = select_lines(text, head=head, max_lines=max_lines)
        gold = gold_rows[i] if i < len(gold_rows) else {}
        records.append({
            "doc_id": doc_id,
            "dataset": "kleister_charity",
            "schema_name": "kleister_charity",
            "split": split,
            "page": 1,
            "lines": lines,        # text-only (no coordinates)
            "key_values": {},
            "tables": [],
            "gold": gold,
        })
    return records


# ---------------------------------------------------------------------------
# Writers / split
# ---------------------------------------------------------------------------


def write_jsonl(records: list[dict[str, Any]], out_path: str | Path) -> int:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(records)


def seeded_split(records: list[dict[str, Any]], test_frac: float = 0.2,
                 seed: int = 42) -> tuple[list, list]:
    """Deterministic train/test split (used when a dataset has no labelled test)."""
    import random

    rng = random.Random(seed)
    idx = list(range(len(records)))
    rng.shuffle(idx)
    n_test = max(1, int(len(records) * test_frac))
    test_ids = set(idx[:n_test])
    train = [records[i] for i in idx[n_test:]]
    test = [records[i] for i in idx[:n_test]]
    for r in train:
        r["split"] = "train"
    for r in test:
        r["split"] = "test"
    return train, test
