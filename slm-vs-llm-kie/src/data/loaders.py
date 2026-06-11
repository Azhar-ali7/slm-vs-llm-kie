"""Pluggable dataset loaders.

Each loader yields *records* in the shape preprocess.from_dataset_record expects,
plus a `gold` dict (target fields) and bookkeeping (`dataset`, `split`, `doc_id`):

    {
      "doc_id": str, "dataset": str, "split": "train"|"test", "page": 1,
      "words": [{"text": str, "bbox": [x0,y0,x1,y1]}, ...],
      "key_values": {k: v}, "tables": [...],
      "gold": {field: value_or_None},
    }

Real datasets (SROIE, Kleister-Charity) are read from data/raw/<name>/. If the
raw files are absent, the loader raises MissingDataError with instructions. A
deterministic synthetic `sample` dataset is always available so the pipeline,
pilot and viva demo run without any download (it is NOT used for real results).

Add a dataset: write load_<name>() and register it in LOADERS below + config.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Callable

from src.config import resolve_path


class MissingDataError(RuntimeError):
    """Raised when a real dataset's raw files are not present."""


# ---------------------------------------------------------------------------
# Synthetic `sample` dataset — deterministic, dependency-free, always available.
# Provides both a simple (receipt) and complex (charity) flavour so the demo and
# pilot exercise the full pipeline. Clearly synthetic; never report as results.
# ---------------------------------------------------------------------------

_SAMPLE_RECEIPTS = [
    ("Maple Cafe", "2021-03-04", "12 Oak Street, Leeds", "18.40"),
    ("Riverside Books", "2020-11-21", "5 Mill Road, Bristol", "7.99"),
    ("Green Grocer", "2021-06-15", "88 High Street, York", "23.10"),
    ("City Pharmacy", "2019-09-02", "3 Market Square, Bath", "11.25"),
    ("Sunrise Bakery", "2021-01-30", "47 Bridge Lane, Hull", "5.60"),
    ("Harbour Diner", "2020-07-19", "9 Quay Side, Dover", "34.75"),
]

_SAMPLE_CHARITIES = [
    ("Hope Outreach Trust", "1098761", "2020-03-31", "452000", "418900",
     "Manchester", "M1 4AB", "21 Albert Road"),
    ("Riverbank Care Foundation", "1123344", "2019-12-31", "1280500", "1190000",
     "Cardiff", "CF10 2EZ", "8 Bute Terrace"),
    ("Brightpath Education Fund", "1156677", "2021-03-31", "98500", "90250",
     "Norwich", "NR2 1RY", "14 Castle Meadow"),
    ("Coastal Wildlife Society", "1066552", "2020-09-30", "330000", "305400",
     "Plymouth", "PL1 2PA", "2 Barbican Approach"),
]


def _receipt_record(idx: int, company: str, date: str, address: str, total: str,
                    split: str) -> dict[str, Any]:
    """Lay receipt fields onto a simple 1-column page as positioned words."""
    rows = [
        company.split(),
        ["Date:", date],
        address.replace(",", " ,").split(),
        ["TOTAL", total],
    ]
    words = []
    y = 0.10
    for row in rows:
        x = 0.10
        for tok in row:
            words.append({"text": tok, "bbox": [x, y, x + 0.06 + 0.012 * len(tok), y + 0.04]})
            x += 0.08 + 0.012 * len(tok)
        y += 0.10
    return {
        "doc_id": f"sample_receipt_{idx:02d}",
        "dataset": "sample",
        "schema_name": "sroie",
        "split": split,
        "page": 1,
        "words": words,
        "key_values": {"Date": date, "Total": total},
        "tables": [],
        "gold": {"company": company, "date": date, "address": address, "total": total},
    }


def _charity_record(idx: int, name: str, number: str, rdate: str, income: str,
                    spending: str, town: str, postcode: str, street: str,
                    split: str) -> dict[str, Any]:
    rows = [
        name.split(),
        ["Charity", "No.", number],
        ["Year", "ended", rdate],
        ["Total", "income", "£" + income],
        ["Total", "expenditure", "£" + spending],
        ["Registered", "office:", street + ",", town, postcode],
    ]
    words = []
    y = 0.08
    for row in rows:
        x = 0.08
        for tok in row:
            words.append({"text": tok, "bbox": [x, y, x + 0.05 + 0.011 * len(tok), y + 0.035]})
            x += 0.07 + 0.011 * len(tok)
        y += 0.09
    return {
        "doc_id": f"sample_charity_{idx:02d}",
        "dataset": "sample",
        "schema_name": "kleister_charity",
        "split": split,
        "page": 1,
        "words": words,
        "key_values": {
            "Charity No.": number, "Year ended": rdate,
            "Total income": "£" + income, "Total expenditure": "£" + spending,
        },
        "tables": [],
        "gold": {
            "charity_name": name,
            "charity_number": number,
            "report_date": rdate,
            "income_annually_in_british_pounds": income,
            "spending_annually_in_british_pounds": spending,
            "address__post_town": town,
            "address__postcode": postcode,
            "address__street_line": street,
        },
    }


def load_sample(cfg: dict[str, Any], split: str = "test") -> list[dict[str, Any]]:
    """Deterministic synthetic dataset (receipts + charities). Always available."""
    records = []
    for i, r in enumerate(_SAMPLE_RECEIPTS):
        records.append(_receipt_record(i, *r, split=split))
    for i, c in enumerate(_SAMPLE_CHARITIES):
        records.append(_charity_record(i, *c, split=split))
    return records


# ---------------------------------------------------------------------------
# Real datasets — read data/raw/<name>/. Absent files -> MissingDataError.
# ---------------------------------------------------------------------------


def _require_dir(cfg: dict[str, Any], dataset: str) -> Path:
    raw_dir = resolve_path(cfg, cfg["datasets"][dataset]["raw_dir"])
    if not raw_dir.exists() or not any(raw_dir.iterdir()):
        raise MissingDataError(
            f"No raw data for '{dataset}' at {raw_dir}.\n"
            f"Run `python scripts/download_data.py` for instructions, then place "
            f"files under {raw_dir}. (You can develop/demo with dataset 'sample'.)"
        )
    return raw_dir


def load_sroie(cfg: dict[str, Any], split: str = "test") -> list[dict[str, Any]]:
    """SROIE receipts.

    Expects a normalised JSONL the preprocessing/download step produced at
    data/raw/sroie/<split>.jsonl, one record per line with keys
    {doc_id, words, key_values?, gold}. (download_data.py documents how to build
    this from the public SROIE OCR + entities files.)
    """
    raw_dir = _require_dir(cfg, "sroie")
    return _load_jsonl_records(raw_dir / f"{split}.jsonl", dataset="sroie", split=split)


def load_kleister(cfg: dict[str, Any], split: str = "test") -> list[dict[str, Any]]:
    """Kleister-Charity financial reports — same normalised JSONL convention."""
    raw_dir = _require_dir(cfg, "kleister_charity")
    return _load_jsonl_records(raw_dir / f"{split}.jsonl", dataset="kleister_charity", split=split)


def _load_jsonl_records(path: Path, dataset: str, split: str) -> list[dict[str, Any]]:
    if not path.exists():
        raise MissingDataError(
            f"Expected normalised records at {path}. See scripts/download_data.py."
        )
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rec.setdefault("dataset", dataset)
            rec.setdefault("schema_name", dataset)
            rec.setdefault("split", split)
            rec.setdefault("page", 1)
            rec.setdefault("key_values", {})
            rec.setdefault("tables", [])
            records.append(rec)
    return records


LOADERS: dict[str, Callable[..., list[dict[str, Any]]]] = {
    "sample": load_sample,
    "sroie": load_sroie,
    "kleister_charity": load_kleister,
}


def load_dataset(cfg: dict[str, Any], name: str, split: str = "test") -> list[dict[str, Any]]:
    if name not in LOADERS:
        raise KeyError(f"Unknown dataset '{name}'. Known: {sorted(LOADERS)}")
    return LOADERS[name](cfg, split=split)


def available_datasets(cfg: dict[str, Any]) -> list[str]:
    """Real datasets whose raw files are present; falls back to ['sample']."""
    present = []
    for name in cfg["datasets"]:
        try:
            _require_dir(cfg, name)
            present.append(name)
        except MissingDataError:
            continue
    return present or ["sample"]
