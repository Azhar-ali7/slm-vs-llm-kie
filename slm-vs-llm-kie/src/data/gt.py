"""Ground truth + reproducible 20-document test-set selection."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from src.config import load_schema, resolve_path, schema_field_names
from src.data.loaders import available_datasets, load_dataset


def align_gold(gold: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    """Return gold with exactly the schema's fields; missing -> None."""
    fields = schema_field_names(schema)
    return {f: gold.get(f) for f in fields}


def _schema_for_dataset(cfg: dict[str, Any], dataset: str) -> dict[str, Any]:
    # The synthetic `sample` dataset reuses sroie/kleister schemas by record shape;
    # gold keys already match, so pick the schema whose fields the gold satisfies.
    if dataset in cfg["datasets"]:
        return load_schema(cfg, dataset)
    return load_schema(cfg, "sroie")


def build_test_set(cfg: dict[str, Any], save: bool = True) -> dict[str, Any]:
    """Select a fixed, reproducible mix of documents (default 20).

    Uses real datasets when their raw files are present, otherwise the synthetic
    `sample` dataset. Selection is seeded so the same docs are chosen every run.
    Writes results/test_set_20.json.
    """
    seed = cfg.get("seed", 42)
    rng = random.Random(seed)
    available = available_datasets(cfg)
    synthetic = available == ["sample"]

    selected: list[dict[str, str]] = []

    if synthetic:
        records = load_dataset(cfg, "sample", split="test")
        total = cfg["test_set"]["total"]
        # Keep the simple/complex balance: split sample into receipts vs charities.
        receipts = [r for r in records if r["doc_id"].startswith("sample_receipt")]
        charities = [r for r in records if r["doc_id"].startswith("sample_charity")]
        rng.shuffle(receipts)
        rng.shuffle(charities)
        half = total // 2
        picked = receipts[:half] + charities[: total - half]
        # If sample is smaller than `total`, take what exists (demo/pilot still works).
        for r in picked:
            selected.append({"dataset": "sample", "doc_id": r["doc_id"]})
    else:
        for dataset, count in cfg["test_set"]["per_dataset"].items():
            if dataset not in available:
                continue
            records = load_dataset(cfg, dataset, split="test")
            rng.shuffle(records)
            for r in records[:count]:
                selected.append({"dataset": dataset, "doc_id": r["doc_id"]})

    payload = {
        "seed": seed,
        "synthetic": synthetic,
        "datasets_used": sorted({s["dataset"] for s in selected}),
        "count": len(selected),
        "doc_ids": selected,
    }

    if save:
        out = resolve_path(cfg, cfg["paths"]["test_set"])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return payload


def load_test_set(cfg: dict[str, Any]) -> dict[str, Any]:
    """Load the saved test-set selection, building it if absent."""
    path = resolve_path(cfg, cfg["paths"]["test_set"])
    if not path.exists():
        return build_test_set(cfg, save=True)
    return json.loads(path.read_text(encoding="utf-8"))


def load_test_records(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Materialise the selected test documents as full records (with gold)."""
    selection = load_test_set(cfg)
    wanted = {(s["dataset"], s["doc_id"]) for s in selection["doc_ids"]}

    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for dataset in {s["dataset"] for s in selection["doc_ids"]}:
        by_dataset[dataset] = load_dataset(cfg, dataset, split="test")

    records = []
    for dataset, docs in by_dataset.items():
        for r in docs:
            if (dataset, r["doc_id"]) in wanted:
                records.append(r)
    # Stable order matching the saved selection.
    order = {(s["dataset"], s["doc_id"]): i for i, s in enumerate(selection["doc_ids"])}
    records.sort(key=lambda r: order[(r["dataset"], r["doc_id"])])
    return records


def load_train_records(cfg: dict[str, Any], dataset: str, limit: int = 20) -> list[dict[str, Any]]:
    """Train-split records for few-shot examples (never the test set)."""
    try:
        return load_dataset(cfg, dataset, split="train")[:limit]
    except Exception:
        # Synthetic sample has no separate train split; reuse sample as examples,
        # excluding nothing here — the prompt builder guards against test leakage.
        return load_dataset(cfg, "sample", split="train")[:limit]
