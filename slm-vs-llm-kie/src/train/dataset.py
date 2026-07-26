"""Build a supervised fine-tuning (SFT) dataset from the TRAIN splits.

The whole point of Phase-3 fine-tuning is a clean before->after on the *same*
frozen 20-doc grid, so the model must be trained on inputs that are BYTE-
IDENTICAL to what the eval loop shows it. We therefore reuse the exact same
`build_prompt()` (src/prompts/builder.py), the same `align_gold()`, and the same
`conditions.max_input_lines` the runner uses (src/runner/run.py):

    prompt      = build_prompt(schema, simple_json, "zero_shot", variant, max_lines)
    completion  = json.dumps(align_gold(record["gold"], schema))

shot_mode is fixed to `zero_shot` — a fine-tuned model has learned the task and
needs no in-context exemplars — but we emit an example for EACH input variant
(`lines_only`, `lines_plus_kv`) so training covers both forms eval will present.

The prompt contains no model identity, so ONE sft.jsonl serves any base model;
the Kaggle notebook applies each model's chat template at train time.

DATA HYGIENE (non-negotiable): built from the train split ONLY. The 20-doc test
set must never leak into training. `assert_no_test_leakage` enforces it, and
`build_sft_examples` also defensively drops any train record whose doc_id
somehow appears in the test selection.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from src.data.gt import align_gold, load_test_set, schema_for_record
from src.data.loaders import load_dataset
from src.data.preprocess import from_dataset_record
from src.prompts.builder import build_prompt

DEFAULT_DATASETS = ["sroie", "kleister_charity"]

# Sentinel: distinguish "caller didn't pass max_lines" (-> use config, matching
# eval) from an explicit `None` (-> no line cap).
_USE_CONFIG = object()


def _test_doc_ids(cfg: dict[str, Any]) -> set[str]:
    return {str(s["doc_id"]) for s in load_test_set(cfg)["doc_ids"]}


def build_sft_examples(
    cfg: dict[str, Any],
    datasets: list[str] | None = None,
    input_variants: list[str] | None = None,
    max_lines: Any = _USE_CONFIG,
    max_per_dataset: int | None = None,
) -> list[dict[str, Any]]:
    """Return SFT examples {prompt, completion, dataset, doc_id, input_variant}.

    `max_lines` defaults to `conditions.max_input_lines` so prompts match eval.
    `max_per_dataset` caps records per dataset (bounds a quick training run) via a
    SEEDED RANDOM SAMPLE — keyed on `cfg["seed"]`, the same seed that fixes the
    20-doc test set — not a head slice, so the 400 are representative of the pool
    yet fully reproducible. Applied after the test-set exclusion.
    """
    datasets = datasets or DEFAULT_DATASETS
    if input_variants is None:
        input_variants = list(cfg["conditions"]["input_variants"])
    if max_lines is _USE_CONFIG:
        max_lines = cfg["conditions"].get("max_input_lines")

    test_ids = _test_doc_ids(cfg)
    examples: list[dict[str, Any]] = []

    for name in datasets:
        records = load_dataset(cfg, name, split="train")
        # Defensive: train/test files are already disjoint, but never trust that.
        records = [r for r in records if str(r["doc_id"]) not in test_ids]
        # Cap via a seeded random sample (reproducible, unbiased) — NOT a head
        # slice, which would over-represent whatever order the loader returns.
        if max_per_dataset is not None and len(records) > max_per_dataset:
            rng = random.Random(cfg.get("seed", 42))
            records = rng.sample(records, max_per_dataset)

        for rec in records:
            schema = schema_for_record(cfg, rec)
            simple_json = from_dataset_record(rec)
            completion = json.dumps(
                align_gold(rec.get("gold", {}), schema), ensure_ascii=False
            )
            for variant in input_variants:
                prompt = build_prompt(
                    schema, simple_json, "zero_shot", variant, None, max_lines=max_lines
                )
                examples.append(
                    {
                        "prompt": prompt,
                        "completion": completion,
                        "dataset": name,
                        "doc_id": str(rec["doc_id"]),
                        "input_variant": variant,
                    }
                )
    return examples


def assert_no_test_leakage(cfg: dict[str, Any], examples: list[dict[str, Any]]) -> bool:
    """Raise if any SFT example comes from a test-set document."""
    test_ids = _test_doc_ids(cfg)
    used = {e["doc_id"] for e in examples}
    leaked = sorted(used & test_ids)
    if leaked:
        raise AssertionError(
            f"TEST-SET LEAKAGE: {len(leaked)} test doc_id(s) found in SFT data "
            f"(e.g. {leaked[:5]}). Fine-tuning must use the train split only."
        )
    return True


def summarize(examples: list[dict[str, Any]]) -> dict[str, Any]:
    """Counts for the CLI: totals, per-dataset, per-variant, unique docs."""
    by_dataset: dict[str, int] = {}
    by_variant: dict[str, int] = {}
    docs: set[tuple[str, str]] = set()
    for e in examples:
        by_dataset[e["dataset"]] = by_dataset.get(e["dataset"], 0) + 1
        by_variant[e["input_variant"]] = by_variant.get(e["input_variant"], 0) + 1
        docs.add((e["dataset"], e["doc_id"]))
    return {
        "total": len(examples),
        "unique_docs": len(docs),
        "by_dataset": by_dataset,
        "by_variant": by_variant,
    }


def write_sft_jsonl(examples: list[dict[str, Any]], path: str | Path) -> Path:
    """Write one JSON object per line. Returns the path written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
    return path
