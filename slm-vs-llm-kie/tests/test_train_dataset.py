"""SFT dataset builder — hygiene and byte-identity guarantees.

The dissertation headline depends on a clean before->after, so these tests pin
the two things a reviewer would doubt: (1) NO test-set document leaks into
training, and (2) the training prompt is byte-identical to the eval prompt.

Requires the real train/test splits on disk; skips cleanly if absent (the
synthetic `sample` dataset shares doc_ids across splits, so it can't exercise
the leakage guard meaningfully).
"""
from __future__ import annotations

import json

import pytest

from src.config import load_config, load_schema, schema_field_names
from src.data.gt import align_gold, load_test_set, schema_for_record
from src.data.loaders import available_datasets, load_dataset
from src.data.preprocess import from_dataset_record
from src.prompts.builder import build_prompt
from src.train import assert_no_test_leakage, build_sft_examples, summarize

cfg = load_config()

_REAL = {"sroie", "kleister_charity"}
_have_real = _REAL.issubset(set(available_datasets(cfg)))
pytestmark = pytest.mark.skipif(
    not _have_real, reason="real sroie + kleister_charity splits not present"
)


@pytest.fixture(scope="module")
def examples():
    # Small per-dataset cap keeps the test fast but exercises both datasets.
    return build_sft_examples(cfg, max_per_dataset=5)


def test_non_empty_both_datasets(examples):
    s = summarize(examples)
    assert s["by_dataset"].get("sroie", 0) > 0
    assert s["by_dataset"].get("kleister_charity", 0) > 0
    # Two input variants emitted per record.
    assert set(s["by_variant"]) == {"lines_only", "lines_plus_kv"}


def test_no_test_leakage(examples):
    # The helper raises on leakage; also assert disjointness directly.
    assert_no_test_leakage(cfg, examples)
    test_ids = {str(s["doc_id"]) for s in load_test_set(cfg)["doc_ids"]}
    used = {e["doc_id"] for e in examples}
    assert used.isdisjoint(test_ids)


def test_completion_is_valid_json_with_schema_keys(examples):
    for e in examples:
        pred = json.loads(e["completion"])  # must parse
        expected = set(schema_field_names(load_schema(cfg, e["dataset"])))
        assert set(pred.keys()) == expected


def test_prompt_is_byte_identical_to_eval(examples):
    """Rebuild the eval prompt for a sampled example's record and compare."""
    max_lines = cfg["conditions"].get("max_input_lines")
    for dataset in _REAL:
        ex = next(e for e in examples if e["dataset"] == dataset)
        rec = next(
            r for r in load_dataset(cfg, dataset, split="train")
            if str(r["doc_id"]) == ex["doc_id"]
        )
        schema = schema_for_record(cfg, rec)
        eval_prompt = build_prompt(
            schema, from_dataset_record(rec), "zero_shot", ex["input_variant"],
            None, max_lines=max_lines,
        )
        assert ex["prompt"] == eval_prompt
        # And the completion equals aligned gold.
        assert json.loads(ex["completion"]) == align_gold(rec.get("gold", {}), schema)


def test_zero_shot_only(examples):
    # A fine-tuned model gets no in-context exemplars.
    assert all("### Example" not in e["prompt"] for e in examples)


def test_capped_sample_is_deterministic():
    """The max_per_dataset cap draws a SEEDED random sample, so two independent
    builds must pick the identical documents (reproducible dissertation record)."""
    docs = lambda exs: {(e["dataset"], e["doc_id"]) for e in exs}
    a = build_sft_examples(cfg, max_per_dataset=5)
    b = build_sft_examples(cfg, max_per_dataset=5)
    assert docs(a) == docs(b)
    # And the cap actually samples a subset (not the whole pool), per dataset.
    for name in _REAL:
        assert len({d for d in docs(a) if d[0] == name}) == 5
