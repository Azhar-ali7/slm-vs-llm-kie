"""metrics.py: normalisation + known TP/FP/FN classification."""
from __future__ import annotations

from src.eval.metrics import (
    aggregate_counts,
    classify_field,
    micro_from_counts,
    norm_date,
    norm_number,
    normalize,
    score_document,
)

SCHEMA = {
    "fields": [
        {"name": "company", "type": "string", "required": True},
        {"name": "date", "type": "date", "required": True},
        {"name": "total", "type": "currency", "required": True},
        {"name": "vat", "type": "currency", "required": False},
    ]
}


def test_normalisation_string_currency_date():
    assert normalize("  ACME  Store ", "string") == "acme store"
    assert norm_number("£1,234.50") == "1234.5"
    assert norm_number("42.00") == "42"
    assert norm_date("29/07/2021") == "2021-07-29"
    assert norm_date("29 Jul 2021") == "2021-07-29"


def test_classify_categories():
    assert classify_field("ACME", "acme", "string") == "tp"          # match after norm
    assert classify_field("ACME", "Other", "string") == "wrong"      # both present, differ
    assert classify_field("ACME", None, "string") == "missing"       # gold present, pred null
    assert classify_field(None, "Spurious", "string") == "hallucinated"
    assert classify_field(None, None, "string") == "tn"
    assert classify_field("£1,234.50", "1234.5", "currency") == "tp"  # currency norm


def test_score_document_counts_and_metrics():
    gold = {"company": "ACME", "date": "2021-07-29", "total": "42.50", "vat": None}
    pred = {"company": "acme", "date": "29/07/2021", "total": "99.99", "vat": "1.00"}
    # company tp, date tp, total wrong, vat hallucinated
    res = score_document(gold, pred, SCHEMA)
    assert res["counts"] == {"tp": 2, "wrong": 1, "missing": 0, "hallucinated": 1, "tn": 0}
    # precision = 2/(2+1+1)=0.5 ; recall = 2/(2+1+0)=0.6667
    assert res["precision"] == 0.5
    assert res["recall"] == 0.6667
    assert res["exact_match_doc"] is False


def test_score_document_perfect():
    gold = {"company": "ACME", "date": "2021-07-29", "total": "42.50", "vat": None}
    pred = {"company": "ACME", "date": "2021-07-29", "total": "42.50", "vat": None}
    res = score_document(gold, pred, SCHEMA)
    assert res["counts"] == {"tp": 3, "wrong": 0, "missing": 0, "hallucinated": 0, "tn": 1}
    assert res["f1"] == 1.0
    assert res["exact_match_doc"] is True


def test_parse_failure_pred_none_all_missing():
    gold = {"company": "ACME", "date": "2021-07-29", "total": "42.50", "vat": None}
    res = score_document(gold, None, SCHEMA)
    # 3 non-null gold -> missing; vat null -> tn
    assert res["counts"] == {"tp": 0, "wrong": 0, "missing": 3, "hallucinated": 0, "tn": 1}
    assert res["recall"] == 0.0


def test_micro_and_aggregate():
    counts = aggregate_counts([
        {"tp": 2, "wrong": 1, "missing": 0, "hallucinated": 1, "tn": 0},
        {"tp": 3, "wrong": 0, "missing": 0, "hallucinated": 0, "tn": 1},
    ])
    assert counts == {"tp": 5, "wrong": 1, "missing": 0, "hallucinated": 1, "tn": 1}
    micro = micro_from_counts(counts)
    # precision = 5/(5+1+1)=0.7143 ; recall = 5/(5+1+0)=0.8333
    assert micro["precision"] == 0.7143
    assert micro["recall"] == 0.8333
