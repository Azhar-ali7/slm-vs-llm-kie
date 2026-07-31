"""Field-level extraction metrics with type-aware normalisation.

Per (document, field) the prediction falls into one category:
  tp           gold non-null, pred non-null, normalised values match
  wrong        gold non-null, pred non-null, values differ
  missing      gold non-null, pred null                     (recall miss / FN)
  hallucinated gold null,     pred non-null                 (precision miss / FP)
  tn           gold null,     pred null                     (correct, both empty)

precision = tp / (tp + wrong + hallucinated)
recall    = tp / (tp + wrong + missing)
exact-match counts tp + tn as correct.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Any

_NULLISH = {"", "null", "none", "n/a", "na", "-", "nil"}

_DATE_FORMATS = [
    "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%Y/%m/%d",
    "%d %b %Y", "%d %B %Y", "%b %d, %Y", "%B %d, %Y", "%d.%m.%Y",
]


def is_null(v: Any) -> bool:
    if v is None:
        return True
    return str(v).strip().lower() in _NULLISH


def norm_string(v: Any) -> str:
    return re.sub(r"\s+", " ", str(v).strip().lower())


def norm_string_lenient(v: Any) -> str:
    """Normalised-match string form: drop ALL non-alphanumerics (whitespace,
    punctuation) and lowercase. Bridges format-only mismatches the strict metric
    rejects — e.g. 'OFF JALAN PUDU' vs 'OFF JALANPUDU' (a missing OCR space) and
    'S/B' vs 'SB'. Symmetric (applied identically to gold + pred), so it preserves
    the fairness guarantee. Reported ALONGSIDE strict, never replacing it."""
    return re.sub(r"[^a-z0-9]", "", str(v).strip().lower())


def norm_number(v: Any) -> str:
    """Canonical numeric string; strips currency symbols, commas, spaces."""
    s = str(v)
    m = re.search(r"[-+]?\d[\d,]*\.?\d*", s)
    if not m:
        return norm_string(v)
    num = m.group(0).replace(",", "")
    try:
        return f"{float(num):g}"
    except ValueError:
        return norm_string(v)


def norm_date(v: Any) -> str:
    s = str(v).strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return norm_string(v)


def normalize(v: Any, ftype: str, lenient: bool = False) -> str:
    if ftype in ("number", "currency"):
        return norm_number(v)
    if ftype == "date":
        return norm_date(v)
    return norm_string_lenient(v) if lenient else norm_string(v)


def _field_types(schema: dict[str, Any]) -> dict[str, str]:
    return {f["name"]: f.get("type", "string") for f in schema["fields"]}


def classify_field(gold: Any, pred: Any, ftype: str, lenient: bool = False) -> str:
    g_null, p_null = is_null(gold), is_null(pred)
    if g_null and p_null:
        return "tn"
    if g_null and not p_null:
        return "hallucinated"
    if not g_null and p_null:
        return "missing"
    return "tp" if normalize(gold, ftype, lenient) == normalize(pred, ftype, lenient) else "wrong"


def score_document(gold: dict[str, Any], pred: dict[str, Any] | None,
                   schema: dict[str, Any], lenient: bool = False) -> dict[str, Any]:
    """Score one document's prediction. A None pred (parse failure) = all missing
    where gold is non-null, all tn where gold is null.

    `lenient=True` uses punctuation/whitespace-insensitive string matching
    (norm_string_lenient) — a normalised-match score reported alongside the strict
    exact-match one. Numbers/dates are unaffected (already canonicalised)."""
    types = _field_types(schema)
    per_field: dict[str, str] = {}
    counts = {"tp": 0, "wrong": 0, "missing": 0, "hallucinated": 0, "tn": 0}

    for field, ftype in types.items():
        g = gold.get(field)
        p = None if pred is None else pred.get(field)
        cat = classify_field(g, p, ftype, lenient)
        per_field[field] = cat
        counts[cat] += 1

    metrics = micro_from_counts(counts)
    n_fields = len(types)
    exact_correct = counts["tp"] + counts["tn"]
    metrics["exact_match_doc"] = exact_correct == n_fields
    metrics["exact_match_fields"] = exact_correct
    metrics["n_fields"] = n_fields
    return {"per_field": per_field, "counts": counts, **metrics}


def micro_from_counts(counts: dict[str, int]) -> dict[str, float]:
    tp = counts["tp"]
    pred_pos = tp + counts["wrong"] + counts["hallucinated"]
    gold_pos = tp + counts["wrong"] + counts["missing"]
    precision = tp / pred_pos if pred_pos else 0.0
    recall = tp / gold_pos if gold_pos else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def aggregate_counts(rows: list[dict[str, int]]) -> dict[str, int]:
    total = {"tp": 0, "wrong": 0, "missing": 0, "hallucinated": 0, "tn": 0}
    for c in rows:
        for k in total:
            total[k] += c.get(k, 0)
    return total
