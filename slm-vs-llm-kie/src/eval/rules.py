"""Rule/regex recall fallback — fills ONLY missing (null) fields, never overrides.

The study's bottleneck is recall: models leave structured fields empty. For fields
with a distinctive surface form (UK postcode, registered-charity number, dates,
£ amounts) a cheap deterministic extractor recovers many of those blanks at $0 and
with no extra model. See FINDINGS [P4]/[E7].

Guarantees (must hold — they protect the fairness/precision contract in metrics.py):
- **Recall-only**: a field is filled only when the model left it null (`is_null`).
  A model-provided value is never touched, so precision on answered fields is intact.
- **Conservative**: an extractor returns a value only on a confident pattern match,
  else ``None`` (the field stays null). No guessing → no new hallucinations.
- **Normalisation-free**: extractors return a RAW string (e.g. ``"£12,500.00"``,
  ``"31 March 2016"``). Scoring's type-aware ``normalize`` canonicalises it exactly
  like gold, so no separate normalisation path is introduced here.

Only fields with a reliable pattern are attempted; fuzzy string fields (company,
charity_name, address street/town) are deliberately skipped — regex there would cost
precision without a recall gain.
"""
from __future__ import annotations

import re
from typing import Any

from src.eval.metrics import is_null

# --- shared patterns -------------------------------------------------------

# UK postcode: 1-2 letters, digit, optional letter/digit, space, digit, 2 letters.
_POSTCODE_RE = re.compile(r"\b([A-Z]{1,2}[0-9][A-Z0-9]?)\s*([0-9][A-Z]{2})\b", re.I)

# 6-8 digit registered-charity number, anchored to a charity/registration keyword so
# we don't grab an arbitrary long number (phone, amount, year-range).
_CHARITY_NUM_RE = re.compile(r"(?:charit|regist)[^\d]{0,40}?(\d{6,8})\b", re.I)
_NUM_6_8_RE = re.compile(r"\b(\d{6,8})\b")
_CHARITY_LINE_RE = re.compile(r"charit", re.I)
_CHARITY_KEY_RE = re.compile(r"number|no\.?\b|reg", re.I)

_MONTHS = "Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec"
_DATE_RES = [
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),                       # 2016-03-31
    re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b"),            # 31/03/2016 31-03-2016
    re.compile(r"\b\d{1,2}\.\d{1,2}\.\d{4}\b"),                # 31.03.2016
    re.compile(rf"\b\d{{1,2}}\s+(?:{_MONTHS})[a-z]*\s+\d{{4}}\b", re.I),   # 31 March 2016
    re.compile(rf"\b(?:{_MONTHS})[a-z]*\s+\d{{1,2}},?\s+\d{{4}}\b", re.I),  # March 31, 2016
]
_DATE_KW_RE = re.compile(r"date|dated|year end|period end|as at|ended|ending|financial year", re.I)

# £ amount, or a bare decimal (SROIE receipts often drop the symbol).
_GBP_RE = re.compile(r"£\s?([\d,]+(?:\.\d{1,2})?)")
_DECIMAL_RE = re.compile(r"\b(\d[\d,]*\.\d{2})\b")

_TOTAL_KW_RE = re.compile(r"\btotal\b", re.I)
_SUBTOTAL_RE = re.compile(r"sub[\s-]*total", re.I)


def _postcode(lines: list[str]) -> str | None:
    for ln in lines:
        m = _POSTCODE_RE.search(ln)
        if m:
            return f"{m.group(1).upper()} {m.group(2).upper()}"
    return None


def _charity_number(lines: list[str]) -> str | None:
    text = "\n".join(lines)
    m = _CHARITY_NUM_RE.search(text)
    if m:
        return m.group(1)
    for ln in lines:  # keyword line without the tight adjacency above
        if _CHARITY_LINE_RE.search(ln) and _CHARITY_KEY_RE.search(ln):
            m2 = _NUM_6_8_RE.search(ln)
            if m2:
                return m2.group(1)
    return None


def _date(lines: list[str]) -> str | None:
    # First pass: a date on a line that also carries a date-ish keyword (more likely
    # the reporting date than an incidental date). Second pass: any date.
    for require_kw in (True, False):
        for ln in lines:
            if require_kw and not _DATE_KW_RE.search(ln):
                continue
            for rgx in _DATE_RES:
                m = rgx.search(ln)
                if m:
                    return m.group(0)
    return None


def _total(lines: list[str]) -> str | None:
    cand = None
    for ln in lines:  # keep the LAST 'total' line (grand total usually last)
        if _TOTAL_KW_RE.search(ln) and not _SUBTOTAL_RE.search(ln):
            m = _GBP_RE.search(ln)
            if m:
                cand = "£" + m.group(1)
            else:
                m2 = _DECIMAL_RE.search(ln)
                if m2:
                    cand = m2.group(1)
    return cand


# Allowlist of fields with a HIGH-PRECISION surface form. Empirically calibrated on
# the study's own predictions (scripts/eval_rule_fallback.py audit): postcode ~89% and
# charity_number ~77% of rule-fills are correct. Deliberately EXCLUDED:
#   income/spending_annually — many £ figures per report; the rule picks the wrong one
#     (~11-53% correct) → pure precision loss, no recall gain (wrong ≠ tp).
#   report_date — several dates per report; the "reporting" one isn't rule-separable (~56%).
#   fuzzy strings (company, charity_name, address street/town) — no reliable pattern.
# This is the finding: rules recover FORMAT-distinctive fields, not SEMANTICALLY-selected ones.
_EXTRACTORS = {
    "address__postcode": _postcode,
    "charity_number": _charity_number,
    "date": _date,      # SROIE receipt date (single, unambiguous)
    "total": _total,    # SROIE receipt total
}


def _extract(name: str, ftype: str, lines: list[str]) -> str | None:
    """Dispatch a field to its extractor via the high-precision allowlist. Fields not
    in the allowlist (ambiguous or fuzzy) return None and stay null."""
    fn = _EXTRACTORS.get(name)
    return fn(lines) if fn else None


def fill_missing_with_rules(
    pred: dict[str, Any] | None,
    simple_json: dict[str, Any],
    schema: dict[str, Any],
) -> dict[str, Any]:
    """Return a copy of ``pred`` with null schema fields filled by rule extraction.

    ``pred`` may be ``None`` (a parse failure) — in that case every field is null and
    the fallback attempts them all, so it can partially rescue an unparseable output.
    Reads the UNCAPPED ``simple_json['lines']`` (not the prompt-truncated view) for
    best recall.
    """
    result: dict[str, Any] = dict(pred) if pred else {}
    lines = [str(x) for x in (simple_json.get("lines") or [])]
    for field in schema["fields"]:
        name = field["name"]
        if not is_null(result.get(name)):
            continue  # recall-only: never overwrite a model-provided value
        val = _extract(name, field.get("type", "string"), lines)
        if val is not None and not is_null(val):
            result[name] = val
    return result
