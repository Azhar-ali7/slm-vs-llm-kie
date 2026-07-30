"""Rule/regex recall fallback (src/eval/rules.py) — a demoted, opt-in helper.

The fallback is NOT part of the headline SLM-vs-LLM comparison (it is model-agnostic
post-processing, measured off to the side by scripts/eval_rule_fallback.py). These tests
just guard the two guarantees that make it safe to keep in the tree: it fills only nulls
with high-precision patterns, and it never overrides a model value.
"""
from src.eval.metrics import score_document
from src.eval.rules import fill_missing_with_rules

SROIE = {"fields": [
    {"name": "company", "type": "string"},
    {"name": "date", "type": "date"},
    {"name": "address", "type": "string"},
    {"name": "total", "type": "currency"},
]}
KLEISTER = {"fields": [
    {"name": "charity_name", "type": "string"},
    {"name": "charity_number", "type": "string"},
    {"name": "report_date", "type": "date"},
    {"name": "income_annually_in_british_pounds", "type": "currency"},
    {"name": "spending_annually_in_british_pounds", "type": "currency"},
    {"name": "address__post_town", "type": "string"},
    {"name": "address__postcode", "type": "string"},
    {"name": "address__street_line", "type": "string"},
]}


def test_postcode_extracted_and_normalised():
    sj = {"lines": ["The Charity Trust", "12 Oak Street, Leeds", "LS1 4AB"]}
    out = fill_missing_with_rules({"address__postcode": None}, sj, KLEISTER)
    assert out["address__postcode"] == "LS1 4AB"


def test_charity_number_extracted_near_keyword():
    sj = {"lines": ["Annual Report", "Registered charity number 1084095", "Trustees"]}
    out = fill_missing_with_rules({"charity_number": None}, sj, KLEISTER)
    assert out["charity_number"] == "1084095"


def test_never_overrides_a_model_value():
    sj = {"lines": ["LS1 4AB", "Registered charity number 1084095"]}
    pred = {"address__postcode": "SW1A 1AA", "charity_number": "9999999"}
    out = fill_missing_with_rules(pred, sj, KLEISTER)
    assert out["address__postcode"] == "SW1A 1AA"   # model value kept
    assert out["charity_number"] == "9999999"


def test_ambiguous_semantic_fields_stay_null():
    # income/spending/report_date are deliberately NOT in the allowlist — several
    # candidate figures/dates, no reliable rule → must remain null.
    sj = {"lines": ["Income for the year £120,000", "Expenditure £95,000",
                    "Year ended 31 March 2016"]}
    pred = {k["name"]: None for k in KLEISTER["fields"]}
    out = fill_missing_with_rules(pred, sj, KLEISTER)
    assert out["income_annually_in_british_pounds"] is None
    assert out["spending_annually_in_british_pounds"] is None
    assert out["report_date"] is None


def test_no_match_leaves_field_null():
    sj = {"lines": ["no useful content here"]}
    out = fill_missing_with_rules({"address__postcode": None}, sj, KLEISTER)
    assert out["address__postcode"] is None


def test_filled_value_scores_as_tp_end_to_end():
    # A recovered postcode must round-trip through the real metrics as a true positive.
    sj = {"lines": ["LS1 4AB"]}
    gold = {"address__postcode": "LS1 4AB", "charity_name": "X"}
    before = score_document(gold, {"address__postcode": None, "charity_name": "X"}, KLEISTER)
    filled = fill_missing_with_rules({"address__postcode": None, "charity_name": "X"}, sj, KLEISTER)
    after = score_document(gold, filled, KLEISTER)
    assert before["per_field"]["address__postcode"] == "missing"
    assert after["per_field"]["address__postcode"] == "tp"
    assert after["recall"] > before["recall"]


def test_handles_none_pred_from_parse_failure():
    sj = {"lines": ["LS1 4AB", "Registered charity number 1084095"]}
    out = fill_missing_with_rules(None, sj, KLEISTER)
    assert out["address__postcode"] == "LS1 4AB"
    assert out["charity_number"] == "1084095"
