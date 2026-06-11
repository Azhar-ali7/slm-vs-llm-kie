"""parse.py must recover JSON from messy model output and fail gracefully."""
from __future__ import annotations

from src.eval.parse import extract_json_text, parse_prediction

SCHEMA = {
    "fields": [
        {"name": "company", "type": "string", "required": True},
        {"name": "date", "type": "date", "required": True},
        {"name": "total", "type": "currency", "required": True},
    ]
}


def test_plain_json():
    pred, err = parse_prediction('{"company": "ACME", "date": "2021-01-01", "total": "9.99"}', SCHEMA)
    assert err is None
    assert pred["company"] == "ACME" and pred["total"] == "9.99"


def test_markdown_fenced_json():
    text = "Here you go:\n```json\n{\"company\": \"ACME\", \"date\": null, \"total\": \"5\"}\n```\nThanks!"
    pred, err = parse_prediction(text, SCHEMA)
    assert err is None
    assert pred["company"] == "ACME"
    assert pred["date"] is None


def test_prose_around_object():
    text = 'The extracted fields are {"company": "Foo", "date": "2020-02-02", "total": "1.00"} as requested.'
    assert extract_json_text(text) == '{"company": "Foo", "date": "2020-02-02", "total": "1.00"}'
    pred, err = parse_prediction(text, SCHEMA)
    assert err is None and pred["company"] == "Foo"


def test_missing_fields_become_null():
    pred, err = parse_prediction('{"company": "OnlyCo"}', SCHEMA)
    assert err is None
    assert pred["company"] == "OnlyCo"
    assert pred["date"] is None and pred["total"] is None


def test_numeric_values_coerced_to_string():
    pred, err = parse_prediction('{"company": "X", "date": "2021-01-01", "total": 42.5}', SCHEMA)
    assert err is None
    assert pred["total"] == "42.5"


def test_extra_keys_noted_and_dropped():
    pred, err = parse_prediction('{"company": "X", "date": null, "total": null, "vat": "1.0"}', SCHEMA)
    assert err is None
    assert "vat" not in pred or pred.get("vat") is None
    assert pred.get("_extras") == ["vat"]


def test_no_json_returns_error():
    pred, err = parse_prediction("I could not find the information.", SCHEMA)
    assert pred is None
    assert err["type"] == "no_json"


def test_broken_json_returns_error():
    pred, err = parse_prediction('{"company": "X", "date": ', SCHEMA)
    assert pred is None
    assert err["type"] in {"no_json", "json_decode"}
