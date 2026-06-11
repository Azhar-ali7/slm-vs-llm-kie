"""Core invariant: the dataset converter and the Textract converter must emit
identical simple JSON for the same document."""
from __future__ import annotations

from src.data.preprocess import from_dataset_record, from_textract

# Canonical word layout (text, x0, y0, x1, y1) — both inputs are derived from this.
WORDS = [
    ("ACME",       0.10, 0.10, 0.20, 0.14),
    ("STORE",      0.22, 0.10, 0.32, 0.14),
    ("Date",       0.10, 0.20, 0.18, 0.24),   # KV key
    ("2021-07-29", 0.25, 0.20, 0.45, 0.24),   # KV value
    ("TOTAL",      0.10, 0.30, 0.20, 0.34),
    ("42.50",      0.30, 0.30, 0.40, 0.34),
    ("Item",       0.10, 0.40, 0.18, 0.44),   # table r1c1
    ("Qty",        0.40, 0.40, 0.48, 0.44),   # table r1c2
    ("Pen",        0.10, 0.50, 0.18, 0.54),   # table r2c1
    ("2",          0.40, 0.50, 0.45, 0.54),   # table r2c2
]

EXPECTED_LINES = ["ACME STORE", "Date 2021-07-29", "TOTAL 42.50", "Item Qty", "Pen 2"]
EXPECTED_KV = {"Date": "2021-07-29"}
EXPECTED_TABLES = [[["Item", "Qty"], ["Pen", "2"]]]


def _dataset_record() -> dict:
    return {
        "doc_id": "doc1",
        "page": 1,
        "words": [{"text": t, "bbox": [x0, y0, x1, y1]} for (t, x0, y0, x1, y1) in WORDS],
        "key_values": {"Date": "2021-07-29"},
        "tables": [[["Item", "Qty"], ["Pen", "2"]]],
    }


def _word_block(wid: str, text: str, x0: float, y0: float, x1: float, y1: float) -> dict:
    return {
        "Id": wid,
        "BlockType": "WORD",
        "Text": text,
        "Geometry": {"BoundingBox": {"Left": x0, "Top": y0, "Width": x1 - x0, "Height": y1 - y0}},
    }


def _textract_response() -> dict:
    blocks: list[dict] = []
    ids = []
    for i, (t, x0, y0, x1, y1) in enumerate(WORDS):
        wid = f"w{i}"
        ids.append(wid)
        blocks.append(_word_block(wid, t, x0, y0, x1, y1))

    # KEY_VALUE_SET: key=w2 ("Date") -> value set -> w3 ("2021-07-29")
    blocks.append({
        "Id": "kv_key", "BlockType": "KEY_VALUE_SET", "EntityTypes": ["KEY"],
        "Relationships": [
            {"Type": "CHILD", "Ids": ["w2"]},
            {"Type": "VALUE", "Ids": ["kv_val"]},
        ],
    })
    blocks.append({
        "Id": "kv_val", "BlockType": "KEY_VALUE_SET", "EntityTypes": ["VALUE"],
        "Relationships": [{"Type": "CHILD", "Ids": ["w3"]}],
    })

    # TABLE: 2x2 from w6..w9
    blocks.append({
        "Id": "table1", "BlockType": "TABLE",
        "Relationships": [{"Type": "CHILD", "Ids": ["c11", "c12", "c21", "c22"]}],
    })
    cells = [
        ("c11", 1, 1, "w6"), ("c12", 1, 2, "w7"),
        ("c21", 2, 1, "w8"), ("c22", 2, 2, "w9"),
    ]
    for cid, r, c, wid in cells:
        blocks.append({
            "Id": cid, "BlockType": "CELL", "RowIndex": r, "ColumnIndex": c,
            "Relationships": [{"Type": "CHILD", "Ids": [wid]}],
        })

    return {"doc_id": "doc1", "Blocks": blocks}


def test_dataset_converter_shape():
    sj = from_dataset_record(_dataset_record())
    assert sj["doc_id"] == "doc1"
    assert sj["page"] == 1
    assert sj["lines"] == EXPECTED_LINES
    assert sj["key_values"] == EXPECTED_KV
    assert sj["tables"] == EXPECTED_TABLES


def test_textract_converter_shape():
    sj = from_textract(_textract_response())
    assert sj["lines"] == EXPECTED_LINES
    assert sj["key_values"] == EXPECTED_KV
    assert sj["tables"] == EXPECTED_TABLES


def test_converters_agree():
    """The whole point: same document → identical simple JSON from both paths."""
    assert from_dataset_record(_dataset_record()) == from_textract(_textract_response())


def test_textract_accepts_bare_block_list():
    resp = _textract_response()
    sj_list = from_textract(resp["Blocks"])
    assert sj_list["lines"] == EXPECTED_LINES  # doc_id falls back, lines identical


def test_deterministic_repeat():
    rec = _dataset_record()
    assert from_dataset_record(rec) == from_dataset_record(rec)


def test_empty_document():
    sj = from_dataset_record({"doc_id": "empty", "words": []})
    assert sj == {"doc_id": "empty", "page": 1, "lines": [], "key_values": {}, "tables": []}
