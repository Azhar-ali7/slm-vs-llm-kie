"""Converters for the real SROIE / Kleister files (parsers + tiny end-to-end)."""
from __future__ import annotations

import json
import lzma

from src.data.convert import (
    kleister_records_from_tsv,
    parse_kleister_expected,
    parse_sroie_box_line,
    select_lines,
    sroie_records_from_dirs,
)
from src.data.preprocess import from_dataset_record


def test_parse_sroie_box_line_keeps_comma_in_transcript():
    text, bbox = parse_sroie_box_line("10,20,100,20,100,50,10,50,ACME STORE, LTD")
    assert text == "ACME STORE, LTD"
    assert bbox == [10.0, 20.0, 100.0, 50.0]


def test_parse_sroie_box_line_rejects_short():
    assert parse_sroie_box_line("not,a,box") is None


def test_parse_kleister_expected_decodes_underscores():
    line = ("address__postcode=WR12_7NL charity_name=Wormington_Village_Society "
            "charity_number=1155074 income_annually_in_british_pounds=10348000.00 "
            "report_date=2018-07-31")
    g = parse_kleister_expected(line)
    assert g["charity_name"] == "Wormington Village Society"
    assert g["address__postcode"] == "WR12 7NL"
    assert g["charity_number"] == "1155074"
    assert g["income_annually_in_british_pounds"] == "10348000.00"
    assert g["report_date"] == "2018-07-31"


def test_select_lines_flat_text_keeps_head_and_cues():
    # No newlines -> chunked into pseudo-lines; cue ('income') line retained.
    text = " ".join(["alpha"] * 200) + " total income 452000 pounds " + " ".join(["beta"] * 200)
    lines = select_lines(text, head=5, max_lines=50, tokens_per_line=10)
    assert len(lines) <= 50
    assert any("income" in ln for ln in lines)


def test_sroie_end_to_end(tmp_path):
    box = tmp_path / "box"
    key = tmp_path / "key"
    box.mkdir()
    key.mkdir()
    (box / "001.txt").write_text(
        "10,10,200,10,200,40,10,40,MAPLE CAFE\n"
        "10,60,200,60,200,90,10,90,TOTAL 18.40\n", encoding="utf-8")
    (key / "001.txt").write_text(
        json.dumps({"company": "MAPLE CAFE", "date": "2021-03-04",
                    "address": "12 Oak Street", "total": "18.40"}), encoding="utf-8")

    records = sroie_records_from_dirs(box, key, split="test")
    assert len(records) == 1
    rec = records[0]
    assert rec["gold"]["company"] == "MAPLE CAFE"
    sj = from_dataset_record(rec)
    assert "MAPLE CAFE" in sj["lines"][0]


def test_kleister_end_to_end(tmp_path):
    in_xz = tmp_path / "in.tsv.xz"
    exp = tmp_path / "expected.tsv"
    # 6 cols: filename, keys, djvu, tesseract, textract, combined
    rows = [
        "doc1.pdf\tcharity_name\t\t\t\tHope Trust registered charity income 452000 year ended 2020",
        "doc2.pdf\tcharity_name\t\t\t\tBright Fund total expenditure 90250 postcode NR2 1RY",
    ]
    with lzma.open(in_xz, "wt", encoding="utf-8") as fh:
        fh.write("\n".join(rows) + "\n")
    exp.write_text(
        "charity_name=Hope_Trust income_annually_in_british_pounds=452000.00 report_date=2020-03-31\n"
        "charity_name=Bright_Fund spending_annually_in_british_pounds=90250.00\n",
        encoding="utf-8")

    records = kleister_records_from_tsv(in_xz, exp, split="test", head=5, max_lines=50)
    assert len(records) == 2
    assert records[0]["doc_id"] == "doc1.pdf"
    assert records[0]["gold"]["charity_name"] == "Hope Trust"
    assert records[0]["lines"]  # non-empty selected lines
    sj = from_dataset_record(records[0])
    assert any("income" in ln.lower() for ln in sj["lines"])
