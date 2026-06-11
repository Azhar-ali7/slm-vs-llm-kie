"""Print instructions for obtaining the datasets. Does NOT scrape or download.

The framework reads a *normalised JSONL* per dataset/split at
data/raw/<name>/<split>.jsonl — one record per line with keys:
    {"doc_id": str,
     "words": [{"text": str, "bbox": [x0,y0,x1,y1]}, ...],
     "key_values": {k: v},            # optional
     "gold": {field: value_or_None}}  # target fields for that dataset's schema

You can develop, run --pilot, and demo entirely on the synthetic `sample`
dataset without any download.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, resolve_path  # noqa: E402

INSTRUCTIONS = {
    "sroie": (
        "SROIE (ICDAR 2019, Task 3 — receipts)\n"
        "  Source: https://rrc.cvc.uab.es/?ch=13  (register, download Task 1/3 data)\n"
        "  Each receipt ships an OCR file (coordinates + text) and an entities JSON\n"
        "  with {company, date, address, total}. Convert each receipt to one JSON\n"
        "  line: words[] from the OCR boxes, gold{} from the entities file.\n"
        "  Write train -> data/raw/sroie/train.jsonl, test -> data/raw/sroie/test.jsonl\n"
    ),
    "kleister_charity": (
        "Kleister-Charity (financial reports)\n"
        "  Source: https://github.com/applicaai/kleister-charity\n"
        "  Provides OCR (words+boxes) and gold for 8 fields incl. income/spending,\n"
        "  charity number, report date, and address__* parts.\n"
        "  Convert to the same JSONL shape ->\n"
        "  data/raw/kleister_charity/train.jsonl and .../test.jsonl\n"
    ),
}


def main() -> None:
    cfg = load_config()
    print("=" * 72)
    print("Dataset acquisition instructions (no automatic download)")
    print("=" * 72)
    for name, text in INSTRUCTIONS.items():
        raw_dir = resolve_path(cfg, cfg["datasets"][name]["raw_dir"])
        present = raw_dir.exists() and any(raw_dir.iterdir())
        status = "PRESENT" if present else "MISSING"
        print(f"\n[{status}] {name}  ->  {raw_dir}")
        print(text)
    print("-" * 72)
    print("Tip: everything runs on the synthetic 'sample' dataset until you add "
          "real data.\nTry: python scripts/run_eval.py --pilot")


if __name__ == "__main__":
    main()
