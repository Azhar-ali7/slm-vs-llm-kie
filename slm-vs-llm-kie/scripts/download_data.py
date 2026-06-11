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
        "SROIE (ICDAR-2019, receipts — company/date/address/total). Text-only;\n"
        "  no registration needed. Sparse-clone just the box+key annotations:\n\n"
        "    git clone --filter=blob:none --sparse --depth 1 \\\n"
        "        https://github.com/zzzDavid/ICDAR-2019-SROIE \\\n"
        "        data/raw/sroie/_src\n"
        "    cd data/raw/sroie/_src && git sparse-checkout set data/box data/key && cd -\n\n"
        "  (Plain `git clone <repo> data/raw/sroie/_src` also works but pulls images.)\n"
        "  Then: python scripts/convert_data.py  ->  data/raw/sroie/{train,test}.jsonl\n"
    ),
    "kleister_charity": (
        "Kleister-Charity (UK charity annual reports — 8 fields). Text-only OCR;\n"
        "  NO git-annex / NO 12GB PDFs needed. Download four small blobs:\n\n"
        "    mkdir -p data/raw/kleister_charity/_src/train data/raw/kleister_charity/_src/dev-0\n"
        "    base=https://raw.githubusercontent.com/applicaai/kleister-charity/master\n"
        "    for f in train/in.tsv.xz train/expected.tsv dev-0/in.tsv.xz dev-0/expected.tsv; do\n"
        "      curl -fSL \"$base/$f\" -o \"data/raw/kleister_charity/_src/$f\"\n"
        "    done\n\n"
        "  Then: python scripts/convert_data.py  ->  data/raw/kleister_charity/{train,test}.jsonl\n"
        "  (dev-0 becomes the test split; test-A gold is withheld upstream.)\n"
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
    print("After downloading, build the JSONL and refresh the test set:")
    print("  python scripts/convert_data.py")
    print("  rm -f results/test_set_20.json   # reselect 20 real docs")
    print("  python scripts/run_eval.py --pilot")
    print("\nTip: until real data is present, everything runs on the synthetic "
          "'sample' dataset.")


if __name__ == "__main__":
    main()
