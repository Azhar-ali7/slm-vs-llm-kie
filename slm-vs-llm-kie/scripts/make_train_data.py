#!/usr/bin/env python3
"""Build the SFT dataset (data/train/sft.jsonl) for Phase-3 QLoRA fine-tuning.

CPU-only, local. Reuses the exact eval-time prompt (src/train/dataset.py), then
asserts no test-set document leaked in before writing. Upload the resulting
sft.jsonl to the Kaggle training notebook (notebooks/03_qlora_finetune.ipynb).

    python scripts/make_train_data.py                 # uses config/train.yaml
    python scripts/make_train_data.py --max-examples 50   # quick smoke build
    python scripts/make_train_data.py --all           # ignore max_per_dataset cap
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, resolve_path  # noqa: E402
from src.train import (  # noqa: E402
    assert_no_test_leakage,
    build_sft_examples,
    summarize,
    write_sft_jsonl,
)

TRAIN_CFG = Path(__file__).resolve().parent.parent / "config" / "train.yaml"


def main() -> None:
    ap = argparse.ArgumentParser(description="Build SFT data from the train splits.")
    ap.add_argument("--train-config", default=str(TRAIN_CFG),
                    help="Path to config/train.yaml.")
    ap.add_argument("--out", default=None, help="Override output path.")
    ap.add_argument("--all", action="store_true",
                    help="Use every train doc (ignore data.max_per_dataset).")
    ap.add_argument("--max-examples", type=int, default=None,
                    help="Truncate the final example list (smoke build).")
    args = ap.parse_args()

    cfg = load_config()
    tcfg = yaml.safe_load(Path(args.train_config).read_text(encoding="utf-8"))
    dcfg = tcfg.get("data", {})

    out = args.out or dcfg.get("out_path", "data/train/sft.jsonl")
    out_path = resolve_path(cfg, out)
    datasets = dcfg.get("datasets")
    max_per_dataset = None if args.all else dcfg.get("max_per_dataset")

    print(f"Building SFT data (datasets={datasets or 'default'}, "
          f"max_per_dataset={max_per_dataset}) …")
    examples = build_sft_examples(
        cfg, datasets=datasets, max_per_dataset=max_per_dataset
    )

    # Hygiene gate BEFORE writing anything.
    assert_no_test_leakage(cfg, examples)
    print("  ✓ leakage check passed — 0 test-set docs in SFT data")

    if args.max_examples is not None:
        examples = examples[: args.max_examples]

    write_sft_jsonl(examples, out_path)

    s = summarize(examples)
    print(f"\nWrote {s['total']} examples ({s['unique_docs']} unique docs) -> {out_path}")
    print(f"  by dataset: {s['by_dataset']}")
    print(f"  by variant: {s['by_variant']}")
    print("\nNext: upload this file to notebooks/03_qlora_finetune.ipynb (Kaggle).")


if __name__ == "__main__":
    main()
