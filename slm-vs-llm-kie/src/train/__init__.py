"""Phase-3 QLoRA fine-tuning support.

Pure, CPU-only helpers that turn the TRAIN splits into a supervised
fine-tuning (SFT) dataset. The actual training runs on a cloud GPU notebook
(notebooks/03_qlora_finetune.ipynb, Kaggle T4x2) — nothing in this package
needs a GPU or any CUDA dependency.
"""
from src.train.dataset import (
    assert_no_test_leakage,
    build_sft_examples,
    summarize,
    write_sft_jsonl,
)

__all__ = [
    "build_sft_examples",
    "assert_no_test_leakage",
    "summarize",
    "write_sft_jsonl",
]
