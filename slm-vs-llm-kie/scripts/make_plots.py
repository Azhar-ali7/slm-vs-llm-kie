"""Aggregate runs.jsonl -> summary.csv + 4 plots + REPORT.md."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.report import build_report  # noqa: E402
from src.config import load_config  # noqa: E402


def main() -> None:
    cfg = load_config()
    out = build_report(cfg)
    print("Wrote:")
    print(f"  {out['summary_csv']}")
    print(f"  {out['report_md']}")
    for name, p in out["plots"].items():
        print(f"  {p}")


if __name__ == "__main__":
    main()
