"""Generate report raw materials from runs.jsonl:
  summary.csv, REPORT.md, the 4 result plots, and the architecture/pipeline
  figures (report/figures/). Paste these into a report you write yourself.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.report import build_report  # noqa: E402
from src.analysis.report_figures import make_figures  # noqa: E402
from src.config import load_config, resolve_path  # noqa: E402


def main() -> None:
    cfg = load_config()
    out = build_report(cfg)
    figures = make_figures(resolve_path(cfg, cfg["paths"]["report_dir"]) / "figures")

    print("Wrote (raw materials for your report):")
    print(f"  {out['summary_csv']}")
    print(f"  {out['report_md']}")
    for _, p in out["plots"].items():
        print(f"  {p}")
    for _, p in figures.items():
        print(f"  {p}")


if __name__ == "__main__":
    main()
