"""Generate report/MidSem_Report.docx in the BITS Mid-Sem format.

If results exist (results/runs.jsonl), the per-model table and result plots are
embedded. Otherwise a structure-complete report with the architecture/pipeline
figures is produced (fill in results later by re-running).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.report_doc import build_docx  # noqa: E402
from src.config import load_config, resolve_path  # noqa: E402


def main() -> None:
    cfg = load_config()
    results = None
    synthetic = True
    runs_path = resolve_path(cfg, cfg["paths"]["runs_jsonl"])
    if runs_path.exists() and runs_path.stat().st_size > 0:
        try:
            from src.analysis.report import build_report
            results = build_report(cfg)
            synthetic = bool(results["per_model"]["model_type"].isin(["mock"]).any()) or \
                _is_sample_only(cfg)
        except Exception as exc:  # report still builds without results
            print(f"[warn] could not build results section: {exc}")

    out = build_docx(cfg, results=results, synthetic=synthetic)
    print(f"Wrote {out}")
    if results is None:
        print("(No results embedded — run `python scripts/run_eval.py --pilot --mock` "
              "or a real run, then re-run this.)")


def _is_sample_only(cfg) -> bool:
    from src.runner.results_store import load_rows
    rows = load_rows(resolve_path(cfg, cfg["paths"]["runs_jsonl"]))
    return bool(rows) and all(r.get("dataset") == "sample" for r in rows)


if __name__ == "__main__":
    main()
