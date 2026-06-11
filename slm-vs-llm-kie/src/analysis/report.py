"""Write results/REPORT.md: summary tables, embedded plots, auto findings."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.analysis.aggregate import (
    load_results_df,
    per_field_accuracy,
    per_model,
    per_model_condition,
    save_summary,
)
from src.analysis.plots import make_all_plots
from src.config import resolve_path


def _df_to_md(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        return "```\n" + df.to_string(index=False) + "\n```"


def _findings(pm: pd.DataFrame) -> list[str]:
    out = []
    if pm.empty:
        return ["No results available."]
    best_f1 = pm.loc[pm["f1_macro"].idxmax()]
    out.append(f"- **Best quality:** `{best_f1['model_id']}` (F1 {best_f1['f1_macro']:.3f}).")

    free = pm[pm["cost_usd_per_doc"] == 0]
    if len(free):
        best_free = free.loc[free["f1_macro"].idxmax()]
        out.append(f"- **Best free (local) model:** `{best_free['model_id']}` "
                   f"(F1 {best_free['f1_macro']:.3f}, $0/doc).")

    paid = pm[pm["cost_usd_per_doc"] > 0]
    if len(paid) and len(free):
        gap = best_f1["f1_macro"] - best_free["f1_macro"]
        out.append(f"- **Quality gap (best API − best local):** {gap:+.3f} F1 — "
                   f"the price of going local.")

    sized = pm[pm["params_b"].notna()].sort_values("params_b")
    if len(sized) >= 2:
        low, high = sized.iloc[0], sized.iloc[-1]
        out.append(f"- **Size trend:** F1 moves from {low['f1_macro']:.3f} "
                   f"({low['model_id']}, {low['params_b']}B) to {high['f1_macro']:.3f} "
                   f"({high['model_id']}, {high['params_b']}B).")

    worst_parse = pm.loc[pm["parse_fail_rate"].idxmax()]
    if worst_parse["parse_fail_rate"] > 0:
        out.append(f"- **JSON reliability:** highest parse-failure rate is "
                   f"`{worst_parse['model_id']}` at {worst_parse['parse_fail_rate']:.0%}.")
    return out


def build_report(cfg: dict[str, Any]) -> dict[str, Any]:
    df = load_results_df(cfg)
    if df.empty:
        raise RuntimeError("No rows in runs.jsonl — run the evaluation first.")

    cond = per_model_condition(df)
    pm = per_model(df, cfg)
    field_df = per_field_accuracy(df)

    save_summary(cfg, cond)
    plots = make_all_plots(cfg, pm, field_df)

    report_path = resolve_path(cfg, cfg["paths"]["report_md"])
    plots_dir = resolve_path(cfg, cfg["paths"]["plots_dir"])

    def rel(p: str) -> str:
        return str(Path(p).relative_to(plots_dir.parent)) if plots_dir.parent in Path(p).parents else p

    synthetic = bool(df["dataset"].eq("sample").all())
    lines = []
    lines.append("# SLM vs LLM — Key Information Extraction: Results\n")
    if synthetic:
        lines.append("> ⚠️ These results are on the **synthetic `sample` dataset** "
                     "(pipeline validation only). Replace with real SROIE / "
                     "Kleister-Charity data for reportable numbers.\n")
    lines.append("## Per-model summary\n")
    lines.append(_df_to_md(pm) + "\n")
    lines.append("## Auto findings\n")
    lines += _findings(pm)
    lines.append("\n## Per-condition detail\n")
    lines.append(_df_to_md(cond) + "\n")
    lines.append("## Plots\n")
    lines.append(f"![F1 vs parameters]({Path(plots['f1_vs_params']).name})\n")
    lines.append(f"![Cost vs F1 frontier]({Path(plots['cost_frontier']).name})\n")
    lines.append(f"![Latency vs parameters]({Path(plots['latency_vs_params']).name})\n")
    if Path(plots["field_heatmap"]).exists():
        lines.append(f"![Per-field F1 heatmap]({Path(plots['field_heatmap']).name})\n")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return {
        "report_md": str(report_path),
        "summary_csv": str(resolve_path(cfg, cfg["paths"]["summary_csv"])),
        "plots": plots,
        "per_model": pm,
        "per_condition": cond,
        "per_field": field_df,
    }
