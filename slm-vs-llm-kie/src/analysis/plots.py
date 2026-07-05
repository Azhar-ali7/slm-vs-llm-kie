"""Produce the four result plots as PNGs. Degrades gracefully with sparse data."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from src.config import resolve_path  # noqa: E402


def _plots_dir(cfg: dict[str, Any]) -> Path:
    d = resolve_path(cfg, cfg["paths"]["plots_dir"])
    d.mkdir(parents=True, exist_ok=True)
    return d


def _label(r) -> str:
    """Human-readable point label; config ids are generic (frontier-llm ->
    GPT-OSS-120B), so prefer the display_name carried by per_model."""
    name = r.get("display_name") if hasattr(r, "get") else None
    return name if name else r["model_id"]


def plot_f1_vs_params(per_model_df, out: Path) -> Path | None:
    sized = per_model_df[per_model_df["params_b"].notna()]
    fig, ax = plt.subplots(figsize=(7, 5))
    if len(sized):
        ax.scatter(sized["params_b"], sized["f1_macro"], s=60, color="#2a6f97")
        for _, r in sized.iterrows():
            ax.annotate(_label(r), (r["params_b"], r["f1_macro"]),
                        fontsize=8, xytext=(4, 4), textcoords="offset points")
        ax.set_xscale("log")
    # API models without a known size: show as a reference band.
    api = per_model_df[(per_model_df["model_type"] == "api") & (per_model_df["params_b"].isna())]
    for _, r in api.iterrows():
        ax.axhline(r["f1_macro"], ls="--", color="#bc4749", alpha=0.7)
        ax.text(0.02, r["f1_macro"], f"{_label(r)} (API)", transform=ax.get_yaxis_transform(),
                color="#bc4749", fontsize=8, va="bottom")
    ax.set_xlabel("Model size (billions of parameters, log scale)")
    ax.set_ylabel("F1 (macro)")
    ax.set_title("Extraction quality vs model size")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_cost_frontier(per_model_df, out: Path) -> Path:
    """Accuracy vs cost per document, log-x so the sub-cent models are readable.

    Billed (DigitalOcean) models have a single real rate, drawn as a point.
    Local models carry a market price BAND (what renting them would cost): the
    point is the band midpoint and the horizontal bar spans low..high. Nothing
    sits at exactly $0 — self-hosting has no per-token bill, but the figure shows
    the going market rate for the same compute.
    """
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for _, r in per_model_df.iterrows():
        is_api = r["model_type"] != "local"
        color = "#bc4749" if is_api else "#2a6f97"
        x = r["cost_usd_per_doc"] if is_api else r["cost_mid_per_doc"]
        if x is None or x <= 0:
            continue
        if is_api:
            ax.scatter(x, r["f1_macro"], s=60, color=color, zorder=3)
        else:
            lo, hi = r["cost_low_per_doc"], r["cost_high_per_doc"]
            ax.errorbar(x, r["f1_macro"], xerr=[[x - lo], [hi - x]], fmt="o",
                        ms=6, color=color, ecolor=color, elinewidth=1.2,
                        capsize=3, alpha=0.85, zorder=3)
        ax.annotate(_label(r), (x, r["f1_macro"]),
                    fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("Cost per document (USD, log scale) — local = market hosting band")
    ax.set_ylabel("F1 (macro)")
    ax.set_title("Accuracy–cost frontier")
    handles = [
        Line2D([0], [0], marker="o", color="#2a6f97", ls="", label="local (hosted-rate band)"),
        Line2D([0], [0], marker="o", color="#bc4749", ls="", label="DigitalOcean (billed)"),
    ]
    ax.legend(handles=handles, fontsize=8, loc="lower right")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_latency_vs_params(per_model_df, out: Path) -> Path:
    sized = per_model_df[per_model_df["params_b"].notna()]
    fig, ax = plt.subplots(figsize=(7, 5))
    if len(sized):
        ax.scatter(sized["params_b"], sized["latency_s"], s=60, color="#386641")
        for _, r in sized.iterrows():
            ax.annotate(_label(r), (r["params_b"], r["latency_s"]),
                        fontsize=8, xytext=(4, 4), textcoords="offset points")
        ax.set_xscale("log")
    ax.set_xlabel("Model size (billions of parameters, log scale)")
    ax.set_ylabel("Mean latency per call (s)")
    ax.set_title("Latency vs model size")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_field_heatmap(field_df, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(max(7, 0.7 * len(field_df.columns) + 3),
                                    max(3, 0.5 * len(field_df.index) + 2)))
    data = field_df.fillna(0).values
    im = ax.imshow(data, cmap="YlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(field_df.columns)))
    ax.set_xticklabels(field_df.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(field_df.index)))
    ax.set_yticklabels(field_df.index, fontsize=8)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title("Per-field accuracy by model")
    fig.colorbar(im, ax=ax, fraction=0.025)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def make_all_plots(cfg: dict[str, Any], per_model_df, field_df) -> dict[str, str]:
    d = _plots_dir(cfg)
    paths = {
        "f1_vs_params": d / "plot_f1_vs_params.png",
        "cost_frontier": d / "plot_cost_frontier.png",
        "latency_vs_params": d / "plot_latency_vs_params.png",
        "field_heatmap": d / "plot_field_heatmap.png",
    }
    plot_f1_vs_params(per_model_df, paths["f1_vs_params"])
    plot_cost_frontier(per_model_df, paths["cost_frontier"])
    plot_latency_vs_params(per_model_df, paths["latency_vs_params"])
    if field_df is not None and len(field_df):
        # Relabel the heatmap rows (model ids) with human-readable names.
        from src.models.registry import model_meta
        meta = model_meta(cfg)
        field_df = field_df.rename(index={mid: m.get("display", mid) for mid, m in meta.items()})
        plot_field_heatmap(field_df, paths["field_heatmap"])
    return {k: str(v) for k, v in paths.items()}
