"""Produce the four result plots as PNGs. Degrades gracefully with sparse data."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402

from src.config import resolve_path  # noqa: E402


def _plots_dir(cfg: dict[str, Any]) -> Path:
    d = resolve_path(cfg, cfg["paths"]["plots_dir"])
    d.mkdir(parents=True, exist_ok=True)
    return d


def plot_f1_vs_params(per_model_df, out: Path) -> Path | None:
    sized = per_model_df[per_model_df["params_b"].notna()]
    fig, ax = plt.subplots(figsize=(7, 5))
    if len(sized):
        ax.scatter(sized["params_b"], sized["f1_macro"], s=60, color="#2a6f97")
        for _, r in sized.iterrows():
            ax.annotate(r["model_id"], (r["params_b"], r["f1_macro"]),
                        fontsize=8, xytext=(4, 4), textcoords="offset points")
        ax.set_xscale("log")
    # API models without a known size: show as a reference band.
    api = per_model_df[(per_model_df["model_type"] == "api") & (per_model_df["params_b"].isna())]
    for _, r in api.iterrows():
        ax.axhline(r["f1_macro"], ls="--", color="#bc4749", alpha=0.7)
        ax.text(0.02, r["f1_macro"], f"{r['model_id']} (API)", transform=ax.get_yaxis_transform(),
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
    fig, ax = plt.subplots(figsize=(7, 5))
    for _, r in per_model_df.iterrows():
        color = "#bc4749" if r["model_type"] == "api" else "#2a6f97"
        ax.scatter(r["cost_usd_per_doc"], r["f1_macro"], s=60, color=color)
        ax.annotate(r["model_id"], (r["cost_usd_per_doc"], r["f1_macro"]),
                    fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Cost per document (USD) — local models = 0")
    ax.set_ylabel("F1 (macro)")
    ax.set_title("Accuracy–cost frontier")
    ax.grid(True, alpha=0.3)
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
            ax.annotate(r["model_id"], (r["params_b"], r["latency_s"]),
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
        plot_field_heatmap(field_df, paths["field_heatmap"])
    return {k: str(v) for k, v in paths.items()}
