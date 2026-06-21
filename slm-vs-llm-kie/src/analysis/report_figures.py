"""Generate the architecture (Figure 1) and pipeline (Figure 2) diagrams."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402


def _flow(boxes: list[str], out: Path, title: str, cols: int = 4,
          color: str = "#2a6f97") -> Path:
    rows = (len(boxes) + cols - 1) // cols
    fig, ax = plt.subplots(figsize=(2.6 * cols, 1.7 * rows + 0.6))
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.axis("off")
    ax.set_title(title, fontsize=12, weight="bold", pad=12)

    centers = []
    for i, label in enumerate(boxes):
        r = i // cols
        c = i % cols
        # serpentine so arrows flow left->right, then wrap
        col = c if r % 2 == 0 else (cols - 1 - c)
        x = col + 0.5
        y = rows - 1 - r + 0.5
        box = FancyBboxPatch((x - 0.42, y - 0.32), 0.84, 0.64,
                             boxstyle="round,pad=0.02", linewidth=1.4,
                             edgecolor=color, facecolor="#eaf2f8")
        ax.add_patch(box)
        ax.text(x, y, label, ha="center", va="center", fontsize=8.5, wrap=True)
        centers.append((x, y, col, r))

    for i in range(len(centers) - 1):
        x0, y0, _, r0 = centers[i]
        x1, y1, _, r1 = centers[i + 1]
        arrow = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="->",
                                mutation_scale=14, color="#555",
                                connectionstyle="arc3,rad=0.0" if r0 == r1 else "arc3,rad=0.2")
        ax.add_patch(arrow)

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def architecture_diagram(out: Path) -> Path:
    boxes = [
        "Data layer\n(loaders, gt)",
        "Preprocess\n→ simple JSON",
        "Prompt builder\n(schema + JSON)",
        "Model registry\n(Ollama / DigitalOcean)",
        "Parse\n(JSON + validate)",
        "Metrics\n(P/R/F1, exact)",
        "Efficiency\n(latency/mem/cost)",
        "Analysis\n(plots, report)",
    ]
    return _flow(boxes, out, "Figure 1: Framework modules", cols=4, color="#2a6f97")


def pipeline_diagram(out: Path) -> Path:
    boxes = [
        "Document",
        "Textract / OCR",
        "Preprocess →\nsimple JSON",
        "Build prompt",
        "Run model\n(local | API)",
        "Parse JSON",
        "Compare to gold",
        "Metrics + plots",
    ]
    return _flow(boxes, out, "Figure 2: Functional pipeline", cols=4, color="#386641")


def make_figures(figures_dir: Path) -> dict[str, Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    return {
        "architecture": architecture_diagram(figures_dir / "fig1_architecture.png"),
        "pipeline": pipeline_diagram(figures_dir / "fig2_pipeline.png"),
    }
