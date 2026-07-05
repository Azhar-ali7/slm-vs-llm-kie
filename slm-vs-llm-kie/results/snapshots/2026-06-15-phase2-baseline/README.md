# Snapshot — 2026-06-15 — Phase 2 (mid-sem) local baseline

Frozen evidence for the dissertation. These files are the exact outputs behind the
[E4] standings in `../../FINDINGS.md` and are committed (the live `results/` copies are
gitignored as regenerable).

| File | What |
|---|---|
| `runs.jsonl` | All 8 local models × 80 cells (20 docs × 4 conditions), structured-output run |
| `summary.csv` | Per-model + per-condition aggregates |
| `REPORT.md` | Auto-generated tables + findings |
| `plot_f1_vs_params.png` | F1 vs model size (central plot) |
| `plot_latency_vs_params.png` | Latency vs size |
| `plot_cost_frontier.png` | Quality vs cost |
| `plot_field_heatmap.png` | Per-field F1 heatmap |

**Scope:** 8 local models on M1 (Ollama, constrained decoding). The large arm
(GPT-4o, Claude Haiku 4.5 on DO serverless) is not yet in this snapshot — it will get its
own snapshot once `DO_INFERENCE_KEY` is set and that run completes.

**Reproduce:** `python3 scripts/run_eval.py` then `python3 scripts/make_plots.py`.
