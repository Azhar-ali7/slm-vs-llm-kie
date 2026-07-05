# Snapshot — 2026-06-15 — Phase 2 (mid-sem) full study WITH large arm

Frozen evidence for the dissertation. These files are the exact outputs behind the
[E5] standings in `../../FINDINGS.md` and are committed (the live `results/` copies are
gitignored as regenerable). This supersedes `2026-06-15-phase2-baseline/` (which froze
the 8-local-model run before the large arm was added).

| File | What |
|---|---|
| `runs.jsonl` | All 11 models × 80 cells (20 docs × 4 conditions) = **880 rows** |
| `summary.csv` | Per-model + per-condition aggregates |
| `REPORT.md` | Auto-generated tables + findings |
| `plot_f1_vs_params.png` | F1 vs model size (central plot, 0.36B–120B) |
| `plot_latency_vs_params.png` | Latency vs size |
| `plot_cost_frontier.png` | Quality vs cost |
| `plot_field_heatmap.png` | Per-field F1 heatmap |

**Scope:** 11 models — 8 local on M1 (Ollama, constrained decoding) + 3 large on
DigitalOcean serverless inference (prompt-instructed JSON):
- gemma4-31b → `gemma-4-31B-it` (31B)
- large-open-llm → `llama3.3-70b-instruct` (70B)
- frontier-llm → `openai-gpt-oss-120b` (120B)

The large arm uses DO-hosted **open** models (GPT-4o/Claude require DO Tier 3+; this
account is Tier 1/2 — see [E5]). Whole large arm cost ~$0.18 on the $200 DO credit.

**Headline:** quality does not track size — gemma4-31b (F1 0.701) beats both the 70B and
the 120B; a 4B local model ties the 120B cloud model. See `../../FINDINGS.md` §2.

**Reproduce:** local arm `python3 scripts/run_eval.py`; large arm
`export DO_INFERENCE_KEY=... && bash scripts/run_large_arm.sh`; then `python3 scripts/make_plots.py`.
