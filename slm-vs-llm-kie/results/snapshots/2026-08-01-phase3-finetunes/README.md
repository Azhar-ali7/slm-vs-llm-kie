# Snapshot — 2026-08-01 — Phase 3 (fine-tunes) full study

Frozen evidence for the dissertation. These files are the exact outputs behind the
Phase-3 findings ([E6] phi4 fine-tune, [E8] mistral fine-tune) in `../../FINDINGS.md`
and are committed (the live `results/` copies are gitignored as regenerable). This
supersedes `2026-06-15-phase2-with-large/` (which froze the pre-fine-tune standings).

| File | What |
|---|---|
| `runs.jsonl` | All 15 models × 80 cells (20 docs × 4 conditions) = **1200 rows** |
| `summary.csv` | Per-model + per-condition aggregates |
| `REPORT.md` | Auto-generated tables + findings |
| `manifest.json` | Run provenance (seed, platform, model list) |
| `plot_f1_vs_params.png` | F1 vs model size (central plot, 0.36B–671B) |
| `plot_latency_vs_params.png` | Latency vs size |
| `plot_cost_frontier.png` | Quality vs cost |
| `plot_field_heatmap.png` | Per-field F1 heatmap |

**Scope — 15 models:**

- **8 local baselines** (M1, Ollama, constrained decoding): smollm2-360m, qwen2.5-0.5b,
  gemma3-1b, llama3.2-1b, gemma2-2b, gemma3-4b, phi4-mini, mistral-7b.
- **2 QLoRA fine-tunes** (Unsloth on Kaggle, trained on the TRAIN split only, then
  imported as GGUF and run through the *identical* 20-doc grid): phi4-mini-ft,
  mistral-7b-ft.
- **5 large cloud** (AWS Bedrock, serverless per-token, prompt-instructed JSON):
  gemma3-27b (`google.gemma-3-27b-it`), deepseek-v3 (`deepseek.v3.2`, 671B), qwen3-235b,
  llama-70b (`us.meta.llama3-3-70b-instruct-v1:0`), llama4-maverick
  (`us.meta.llama4-maverick-17b-instruct-v1:0`, 400B).

**Fine-tune headlines (before → after, same frozen grid):**

Numbers below are F1_macro = mean per-cell F1 over the 80 cells (matches FINDINGS
[E6]/[E8]; do **not** use `summary.csv`'s `f1_macro_mean`, which averages the four
per-condition rows and gives slightly different figures).

- **phi4-mini → phi4-mini-ft** — the clean win ([E6]). F1_macro **0.528 → 0.669** (+0.141).
  A task-tuned 3.8B model that started as the weakest local baseline jumps to beat several
  much larger models — the [P2] dissertation claim.
- **mistral-7b → mistral-7b-ft** — mixed/instructive ([E8]). F1_macro **0.642 → 0.661**
  (+0.019, not significant), and this hides a trade-off: **SROIE +0.106** (exact-match
  8→26, tripled) while **Kleister −0.067**. The already-strong baseline had little headroom,
  and 4096-token training truncation dropped most long Kleister docs, so the model
  specialised to SROIE at Kleister's expense. (Under pooled *micro* F1 this is a slight
  regression, 0.607→0.590 — see [E8].)

**Reproduce:** local arm `python3 scripts/run_eval.py`; fine-tunes
`python3 scripts/run_eval.py --models phi4-mini-ft mistral-7b-ft` (after
`bash scripts/import_finetuned.sh …`); large arm
`export AWS_BEARER_TOKEN_BEDROCK=… && bash scripts/run_large_arm.sh`; then
`python3 scripts/make_plots.py`.
