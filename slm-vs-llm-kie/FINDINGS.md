# Experiment Log — SLM vs LLM for Key Information Extraction

> **Living document.** This is the lab notebook for the dissertation. It records the
> initial conditions, every change made to the system, and the measured effect of that
> change (before → after), with pointers to the artifact that proves it. Newest entries
> are appended to the **Changelog** at the bottom. Auto-generated numbers live in
> `results/REPORT.md` (overwritten each run); the *narrative* lives here.

- **Project:** Comparing small local LLMs (0.36B–7B, Ollama) vs large cloud LLMs on
  Key Information Extraction (KIE) from financial documents.
- **Author:** Azhar Ali · BITS ID 2024AA05791 · AIMLCZG628T (M.Tech AI & ML)
- **Last updated:** 2026-06-13

---

## 1. Initial conditions (the fixed experimental setup)

These do not change between runs — they define the bench.

| Aspect | Setting |
|---|---|
| **Hardware** | Apple M1, 8 GB unified memory (RAM near-full); all data/code on USB SSD |
| **Local inference** | Ollama REST `/api/generate`, `temperature=0.0`, `seed=42`, `num_predict=512` |
| **Datasets** | **SROIE** (ICDAR-2019 receipts, 4 fields, *simple*) + **Kleister-Charity** (UK charity annual reports, 8 fields, *complex*) — real data, text-only (no images/PDFs) |
| **Test set** | 20 docs, seed-fixed: 10 SROIE + 10 Kleister (`results/test_set_20.json`) |
| **Condition grid** | 2 shot modes (zero/few, k=2) × 2 input variants (`lines_only` / `lines_plus_kv`) × n_samples=1 = **4 cells/doc** |
| **Cells per model** | 20 docs × 4 conditions = **80 rows** |
| **Long-doc cap** | `max_input_lines=120` (bounds Kleister context for 8 GB + cost) |
| **Metrics** | Precision / Recall / F1 (macro & micro), exact-match rate, parse-fail rate; efficiency: latency, peak RSS, tokens/s, cost |

**Models under test**

| id | params | where it runs | status |
|---|---|---|---|
| smollm2-360m | 0.36B | local (Ollama) | ✅ run |
| qwen2.5-0.5b | 0.5B | local (Ollama) | ✅ run |
| llama3.2-1b | 1.0B | local (Ollama) | ✅ run |
| gemma2-2b | 2.0B | local (Ollama) | ✅ run |
| phi4-mini | 3.8B | DO GPU droplet (Ollama, destroy after) | ⏳ pending |
| mistral-7b | 7.0B | DO GPU droplet (Ollama, destroy after) | ⏳ pending |
| frontier-llm (**GPT-4o**) | frontier | OpenRouter | ⏳ pending |
| large-open-llm (**Claude Haiku 4.5**) | frontier | OpenRouter | ⏳ pending |

---

## 2. Current standings (4 local models, structured-output run)

Source: `results/runs.jsonl` → `results/summary.csv` / `results/REPORT.md`.

| Model | F1 (macro) | Exact-match | Parse-fail | Latency | Peak mem |
|---|---|---|---|---|---|
| smollm2-360m | 0.306 | 0/80 | **0** | ~3.9 s | ~1.1 GB |
| qwen2.5-0.5b | 0.381 | 0/80 | 0 | ~2.0 s | ~1.5 GB |
| llama3.2-1b | 0.277 | 0/80 | 0 | ~4.0 s | ~2.1 GB |
| gemma2-2b | **0.611** | 9/80 | 0 | ~6.9 s | ~2.5 GB |

**Headline so far:** clear size→quality trend among small models; gemma2-2b is the best
free local model. Frontier/cloud models still needed to complete the "small vs large" axis.

---

## 3. Artifact index (what proves what)

| Artifact | What it is |
|---|---|
| `results/runs.jsonl` | Current run, one row per cell (320 rows) — **structured-output** |
| `results/runs_baseline.jsonl` | Frozen baseline (320 rows) — **no structured output**, for the A/B |
| `results/summary.csv` | Per-model + per-condition aggregates |
| `results/REPORT.md` | Auto-generated tables + auto-findings (regenerated each run) |
| `results/plot_f1_vs_params.png` | F1 vs model size (the central plot) |
| `results/plot_latency_vs_params.png` | Latency vs size |
| `results/plot_cost_frontier.png` | Quality vs cost trade-off |
| `results/plot_field_heatmap.png` | Per-field F1 heatmap (which fields are hard) |
| `results/manifest.json` | Run provenance (seed, platform, models, datasets) |

---

## 4. Changelog (interventions & their measured effect)

Each entry: **what changed**, **why (hypothesis)**, **result (before→after)**, **artifact**.

### [E1] 2026-06-12 — Baseline: 4 local models, free-form JSON output
- **What:** First full run of the 4 small local models across the full 20×4 grid, asking
  the model to emit JSON in plain text (no decoding constraint).
- **Hypothesis:** Establish a reference point; expect tiny models to struggle with valid JSON.
- **Result:** F1 0.250 / 0.287 / 0.251 / 0.592 (smollm2 / qwen / llama / gemma).
  **smollm2-360m failed to produce parseable JSON on 31/80 cells (39%)** — a major
  reliability problem for the smallest model. Others parsed cleanly.
- **Artifact:** `results/runs_baseline.jsonl`.

### [E2] 2026-06-12 — Constrained JSON decoding (Ollama structured output)
- **What:** Enabled `ollama.structured_output: true`. The runner now passes a JSON
  **schema** as Ollama's `format` field, so decoding is grammar-constrained to emit an
  object with exactly the schema's fields (all keys present, valid JSON guaranteed).
  Implemented `json_schema_format()` in `src/prompts/builder.py`; plumbed `response_format`
  through every runner. Applies to **local models only**.
- **Hypothesis:** Forcing the grammar will eliminate parse failures and lift the weakest
  models most (their errors were largely malformed output, not wrong content).
- **Result — improvement confirmed (A/B vs E1):**

  | Model | F1 before → after | Δ | Parse errors |
  |---|---|---|---|
  | smollm2-360m | 0.250 → **0.306** | **+5.6 pp** | **31 → 0** |
  | qwen2.5-0.5b | 0.287 → **0.381** | **+9.4 pp** | 0 → 0 |
  | llama3.2-1b | 0.251 → **0.277** | +2.6 pp | 0 → 0 |
  | gemma2-2b | 0.592 → **0.611** | +1.9 pp | 0 → 0 |

- **Interpretation:** Every model improved; gains are **largest on the smallest/weakest
  models** and smallest on the already-reliable gemma2-2b — consistent with the theory
  that constrained decoding closes the *reliability* gap, not the *reasoning* gap. The
  smollm2 31→0 parse-error collapse is the cleanest evidence. **Zero extra cost.**
- **Artifact:** `results/runs.jsonl` (after) vs `results/runs_baseline.jsonl` (before).
- **Dissertation value:** A self-contained finding for the Discussion — *"constrained
  decoding is a free, high-ROI intervention that disproportionately benefits sub-1B models."*

### [E3] 2026-06-15 — Compute scope locked to **local M1 + DigitalOcean only**
- **What / decision:** Every model runs on either the M1 (Ollama) or DigitalOcean —
  no OpenRouter, no Azure, no direct OpenAI/Anthropic. Final allocation:
  - **Small (smollm2, qwen, llama, gemma)** → M1 Ollama (done).
  - **Mid (phi4-mini 3.8B, mistral-7b 7B)** → **M1 Ollama** — same constrained-decoding
    stack as the small models (most apples-to-apples); not in DO's catalog anyway.
    Switched both `type: foundry` → `type: local`.
  - **Large (GPT-4o, Claude Haiku 4.5)** → **DigitalOcean serverless inference**.
- **Why this is optimal under the constraint:** DO serverless inference is OpenAI-compatible
  (`https://inference.do-ai.run/v1`), hosts GPT-4o (`openai-gpt-4o`) and Claude Haiku 4.5
  (`anthropic-claude-haiku-4.5`) directly, is **pay-per-token (no idle/hourly billing)**, and
  bills to the **$200 credit** — keeping the strong frontier large arm with **no GPU droplet**
  to provision or remember to destroy. (The droplet idea was dropped for exactly that
  hourly-billing risk; the earlier OpenRouter wiring was removed per the no-OpenRouter rule.)
- **Code:** Added `src/models/do_runner.py` (`DigitalOceanRunner`, trimmed OpenAI-compatible
  client) + `type: do_serverless` in the registry; `api.digitalocean` block in config. Made
  the Ollama host env-overridable (`OLLAMA_HOST`) — harmless, kept for flexibility.
- **Note (method):** Constrained/structured decoding fires only for `kind == "local"`
  (`run.py:128`); the large models use prompt-instructed JSON. Intentional, and itself a
  finding — *constrained decoding is the SLM equalizer; frontier models don't need it.*
- **Cost:** mid arm **$0** (local); large arm ~120K in / ~40K out per model → **~$1 total**
  on the DO credit.
- **Prereq:** create a DO **model access key** in the Control Panel → `export DO_INFERENCE_KEY=...`.
- **Result:** ⏳ pending run.
- **Artifact:** `config/config.yaml`, `src/models/do_runner.py`, `src/models/registry.py`.

---

## 5. Planned / candidate experiments (not yet run)

- **[P1] Run the mid + large arms** to complete the small-vs-large axis. *Placement decided
  (see [E3]):* mid (phi4-mini, mistral-7b) on M1 Ollama; large (GPT-4o, Claude Haiku 4.5) on
  DO serverless. Remaining: `export DO_INFERENCE_KEY`, `ollama pull phi4-mini mistral`, run.
- **[P2] Drop few-shot for the smallest models** — llama3.2-1b collapsed under few_shot
  (F1 0.025 vs 0.47 zero-shot; the 2 examples blew its context). Hypothesis: tiny models
  do better zero-shot. Worth an explicit per-model shot-mode comparison.
- **[P3] Self-consistency voting** (n_samples=3, majority vote per field) — does sampling
  + vote beat a single greedy decode for small models?
- **[P4] Hybrid regex fallback** — fill nulls with rule-based extraction (dates, £ amounts,
  postcodes) when the model leaves a field empty; measure recall lift.
- **[P5] Per-field error analysis** — use `plot_field_heatmap.png` to identify which fields
  (e.g. Kleister income/expenditure) drive most of the loss.
