# Experiment Log — SLM vs LLM for Key Information Extraction

> **Living document.** This is the lab notebook for the dissertation. It records the
> initial conditions, every change made to the system, and the measured effect of that
> change (before → after), with pointers to the artifact that proves it. Newest entries
> are appended to the **Changelog** at the bottom. Auto-generated numbers live in
> `results/REPORT.md` (overwritten each run); the *narrative* lives here.

- **Project:** Comparing small local LLMs (0.36B–7B, Ollama) vs large cloud LLMs on
  Key Information Extraction (KIE) from financial documents.
- **Author:** Azhar Ali · BITS ID 2024AA05791 · AIMLCZG628T (M.Tech AI & ML)
- **Last updated:** 2026-07-28
- **Dissertation phases:** (1) Abstract ✅ done · (2) Mid-sem ✅ done (baseline
  small-vs-large comparison + constrained decoding) · (3) **Final viva — started**
  (fine-tuning & recall-lift improvements; QLoRA [P2] phi4-mini done, see [E6]). This
  document is the single running record across all three.

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
| gemma3-1b | 1.0B | local (Ollama) | ✅ run |
| llama3.2-1b | 1.0B | local (Ollama) | ✅ run |
| gemma2-2b | 2.0B | local (Ollama) | ✅ run |
| gemma3-4b | 4.0B | local (Ollama) | ✅ run |
| phi4-mini | 3.8B | local (Ollama, M1) | ✅ run |
| mistral-7b | 7.0B | local (Ollama, M1) | ✅ run |
| gemma4-31b (**gemma-4-31B-it**) | 31B | DO serverless | ✅ run |
| large-open-llm (**llama3.3-70b-instruct**) | 70B | DO serverless | ✅ run |
| frontier-llm (**openai-gpt-oss-120b**) | 120B | DO serverless | ✅ run |

> **Large-arm model choice (revised):** GPT-4o and Claude Haiku 4.5 require DO subscription
> **Tier 3+**; this account is Tier 1/2, so the large arm uses DO-hosted **open** models
> instead (gemma-4-31B, llama3.3-70b, gpt-oss-120b). This is also a *methodological gain* —
> fully open/reproducible large models, no closed-weights dependency. See [E5].

---

## 2. Current standings — 11 models (8 local on M1 + 3 large on DO serverless)

Source: `results/runs.jsonl` → `results/summary.csv` / `results/REPORT.md`. All 80 cells
(20 docs × 4 conditions) per model; **0 parse failures except large-open-llm (1/80)**.
Ranked by F1. Peak mem is local-only (RSS on M1); the large arm runs remotely (`n/a`).
`$tot` is the whole-grid DigitalOcean cost on the $200 credit (local = free).

| Model | arm | params | F1 | P / R | Exact-match | Med. latency | Peak mem | $tot | zero / few-shot |
|---|---|---|---|---|---|---|---|---|---|
| **gemma4-31b** | large | 31B | **0.701** | .75 / .67 | **15/80** | **1.8 s** | n/a | 0.043 | 0.683 / 0.718 |
| large-open-llm (llama3.3) | large | 70B | 0.667 | .72 / .63 | 10/80 | 11.6 s | n/a | 0.083 | 0.601 / **0.733** |
| frontier-llm (gpt-oss) | large | 120B | 0.655 | .71 / .62 | 3/80 | 13.4 s | n/a | 0.050 | 0.646 / 0.665 |
| mistral-7b | local | 7.0B | 0.642 | .68 / .62 | 8/80 | 18.7 s | 5.0 GB | free | 0.578 / 0.705 |
| gemma3-4b | local | 4.0B | 0.639 | .66 / .63 | 7/80 | 8.9 s | 2.6 GB | free | 0.647 / 0.632 |
| gemma2-2b | local | 2.0B | 0.611 | .64 / .59 | 9/80 | 8.1 s | 2.8 GB | free | 0.573 / 0.650 |
| phi4-mini | local | 3.8B | 0.528 | .58 / .51 | 10/80 | 7.1 s | 3.6 GB | free | 0.587 / 0.469 |
| gemma3-1b | local | 1.0B | 0.417 | .42 / .41 | 1/80 | 3.4 s | 2.2 GB | free | 0.428 / 0.406 |
| qwen2.5-0.5b | local | 0.5B | 0.381 | .41 / .37 | 0/80 | 3.1 s | 1.5 GB | free | 0.337 / 0.425 |
| smollm2-360m | local | 0.36B | 0.306 | .31 / .31 | 0/80 | 3.1 s | 1.5 GB | free | 0.350 / 0.263 |
| llama3.2-1b | local | 1.0B | 0.277 | .30 / .27 | 0/80 | 5.1 s | 2.2 GB | free | **0.529 / 0.025** |

**Headlines:**
1. **Quality is INVERSELY related to size across the large arm: 31B (0.701) > 70B (0.667)
   > 120B (0.655).** The smallest large model wins outright. Combined with the local arm
   (4B gemma3 ≈ 7B mistral; 2B gemma2 > 3.8B phi4-mini), the whole study delivers one
   verdict: **parameter count is a poor predictor of KIE quality — architecture & training
   dominate.** This is the dissertation's central result.
2. **gemma4-31b dominates the entire study** — best F1 (0.701), best exact-match (15/80,
   ~5× the 120B's 3/80), *fastest* model of all 11 (1.8 s median; DO serves it very fast),
   and cheapest of the large arm ($0.043).
3. **The SLM↔LLM gap is small.** The best local model (mistral-7b, 0.642) is within
   **0.06 F1** of the best cloud model and **beats the 120B gpt-oss on exact-match**
   (8/80 vs 3/80). A **4B** model on the M1 (gemma3-4b, 0.639) effectively ties a **120B**
   cloud model (0.655) — for free, locally, on 8 GB. The strongest small-vs-large argument.
4. **The Gemma family leads at every size** (1B/2B/4B/31B = 0.417/0.611/0.639/0.701) — a
   clean single-architecture scaling curve from edge to cloud; Gemma 3 also beats the prior
   gen at 1B (0.417 vs llama3.2-1b 0.277).
5. **Few-shot is model-dependent, not universally good.** It *helps* mistral (+0.13),
   llama3.3-70b (+0.13) and gemma2 (+0.08) but *destroys* llama3.2-1b (0.529→0.025, context
   overflow) and hurts phi4-mini (−0.12). Tiny models prefer zero-shot. (→ Phase-3 [P5].)
6. **Recall < precision for every model, large arm included** — the bottleneck is *missed*
   fields, not hallucinations, even at 120B (→ Phase-3 [P4] regex fallback).

---

## 3. Artifact index (what proves what)

| Artifact | What it is |
|---|---|
| `results/runs.jsonl` | Current run, one row per cell (880 rows = 11 models × 80) — local **structured-output** + large arm prompt-JSON |
| `results/runs_baseline.jsonl` | Frozen baseline (320 rows) — **no structured output**, for the A/B |
| `results/summary.csv` | Per-model + per-condition aggregates |
| `results/REPORT.md` | Auto-generated tables + auto-findings (regenerated each run) |
| `results/plot_f1_vs_params.png` | F1 vs model size (the central plot) |
| `results/plot_latency_vs_params.png` | Latency vs size |
| `results/plot_cost_frontier.png` | Quality vs cost trade-off |
| `results/plot_field_heatmap.png` | Per-field F1 heatmap (which fields are hard) |
| `results/manifest.json` | Run provenance (seed, platform, models, datasets) |
| `results/snapshots/2026-06-15-phase2-baseline/` | **Versioned** frozen copy of the 8-local-model run (the pre-large-arm baseline) |
| `results/snapshots/2026-06-15-phase2-with-large/` | **Versioned** frozen copy of the full 11-model run incl. large arm (runs.jsonl, summary.csv, REPORT.md, plots) — the live `results/` files are gitignored as regenerable |
| `scripts/run_large_arm.sh` | One-command large-arm run (gpt-oss-120b + llama3.3-70b on DO serverless); needs `DO_INFERENCE_KEY` |

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

### [E4] 2026-06-15 — Mid arm run on M1 + Gemma 3 family added (8 local models complete)
- **What:** Ran phi4-mini (3.8B) and mistral-7b (7B) locally on the M1 (Ollama, same
  constrained-decoding stack as the small models). Added the latest **Gemma 3** models to
  the ladder — `gemma3-1b` (1B) and `gemma3-4b` (4B) — pulled and run on M1. All eight
  local models now have a full 80-cell run; regenerated `results/` artifacts.
- **Why:** Completes the local half of the small-vs-large axis and adds a current-generation
  family across three sizes, enabling generational (Gemma 3 vs Gemma 2) and same-size
  (gemma3-4b vs phi4-mini) head-to-heads.
- **Result (see §2 for the full table):**
  - **gemma3-4b (4B) ties mistral-7b (7B):** F1 0.639 vs 0.642, at **half the latency
    (8.9 s vs 18.7 s) and half the memory (2.6 GB vs 5.0 GB)** → the efficiency winner.
  - **Size ≠ quality:** 2B gemma2 (0.611) > 3.8B phi4-mini (0.528); 1B gemma3 (0.417) >
    1B llama3.2 (0.277). Architecture/training dominate.
  - **Few-shot is model-specific:** helps mistral (+0.13) / gemma2 (+0.08); destroys
    llama3.2-1b (0.529→0.025) and hurts phi4-mini (−0.12).
  - 0 parse failures across all 8 models; mistral-7b ran clean on 8 GB but slow (swap).
- **Cost:** $0 (all local).
- **Artifact:** `results/runs.jsonl`, `results/summary.csv`, `results/REPORT.md`, plots.
- **Dissertation value:** the core mid-sem result — *"for KIE, a well-trained 4B open model
  matches a 7B at half the cost, and parameter count is a poor predictor of quality."*

### [E5] 2026-06-15 — Large arm complete on DO serverless (11-model study finished)
- **What:** Ran the three large models on DigitalOcean serverless inference, full 80-cell
  grid each: **gemma-4-31B-it** (31B), **llama3.3-70b-instruct** (70B), **openai-gpt-oss-120b**
  (120B). Regenerated all `results/` artifacts. The mid-sem study is now complete: 11 models
  spanning 0.36B → 120B, ~3 orders of magnitude.
- **Model-choice revision (important):** the plan was GPT-4o + Claude Haiku 4.5, but those
  require DO subscription **Tier 3+** and this account is **Tier 1/2** (they return
  *"not available for your subscription tier"*). Pivoted the large arm to DO-hosted **open**
  models. This is a **methodological upgrade, not a compromise** — the large arm is now fully
  open-weights and reproducible, with no closed-model dependency, which strengthens the
  comparison's repeatability.
- **Method note:** large models use **prompt-instructed JSON** (no grammar constraint —
  constrained decoding fires only for `kind == "local"`). Despite that, only 1 parse failure
  occurred across 240 large-arm cells (llama, 1/80) — frontier models don't need the SLM
  equalizer, itself a finding.
- **Result (full table in §2):**
  - **Quality is *inverse* to size in the large arm: 31B 0.701 > 70B 0.667 > 120B 0.655.**
    The smallest large model wins; size is not the driver.
  - **gemma4-31b is the overall study winner** — top F1, top exact-match (15/80), fastest of
    all 11 models (1.8 s median), cheapest large model.
  - **Small↔large gap is small:** best local mistral-7b (0.642) is within 0.06 F1 of the best
    cloud model and beats the 120B on exact-match; 4B gemma3-4b ≈ 120B gpt-oss on F1, for free.
  - **Few-shot helps llama3.3-70b (+0.13)** like it helps mistral/gemma2.
- **Cost:** whole large arm **~$0.18 total** on the DO credit (gemma4 $0.043 + llama $0.083 +
  gpt-oss $0.050) — far under the ~$1 estimate; the $200 credit is barely touched.
- **Artifact:** `results/runs.jsonl` (880 rows), `results/summary.csv`, `results/REPORT.md`,
  plots; frozen at `results/snapshots/2026-06-15-phase2-with-large/`.
- **Dissertation value:** completes the mid-sem deliverable — *"across 0.36B–120B on KIE,
  parameter count does not predict quality; a 31B open model beats a 120B one, and a 4B local
  model matches a 120B cloud model at zero cost."*

### [E6] 2026-07-28 — Phase-3 [P2]: QLoRA fine-tuning of phi4-mini (before→after)
- **What:** Fine-tuned **phi4-mini (3.8B)** with QLoRA (Unsloth, 4-bit) on the **train
  splits only** of SROIE + Kleister-Charity, exported to a quant-matched **Q4_K_M GGUF**,
  imported back into the Ollama harness as `phi4-mini-ft` (`type: local`), and re-ran the
  identical frozen 20×4 grid. Two runs: Run 1 (initial config) and Run 2 (corrected, final).
- **Hypothesis:** task-specific fine-tuning on the narrow KIE schema should substantially
  lift a weak-but-cheap local model — phi4-mini was the *worst* mid-size local model in
  Phase 2 (0.528) — at unchanged inference cost.
- **Result — Run 2 (final), before→after on identical docs/conditions:**

  | Metric | phi4-mini (before) | phi4-mini-ft (after) | Δ |
  |---|---|---|---|
  | **F1_macro** | 0.528 | **0.669** | **+0.141** |
  | **Exact-match** | 0.125 (10/80) | **0.312 (25/80)** | **+0.187** |
  | Parse failures | 0/80 | 0/80 | — |
  | SROIE F1 | 0.713 | **0.844** | +0.131 |
  | Kleister F1 | 0.343 | **0.494** | +0.151 |

  - **Significance:** per-doc cluster bootstrap (10k resamples) ΔF1 = **+0.141, 95% CI
    [+0.067, +0.218], p(Δ≤0)=0.000**; **14/20 docs improved, 2 regressed, 4 tie.**
  - phi4-mini jumps from the **worst** mid-size local model (0.528) to **0.669** — above
    gemma3-4b / mistral-7b (~0.64) and level with the **70B/120B large arm** (0.667 / 0.655,
    see §2). A task-tuned **3.8B local** model matches the frontier large arm on this KIE
    task, for **free**, on an 8 GB laptop. Exact-match (0.312) also beats every Phase-2
    model except gemma4-31b (15/80).
  - Fine-tuning **repaired phi4-mini's few-shot collapse** (baseline few-shot·lines_only
    0.426 → 0.650) — relevant to [P5].
- **Key methodological finding (Run 1 → Run 2):** Run 1 gained only **+0.079** (CI
  [+0.021, +0.136]) because the LoRA attached to **only 2 of the 7 intended modules**.
  Phi-3/Phi-4-mini **fuse** attention into `qkv_proj` and MLP into `gate_up_proj`, so
  Unsloth's default target names `q/k/v/gate/up_proj` matched **nothing** — only `o_proj`
  + `down_proj` were adapted. Setting `target_modules=[qkv_proj, o_proj, gate_up_proj,
  down_proj]` (+ rank 16→64, epochs 3→5, cap→full data) roughly **doubled the delta** to
  +0.141. *Lesson: verify the adapter actually attached to the intended (fused) modules
  before trusting a weak fine-tuning result.*
- **Caveat — truncation:** at `max_seq_len=4096`, **2992/4460 training examples (67%) were
  dropped as truncated** (long Kleister `lines_plus_kv` whose key-value dumps exceed 4096
  tokens), so Run 2 effectively trained on SROIE + short Kleister. The gain held regardless;
  a longer sequence length is the obvious remaining lever if more is wanted.
- **Caveat — latency not comparable:** measured ft latency (19.9 s) vs baseline (9.1 s) is a
  **measurement artifact** — the ft re-eval ran under severe memory/disk pressure (swap
  thrashing on the 8 GB M1 with a near-full disk). Same architecture/quant ⇒ inference cost
  is expected ~equal; latency needs a clean re-measure. Quality metrics (temp=0, seed=42)
  are deterministic and unaffected.
- **Data hygiene:** SFT built by `scripts/make_train_data.py` from **train splits only**;
  leakage assert passed (0 test-set doc_ids in 2230 training docs) + determinism test green.
  The frozen 20-doc test set never entered training.
- **Infra (deviation from plan):** trained on **Kaggle T4×2** (Unsloth), not the DO GPU
  droplet [P2] originally anticipated — free, nothing to destroy. Driven **headlessly** via
  the Kaggle CLI (`kernels push/status/output`); GPU pinned with
  `machine_shape=NvidiaTeslaT4` (plain `enable_gpu` yields an incompatible P100). GGUF
  export runs in a **separate CPU-only kernel** (tensor-level LoRA merge +
  `tokenizer_class=GPT2Tokenizer` to route llama.cpp to Phi-4-mini's gpt-4o BPE path,
  which otherwise demands a nonexistent `tokenizer.model`).
- **Artifact:** `results/runs.jsonl` (phi4-mini-ft, 80 rows) vs the frozen phi4-mini
  baseline; `config/train.yaml`, `notebooks/03_qlora_finetune.ipynb`; trained adapter as
  Kaggle dataset `azharali7/phi4-mini-ft-adapter` (v2); plots regenerated.
- **Dissertation value:** the viva headline — *"task-specific QLoRA turns the worst
  mid-size local model (3.8B, F1 0.528) into one that matches a 70–120B cloud model
  (0.669) and doubles exact-match, at zero cost on an 8 GB laptop."*

### [E7] 2026-07-29 — Phase-3 [P4]: rule/regex recall fallback — a supporting NEGATIVE result
- **What:** A model-agnostic post-processor (`src/eval/rules.py`) that fills only *null*
  fields with deterministic pattern extraction over the document lines, measured post-hoc
  on the stored predictions (`scripts/eval_rule_fallback.py`, no re-inference).
- **Why demoted:** this is **off the SLM-vs-LLM axis** — it applies the same rules to every
  model, so it can't speak to small-vs-large. Kept as an **opt-in helper**, not part of the
  headline comparison.
- **Result — deliberately kept because the near-null finding *reinforces* the thesis:**
  - **Overall ΔF1 = +0.005** (95% CI [+0.002, +0.008]) — statistically nonzero, practically
    negligible. **The six most capable models (gemma2-2b, mistral-7b, gemma3-4b, frontier-llm,
    gemma4-31b, phi4-mini-ft) gained exactly 0.000.** Only the *weakest* mid model moved
    (phi4-mini +0.033). Total across the whole 12-model × 20-doc study: **27 correct fills.**
  - **Format-distinctive fields are rule-recoverable; semantically-selected fields are not.**
    Per-fill precision: `address__postcode` 89% (17/19), `charity_number` 77% (10/13) — kept;
    `report_date` 56%, `income` 53%, `spending` **11%** (grabs the wrong £ figure) — excluded,
    because a wrong fill costs precision with no recall gain (wrong ≠ tp).
- **Interpretation (the point):** cheap post-processing does **not** close the gap — quality
  comes from the *model*, not from rules; and like constrained decoding [E2], what little it
  gives helps only the weakest model. The on-thesis improvement lever is fine-tuning [E6],
  not this.
- **Artifact:** `src/eval/rules.py`, `scripts/eval_rule_fallback.py`, `tests/test_rules.py`
  (7 tests). The eval loop and frozen baseline are untouched.

---

## 5. Roadmap — Phase 2 (mid-sem, now) vs Phase 3 (final viva, reserved)

The work is deliberately split so **mid-sem is a complete, standalone result** (the
baseline small-vs-large comparison) while the **improvement story is reserved for the
viva** — so there is substantial, novel work left to present in Phase 3. The frozen
`runs_baseline.jsonl` + this changelog are the bridge: every Phase-3 change is measured
as a before→after delta against the Phase-2 baseline using the identical A/B method.

### Phase 2 — Mid-sem (finish now; no fine-tuning)
Goal: answer the core research question end-to-end on a fixed bench.
- **[P1] ✅ DONE — Complete the 11-model baseline** across the small→large ladder
  (0.36B–120B). *Placement (see [E3]/[E5]):* small + mid (phi4-mini, mistral-7b) on M1
  Ollama; large (gemma-4-31B, llama3.3-70b, gpt-oss-120b) on DO serverless. All 11 models
  ran the full 80-cell grid; see §2.
- **[P1b] ✅ DONE — Efficiency/cost frontier** — latency, peak memory, tokens/s and $ per
  model captured; quality-vs-cost trade-off plot regenerated (`plot_cost_frontier.png`).
- **Status:** **Mid-sem deliverable complete.** Constrained-decoding finding ([E2]) ✅;
  full 11-model study ([E4]+[E5]) ✅; efficiency frontier ✅. Nothing from Phase 3 below
  is started (by design — it's the reserved viva work).

### Phase 3 — Final viva (reserved; do NOT start before mid-sem is submitted)
Goal: the *improvement* narrative — how to make small models competitive.
- **[P2] ⭐ Headline: QLoRA fine-tuning** of phi4-mini + mistral-7b on the **train splits
  only** (SROIE + Kleister), then re-run the **same frozen 20×4 grid** for a clean
  before→after delta vs the Phase-2 baseline.
  - **✅ phi4-mini DONE — see [E6]:** F1 0.528→0.669 (ΔF1 +0.141, 95% CI [+0.067,+0.218],
    p≈0), exact-match 0.125→0.312. A tuned 3.8B local model reaches the 70–120B large arm.
    Trained on Kaggle T4×2 (Unsloth), not the DO droplet. **mistral-7b-ft still pending.**
  - *Why it's the viva centrepiece:* targets the punchline *"a fine-tuned 3.8B/7B matches
    or beats GPT-4o zero-shot on this narrow KIE task at a fraction of cost/latency"* —
    turning the thesis from "big beats small" into "small + task-tuning closes the gap."
  - *Infra:* this is where the **DO GPU droplet** is justified (QLoRA 7B fits ~16–24 GB
    VRAM; a couple of hours ≈ a few $ of the credit; **destroy the droplet after**).
  - *Stack:* Unsloth (single-GPU, exports **GGUF** → drops straight back into the Ollama
    harness as e.g. `phi4-mini-ft`, `type: local`). Axolotl/PEFT as alternatives.
  - *Data hygiene (non-negotiable):* fine-tune on train splits ONLY; the frozen 20-doc
    test set must never leak in, or the result is worthless. `build_test_set` already
    separates them — keep it that way.
  - *Scope note:* fine-tuning was an out-of-scope *limitation* in the original proposal;
    moving it in is a deliberate Phase-3 expansion — confirm with supervisor it fits the
    viva timeline.
- **[P3] Self-consistency voting** (n_samples=3, majority vote per field) — does sampling
  + vote beat a single greedy decode for small models?
- **[P4] Hybrid regex fallback** — ✅ **DONE but DEMOTED (see [E7]):** measured, near-null
  lift (+0.005 F1; 0.000 for the six most capable models), and off the SLM-vs-LLM axis
  (model-agnostic). Kept as an opt-in helper + a supporting *negative* finding — *rules
  don't close the gap; the model does.* Not a headline.
- **[P5] Drop few-shot for the smallest models** — llama3.2-1b collapsed under few_shot
  (F1 0.025 vs 0.47 zero-shot; the 2 examples blew its context). Explicit per-model
  shot-mode comparison.
- **[P6] Per-field error analysis** — use `plot_field_heatmap.png` to identify which fields
  (e.g. Kleister income/expenditure) drive most of the loss; motivates [P4].
