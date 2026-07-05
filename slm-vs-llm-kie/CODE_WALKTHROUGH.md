# Code Walkthrough — how to explain this codebase in the viva

A narrated tour of the code, following **one document as it flows through the
system**. Read top to bottom and you can explain any part of the pipeline aloud.
Companion to `README.md` (setup/run) and `FINDINGS.md` (results).

> **The one-sentence architecture:** *A document is normalised to a simple JSON,
> turned into a prompt from a schema, sent to a model runner, the model's text is
> parsed back into fields, and those fields are scored against gold — every stage
> is a small pure module, and `config.yaml` is the only place constants live.*

---

## 0. The 30-second mental model

```
       config.yaml  (single source of truth: models, datasets, conditions, seed)
            │
 raw data ──▶ loaders ──▶ preprocess ──▶ prompt builder ──▶ MODEL RUNNER ──▶ parse ──▶ metrics
 (SROIE/     (dataset    (canonical      (schema + doc      (Ollama /        (JSON    (P/R/F1
  Kleister)   → records)  "simple JSON")  → prompt text)     DO / mock)       → dict)  per field)
                                                                                          │
                                                              runner/run.py orchestrates all of it
                                                                                          │
                                                              results/runs.jsonl ──▶ analysis ──▶ plots/report
```

**If you remember one thing:** each arrow is a module boundary with a pure
function. You can point at any arrow and name the file.

---

## 1. Configuration — `src/config.py` + `config/config.yaml`

*"No magic constants live in code."* Everything tunable is in `config.yaml`:
`seed: 42`, `paths:`, `datasets:`, `test_set:`, `conditions:`, `models:`,
`ollama:`, `api:`, `run:`, `report:`.

`config.py` loads it:
- `load_config()` — reads the YAML, also loads `.env` (API keys) via `_load_dotenv`.
- `resolve_path(cfg, rel)` — turns a config-relative path into an absolute one.
- `load_schema(cfg, dataset)` — loads the field schema (`config/schemas/sroie.json`
  = 4 fields, `kleister.json` = 8 fields).

**Say in the viva:** "Adding a model or dataset is config-only — no code changes."

---

## 2. Loading raw data → records — `src/data/loaders.py` + `convert.py`

Two real datasets: **SROIE** (receipts, 4 fields, simple) and
**Kleister-Charity** (UK charity reports, 8 fields, complex).

- `scripts/download_data.py` fetches the raw files into `data/raw/`.
- `src/data/convert.py` (pure, no network) parses those raw files into normalised
  **JSONL records** — `sroie_records_from_dirs()` and `kleister_records_from_tsv()`.
  `scripts/convert_data.py` drives it.
- `src/data/loaders.py` reads those JSONL records back as Python dicts:
  `load_sroie()`, `load_kleister()`, `load_dataset(name)`.

A **record** carries: `doc_id`, `dataset`, OCR `words` (+bboxes), optional
`key_values`/`tables`, and the `gold` dict (the correct field values).

---

## 3. The reproducible test set — `src/data/gt.py`

*"Ground truth + reproducible 20-document test-set selection."*
- `build_test_set()` — seed-fixed pick of **20 docs (10 SROIE + 10 Kleister)**,
  saved to `results/test_set_20.json` so every run scores the same documents.
- `load_test_records()` — loads those 20 docs for a run.
- `align_gold(gold, schema)` — lines the gold values up with the schema's fields.
- `schema_for_record(record)` — picks the right schema (sroie vs kleister) per doc.
- `few_shot_pool()` — draws few-shot examples **from the train split only** (never
  the test set — this prevents data leakage; a likely viva question).

---

## 4. Normalising to "simple JSON" — `src/data/preprocess.py`

The key abstraction. Every document, whatever its source, becomes the **same
canonical shape** so the rest of the pipeline is source-agnostic:

```json
{"doc_id": "...", "page": 1,
 "lines": ["FARMASI MALURI S/B", "TOTAL 79.35", ...],
 "key_values": {"DETECTED_KEY": "value"},
 "tables": [[["r1c1","r1c2"]]]}
```

- `from_dataset_record(record)` — the path used in the study: OCR words+boxes are
  grouped into text **lines** (`_words_to_lines` sorts by vertical position).
- `from_textract(blocks)` — an alternate entry point for raw AWS Textract output.

**Both entry points produce identical simple JSON** — that's the design guarantee.
This simple JSON is exactly what the demo prints in the document panel.

---

## 5. Building the prompt — `src/prompts/builder.py`

Turns *(schema + simple JSON)* into the text sent to the model. Two
config-controlled conditions:
- **shot_mode**: `zero_shot` | `few_shot` (k=2 examples from train split).
- **input_variant**: `lines_only` | `lines_plus_kv`.

Functions:
- `build_prompt(schema, simple_json, shot_mode, input_variant, max_lines)` — the
  main entry. Instruction text lives in `src/prompts/templates/instruction.txt`
  (inspectable, not hard-coded).
- `field_spec(schema)` — lists the fields the model must return.
- `render_input_block()` — formats the document LINES (+KV) into the prompt.
- `json_schema_format(schema)` — **the constrained-decoding piece**: produces the
  JSON schema handed to Ollama's `format` field so decoding is grammar-constrained
  to emit exactly the schema's keys. *(This is the local-arm advantage discussed
  in FINDINGS [E2] — worth being ready to defend as a fairness point.)*

---

## 6. Calling the model — `src/models/` (the runner abstraction)

The cleanest part of the design: **one interface, many backends.**

- `base.py` — defines `ModelRunner` (abstract: `run()`, `ensure_available()`) and
  the `RunResult` dataclass (raw text + latency + tokens + peak memory + error).
- `registry.py` — `build_runners(cfg, only=[...])` reads `config.yaml models:` and
  instantiates the right runner per model. **Adding a model never touches the loop.**

The concrete runners (all implement the same `run()` signature):

| Runner | Backend | Used for |
|--------|---------|----------|
| `ollama_runner.py` | Local Ollama REST | **the 8 local models** — samples peak RSS of the ollama process; `keep_alive=0` unloads after each call (8 GB-friendly) |
| `do_runner.py` | DigitalOcean serverless (OpenAI-compatible) | **the 3 large-arm models** (gemma-4-31B, llama3.3-70b, gpt-oss-120b) |
| `api_runner.py` | Azure OpenAI | alternate cloud path (GPT models) |
| `foundry_runner.py` | Azure AI Foundry | alternate cloud path |
| `openrouter_runner.py` | OpenRouter | alternate cloud path |
| `mock_runner.py` | Offline regex heuristic | tests + demo seeding without any model |

**Say in the viva:** "Every model — local or cloud — is behind one `run(prompt)`
interface returning a `RunResult`, so the eval loop is identical for a 0.36B local
model and a 120B cloud model."

---

## 7. Parsing the model's output — `src/eval/parse.py`

Models return **text**; we need a clean field dict. This module is defensive:
- `extract_json_text()` / `_first_json_object()` / `_strip_fences()` — pull the JSON
  object out of possibly-messy output (code fences, prose around it).
- `parse_prediction(text, schema)` — validates against the schema and coerces
  scalars. **On failure it returns a structured `parse_error`, never raises** — so
  one bad output can't crash an 880-cell run. (Free-form JSON parse-failures were
  the smollm2 problem in FINDINGS [E1], fixed by constrained decoding in [E2].)

---

## 8. Scoring — `src/eval/metrics.py` (the heart of the evaluation)

Per *(document, field)*, the prediction falls into one category:

| Category | Condition | Meaning |
|----------|-----------|---------|
| `tp` | gold≠∅, pred≠∅, values match | correct |
| `wrong` | gold≠∅, pred≠∅, differ | wrong value |
| `missing` | gold≠∅, pred=∅ | recall miss (FN) |
| `hallucinated` | gold=∅, pred≠∅ | precision miss (FP) |
| `tn` | gold=∅, pred=∅ | correct empty |

- `normalize(v, ftype)` — **type-aware**: case/whitespace for strings, `%g` for
  numbers, `YYYY-MM-DD` for dates. So `$79.35` == `79.35`. Applied *identically* to
  gold and prediction (so it can't bias any model — a fairness point).
- `classify_field()` → one of the categories above.
- `score_document(gold, pred, schema)` → `{per_field, counts, precision, recall, f1,
  exact_match_doc, exact_match_fields, n_fields}`. This is what colours the demo's
  ✓/✗ and produces the F1. (`exact_match_doc` = all fields correct on that document;
  the study's "15/80" figures count these.)
- `precision = tp/(tp+wrong+halluc)`, `recall = tp/(tp+wrong+missing)`.

---

## 9. Efficiency — `src/eval/efficiency.py`

- `efficiency_metrics(result, meta, kind, ...)` → latency, tokens/s, peak memory,
  and **cost**. Cloud: `prompt_tokens*price_in + completion_tokens*price_out`
  (prices per 1M tokens in config). Local: `cost_usd = 0.0` but latency + peak RSS
  still recorded. This is why the demo shows `$0.00000` for local, real cents for cloud.

---

## 10. The orchestrator — `src/runner/run.py` + `results_store.py`

`run_eval()` is the main loop: **documents × models × shot_mode × input_variant ×
samples**.
- **Models iterated outermost** so each loads once and is unloaded before the next
  (8 GB-friendly).
- **Resumable**: `results_store.completed_keys()` skips already-done cells; results
  are appended incrementally to `results/runs.jsonl` (one JSON row per cell).
- **Retry with backoff**: `_run_with_retry` + `_is_transient` re-try flaky network
  errors so a blip doesn't lose a run.
- `_write_manifest()` records provenance (seed, platform, models, datasets).

`scripts/run_eval.py` is the CLI entry (`--pilot`, `--mock`, etc.);
`scripts/run_large_arm.sh` runs just the 3 DO models.

**Result:** `results/runs.jsonl` = 11 models × 80 cells = **880 rows**, each with
prediction, per-field scores, latency, cost.

---

## 11. Analysis & outputs — `src/analysis/`

Turns `runs.jsonl` into tables and figures:
- `aggregate.py` — `load_results_df()` flattens the JSONL to a DataFrame;
  `per_model()` / `per_model_condition()` / `per_field_accuracy()` compute the
  standings; `save_summary()` → `results/summary.csv`.
- `report.py` — `build_report()` → `results/REPORT.md` (auto tables + auto-findings).
- `plots.py` — `plot_f1_vs_params` (the central plot), `plot_cost_frontier`,
  `plot_latency_vs_params`, `plot_field_heatmap`.
- `report_figures.py` — the architecture + pipeline **diagrams** for the report/slides.

`scripts/make_plots.py`, `make_report_doc.py`, `make_slides.py` drive these into the
`.docx` report and `.pptx` deck.

---

## 12. Tests — `tests/`

Pure-function units, run with `pytest`:
- `test_metrics.py` — the scoring categories & normalisation.
- `test_parse.py` — JSON extraction from messy text.
- `test_preprocess.py` — words → lines → simple JSON.
- `test_convert.py` — raw dataset files → records.

**Say in the viva:** "The scoring, parsing, and normalisation are unit-tested — the
parts a reviewer would most doubt are the parts with tests."

---

## How to answer "explain your code" in 90 seconds

Trace one receipt, naming the file at each step:

1. **"config.yaml** defines the 11 models, 2 datasets, and conditions — one source
   of truth."
2. "The **loader** reads the receipt as a record; **preprocess.py** normalises it to
   a *simple JSON* of text lines — every document ends up in this one shape."
3. "The **prompt builder** combines that with the field schema; for local models it
   also emits a JSON schema so **Ollama's decoding is grammar-constrained**."
4. "It goes to a **model runner** — every backend, local or cloud, is behind one
   `run(prompt)` returning a `RunResult`, so the loop is identical across all 11."
5. "The model's text is **parsed** back to a field dict — defensively, a bad output
   returns an error instead of crashing."
6. "**metrics.py** scores each field as tp/wrong/missing/hallucinated against gold,
   with type-aware normalisation, and computes precision/recall/F1."
7. "The **runner loop** does this for every doc×model×condition, writes
   `runs.jsonl`, and **analysis** turns that into the tables and plots in my report."

Then offer: *"I can open any of those files — e.g. `metrics.py` for the scoring, or
`registry.py` for how models plug in."*

---

## Fast file-finder (when they point at a claim)

| They ask about… | Open… |
|-----------------|-------|
| "how is F1 computed" | `src/eval/metrics.py` |
| "how do you add a model" | `config.yaml` + `src/models/registry.py` |
| "the constrained decoding" | `src/prompts/builder.py` → `json_schema_format` |
| "how is a document represented" | `src/data/preprocess.py` |
| "how is the test set chosen" | `src/data/gt.py` → `build_test_set` |
| "what if the model outputs garbage" | `src/eval/parse.py` |
| "how is cost measured" | `src/eval/efficiency.py` |
| "the main experiment loop" | `src/runner/run.py` → `run_eval` |
| "how the plots are made" | `src/analysis/plots.py` |
| "the demo" | `scripts/demo.py` (+ `DEMO_GUIDE.md`) |
