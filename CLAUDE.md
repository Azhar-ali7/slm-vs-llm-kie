# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A reproducible research framework for an M.Tech dissertation comparing **small local
language models** (via Ollama) against **large cloud API models** at Key Information
Extraction (KIE) — extracting fixed schema fields as JSON from financial documents —
scored on both **quality** (precision/recall/F1/exact-match) and **efficiency**
(latency, peak memory, cost). The end product is an accuracy–cost frontier plus the
figures, tables, `.docx` report, and `.pptx` deck that go into the dissertation.

The actual project lives in the **`slm-vs-llm-kie/`** subdirectory. Run everything from
there. The repo root only holds the top-level proposal/summary docs.

## Commands

```bash
cd slm-vs-llm-kie
python3.11 -m venv .venv && source .venv/bin/activate   # first time; .venv/ already uses 3.12
pip install -r requirements.txt

# --- run the study (writes results/runs.jsonl incrementally, resumable) ---
python scripts/run_eval.py --pilot        # 3 docs × 2 models — validate wiring cheaply
python scripts/run_eval.py --mock         # no models needed; mock_runner seeds results
python scripts/run_eval.py                # full run
python scripts/run_eval.py --skip-large   # full run minus the 7B+ models (8GB-safe)
python scripts/run_eval.py --models smollm2:360m qwen2.5:0.5b   # subset of models
python scripts/run_eval.py --datasets sroie                     # subset of datasets
bash scripts/run_large_arm.sh             # only the DigitalOcean large-arm models (billed; needs DO_INFERENCE_KEY)

# --- analysis & report artifacts (read runs.jsonl, regenerate outputs) ---
python scripts/make_plots.py              # -> results/summary.csv, 4 PNGs, results/REPORT.md
python scripts/make_report_doc.py         # -> report/Mid_Sem_Report.docx (tables pulled live)
python scripts/make_slides.py             # -> report/*.pptx (needs python-pptx)

# --- data (real datasets: you download, framework converts) ---
python scripts/download_data.py           # prints the clone/curl commands to run yourself
python scripts/convert_data.py            # data/raw/<name>/_src/ -> {train,test}.jsonl

# --- viva demo ---
python scripts/demo.py --replay                     # instant, from saved results
python scripts/demo.py --live --doc <id> [--local-only]

# --- tests ---
pytest -q                                 # all
pytest tests/test_metrics.py -q           # one file
pytest tests/test_metrics.py::<name> -q   # one test
```

Local models run under Ollama with `OLLAMA_KEEP_ALIVE=0` (one model loaded at a time,
unloaded after each call — this is what makes an 8GB M1 viable). Set that env var
before running. Cloud arms need keys in `.env` (copy `.env.example`); which keys depend
on which runner `config.yaml` selects.

## Architecture

**The whole pipeline is a chain of small pure modules, and `config/config.yaml` is the
single source of truth** — no magic constants in code. Adding a model or dataset is a
config edit, not a code change. `CODE_WALKTHROUGH.md` is the definitive narrated tour;
read it before making structural changes. The flow:

```
config.yaml → loaders → preprocess (→ "simple JSON") → prompt builder → MODEL RUNNER
           → parse → metrics → runner/run.py writes results/runs.jsonl → analysis → plots/report
```

Key design invariants — preserve these when editing:

- **`preprocess.py` normalises every document to one canonical "simple JSON"**
  (`{doc_id, lines, key_values, tables}`). Both entry points (`from_dataset_record`
  and `from_textract`) must produce the identical shape — the rest of the pipeline is
  source-agnostic because of this.
- **`src/models/` is one interface, many backends.** `base.py` defines the abstract
  `ModelRunner` (`run()`, `ensure_available()`) and the `RunResult` dataclass;
  `registry.py::build_runners()` instantiates runners from `config.yaml`. Concrete
  runners: `ollama_runner` (local), `do_runner` (DigitalOcean serverless),
  `api_runner`/`foundry_runner`/`openrouter_runner` (cloud), `mock_runner` (offline,
  for tests/demo). The eval loop is identical for a 0.36B local and a 120B cloud model
  — never special-case a backend in the loop; add a runner instead.
- **`parse.py` is defensive and never raises** — a bad model output returns a
  structured `parse_error` so one cell can't crash the run.
- **`metrics.py` normalisation is applied identically to gold and prediction**
  (type-aware: strings, numbers via `%g`, dates as `YYYY-MM-DD`). This is a fairness
  guarantee — don't introduce per-model or asymmetric normalisation.
- **`runner/run.py::run_eval()` iterates models outermost** (load-once, unload-before-next)
  and is **resumable** via `results_store.completed_keys()` — results append one JSON
  row per (doc × model × shot_mode × input_variant × sample) cell to `runs.jsonl`.
- **Few-shot examples come from the train split only** (`gt.py::few_shot_pool`) — never
  the 20-doc test set. Preserving this no-leakage boundary matters.

Constrained decoding: for local models, `builder.py::json_schema_format` emits a JSON
schema into Ollama's `format` field so decoding is grammar-constrained to the schema
keys (see FINDINGS [E2]).

## Data & results conventions

- `config/schemas/{sroie,kleister}.json` define the extraction fields (4 and 8
  respectively). Two datasets: SROIE (receipts, simple) and Kleister-Charity (charity
  reports, complex).
- `results/test_set_20.json` is the seed-fixed 20-doc test set; delete it to reselect
  after loading real data.
- `runs.jsonl`, `summary.csv`, PNGs, and `REPORT.md` are **regenerable and gitignored**.
  `results/snapshots/**` is the exception — frozen evidence snapshots ARE versioned as
  the dissertation record. Don't commit the regenerable outputs; do preserve snapshots.
- `report/*.docx` and `report/figures/` are generated (gitignored); the prose in the
  report/deck is hand-written, the scripts only lay it out and pull result tables live.

## Cost & safety

The local models are free; cloud arms (Azure / DigitalOcean) are **billed per token**.
Always validate with `--pilot` or `--mock` before a full billed run, keep `n_samples`
and doc count low, and check the summed `cost_usd` in `runs.jsonl` stays within budget.
Never commit `.env` or real keys.
