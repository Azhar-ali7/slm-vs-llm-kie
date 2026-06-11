# Claude Code Build Prompt — SLM vs LLM for Key Information Extraction

> **How to use this:** Create an empty folder, open it in Claude Code, paste everything
> below (from "PROJECT BRIEF" onward) as your first message. Claude Code will scaffold and
> build the project. Build it in the phase order given so each part is testable.

---

## PROJECT BRIEF

Build a **reproducible Python framework** that compares **small (local) vs large (API) language
models** on **Key Information Extraction (KIE) from financial documents**. The model input is a
**preprocessed "simple JSON"** (the flattened form of an AWS-Textract-style output), and the model
must return the target fields as JSON. Measure both **extraction quality** and **operational
efficiency**, across **zero-shot / few-shot** and **input-variant** conditions.

### Research goal this code serves
Quantify the accuracy–efficiency trade-off as model size grows (0.36B → 70B+), find where small
models stop being viable as documents get more complex, and produce an accuracy-vs-cost frontier.

---

## 1. Tech stack & environment
- Python 3.11, packaged with a `requirements.txt` and a `README.md`.
- **Local models:** run via **Ollama** (HTTP API at `http://localhost:11434`).
- **API models:** OpenAI-compatible client and an Anthropic client, selected by config; read keys
  from environment variables only (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.). Never hardcode keys.
- Libraries: `pydantic` (schema validation), `pandas`, `matplotlib`, `pyyaml`, `requests`,
  `pdfplumber` (for raw PDFs), `psutil` and `pynvml` (memory), `tiktoken` (token estimate),
  `tenacity` (retries), `pytest` (tests).
- Everything driven by a single `config/config.yaml`. No magic constants in code.

## 2. Repository structure (create exactly this)
```
slm-vs-llm-kie/
  README.md
  requirements.txt
  config/
    config.yaml
    schemas/{sroie.json, kleister.json}
  src/
    data/{loaders.py, preprocess.py, gt.py}
    models/{base.py, ollama_runner.py, api_runner.py, registry.py}
    prompts/{builder.py, templates/}
    eval/{parse.py, metrics.py, efficiency.py}
    runner/{run.py, results_store.py}
    analysis/{aggregate.py, plots.py, report.py}
  scripts/{download_data.py, run_eval.py, make_plots.py}
  data/        # raw + processed (gitignored)
  results/     # run outputs (gitignored)
  tests/{test_preprocess.py, test_metrics.py, test_parse.py}
```

## 3. Data layer
- **Datasets (start with two):** `SROIE` (receipts, 4 fields — simple) and `Kleister-Charity`
  (financial reports, 8 fields — complex). Make the loader pluggable so `FUNSD`, `CORD`, `DocILE`,
  `Form-NLU` can be added later.
- `download_data.py`: document where to obtain each dataset and place raw files under `data/raw/<name>/`.
  Do **not** scrape; print clear instructions if files are missing.
- **`preprocess.py` — the core normalisation step.** Provide two functions that both output the SAME
  **simple JSON**:
  1. `from_dataset_record(record) -> simple_json` — convert a dataset's OCR words+boxes (+ any
     key-value annotations) into the simple JSON.
  2. `from_textract(blocks_json) -> simple_json` — convert a raw AWS Textract `Blocks` response
     (LINE / WORD / KEY_VALUE_SET / TABLE / CELL) into the same simple JSON.
  - **simple JSON schema:**
    ```json
    {"doc_id": "str", "page": 1,
     "lines": ["text line 1", "text line 2"],
     "key_values": {"DETECTED_KEY": "detected value"},
     "tables": [[["r1c1","r1c2"],["r2c1","r2c2"]]]}
    ```
  - Must be deterministic (stable ordering by reading order: top-to-bottom, left-to-right).
- **`gt.py`:** load each dataset's gold field values into the target schema; build the **20-document
  test set** (configurable: a reproducible mix of simple + complex, fixed by seed) and save the
  selected `doc_id`s to `results/test_set_20.json`.

## 4. Target field schemas (`config/schemas/`)
- `sroie.json`: `["company", "date", "address", "total"]`
- `kleister.json`: `["charity_name", "charity_number", "report_date",
  "income_annually_in_british_pounds", "spending_annually_in_british_pounds",
  "address__post_town", "address__postcode", "address__street_line"]`
- Schemas declare each field's name, type (string/number/date/currency), and whether required.

## 5. Models layer
- `base.py`: define `ModelRunner` with `run(prompt: str) -> RunResult`, where
  `RunResult = {text, latency_s, prompt_tokens, completion_tokens, peak_mem_mb, error}`.
- `ollama_runner.py`: call Ollama REST; set temperature, num_predict, seed from config; capture
  wall-clock latency; capture peak memory via `pynvml` (GPU) or `psutil` (CPU/RSS) sampled during the call.
- `api_runner.py`: OpenAI-compatible + Anthropic; capture latency and token usage from the API response.
- `registry.py`: build the active runners from config. **Locked model set:**
  | id | type | ollama tag / api name |
  |---|---|---|
  | smollm2-360m | local | `smollm2:360m` |
  | qwen2.5-0.5b | local | `qwen2.5:0.5b` |
  | llama3.2-1b | local | `llama3.2:1b` |
  | gemma2-2b | local | `gemma2:2b` |
  | phi4-mini | local | `phi4-mini` |
  | mistral-7b | local | `mistral` |
  | frontier-llm | api | (placeholder — user fills, e.g. a GPT/Claude model name) |
  | large-open-llm | api | (placeholder — e.g. a hosted Llama-70B / DeepSeek) |
  - Verify exact Ollama tags exist before pulling; print a helpful error if a model is missing.

## 6. Prompts layer
- `builder.py`: build the prompt from (a) the target schema and (b) the simple JSON. The instruction
  must tell the model to return **only** a JSON object with exactly the schema fields, using `null`
  for fields it cannot find.
- Support two **conditions**, both config-controlled:
  - `shot_mode`: `zero_shot` | `few_shot` (few-shot inserts K worked examples drawn from the **train**
    split — never from the 20-doc test set, to avoid leakage).
  - `input_variant`: `lines_only` (feed only `lines`) | `lines_plus_kv` (feed `lines` + `key_values`).
- Keep prompts in `templates/` as text files so they're easy to inspect and tweak.

## 7. Evaluation layer
- `parse.py`: robustly extract a JSON object from model output — strip markdown fences, grab the first
  balanced `{...}`, `json.loads`, then validate against the schema with pydantic. On failure, return a
  structured `parse_error` (do not crash the run).
- `metrics.py`: **field-level** comparison with normalisation before matching (trim, lowercase for
  strings; normalise numbers/currency to a canonical numeric; normalise dates to ISO). Compute per
  field and aggregate: **precision, recall, F1 (micro and macro)** and **exact-match accuracy** (per
  field and per document). Count missing vs hallucinated fields separately for error analysis.
- `efficiency.py`: latency (s/doc), throughput (tokens/s), peak memory (MB), and **cost**: for API
  models `cost_usd = tokens × price` from a price table in config; for local models record `cost_usd =
  0` (mark `local`) but still log latency and memory.

## 8. Runner
- `run.py`: main loop over `documents × models × shot_mode × input_variant × n_samples`. Write results
  **incrementally** to `results/runs.jsonl` (resumable — skip already-completed cells). Retry transient
  errors with backoff (`tenacity`). Log progress to console. Each result row records: doc_id, dataset,
  model_id, shot_mode, input_variant, sample_idx, predicted_json, gold_json, all metrics, latency,
  tokens, memory, cost, error, timestamp, and a **run manifest** (config snapshot, model versions, seeds).
- Provide a `--pilot` flag that runs a tiny subset (e.g. 3 docs × 2 models) to validate the pipeline end to end.

## 9. Analysis & outputs
- `aggregate.py`: aggregate `runs.jsonl` into per-model and per-condition tables (mean ± std over
  samples); save `results/summary.csv`.
- `plots.py`: produce PNGs — **(a)** F1 vs parameters (log-scale x), **(b)** cost-per-doc vs F1
  frontier, **(c)** latency vs parameters, **(d)** per-field F1 heatmap by model.
- `report.py`: write `results/REPORT.md` with the summary tables, the plots embedded, and an
  auto-generated findings section (which models lead on quality, on cost, where small models break down).

## 10. Reproducibility, quality, docs
- Pin all versions in `requirements.txt`. Set and record seeds. Save a per-run manifest.
- Unit tests (`pytest`): `preprocess` (dataset & Textract → identical simple JSON), `metrics`
  (known TP/FP/FN cases), `parse` (messy outputs incl. markdown-wrapped JSON and missing fields).
- `README.md`: install Ollama + `ollama pull` each model; `pip install -r requirements.txt`; set API
  keys; download datasets; `python scripts/run_eval.py --pilot` then full run; where to read results.

## Build order (milestones — implement and test in this sequence)
1. Repo scaffold + config + schemas + README skeleton.
2. `preprocess.py` + tests (both converters → identical simple JSON).
3. `loaders.py` + `gt.py` + 20-doc test-set selection.
4. `base.py` + `ollama_runner.py` (get ONE local model answering on ONE doc).
5. `prompts/builder.py` (zero-shot, lines_only).
6. `parse.py` + `metrics.py` + tests.
7. `runner/run.py` with `--pilot`; confirm an end-to-end pilot result row.
8. `api_runner.py` + add the two API models.
9. Add few-shot and `lines_plus_kv` variants.
10. `efficiency.py` wired into the runner (latency, memory, cost).
11. Full run → `aggregate.py` → `plots.py` → `report.py`.

## Acceptance criteria
- `python scripts/run_eval.py --pilot` completes and writes valid rows to `results/runs.jsonl`.
- All `pytest` tests pass.
- `results/summary.csv`, the four plots, and `results/REPORT.md` are generated from a real run.
- Adding a new model or dataset requires only config + a loader — no changes to the runner.
- No API keys, secrets, or dataset contents are committed; `data/` and `results/` are gitignored.
