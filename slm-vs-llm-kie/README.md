# SLM vs LLM for Key Information Extraction (KIE) from Financial Documents

A reproducible framework comparing **small local language models** (via Ollama)
against **large API models** (Azure OpenAI) at extracting target fields as JSON
from a preprocessed, AWS-Textract-style "simple JSON". It measures both
**extraction quality** (precision / recall / F1 / exact-match) and **operational
efficiency** (latency, peak memory, cost) across zero-/few-shot and input-variant
conditions, and produces an accuracy–cost frontier.

> M.Tech (AI & ML) dissertation — Azhar Ali, BITS ID 2024AA05791, course AIMLCZG628T.

## Pipeline

```
Document → Textract/OCR → preprocess → simple JSON {lines, key_values, tables}
        → build prompt (schema + simple JSON) → run model (Ollama local | Azure API)
        → parse JSON → compare to gold → metrics + plots + report
```

## 1. Install

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Local models (Ollama) — Mac M1 / 8GB notes

Install Ollama (https://ollama.com), then pull the locked model set. **On an 8GB
M1, close other apps first and pull from smallest up.** The framework loads one
model at a time and unloads it immediately (`OLLAMA_KEEP_ALIVE=0`).

```bash
export OLLAMA_KEEP_ALIVE=0
ollama pull smollm2:360m
ollama pull qwen2.5:0.5b
ollama pull llama3.2:1b
ollama pull gemma2:2b
ollama pull phi4-mini      # ~3GB — tight on 8GB
ollama pull mistral        # ~5GB — will swap; opt-in, run last (see --skip-large)
```

`gemma2:2b` is fine with apps closed; `phi4-mini` is tight; `mistral` (7B) will
swap hard on 8GB — treat it as opt-in (run overnight with everything closed, or
skip with `--skip-large`). The 0.36B→2B curve already shows the accuracy-vs-size
trend for the viva.

## 3. Azure OpenAI (the two API models)

```bash
cp .env.example .env     # then fill AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_API_KEY
```

Set your deployment names in `config/config.yaml` under `api.azure.models`
(`frontier-llm` = a GPT-4o/4.1 deployment; `large-open-llm` = a mini deployment).
**Budget:** 6 local models are free; only these two Azure deployments are billed.
A full 20-doc run is ~$5–$12 — keep `n_samples` low and avoid `gpt-4-32k`.

## 4. Data (real datasets — you download, the framework converts)

Both proposal datasets are **text-only** (no images, no 12 GB PDFs). Print the
exact download commands, run them yourself, then convert:

```bash
python scripts/download_data.py        # prints the git clone / curl commands
# ...run those commands to fetch into data/raw/<name>/_src/ ...
python scripts/convert_data.py         # -> data/raw/<name>/{train,test}.jsonl
rm -f results/test_set_20.json         # reselect the 20-doc set on real data
```

- **SROIE** (receipts, 4 fields): sparse-clone `zzzDavid/ICDAR-2019-SROIE`
  (box+key only) → seeded train/test split.
- **Kleister-Charity** (charity reports, 8 fields): four raw blobs
  (`train`/`dev-0` × `in.tsv.xz`+`expected.tsv`) — OCR text + gold, **no
  git-annex**. dev-0 becomes the test split. Long reports are line-capped
  (`conditions.max_input_lines`) to stay cheap on Azure and fit 8 GB locally.

Until real data is present, everything runs on the synthetic `sample` dataset.

## 5. Run

```bash
python scripts/run_eval.py --pilot          # 3 docs × 2 models — validate wiring (cents)
python scripts/run_eval.py                  # full run (use --skip-large on 8GB)
python scripts/make_plots.py                # summary.csv + 4 PNGs + REPORT.md
```

Check the running cost: the summed `cost_usd` in `results/runs.jsonl` should stay
well under your $25 budget before raising `n_samples` or doc count.

## 6. Demo script for the viva

```bash
python scripts/demo.py --replay             # instant, from saved results — can't fail live
python scripts/demo.py --live --doc <id>    # real local + Azure call on one doc
```

Fallback order if anything is flaky: `--replay` → `--live --local-only` → `--live`.

## 7. Report materials (you write the report)

`make_plots.py` produces everything you need to paste into your own report:
`results/summary.csv`, `results/REPORT.md` (tables + auto-findings), the four
result plots, and `report/figures/` (architecture + pipeline diagrams).

`scripts/make_report_doc.py` assembles the BITS mid-semester report as a Word
document (`report/Mid_Sem_Report.docx`) in the prescribed format. The prose is
hand-written; the script lays it out and pulls the result tables live from
`results/runs.jsonl`, so re-running it refreshes the numbers.

## Tests

```bash
pytest -q
```

## Layout

See `config/config.yaml` (single source of truth) and `src/` modules:
`data/` (preprocess, loaders, gt), `models/` (runners, registry),
`prompts/` (builder), `eval/` (parse, metrics, efficiency),
`runner/` (run loop), `analysis/` (aggregate, plots, report, report_figures).
