# Phase-3 QLoRA fine-tuning — runbook

The Phase-3 headline (FINDINGS `[P2]`): fine-tune a small model on the task's
**train split only**, re-run the **same frozen 20×4 grid**, and report a clean
before→after delta. This document is the end-to-end round-trip. We start with
**phi4-mini** (3.8B); mistral-7b is added once this proves out.

No GPU is needed locally — training runs on **Kaggle (free T4×2)** via Unsloth,
which exports **GGUF** that drops back into the existing Ollama harness as a
`type: local` model. The eval loop, metrics, and plots then work unchanged.

> **Data hygiene (non-negotiable).** The SFT set is built from the train split
> only. The 20-doc test set must never enter training, or the result is
> worthless. `src/train/dataset.py` excludes test doc_ids, `make_train_data.py`
> asserts it before writing, and `tests/test_train_dataset.py` enforces it.

## The pipeline at a glance

```
config/train.yaml
       │
scripts/make_train_data.py ──▶ data/train/sft.jsonl   (CPU, local; train split only)
       │  (upload)
notebooks/03_qlora_finetune.ipynb  (Kaggle T4×2, Unsloth) ──▶ phi4-mini-ft GGUF + Modelfile
       │  (download)
scripts/import_finetuned.sh ──▶ `ollama create phi4-mini-ft`
       │
scripts/run_eval.py --models phi4-mini-ft ──▶ appends 80 cells to results/runs.jsonl
       │
scripts/make_plots.py ──▶ before→after delta in results/REPORT.md + plots
```

## Step 1 — Build the SFT data (local, CPU)

```bash
cd slm-vs-llm-kie && source .venv/bin/activate
python scripts/make_train_data.py           # -> data/train/sft.jsonl
```

Each line is `{prompt, completion, dataset, doc_id, input_variant}`. The `prompt`
is produced by the **same** `build_prompt()` the eval loop uses (zero-shot, both
input variants, same `conditions.max_input_lines`), so the model trains on inputs
byte-identical to what it is later tested on. Knobs (datasets, `max_per_dataset`)
live in `config/train.yaml`; `--all` ignores the cap, `--max-examples N` makes a
smoke set. `data/train/` is gitignored (regenerable).

## Step 2 — Fine-tune on Kaggle (GPU)

1. New Kaggle notebook → **Settings → Accelerator: GPU T4 ×2**, Internet **On**.
2. Upload `notebooks/03_qlora_finetune.ipynb` (or paste it) and `data/train/sft.jsonl`
   (Add Data → upload), or `git clone` this repo inside the notebook.
3. **Run All.** It installs Unsloth, loads `unsloth/Phi-4-mini-instruct` in 4-bit,
   attaches a LoRA adapter, trains on the completions only
   (`train_on_responses_only`), runs a sanity generation, then merges and exports
   **GGUF** (`q4_k_m`) plus an Ollama **Modelfile**, and zips them under `artifacts/`.
4. Download the artifacts zip.

Hyperparameters are read from `config/train.yaml` (`lora:`, `sft:`, `export:`).
Whole run is ~1–2 h on T4×2 with the default `max_per_dataset: 400`.

## Step 3 — Import the GGUF into Ollama (local)

```bash
# after unzipping the Kaggle artifacts into ./artifacts/
bash scripts/import_finetuned.sh artifacts/phi4-mini-ft.Q4_K_M.gguf artifacts/Modelfile
# -> ollama create phi4-mini-ft ; then a smoke `ollama run` on one prompt
```

The Modelfile inherits phi4-mini's chat TEMPLATE/PARAMETERS so `/api/generate`
wrapping at eval time matches how the model was trained.

## Step 4 — Re-run the frozen grid (additive)

```bash
python scripts/run_eval.py --models phi4-mini-ft   # 20 docs × 4 conditions = 80 cells
```

`phi4-mini-ft` is a normal `type: local` entry in `config/config.yaml`. Running
with `--models` appends **only** its rows to `results/runs.jsonl`; the frozen
11-model baseline is untouched (the loop is resumable and keyed per cell).

## Step 5 — Compare

```bash
python scripts/make_plots.py     # summary.csv + plots + REPORT.md, now incl. phi4-mini-ft
```

The before→after story is `phi4-mini` vs `phi4-mini-ft` on the same 20 docs and
conditions: F1 delta, plus latency/cost (both ~unchanged — same size, local) to
support "task-tuning closes the gap at a fraction of a frontier LLM's cost." The
detailed demo notebook (`notebooks/05_demo_detailed.ipynb`) shows the two on a
single document side by side.

## Provenance to record for the thesis

- Base checkpoint + revision, LoRA config, epochs/lr/seed (all in `config/train.yaml`).
- SFT size and composition (printed by `make_train_data.py`).
- The leakage guarantee (this doc + the passing test).
- GGUF quant used (`q4_k_m`) and the Ollama tag (`phi4-mini-ft`).

## Adding mistral-7b next

Uncomment the `mistral-7b-ft` block in `config/train.yaml`, add the matching
`mistral-7b-ft` entry to `config/config.yaml` (copy the phi4-mini-ft entry), then
repeat Steps 2–5. The SFT data is model-agnostic — no rebuild needed.
