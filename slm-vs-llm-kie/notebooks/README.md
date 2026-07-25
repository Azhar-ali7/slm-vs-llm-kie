# Notebooks — demo & walkthrough

Self-contained notebooks that visualise the study. All read the **real** project code and
data (`src/`, `data/raw/`, `results/runs.jsonl`) — no toy stubs — so every table and chart
is the actual data the models see and produce.

| Notebook | What it shows | Runtime |
|---|---|---|
| **`01_eda.ipynb`** | Exploratory data analysis of both datasets (SROIE + Kleister): sizes, schemas, document-length distribution, field fill rates, value formats — and how each observation justified a design decision. | instant (local data only) |
| **`02_inference.ipynb`** | One document walked through **every stage of inference** — OCR → simple JSON → prompt → model call → parse → score → efficiency — for a local SLM (Ollama), an AWS Bedrock LLM, and offline replay. Ends with the 20-doc SLM-vs-LLM comparison. | instant in replay; live cells optional |
| **`03_qlora_finetune.ipynb`** | **Phase-3, Kaggle T4×2.** QLoRA fine-tune of phi4-mini on the train split (Unsloth) → GGUF + Ollama Modelfile. GPU notebook; see `docs/QLORA.md`. | ~1–2 h on Kaggle GPU |
| **`04_eda_presentation.ipynb`** | **Presentation-grade** EDA deck: the same corpus facts as a clean, projector-ready story of *observation → design decision*. | instant (local data only) |
| **`05_demo_detailed.ipynb`** | **Maximally verbose** data→result demo: every intermediate printed for one document, then the 20-doc standings and the Phase-3 fine-tuned before→after (replay fallback until `phi4-mini-ft` exists). | instant in replay; live cells optional |

`01`, `02`, `04`, `05` ship **with outputs already executed**, so they render fully before
you re-run them. `03` is a GPU training skeleton (run it on Kaggle).

## Run locally

```bash
cd slm-vs-llm-kie
source .venv/bin/activate
pip install jupyter            # if not already installed
jupyter lab notebooks/         # or: jupyter notebook
```

Then **Kernel → Restart & Run All**. Everything runs from saved results and local data.

## Live cells (optional, in `02_inference.ipynb`)

- **Local SLM (Ollama):** start Ollama (`ollama serve`) and `ollama pull gemma3:4b`. The
  Stage-3(a) cell then makes a real on-device call (latency + peak memory captured live).
- **AWS Bedrock LLM:** `pip install boto3`, set AWS credentials, and enable the model in the
  Bedrock console (*Model access*). The Stage-3(b) cell then calls the model via the Converse
  API. Swap `BEDROCK_MODEL` for any Bedrock id (Llama, Qwen3, DeepSeek-R1, …).

If a live backend isn't configured, its cell **skips gracefully** and the notebook falls back
to the saved result — so *Run All* never fails on stage.

## Run on Colab / Kaggle

`git clone` the repo, then open the notebook. The path bootstrap in cell 0 finds the repo root
automatically. The EDA notebook needs only the `data/raw/` files; the inference notebook needs
`results/runs.jsonl` for the replay/comparison cells.
