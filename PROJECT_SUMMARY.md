# Project Summary — SLM vs LLM for Key Information Extraction from Financial Documents

**Student:** Azhar Ali · BITS ID 2024AA05791 · M.Tech AI & ML · Course AIMLCZG628T
**One-line:** Can small, locally-run language models match large LLMs at extracting key fields from
financial documents — and at what cost, speed, and memory?

---

## 1. What the finished project is
A reproducible Python framework that feeds the **same financial documents** (as a preprocessed,
Textract-style "simple JSON") to **eight models** ranging from 0.36B to 70B+ parameters, asks each to
extract a fixed set of fields as JSON, and then scores every model on **accuracy** and **efficiency**.
The output is an **accuracy–efficiency frontier** plus deployment guidance.

## 2. What it produces (your deliverables)
- `results/runs.jsonl` — every model's answer on every document, with all metrics.
- `results/summary.csv` — per-model, per-condition results (mean ± std).
- Four plots — F1 vs model size, cost vs F1 frontier, latency vs size, per-field F1 heatmap.
- `results/REPORT.md` — tables + plots + an auto-written findings section.
- A clean, documented codebase you can demo live in the viva.

## 3. How the pieces fit (the pipeline)
1. **Document** → 2. **Textract / OCR → preprocess into simple JSON** (`{lines, key_values, tables}`)
→ 3. **Build prompt** (schema + simple JSON) → 4. **Run model** (local via Ollama or API)
→ 5. **Parse JSON output** → 6. **Compare to gold** → 7. **Metrics + plots**.

## 4. The experiment grid
- **Models (8):** SmolLM2-360M, Qwen2.5-0.5B, Llama-3.2-1B, Gemma-2-2B, Phi-4-mini, Mistral-7B
  (all local, free, via Ollama) + 2 large API models (you choose).
- **Datasets (2):** SROIE (simple, 4 fields) and Kleister-Charity (complex financial, 8 fields).
- **Conditions:** zero-shot vs few-shot; and `lines_only` vs `lines + key_values` input.
- **Test set:** a fixed, reproducible 20-document mix (simple + complex).
- **Metrics:** precision, recall, F1, exact-match (quality); latency, memory, cost (efficiency).

## 5. How it maps to your first-viva feedback
| Feedback point | Where it lives in the project |
|---|---|
| 5 "agents" / models | The 8-model registry (`src/models/registry.py`) |
| Literature review | Your proposal doc + deck (already built) |
| Metrics (one per slide) | `src/eval/metrics.py` + `efficiency.py` |
| Framework architecture (mid-viva) | The whole `src/` pipeline + the architecture slide |
| 20 inputs + correct output (GT) | `src/data/gt.py` → `results/test_set_20.json` |
| Test & evaluate | `src/runner/run.py` → `results/REPORT.md` |

## 6. What YOU need to supply / do (checklist)
- [ ] A machine that can run Ollama (a modern laptop is fine; a GPU helps for the 7B model).
- [ ] Install Ollama and `ollama pull` the six local models (verify exact tags on ollama.com).
- [ ] Pick the **two large API models** and add their names + API keys as environment variables.
- [ ] Download the datasets (SROIE, Kleister-Charity) into `data/raw/` (instructions printed by the script).
- [ ] (Optional) An AWS account if you want *genuine* Textract JSON; otherwise the dataset OCR is converted for you.
- [ ] Run the pilot first (`--pilot`), confirm it works, then the full run.
- [ ] Replace the deck's "representative" dataset sample with one real record before the viva.

## 7. Cost & time expectations
- The six local models cost **nothing** to run (just your electricity/time).
- The two API models incur small usage fees — keep the test set to 20 docs to stay cheap.
- A full run over 20 docs × 8 models × conditions is a matter of hours, not days, on a normal laptop.

## 8. Honest limitations to state in your thesis
- Small public sample (20 docs) — report variance and treat results as indicative.
- Mixed model families means size isn't perfectly isolated (you compare *best-in-class per size*).
- Public benchmarks may overlap with training data; your custom held-out set mitigates this partially.
- No large public dataset ships *natively* as Textract JSON — you convert equivalent OCR+KV data.
- Text-in only (no image/vision); fine-tuning is out of scope (optional extension).

## 9. Suggested next steps after the build runs
1. Run the pilot, fix any model-tag issues, then the full run.
2. Drop the real numbers into the deck's "Expected Outcomes" slide (replace the illustrative chart).
3. Write the results chapter around `REPORT.md`'s findings.
4. If time allows, add a third dataset (DocILE or Form-NLU) or the optional fine-tuning experiment.

## 10. Files you now have
- `Dissertation_Proposal_SLM_vs_LLM.docx` — full proposal in your institution's format.
- `SLM_vs_LLM_Proposal_Deck.pptx` — 20-slide content-rich deck.
- `CLAUDE_CODE_BUILD_PROMPT.md` — paste into Claude Code to build the project.
- `PROJECT_SUMMARY.md` — this document.
