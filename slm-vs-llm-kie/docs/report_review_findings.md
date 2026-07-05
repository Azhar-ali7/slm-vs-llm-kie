# Mid-Semester Report — Review Findings

Working notes from the report review sessions (June 2026). Each entry records what
was wrong or unclear, what the correct picture is, and how it was resolved in
`scripts/make_report_doc.py` (the report generator). The last section (evaluation
methodology) is analysis only — no code changed yet.

Project: *SLM vs LLM for Key Information Extraction from Financial Documents*
(M.Tech, BITS, 2024AA05791). Phase 2 = 11-model baseline; Phase 3 = QLoRA + recall
techniques (reserved).

---

## 1. Preprocessing description was self-contradictory

**Problem.** The report described preprocessing as "converting raw documents into
clean text lines." The project never sees raw documents — every input arrives
**already OCR'd**, so the wording was wrong.

**Correct picture.** Input arrives in one of two forms:

- **Positioned words with bounding boxes (SROIE receipts)** — grouped into
  reading-order lines by clustering on vertical position
  (`src/data/preprocess.py::_words_to_lines`, tolerance = median height × 0.6).
- **Flat OCR text (Kleister-Charity reports)** — split into lines and the
  financially relevant ones selected (`src/data/convert.py::select_lines`,
  head=60, max_lines=250, financial-cue matching).

Both paths emit the **same simple JSON**: a list of text lines plus, for receipts,
an optional set of key–value pairs. Long reports are capped at `max_input_lines:
120` so the prompt fits the smallest models' context and bounds cloud cost.

**Fix.** Rewrote module (b) in §1 and its summary line. The docx was regenerated.

---

## 2. Prompt structure (zero-shot vs few-shot)

Both datasets use the same template (`src/prompts/builder.py::build_prompt`):

```
<instruction + field spec>          # templates/instruction.txt

### Example 1                        # few-shot ONLY (k=2, TRAIN split, no leakage)
LINES:
 ...
[KEY_VALUES: ...]                    # only in the lines_plus_kv variant
Output:
{"company": "...", "date": "...", ...}

### Example 2
 ...

### Document                         # the test document
LINES:
 ...
Output:
```

- **Zero-shot** omits the `### Example` blocks entirely — instruction + document only.
- **Few-shot** prepends `few_shot_k: 2` worked examples drawn from the **train
  split only**; the 20-doc test set is never used as an example (no leakage).
- The `input_variant` toggles whether a `KEY_VALUES:` block is appended
  (`lines_only` vs `lines_plus_kv`).

---

## 3. The results table mixed shot modes (metric reconciliation)

**Problem.** Table 2 showed a single precision, recall, and F1 per model. These were
a **pooled average over all 80 rows** (both shot modes × both input variants), not
zero-shot or few-shot specifically — so the reader could not tell what the number
meant, and the columns did not correspond to a single set of runs.

**Fix.**

- F1 is now split into **`F1 (zero)`** and **`F1 (few)`** columns
  (`aggregate.py::per_model` now computes `f1_zero_shot` / `f1_few_shot`).
- The remaining columns (**P, R, exact, latency, cost**) are taken from the
  **few-shot runs only**, so the row is internally consistent — every metric comes
  from the same set of runs. Zero-shot F1 is carried alongside as a labelled
  reference column for the shot-mode contrast.
- Table 2 is now **ordered by few-shot F1**, and the surrounding prose was rewritten
  to match that ordering. The abstract's pooled claim ("strongest model is
  Gemma-4-31B") is explicitly labelled **"averaged across the four conditions"** so
  it does not contradict the few-shot Table 2.

**Shot-mode contrasts worth noting.** Few-shot helps the larger models (Llama-3.3
+0.13, Mistral-7B +0.13) but *hurts* several small ones — Llama-3.2-1B collapses
0.53 → 0.03 by copying the example's answer rather than extracting from the document.

---

## 4. "Constrained decoding" and "baseline" — definitions

**Constrained decoding.** The local models are run with JSON-schema-constrained
decoding via Ollama's `format` parameter (`config: structured_output: true`;
`builder.py::json_schema_format` builds an object schema with every target field a
required string; `ollama_runner.py` passes it as `payload["format"]`). It forces
syntactically valid JSON with all keys present. Effect: the smallest model's format
failures dropped from 31/80 to 0. **Local models only** — cloud models are merely
prompt-instructed to emit JSON.

**"Baseline" is used in two distinct senses** in the report (a known source of
confusion — see open item below):

1. **Unconstrained control** — a frozen run *without* the JSON constraint, kept as
   the A/B comparison point to isolate the effect of constrained decoding.
2. **Pre-fine-tuning standings** — the Phase-2 results in Table 2 that Phase-3 QLoRA
   fine-tuning will be measured against.

---

## 5. Evaluation methodology — is the design correct, is n=20 enough, what's the split?

(Analysis only; nothing changed in the report yet.)

### Design
Every model runs the **same fixed grid**: 20 docs × 2 shot modes × 2 input
variants × 1 sample = **80 rows/model**, 880 total. The 20 docs are frozen in
`results/test_set_20.json` (seed 42, not synthetic) and stratified
`sroie: 10, kleister_charity: 10`. Verified against `runs.jsonl`: identical 10 + 10
doc set for all 11 models.

The design is **methodologically sound for a comparative study**:

- **Matched/paired** — identical documents, prompts, conditions, seed, and
  temperature 0 across all models, so differences are attributable to the model.
- **No leakage** — few-shot examples come from the train split; the test set is
  never used as an example.
- **Field-level micro-F1** — partial credit captured, not just doc-level accuracy.

### Is 20 enough?
Acceptable for a mid-semester comparative pilot; **not** sufficient for absolute or
fine-grained claims.

- The scoring unit is **field instances**, not documents: ~**120 present gold
  fields per condition** (40 SROIE = 10×4 fields, 80 Kleister = 10×~8 fields);
  ~480 field decisions pooled across the four conditions per model.
- But documents are the unit of independence, and 20 is small. At F1 ≈ 0.70 with
  ~120 field instances, the 95% CI is roughly **±0.08**.
- **Safe to claim:** broad trends — params don't predict quality; a 4B local model
  rivals a 120B cloud model; few-shot helps big models and hurts tiny ones.
- **Not safe to claim:** that 0.73 beats 0.72 beats 0.71 — adjacent rows are within
  noise (statistically tied).
- *Recommendation for Phase 3 / final write-up:* expand to ~50–100 docs/dataset, or
  report bootstrap confidence intervals so the overlap is explicit.

### Dataset distribution within the 20
- **Document level: balanced** — 10 SROIE / 10 Kleister (stratified, seed-fixed).
- **Field level: imbalanced** — SROIE has 4 target fields, Kleister has 8, so
  micro-F1 is weighted **~⅓ SROIE / ~⅔ Kleister** field instances (40 vs 80).
  Kleister is also the harder dataset (flat OCR, longer docs), so the pooled
  headline F1 leans toward — and is depressed by — Kleister performance.

---

## Open items (not yet actioned)

- **Add a limitations / threats-to-validity note** to the report covering (a) n=20
  supports trends not fine rankings → report CIs, and (b) micro-F1 is field-weighted
  ~2:1 toward Kleister.
- **Disambiguate "baseline"** in the document — e.g. rename sense 1 to "unconstrained
  control" and reserve "baseline" for sense 2 (pre-fine-tuning standings).
- **Figure 4 (cost frontier)** is still computed from the pooled aggregation, so its
  per-doc costs sit below the few-shot Table 2 costs — consistent but a different
  view. Consider regenerating it few-shot-only to match Table 2.
