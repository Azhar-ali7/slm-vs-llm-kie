# Viva Q&A Prep — SLM vs LLM for Key Information Extraction

Anticipated examiner questions with suggested answers, grounded in this project's
findings (`FINDINGS.md`) and results. **⚠️ = trap question**: a genuine soft spot
in the design — rehearse these cold, each has a clean, honest defence.

---

## A. The core result (they *will* press on this)

**Q: What's your one-sentence finding?**
Parameter count is a poor predictor of KIE quality — architecture and training
dominate. The large arm proves it backwards: 31B (0.701) > 70B (0.667) > 120B
(0.655), and a 4B *local* model (0.639) effectively ties the 120B cloud model.

**Q: Why does the 120B lose to the 31B?**
Mind the boundary: the large arm is cross-family (gpt-oss vs llama3.3 vs gemma),
so it's not a clean size sweep. The *clean* scaling evidence is the **Gemma
family** (1B→2B→4B→31B = 0.417→0.611→0.639→0.701). The cross-family inversion
suggests instruction-tuning and JSON-adherence matter more than raw size for a
*structured, narrow* task. Don't overclaim it as a universal law.

**Q: Isn't 0.70 F1 low? Is any of this production-usable?**
Frame it: zero fine-tuning, text-only, capped context, single sample. The point
is *relative* comparison. Recall (missed fields) is the ceiling — Phase 3 targets it.

---

## B. Methodology / validity (the trap questions)

**⚠️ Q: You gave local models grammar-constrained JSON decoding but the large arm
only prompt-JSON. Isn't that an unfair confound?**
Biggest exposure. Ollama's `format=schema` isn't available on the DO serverless
endpoint, so the large arm couldn't get it. That actually *handicaps the local
arm's competitor* — the large models had the easier deal (no grammar constraint)
and still didn't dominate, so the small-model conclusion is **conservative, not
inflated**. It's a documented asymmetry (see [E2]), not a hidden one.

**⚠️ Q: 20 documents — is that enough to conclude anything? Where are your
confidence intervals?**
Concede it's pilot-scale (20 docs × 4 conditions = 80 cells/model). No
significance test yet; n_samples=1 at temp 0 gives determinism but no variance
estimate. Scaling the test set + significance testing is a Phase-3 item. But the
*effect sizes* (0.30 vs 0.70) are large relative to the sample.

**⚠️ Q: You used open DO-hosted models, not GPT-4o/Claude. Can you call this an
SLM-vs-LLM study?**
Turn the constraint into a virtue: fully open/reproducible large models, no
closed-weights dependency — driven by the DO account tier limit. But scope it
explicitly: it covers *open* large models, not frontier proprietary LLMs.

**Q: `max_input_lines=120` — doesn't truncating Kleister docs unfairly hurt
recall, especially for big models that could use more context?**
Yes — a deliberate cost/memory bound (8 GB M1). Applied equally to all models,
so fair *within* the study, but it does cap the large models' context advantage.
Name it as a limitation.

**Q: Text-only input — so this isn't really document AI, just extraction from
clean OCR?**
Correct — it isolates the *language-model* extraction step, not OCR/layout. A
scoping choice to compare models fairly, not an end-to-end DocAI system.

**Q: How is a field scored "correct"? Couldn't your normalization inflate scores?**
The normalize step (case / whitespace / `$` / date formats) is applied
*identically* to gold and prediction for every model, so it can't bias one model
over another.

---

## C. Technical implementation

**Q: Walk me through the pipeline for one document.**
doc → `simple_json` (lines + key-values) → prompt (`build_prompt`, schema +
shot-mode) → model → `parse_prediction` → `score_document` vs gold → P/R/F1.

**Q: Why temperature 0 and seed 42?**
Determinism / reproducibility; every run is replayable from `runs.jsonl`.

**Q: What is F1 here — token-level or field-level?**
Field-level: TP / wrong / missing / hallucinated over the schema's fields.
precision = tp/(tp+wrong+halluc), recall = tp/(tp+wrong+missing).

**Q: Why did the smallest model improve so much between runs?**
Constrained decoding [E2]: smollm2 went from 39% parse-failures (free-form JSON)
to valid JSON guaranteed. Reliability gain, not capability.

**Q: How do you measure cost for local vs cloud?**
Local = free (peak RSS on M1 reported instead); cloud = real DO token pricing.

---

## D. Results interpretation

**Q: Why is recall below precision for every model, even 120B?**
The bottleneck is *missed* fields, not hallucination — models are cautious.
That's why Phase 3 adds a regex fallback to lift recall [P4].

**Q: Why does few-shot help some models and destroy others?**
Model-dependent: helps mistral/llama3.3/gemma2; catastrophic for llama3.2-1b
(0.529→0.025, context overflow) — small context windows can't absorb exemplars.
Tiny models prefer zero-shot.

**Q: Which fields are hardest?**
Point to the per-field heatmap (`plot_field_heatmap.png`) — the complex 8-field
Kleister schema, address / multi-token fields.

---

## E. Demo-specific

**Q: Is the demo live or cached?**
Both — replay (from `runs.jsonl`, deterministic) vs live (real Ollama call). Show
a live gemma3-4b run to prove it.

**Q: Show me a case where a small model fails and a bigger one succeeds.**
Run smollm2-360m then gemma3-4b on receipt 132 — 0.25 vs 1.00 on the same input.

---

## F. Future work (Phase 3 — have this ready)

- QLoRA fine-tuning of a small model to close the gap.
- Recall-lift: regex / pattern fallback for missed fields [P4].
- Larger test set + significance testing.
- Per-model prompt / shot-mode tuning (few-shot is model-dependent).

---

## Reference numbers (memorise the headline row)

| Model | arm | params | F1 | exact | latency | cost |
|---|---|---|---|---|---|---|
| gemma4-31b | large | 31B | **0.701** | 15/80 | 1.8 s | $0.043 |
| llama3.3-70b | large | 70B | 0.667 | 10/80 | 11.6 s | $0.083 |
| gpt-oss-120b | large | 120B | 0.655 | 3/80 | 13.4 s | $0.050 |
| mistral-7b | local | 7B | 0.642 | 8/80 | 18.7 s | free |
| **gemma3-4b** | local | 4B | **0.639** | 7/80 | 8.9 s | free |
| smollm2-360m | local | 0.36B | 0.306 | 0/80 | 3.1 s | free |

**Three sentences to have ready:**
1. A 4B local model on 8 GB ties a 120B cloud model — for free.
2. Best local (mistral-7b, 0.642) is within 0.06 F1 of the best cloud model and
   *beats* the 120B on exact-match (8/80 vs 3/80).
3. The Gemma family scales cleanly from edge to cloud (0.417→0.701); size alone
   doesn't explain quality — architecture and training do.

---

## The three traps — rehearse cold
1. **Constrained-decoding asymmetry** → it handicaps the local arm; conclusion is conservative.
2. **20-doc sample** → pilot-scale, large effect sizes, significance testing is Phase 3.
3. **Open ≠ frontier LLMs** → reproducibility gain, scoped to open large models.
