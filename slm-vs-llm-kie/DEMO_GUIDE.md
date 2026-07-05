# Viva Demo Guide — SLM vs LLM for Key Information Extraction

This guide covers **every way to run the demo** (`scripts/demo.py`): setup, both
modes, all flags, which models you can run, how to read the output, the
recommended viva flow, and troubleshooting.

> **What the demo does:** takes **one document**, has one or more models extract
> its fields, and prints each model's answer next to the gold (correct) answer —
> every field marked ✓ match / ✗ wrong / – missing / + hallucinated — plus F1,
> latency, and cost per model.

---

## 0. One-time setup (do this before you start)

```bash
# 1. Go to the project (internal storage — NOT the USB drive)
cd ~/Desktop/projects/slm-llm/slm-vs-llm-kie

# 2. Activate the virtual environment
source .venv/bin/activate
# your prompt should now start with (.venv)

# 3. Make sure Ollama is running with the local models
ollama list        # should list gemma3:4b, smollm2:360m, etc.
# if the command hangs or is empty, open the Ollama app (or run: ollama serve)
```

**Interpreter note:** run everything with the `(.venv)` Python. If `rich` ever
breaks again (only happens on the USB drive), fall back to
`/opt/anaconda3/bin/python3 scripts/demo.py ...` — it has all the same
dependencies.

---

## 1. The two modes

| Mode | Flag | What it does | Can it fail? |
|------|------|--------------|--------------|
| **Replay** | `--replay` (default) | Reads saved results from `results/runs.jsonl`. **No models are called.** | No — instant & deterministic |
| **Live** | `--live` | **Actually calls** the models on the document right now. | Yes — needs Ollama / network |

Grading (the ✓/✗ marks and F1) is **recomputed live in both modes**, so replay
and live produce the exact same table layout.

---

## 2. Every command you can run

### A. Replay — all models at once (the opening slide)
```bash
python scripts/demo.py --replay
```
Shows **all 11 models** on the default document (receipt 132): F1 climbs from
0.25 (smollm2-360m) to 1.00 (gemma3-4b), with latency and cost each. Safest
possible run.

### B. Live — one local model (the "it's real" proof)
```bash
python scripts/demo.py --live --local-only --models gemma3-4b
```
Calls gemma3-4b for real (~14 s) → **F1=1.00, all 4 fields ✓**. This is the
money moment.

### C. Live — the smallest model (shows the low end)
```bash
python scripts/demo.py --live --local-only --models smollm2-360m
```
~3–9 s, **F1≈0.25** (only `total` correct). Demonstrates why tiny models struggle.

### D. Live — several models side by side
```bash
python scripts/demo.py --live --local-only --models smollm2-360m qwen2.5-0.5b llama3.2-1b gemma3-4b
```
Runs each in turn on the same document → you watch the score climb with size.

### E. Live — a cloud/frontier model (billed, needs internet)
```bash
python scripts/demo.py --live --models frontier-llm
```
Calls the DigitalOcean-hosted 120B model. Costs a fraction of a cent per call.

### F. Replay a chosen subset (compare without calling anything)
```bash
python scripts/demo.py --replay --models gemma3-4b frontier-llm
```
Instantly contrasts a small local model vs the frontier API from saved results.

### G. Pick a different document (works in any mode)
```bash
python scripts/demo.py --replay --doc 480
python scripts/demo.py --live --local-only --models gemma3-4b --doc 074
```

### H. Verbose — show what's happening under the hood
```bash
python scripts/demo.py --replay --verbose
```
Adds **precision / recall / exact-match** (not just F1) to each model's title, plus
a **detail panel** per model showing the **condition** (shot-mode + input-variant),
**token counts**, and **peak memory**. Great for "how is this scored / measured?".

### I. Verbose live — raw output + resource use in real time
```bash
python scripts/demo.py --live --local-only --models gemma3-4b --verbose
```
Same as above **and** prints the model's **raw text output** (before parsing) — the
"here's literally what the model returned" moment.

### J. Show the exact prompt the model receives (live)
```bash
python scripts/demo.py --live --local-only --models gemma3-4b --show-prompt
```
Dumps the full prompt string sent to every model — instruction + field schema +
document lines. Use this to answer "show me exactly what the LLM sees."

---

## 3. The flags, in plain English

| Flag | Meaning |
|------|---------|
| `--replay` | (default) use saved results, call nothing |
| `--live` | actually call the models now |
| `--models A B C` | run/show **only** these model IDs (space-separated) |
| `--local-only` | in live mode, don't auto-add a cloud model (safe/offline) |
| `--doc ID` | show this specific document instead of the default receipt |
| `-v` / `--verbose` | show P/R/exact-match, condition, tokens, peak memory (+ raw output in live) |
| `--show-prompt` | (live) print the full prompt sent to the models |

**Key point:** the demo processes **one file per run**. If you pass three
models, that's *1 document × 3 models*, all on the same input — an
apples-to-apples comparison.

---

## 4. Which models can you run?

### Live-runnable right now — 8 local models (via Ollama)
| Model ID | Size | Typical F1 on receipt 132 |
|----------|------|---------------------------|
| `smollm2-360m` | 0.36B | 0.25 |
| `qwen2.5-0.5b` | 0.5B | 0.25 |
| `llama3.2-1b` | 1B | 0.50 |
| `gemma3-1b` | 1B | 0.50 |
| `gemma2-2b` | 2B | 0.75 |
| `gemma3-4b` | 4B | **1.00** |
| `phi4-mini` | 3.8B | 0.75 |
| `mistral-7b` | 7B | 0.75 |

### Live-runnable but billed + needs internet — 3 cloud models
| Model ID | Backend |
|----------|---------|
| `frontier-llm` | openai-gpt-oss-120b |
| `large-open-llm` | llama3.3-70b-instruct |
| `gemma4-31b` | gemma-4-31B-it |

### Replay-only — not pulled in Ollama
`gemma4-e2b`, `gemma4-e4b` — to run these live first do
`ollama pull gemma4:e2b-it-qat` (and `e4b-it-qat`). Otherwise they only appear
in replay if saved.

---

## 5. Which documents can you show?

**One file per run.** Choose with `--doc`:

- **SROIE receipts (clean, best for a projector):**
  `132` (default) `480` `074` `077` `506` `062` `585` `282` `500` `251`
- **Kleister charity PDFs (longer, harder — SLM gap widens):** 10 available
  (long hash filenames; use receipts for the live demo).

---

## 6. How to read the output

For each model you get a table: **Field | Gold | Predicted | Status**.

| Status | Meaning |
|--------|---------|
| ✓ match | prediction equals gold (after normalizing case/spaces/dates/`$`) |
| ✗ wrong | predicted something, but wrong |
| – missing | field left blank but gold had a value |
| + hallucinated | invented a value where gold was empty |
| · both empty | correctly left blank |

The header line reads: `model-id  F1=x.xx  lat=Ns  $cost`.
- **F1** = harmonic mean of precision & recall over the fields (1.00 = perfect).
- Matching is lenient: `$79.35` counts the same as `79.35`, and dates are
  normalized, so formatting differences don't unfairly penalize a model.

---

## 7. Recommended viva flow (2 commands)

```bash
# 1) Open: all 11 models on one receipt (safe, instant)
python scripts/demo.py --replay

# 2) Prove it's live: a 4B local model nails it in real time
python scripts/demo.py --live --local-only --models gemma3-4b
```

**Pre-warm gemma3-4b right before you present** so the live run is snappy:
```bash
ollama run gemma3:4b "hi" >/dev/null
```

**Fallback ladder if something misbehaves:**
`--replay` (never fails) → `--live --local-only` (needs Ollama) →
`--live` (needs internet).

---

## 8. Troubleshooting

| Symptom | Fix |
|---------|-----|
| `ModuleNotFoundError: No module named 'rich...'` | You're on the USB drive (exFAT corruption). Use the internal-storage copy, or `pip install --force-reinstall rich==13.9.4`, or run with `/opt/anaconda3/bin/python3`. |
| `smollm2-360m unavailable` / preflight ✗ | Ollama isn't running or the model isn't pulled. `ollama list`, then `ollama serve` or `ollama pull <tag>`. |
| Live cloud model errors | No internet or DO key missing. Use `--local-only`, or fall back to `--replay`. |
| First live call is slow | The model was cold. Pre-warm it (section 7). |
| `doc_id ... not found` | Use a valid ID from section 5. |

---

*Generated for the mid-semester viva demo. The demo is the one-document,
on-stage version of the full 20-document × 11-model study in `results/`.*
