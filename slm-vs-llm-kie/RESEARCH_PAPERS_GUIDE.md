# Research Papers Guide — the four papers in your deck

Everything you need to understand the four papers cited in the Literature Review
(slides 8–12 of `SLM_vs_LLM_Mid_Sem_Review.pptx`): a clickable link to each,
a plain-English **detailed summary** you can actually understand, and a
**"how it grounds my dissertation"** line for the viva.

Read order for understanding: **4 → 3 → 1 → 2** (the two datasets first — they're
concrete — then the two argument papers). The deck presents them 1→4.

Quick reference — where each sits in your story:

| # | Short name | Type | Role in your dissertation |
|---|------------|------|---------------------------|
| 1 | Belcak et al. (NVIDIA) | Position / argument | **Why** small models are worth testing at all |
| 2 | Sinha, Jain, Chadha | Evaluation framework | **How** to run a fair small-vs-large comparison |
| 3 | Kleister | Dataset + benchmark | Your **complex** financial data (the hard end) |
| 4 | SROIE | Dataset + benchmark | Your **simple** data + the F1 scoring protocol |

---

## Paper 1 — Small Language Models are the Future of Agentic AI

**Belcak, Heinrich, Diao, Fu, Dong, Muralidharan, Lin, Molchanov (NVIDIA), 2025**

- **arXiv page:** https://arxiv.org/abs/2506.02153
- **PDF:** https://arxiv.org/pdf/2506.02153
- **HTML (full text):** https://arxiv.org/html/2506.02153
- Position paper (NVIDIA Research), revised Sept 2025.

### How they define an "SLM"
Their definition is deliberately **hardware-relative, not a fixed number**: an SLM
is *"a language model that can fit onto a common consumer electronic device and
perform inference with latency low enough to be practical when serving one user's
agentic requests."* As of 2025 that's roughly **anything under ~10B parameters**.
They keep it timeless (no specific GPU) on purpose. **This matters for your viva:
their SLM definition is the same *deployability* criterion your study uses — "runs
locally on a consumer device" — not an arbitrary parameter cutoff.**

### What it argues
This is a **position paper** — it makes an argument, it does not run an experiment.
It defends one claim through three "value statements":

**V1 — Sufficiently powerful.** Small models are already capable enough for the
*narrow, repetitive* sub-tasks agents actually perform. Their line: *"capability,
not parameter count, is the binding constraint."* They back this with named
models — worth memorising two or three:
- **Phi-2 (2.7B)** — code generation "on par with 30B models" while running ~15× faster.
- **Nemotron-H (2–9B)** — instruction-following + code accuracy "comparable to dense
  30B LLMs" at ~an order of magnitude fewer inference FLOPs.
- **SmolLM2 (125M–1.7B)** — language understanding / tool-calling / instruction
  following rivalling 14B models of the previous generation.
- **DeepSeek-R1-Distill-7B** — reportedly outperforms Claude-3.5-Sonnet and GPT-4o
  on reasoning; **Salesforce xLAM-2-8B** — SOTA tool-calling, beating GPT-4o.

**V2 — Inherently more suitable.** Agents expose only a *narrow slice* of what a
model can do, and call it repeatedly. Small specialist models fit that shape better
than one giant generalist. They argue for **heterogeneous systems**: default to a
small model, escalate to a big LLM only when a task genuinely needs broad reasoning.
Agentic systems also naturally *generate their own fine-tuning data* (every logged
call is a training example).

**V3 — Necessarily more economical.** The headline numbers:
- Serving a **7B SLM is 10–30× cheaper** than a 70–175B LLM in latency, energy, and FLOPs.
- Fine-tuning an SLM takes **a few GPU-hours**, versus weeks for an LLM.
- **10k–100k examples** are enough to fine-tune a small model for a task.
- SLMs run **on-device / on-premise** — no per-token cloud bill, and no data leaving
  the machine.

### The LLM→SLM conversion algorithm (6 steps — S1–S6)
Their practical recipe for replacing big-model calls with small specialists:
1. **S1 — collect usage data**: log the agent's real calls (prompts, responses,
   tool calls, latency), securely.
2. **S2 — curate/filter**: strip PII/PHI, keep ~10k–100k clean examples.
3. **S3 — cluster tasks**: unsupervised clustering to find the recurring request types.
4. **S4 — select an SLM**: by capability fit, benchmark scores, licence, footprint.
5. **S5 — fine-tune**: parameter-efficient methods — **LoRA / QLoRA** — or full
   fine-tuning, optionally with **knowledge distillation** from the original LLM.
6. **S6 — iterate**: retrain periodically as usage drifts.

> **⭐ Direct link to your Phase 3:** step **S5 explicitly recommends QLoRA** — which
> is exactly your planned Phase-3 fine-tuning of phi4-mini / mistral-7b. So your
> Phase 3 isn't an ad-hoc idea; it *instantiates the conversion algorithm this paper
> proposes*. Say that in the viva — it frames your future work as executing a
> published methodology, not improvising.

### Case studies (how much could be SLMs?)
They audit three real agent systems and estimate the share of LLM calls a
specialised SLM could handle: **MetaGPT ≈ 60%**, **Open Operator ≈ 40%**,
**Cradle ≈ 70%**. Routine/templated sub-tasks → SLM; complex multi-step reasoning
→ keep the LLM.

### Counterarguments they pre-empt (useful if an examiner plays devil's advocate)
1. *"Big models are just generally better."* → Scaling laws assume a *fixed*
   architecture; different sizes suit different architectures. And agents decompose
   problems into simple sub-tasks where general ability adds little.
2. *"Centralised LLM serving is cheaper (economies of scale)."* → Inference
   scheduling + modular serving + falling infra costs are eroding that advantage.
3. *"LLMs already have industry momentum."* → They concede this is real, but argue
   the weight of the other arguments can overturn it.

### Why it grounds *your* dissertation
This is your **motivation** — the "recent work argues small models are good enough"
claim on your Research Gap slide, now with specifics. **The gap you fill:** this
paper argues from *economics and architecture*, mostly about *agentic* tasks — it
does **not** empirically test small models on **real financial document
extraction**. Your study supplies that missing evidence, and your Phase 3
*implements its S5 (QLoRA) conversion step*. Viva line: *"Paper 1 makes the
economic and architectural case and even prescribes a QLoRA conversion recipe; my
Phase 2 tests whether the 'good enough' claim holds on real KIE, and my Phase 3
executes their conversion step."*

---

## Paper 2 — Are Small Language Models Ready to Compete with LLMs for Practical Applications?

**Sinha, Jain, Chadha — 2024 (rev. 2025)**

- **arXiv page:** https://arxiv.org/abs/2406.11402
- **PDF:** https://arxiv.org/pdf/2406.11402
- **HTML (full text):** https://arxiv.org/html/2406.11402
- **Published:** TrustNLP 2025 workshop @ NAACL 2025 (peer-reviewed venue — worth
  saying aloud; it's not just a preprint).

### What it does
Unlike Paper 1, this one **runs experiments**. It builds an **evaluation
framework** to answer: *when can a small open model actually replace a big
proprietary one?* — judging outputs along three axes: **task types**, **application
domains**, and **reasoning types**.

**The 10 small models they evaluate** (this is the concrete list — good to know):
| Base (pre-trained) | Instruction-tuned |
|--------------------|-------------------|
| Gemma-2B | Gemma-2B-I |
| Gemma-7B | Gemma-7B-I |
| Llama-3-8B | Llama-3-8B-I |
| Mistral-7B-v0.3 | Mistral-7B-I |
| Falcon-2-11B | SmolLM-1.7B-I |

**Note the overlap with your own registry:** Gemma-2B, Mistral-7B, and the
SmolLM/Llama families appear in *both* their study and yours — so you're testing
several of the *same* models, just on a different task (financial KIE vs general NLP).

### How they run it (the experimental grid)
- **Benchmark:** the test split of **Super-Natural Instructions** (a meta-dataset of
  many standard NLP benchmarks). They sample ≤100 instances per task → **11,810
  instances** across **12 task types, 36 domains, 18 reasoning types**.
- **Prompt styles (8 configurations):** with/without a chat-style task definition ×
  **0, 2, 4, or 8 in-context examples**; plus paraphrased and *adversarial* task
  definitions to test robustness.
- **Metric:** **BERTScore recall** (roberta-large) to measure *semantic* correctness
  — chosen over n-gram metrics (ROUGE/METEOR) because it rewards meaning, not
  surface word overlap.

### Key findings
- **No single winner.** Best model depends on task, domain, and prompt. Among base
  models, **Gemma-2B** won on ~50% of task types, **Falcon-2-11B** on the rest — and
  performance was *"very sensitive across domains"* (Gemma-7B strong on
  health/medical where Falcon-2-11B collapsed).
- **Instruction-tuned Mistral-7B-I was the standout** — best on *all* task types
  among the IT models; Gemma-2B-I and SmolLM-1.7B-I fought for second.
- **⭐ In-context examples plateau after ~2.** Adding a task definition helps, but
  gains *"plateaued after 2 examples."* **This directly justifies your `few_shot_k=2`
  config choice** — you're not guessing at k, you're using the value this paper found
  to be the sweet spot.
- **Small models genuinely compete with SOTA.** vs the frontier models (GPT-4o,
  GPT-4o-mini, Gemini-1.5-Pro, DeepSeek-v2): **Mistral-7B-I was only ~4.9% behind
  Gemini-1.5-Pro and ~2.1% behind GPT-4o**; small models *outperformed*
  GPT-4o-mini, Gemini-1.5-Pro, and DeepSeek-v2 in many categories.

### Why it grounds *your* dissertation
This is your **methodological template**. Your study mirrors its structure:
multiple small models × multiple conditions (zero-shot / few-shot) × a scored task,
compared fairly. Three of its results map straight onto your design and findings:
- *"No single small model wins everywhere"* → your finding that **parameter count
  doesn't track quality**.
- *"Gains plateau after 2 in-context examples"* → **your `few_shot_k=2` is
  evidence-based**, and your finding that **few-shot helps big models but can collapse
  tiny ones** (Llama-3.2-1B, 0.53 → 0.03) is the sharper, task-specific version of
  their "prompt style matters" result.
- *"Small models compete with GPT-4o-class models"* → the general-domain analogue of
  your headline (a 4B local model ties a 120B cloud model on KIE).

**How you extend it:** they measure *general* NLP with a *semantic* metric
(BERTScore) and **no efficiency measurement**. You narrow to **one domain (financial
KIE)**, use **strict field-level F1** (exact values matter for money/dates), and add
the **efficiency axis they omit** (latency, memory, cost). Viva line: *"Paper 2 is my
evaluation blueprint; I specialise it to financial KIE, swap semantic similarity for
strict field-level F1, and add the cost/latency/memory axis it doesn't measure."*

---

## Paper 3 — Kleister: KIE Datasets Involving Long Documents with Complex Layouts

**Stanisławek et al. — ICDAR 2021**

- **arXiv page:** https://arxiv.org/abs/2105.05796
- **PDF:** https://arxiv.org/pdf/2105.05796
- Peer-reviewed: *Proceedings of ICDAR 2021* (a top document-analysis conference).

### What it is
A **benchmark paper** — it introduces datasets, not a model. Most KIE benchmarks
before it used *short* documents (receipts, single forms). Kleister argues that
**real** key-information extraction means **long, multi-page documents with messy
layouts** (tables, headers, footnotes, values split across pages) — genuinely
harder. It introduces two datasets:

- **Kleister-Charity** — annual financial reports of UK charities, with gold
  labels for **8 entity types**: charity name, charity number, report date,
  annual income, annual spending, and address components (post town, postcode,
  street). **This is the one you use.**
- **Kleister-NDA** — non-disclosure agreements (a legal-document counterpart).

The hard part isn't just *finding* the entity — it's that the same value may
appear many times, in different formats, and must be **normalised** (a date, a
monetary amount). They provide OCR + NER pipeline **baselines** (BERT/RoBERTa-era
models) and show that even strong models leave a lot on the table — long-document
KIE is unsolved.

### Why it grounds *your* dissertation
Kleister-Charity is your **"complex" end** and your **ground-truth source** for the
financial half of the study. It's why your Research Question talks about *"the gap
widening as documents grow complex."* Your own results confirm the paper's core
claim: **recall collapses on exactly the hard Kleister fields** — the income and
spending monetary figures buried in long reports (slide 22). In the viva, if asked
"why is your F1 only ~0.70?", part of the honest answer is *"because half my test
set is Kleister, which this paper established is genuinely hard even for
purpose-built models."*

---

## Paper 4 — SROIE: Scanned Receipt OCR and Information Extraction

**Huang, Chen, He, Bai, Karatzas, Lu, Jawahar — ICDAR 2019 Competition**

- **arXiv page:** https://arxiv.org/abs/1905.13538
- **PDF:** https://arxiv.org/pdf/1905.13538
- Peer-reviewed: *ICDAR 2019 Robust Reading Competition* — a very widely-cited,
  standard benchmark.

### What it is
The paper that defined the **SROIE competition and dataset** — now one of the most
common KIE benchmarks. It uses ~**1,000 scanned receipt images** and defines
**three tasks**: (1) text localization, (2) OCR text recognition, and (3) **key
information extraction** — pull **4 fields**: `company`, `date`, `address`,
`total`. **You use Task 3.**

Its lasting contribution is a **clean, agreed scoring protocol**: an
**entity-level F1** where an extracted field counts as correct only if its value
exactly matches the gold value (after light normalisation). Because the field set
is small and unambiguous, SROIE is a **clean measure of extraction accuracy** —
little room to argue about what "correct" means.

### Why it grounds *your* dissertation
SROIE is your **"simple" end** — the easy baseline that even small models should
handle — and, importantly, the **origin of your scoring protocol**. Your
`metrics.py` implements exactly this idea: entity/field-level precision, recall,
and F1 with value normalisation. In the viva, if asked "why is your F1 fair?",
point here: *"I'm using the SROIE entity-level F1 convention — an established,
peer-reviewed protocol — not a metric I invented."* SROIE vs Kleister is your
**simple-vs-complex axis**: the pairing is what reveals *where* the SLM–LLM gap
opens up.

---

## The one-paragraph synthesis (say this if asked "how do these fit together?")

> "Papers 1 and 2 make the case that small models are worth taking seriously —
> Paper 1 from economics and agent architecture, Paper 2 with an actual
> multi-model evaluation framework, which I adopt as my method. But neither tests
> **financial document extraction** specifically. Papers 3 and 4 give me the two
> ends of that task: SROIE (Paper 4) is the simple, clean benchmark and the source
> of my F1 scoring protocol; Kleister-Charity (Paper 3) is the hard, long,
> multi-page financial benchmark. My dissertation sits in the gap between them —
> it takes Paper 2's method, applies it to Papers 3 and 4's data, and empirically
> tests Paper 1's claim on a real KIE task."

---

## 30-second citation lookup (for slides / the report bibliography)

| # | Cite | Link |
|---|------|------|
| 1 | Belcak et al., *Small Language Models are the Future of Agentic AI*, arXiv:2506.02153, 2025 | https://arxiv.org/abs/2506.02153 |
| 2 | Sinha, Jain, Chadha, *Are Small Language Models Ready to Compete with LLMs...*, arXiv:2406.11402, TrustNLP @ NAACL 2025 | https://arxiv.org/abs/2406.11402 |
| 3 | Stanisławek et al., *Kleister: KIE Datasets Involving Long Documents with Complex Layouts*, ICDAR 2021, arXiv:2105.05796 | https://arxiv.org/abs/2105.05796 |
| 4 | Huang et al., *ICDAR2019 Competition on Scanned Receipt OCR and Information Extraction (SROIE)*, ICDAR 2019, arXiv:1905.13538 | https://arxiv.org/abs/1905.13538 |

*Companion to `VIVA_QA.md` (examiner Q&A), `CODE_WALKTHROUGH.md` (the code), and
`DEMO_GUIDE.md` (running the demo).*
