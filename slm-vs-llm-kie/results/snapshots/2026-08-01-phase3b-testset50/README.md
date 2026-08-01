# Snapshot — 2026-08-01 — Phase 3b — test-set expansion (two-tier bench)

Frozen evidence behind FINDINGS **[E9]** (and the corrected §1–§2 standings). Extends the
Phase-3 bench after a critical review found two headline claims unsupported by the data.
Committed; live `results/` copies are gitignored/regenerable.

## Why this snapshot exists
A hostile-examiner code review (2026-08-01) found:
- The old "inverse scaling / smallest large model wins outright" headline was **false** for
  the current Bedrock large arm (the curve is flat).
- "Recall &lt; precision → the bottleneck is missed fields, not hallucinations" was a
  **test-set artifact**: the frozen 20-doc set has **zero null-gold fields**, so
  hallucination was structurally impossible to observe.

Both are large-arm-internal questions, so the fix is a **large-arm-only** test-set expansion.

## The two-tier design
| Tier | Docs | Models | Purpose |
|---|---|---|---|
| **A** | 20 (`test_set_20.json`) | all 15 | unified small-vs-large standings |
| **B** | 50 (`test_set_50.json`, seed-42 **superset** of the 20) | 5 Bedrock only | size-sensitivity + null-gold hallucination |

Tier B re-runs **only the 5 cloud models** on the +30 new docs (600 cells, $0.42, 0 errors);
the 15-model Tier-A comparison is unchanged (the 20 remain a subset). Local models were
**not** re-scored on the 30 new docs — a stated limitation.

| File | What |
|---|---|
| `runs.jsonl` | 1800 rows: 15 models × 20 docs + 5 cloud × 30 extra docs |
| `test_set_20.json` / `test_set_50.json` | the two seed-42 selections (50 ⊃ 20) |
| `tier_analysis.txt` | `scripts/analyze_tiers.py` output — the tables below |
| `manifest.json` | run provenance |

## Headline results (see `tier_analysis.txt`)
- **[②] Size-insensitive across the large arm (50 docs):** 27B 0.697 · 70B 0.671 · 235B 0.680
  · 400B 0.681 · 671B 0.688. All 95% CIs overlap; top−bottom spread 0.026 (paired CI
  [−0.002, 0.056]); **r(log-params, F1) = −0.21**. Bigger is not better — the false "inverse
  scaling" is replaced by a defensible "size-insensitive."
- **[③] Hallucination is real once null-gold exists:** on the 50-doc set, the large models
  over-extract a spurious value into **36 of 100 null-gold field-instances (36%)** — vs
  **0** on the null-gold-free 20-doc subset. A measured failure mode, not an artifact.
- **Tier-A unified standings** (20 docs, all 15 models): best cloud qwen3-235b **0.714**;
  best local **phi4-mini-ft 0.669** (within **0.045**) — and the fine-tuned locals **win
  exact-match outright** (0.312 / 0.325 vs cloud ≤ 0.225).

**Reproduce:** `python3 scripts/analyze_tiers.py` (read-only over `results/runs.jsonl`);
cloud re-run `bash scripts/run_large_arm.sh` with `test_set` → `test_set_50.json`.
