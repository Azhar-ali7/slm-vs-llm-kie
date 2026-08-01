#!/usr/bin/env python3
"""Two-tier analysis of results/runs.jsonl (see FINDINGS [E9]).

Tier A — the common 20-doc set (test_set_20.json), ALL models: the unified
         small-vs-large standings.
Tier B — the 50-doc set (test_set_50.json = seed-42 superset of the 20), the 5
         Bedrock large-arm models only: two large-arm-internal questions that need
         more docs than Tier A provides —
           (2) is quality size-sensitive within the large arm?  (F1 + bootstrap CI)
           (3) do the models hallucinate on genuinely-null gold fields?  (over-extraction)

Pure read-only; prints tables. Bootstrap clusters by doc_id (10k resamples, seeded).
"""
from __future__ import annotations

import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "results" / "runs.jsonl"
FROZEN20 = ROOT / "results" / "test_set_20.json"

CLOUD = ["gemma3-27b", "llama-70b", "qwen3-235b", "llama4-maverick", "deepseek-v3"]
PARAMS_B = {"gemma3-27b": 27, "llama-70b": 70, "qwen3-235b": 235,
            "llama4-maverick": 400, "deepseek-v3": 671}
NULLISH = (None, "", "null")


def load_rows() -> list[dict]:
    return [json.loads(l) for l in RUNS.open()]


def perdoc_f1(rows, model, docset=None) -> dict[str, float]:
    """Mean F1 per doc_id (averages the 4 conditions per doc)."""
    d = defaultdict(list)
    for r in rows:
        if r["model_id"] == model and (docset is None or r["doc_id"] in docset):
            d[r["doc_id"]].append(r["f1"])
    return {k: statistics.mean(v) for k, v in d.items()}


def boot_ci(perdoc: dict[str, float], docs: list[str], n=10000, seed=42):
    rng = random.Random(seed)
    means = []
    for _ in range(n):
        s = [rng.choice(docs) for _ in docs]
        means.append(statistics.mean(perdoc[d] for d in s))
    means.sort()
    return means[int(0.025 * n)], means[int(0.975 * n)]


def tier_a(rows):
    frozen = {x["doc_id"] for x in json.loads(FROZEN20.read_text())["doc_ids"]}
    models = sorted({r["model_id"] for r in rows})
    print("=== TIER A — unified standings, common 20-doc set, all models ===")
    print(f"{'model':16} {'type':6} {'F1':>6}  {'exact':>6}")
    out = []
    for m in models:
        pd = perdoc_f1(rows, m, frozen)
        if len(pd) < 20:  # only models that actually ran all 20
            continue
        f1 = statistics.mean(pd.values())
        em_rows = [r for r in rows if r["model_id"] == m and r["doc_id"] in frozen]
        em = sum(1 for r in em_rows if r["exact_match_doc"]) / len(em_rows)
        typ = em_rows[0]["model_type"]
        out.append((f1, m, typ, em))
    for f1, m, typ, em in sorted(out, reverse=True):
        print(f"{m:16} {typ:6} {f1:6.3f}  {em:6.3f}")


def tier_b(rows):
    print("\n=== TIER B — large arm on 50 docs ===")
    pd = {m: perdoc_f1(rows, m) for m in CLOUD}
    docs = sorted(set().union(*[set(pd[m]) for m in CLOUD]))
    res = {}
    print(f"[2] size-sensitivity  ({len(docs)} docs)")
    for m in sorted(CLOUD, key=lambda x: PARAMS_B[x]):
        f1 = statistics.mean(pd[m][d] for d in docs)
        lo, hi = boot_ci(pd[m], docs)
        res[m] = f1
        print(f"    {m:16} {PARAMS_B[m]:>4}B  F1={f1:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
    top = max(CLOUD, key=lambda m: res[m])
    bot = min(CLOUD, key=lambda m: res[m])
    rng = random.Random(1)
    diffs = []
    for _ in range(10000):
        s = [rng.choice(docs) for _ in docs]
        diffs.append(statistics.mean(pd[top][d] for d in s)
                     - statistics.mean(pd[bot][d] for d in s))
    diffs.sort()
    xs = [math.log(PARAMS_B[m]) for m in CLOUD]
    ys = [res[m] for m in CLOUD]
    mx, my = statistics.mean(xs), statistics.mean(ys)
    r = (sum((x - mx) * (y - my) for x, y in zip(xs, ys))
         / (math.sqrt(sum((x - mx) ** 2 for x in xs)) * math.sqrt(sum((y - my) ** 2 for y in ys))))
    print(f"    top({top} {res[top]:.3f}) - bottom({bot} {res[bot]:.3f}) = {res[top]-res[bot]:.3f}; "
          f"paired ΔF1 95% CI [{diffs[250]:.3f}, {diffs[9750]:.3f}]")
    print(f"    Pearson r(log-params, F1) = {r:.3f}  (~0 ⇒ size-insensitive)")

    print("[3] hallucination on null-gold fields (over-extraction)")
    hall = defaultdict(int)
    nullgold = 0
    for r in rows:
        if r["model_id"] not in CLOUD:
            continue
        hall[r["model_id"]] += r["counts"].get("hallucinated", 0)
        nullgold += sum(1 for v in r["gold_json"].values() if v in NULLISH)
    total = sum(hall.values())
    print(f"    null-gold field-instances (cloud rows): {nullgold}; hallucinated: {total} "
          f"({total/nullgold:.0%} over-extraction)")
    print("    by model:", dict(hall))


if __name__ == "__main__":
    rows = load_rows()
    tier_a(rows)
    tier_b(rows)
