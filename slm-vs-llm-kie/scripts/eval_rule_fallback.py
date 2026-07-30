"""Post-hoc measurement of the rule/regex recall fallback [P4].

Applies `fill_missing_with_rules` to the predictions already stored in
results/runs.jsonl and re-scores with the SAME metrics — NO model re-inference. Prints
a per-model before->after table (F1 / precision / recall), a paired per-doc bootstrap CI
on ΔF1, and a per-field audit of what the rules filled (correct vs wrong), so a
net-negative extractor can be spotted and disabled.

    python scripts/eval_rule_fallback.py            # all models
    python scripts/eval_rule_fallback.py --models phi4-mini-ft
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, resolve_path  # noqa: E402
from src.data.gt import align_gold, load_test_records, schema_for_record
from src.data.preprocess import from_dataset_record
from src.eval.metrics import score_document
from src.eval.rules import fill_missing_with_rules


def _load_doc_context(cfg):
    """(dataset, doc_id) -> (simple_json, schema). Rebuilt from the test records the
    exact same way run.py does, so the fallback sees identical inputs."""
    ctx = {}
    for rec in load_test_records(cfg):
        schema = schema_for_record(cfg, rec)
        ctx[(rec["dataset"], rec["doc_id"])] = (from_dataset_record(rec), schema)
    return ctx


def _bootstrap_ci(doc_deltas: dict, n=10000, seed=42):
    docs = list(doc_deltas)
    if not docs:
        return (0.0, 0.0, 0.0, 1.0)
    obs = st.mean(doc_deltas.values())
    rng = random.Random(seed)
    boot = sorted(
        st.mean(doc_deltas[rng.choice(docs)] for _ in docs) for _ in range(n)
    )
    lo, hi = boot[int(0.025 * n)], boot[int(0.975 * n)]
    p_le0 = sum(b <= 0 for b in boot) / n
    return (obs, lo, hi, p_le0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    args = ap.parse_args()

    cfg = load_config()
    runs_path = resolve_path(cfg, cfg["paths"]["runs_jsonl"])
    ctx = _load_doc_context(cfg)

    rows = [json.loads(l) for l in open(runs_path)]
    if args.models:
        rows = [r for r in rows if r["model_id"] in args.models]

    # per-model accumulators (means over cells)
    agg = collections.defaultdict(lambda: {"f1_b": [], "f1_a": [], "p_b": [], "p_a": [],
                                           "r_b": [], "r_a": []})
    # per-model per-doc ΔF1 for the bootstrap (cluster by doc, mean over its cells)
    doc_delta = collections.defaultdict(lambda: collections.defaultdict(list))
    # per (dataset, field) audit of fills: filled -> new category
    fill_audit = collections.defaultdict(collections.Counter)

    for r in rows:
        key = (r["dataset"], r["doc_id"])
        if key not in ctx:
            continue
        simple_json, schema = ctx[key]
        gold = r.get("gold_json") or align_gold({}, schema)
        pred = r.get("predicted_json")

        before = score_document(gold, pred, schema)
        filled = fill_missing_with_rules(pred, simple_json, schema)
        after = score_document(gold, filled, schema)

        m = r["model_id"]
        a = agg[m]
        a["f1_b"].append(before["f1"]); a["f1_a"].append(after["f1"])
        a["p_b"].append(before["precision"]); a["p_a"].append(after["precision"])
        a["r_b"].append(before["recall"]); a["r_a"].append(after["recall"])
        doc_delta[m][key].append(after["f1"] - before["f1"])

        # audit: fields the rules changed from null -> value. missing->tp/wrong shows
        # recall recovery/error; tn->hallucinated shows a fill that hurt precision.
        pf_before, pf_after = before["per_field"], after["per_field"]
        for f, cat_b in pf_before.items():
            if cat_b in ("missing", "tn") and pf_after[f] != cat_b:
                fill_audit[(r["dataset"], f)][pf_after[f]] += 1

    # ---- per-model table ----
    print(f"\n{'model':16} {'F1 before→after':>18} {'Δ':>7} {'prec b→a':>15} {'rec b→a':>15}")
    print("-" * 76)
    overall_doc_delta = collections.defaultdict(list)
    for m in sorted(agg, key=lambda k: -(st.mean(agg[k]['f1_a']) - st.mean(agg[k]['f1_b']))):
        a = agg[m]
        f1b, f1a = st.mean(a["f1_b"]), st.mean(a["f1_a"])
        pb, pa = st.mean(a["p_b"]), st.mean(a["p_a"])
        rb, ra = st.mean(a["r_b"]), st.mean(a["r_a"])
        flag = "  ⚠ prec↓" if pa < pb - 1e-4 else ""
        print(f"{m:16} {f1b:7.4f}→{f1a:7.4f} {f1a-f1b:+7.4f} "
              f"{pb:6.3f}→{pa:6.3f} {rb:6.3f}→{ra:6.3f}{flag}")
        for d, dv in doc_delta[m].items():
            overall_doc_delta[d].append(st.mean(dv))

    # ---- overall bootstrap on per-doc ΔF1 (averaged across models per doc) ----
    od = {d: st.mean(v) for d, v in overall_doc_delta.items()}
    obs, lo, hi, p = _bootstrap_ci(od)
    print(f"\nOverall per-doc ΔF1 = {obs:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  p(Δ≤0)={p:.3f}")

    # ---- per-field fill audit (which extractors help vs hurt) ----
    print(f"\n{'dataset':16} {'field':34} {'filled':>7} {'→tp':>5} {'→wrong':>7} {'→halluc':>8}")
    print("-" * 82)
    for (d, f), cnt in sorted(fill_audit.items(), key=lambda kv: -sum(kv[1].values())):
        tot = sum(cnt.values())
        print(f"{d:16} {f:34} {tot:7} {cnt['tp']:5} {cnt['wrong']:7} {cnt['hallucinated']:8}")


if __name__ == "__main__":
    main()
