#!/usr/bin/env python3
"""Post-hoc: recompute strict vs normalised-match (lenient) F1 from stored
predictions in results/runs.jsonl — NO re-inference. Shows how much of each
model's apparent error is format-only (punctuation/whitespace) rather than
genuine. Strict is the headline number; lenient is reported alongside.

    python scripts/lenient_rescore.py
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.eval.metrics import score_document  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SCHEMAS = {
    "sroie": json.loads((ROOT / "config/schemas/sroie.json").read_text()),
    "kleister_charity": json.loads((ROOT / "config/schemas/kleister.json").read_text()),
}


def main() -> None:
    runs = ROOT / "results/runs.jsonl"
    strict = defaultdict(list)
    lenient = defaultdict(list)
    for line in runs.open():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("error") or r.get("parse_error"):
            # keep parse failures as F1 0 in both (already reflected in stored rows)
            strict[r["model_id"]].append(0.0)
            lenient[r["model_id"]].append(0.0)
            continue
        schema = SCHEMAS.get(r.get("schema_name"))
        gold = r.get("gold_json") or {}
        pred = r.get("predicted_json")
        if schema is None or pred is None:
            continue
        strict[r["model_id"]].append(score_document(gold, pred, schema)["f1"])
        lenient[r["model_id"]].append(score_document(gold, pred, schema, lenient=True)["f1"])

    rows = []
    for m in strict:
        s = sum(strict[m]) / len(strict[m]) if strict[m] else 0.0
        l = sum(lenient[m]) / len(lenient[m]) if lenient[m] else 0.0
        rows.append((m, s, l, l - s, len(strict[m])))
    rows.sort(key=lambda x: -x[2])

    print(f"{'model':18} {'strict':>7} {'lenient':>8} {'Δ':>7} {'cells':>6}")
    print("-" * 50)
    for m, s, l, d, n in rows:
        print(f"{m:18} {s:7.3f} {l:8.3f} {d:+7.3f} {n:6}")


if __name__ == "__main__":
    main()
