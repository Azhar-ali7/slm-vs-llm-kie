"""Aggregate runs.jsonl into per-model and per-condition tables."""
from __future__ import annotations

from typing import Any

import pandas as pd

from src.config import resolve_path
from src.eval.metrics import micro_from_counts
from src.models.registry import model_meta

_COUNT_KEYS = ("tp", "wrong", "missing", "hallucinated", "tn")


def load_results_df(cfg: dict[str, Any]) -> pd.DataFrame:
    path = resolve_path(cfg, cfg["paths"]["runs_jsonl"])
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(_flatten_row(line))
    return pd.DataFrame(rows)


def _flatten_row(line: str) -> dict[str, Any]:
    import json

    r = json.loads(line)
    counts = r.get("counts") or {}
    flat = {k: v for k, v in r.items() if k not in ("counts", "per_field", "predicted_json",
                                                    "gold_json", "parse_error")}
    for ck in _COUNT_KEYS:
        flat[f"c_{ck}"] = counts.get(ck, 0)
    flat["parse_failed"] = bool(r.get("parse_error"))
    flat["per_field"] = r.get("per_field") or {}
    return flat


def _micro(group: pd.DataFrame) -> dict[str, float]:
    counts = {ck: int(group[f"c_{ck}"].sum()) for ck in _COUNT_KEYS}
    return micro_from_counts(counts)


def per_model_condition(df: pd.DataFrame) -> pd.DataFrame:
    """Mean ± std over samples, grouped by model × shot_mode × input_variant."""
    keys = ["model_id", "model_type", "shot_mode", "input_variant"]
    out = []
    for vals, g in df.groupby(keys, dropna=False):
        micro = _micro(g)
        out.append({
            **dict(zip(keys, vals)),
            "n": len(g),
            "f1_macro_mean": round(g["f1"].mean(), 4),
            "f1_macro_std": round(g["f1"].std(ddof=0), 4),
            "f1_micro": micro["f1"],
            "precision_micro": micro["precision"],
            "recall_micro": micro["recall"],
            "exact_match_rate": round(g["exact_match_doc"].mean(), 4),
            "latency_s_mean": round(g["latency_s"].mean(), 3),
            "tokens_per_s_mean": round(g["tokens_per_s"].dropna().mean(), 2)
            if g["tokens_per_s"].notna().any() else None,
            "peak_mem_mb_mean": round(g["peak_mem_mb"].dropna().mean(), 1)
            if g["peak_mem_mb"].notna().any() else None,
            "cost_usd_per_doc": round(g["cost_usd"].mean(), 6),
            "parse_fail_rate": round(g["parse_failed"].mean(), 4),
            "missing": int(g["c_missing"].sum()),
            "hallucinated": int(g["c_hallucinated"].sum()),
        })
    return pd.DataFrame(out).sort_values(["model_id", "shot_mode", "input_variant"])


def per_model(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """One row per model across all conditions (for the size/cost plots)."""
    meta = model_meta(cfg)
    out = []
    for model_id, g in df.groupby("model_id"):
        micro = _micro(g)
        m = meta.get(model_id, {})
        # Cost-per-doc range from REAL mean token counts × the model's price band.
        # Billed models: low == high (one real rate). Local models: a market band
        # (what renting them would cost) — see model_meta / config hosted_price.
        pt = g["prompt_tokens"].dropna().mean() or 0.0
        ct = g["completion_tokens"].dropna().mean() or 0.0
        cost_low = (pt * m.get("price_in_low", 0.0) + ct * m.get("price_out_low", 0.0)) / 1e6
        cost_high = (pt * m.get("price_in_high", 0.0) + ct * m.get("price_out_high", 0.0)) / 1e6
        # F1 split by shot mode — the pooled f1_macro averages the two and hides
        # that few-shot helps the larger models but hurts several small ones.
        zs = g[g["shot_mode"] == "zero_shot"]["f1"]
        fs = g[g["shot_mode"] == "few_shot"]["f1"]
        out.append({
            "model_id": model_id,
            "display_name": m.get("display", model_id),
            "model_type": g["model_type"].iloc[0],
            "params_b": m.get("params_b"),
            "f1_macro": round(g["f1"].mean(), 4),
            "f1_zero_shot": round(zs.mean(), 4) if len(zs) else None,
            "f1_few_shot": round(fs.mean(), 4) if len(fs) else None,
            "f1_micro": micro["f1"],
            "precision": micro["precision"],
            "recall": micro["recall"],
            "exact_match_rate": round(g["exact_match_doc"].mean(), 4),
            "latency_s": round(g["latency_s"].mean(), 3),
            "peak_mem_mb": round(g["peak_mem_mb"].dropna().mean(), 1)
            if g["peak_mem_mb"].notna().any() else None,
            "cost_usd_per_doc": round(g["cost_usd"].mean(), 6),
            "cost_low_per_doc": round(cost_low, 6),
            "cost_high_per_doc": round(cost_high, 6),
            "cost_mid_per_doc": round((cost_low + cost_high) / 2, 6),
            "parse_fail_rate": round(g["parse_failed"].mean(), 4),
        })
    df_out = pd.DataFrame(out)
    # Order by params where known, then by F1.
    return df_out.sort_values(["params_b", "f1_macro"], na_position="last").reset_index(drop=True)


def per_field_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    """model × field accuracy = fraction of (doc,sample) where the field is tp or tn."""
    records: dict[str, dict[str, list[int]]] = {}
    for _, row in df.iterrows():
        model = row["model_id"]
        for field, cat in (row["per_field"] or {}).items():
            records.setdefault(model, {}).setdefault(field, []).append(1 if cat in ("tp", "tn") else 0)
    table = {}
    for model, fields in records.items():
        table[model] = {f: round(sum(v) / len(v), 3) for f, v in fields.items()}
    return pd.DataFrame(table).T  # rows=models, cols=fields


def save_summary(cfg: dict[str, Any], df_condition: pd.DataFrame) -> str:
    out = resolve_path(cfg, cfg["paths"]["summary_csv"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df_condition.to_csv(out, index=False)
    return str(out)
