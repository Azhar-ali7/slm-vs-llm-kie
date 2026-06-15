"""Main evaluation loop: documents × models × shot_mode × input_variant × samples.

Writes results incrementally to results/runs.jsonl (resumable — completed cells
are skipped). Models are iterated outermost so each loads once and is unloaded
before the next (8GB-friendly). Transient errors are retried with backoff.
"""
from __future__ import annotations

import platform
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import load_config, resolve_path
from src.data.gt import (
    align_gold,
    few_shot_pool,
    load_test_records,
    load_test_set,
    schema_for_record,
)
from src.data.preprocess import from_dataset_record
from src.eval.efficiency import efficiency_metrics
from src.eval.metrics import score_document
from src.eval.parse import parse_prediction
from src.models.base import RunResult
from src.models.registry import build_runners, model_meta
from src.prompts.builder import build_prompt, json_schema_format
from src.runner.results_store import append_row, cell_key, completed_keys, write_manifest

_TRANSIENT = ("timeout", "ratelimit", "rate_limit", "connection", "429", "503",
              "overloaded", "temporarily")


def _is_transient(error: str) -> bool:
    e = error.lower()
    return any(t in e for t in _TRANSIENT)


_RETRY_AFTER_RE = re.compile(
    r"retry[_-]after(?:_seconds)?['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)", re.IGNORECASE
)


def _retry_delay(error: str, attempt: int, backoff: float, cap: float = 90.0) -> float:
    """Honor a provider Retry-After if present (e.g. OpenRouter free 429s),
    otherwise exponential backoff."""
    m = _RETRY_AFTER_RE.search(error or "")
    if m:
        return min(float(m.group(1)) + 1.0, cap)
    return min(backoff * (2 ** (attempt - 1)), cap)


def _run_with_retry(runner, prompt: str, attempts: int, backoff: float,
                    response_format: dict | None = None) -> RunResult:
    result = runner.run(prompt, response_format)
    for attempt in range(1, attempts):
        if not result.error or not _is_transient(result.error):
            return result
        time.sleep(_retry_delay(result.error, attempt, backoff))
        result = runner.run(prompt, response_format)
    return result


def run_eval(
    config_path: str | Path | None = None,
    pilot: bool = False,
    skip_large: bool = False,
    only_models: list[str] | None = None,
    only_datasets: list[str] | None = None,
    use_mock: bool = False,
) -> dict[str, Any]:
    cfg = load_config(config_path) if config_path else load_config()
    runs_path = resolve_path(cfg, cfg["paths"]["runs_jsonl"])

    records = load_test_records(cfg)
    if only_datasets:
        records = [r for r in records if r["dataset"] in only_datasets]

    shot_modes = list(cfg["conditions"]["shot_modes"])
    variants = list(cfg["conditions"]["input_variants"])
    n_samples = int(cfg["conditions"]["n_samples"])
    few_shot_k = int(cfg["conditions"]["few_shot_k"])
    max_lines = cfg["conditions"].get("max_input_lines")
    structured = cfg["ollama"].get("structured_output", False)  # local JSON-constrained decoding

    if pilot:
        records = records[: cfg["run"]["pilot_docs"]]
        only_models = only_models or list(cfg["run"]["pilot_models"])
        shot_modes, variants, n_samples = ["zero_shot"], ["lines_only"], 1

    if use_mock:
        from src.models.mock_runner import MockRunner
        runners = [MockRunner()]
    else:
        runners = build_runners(cfg, only=only_models, skip_large=skip_large)
    meta = model_meta(cfg)
    meta.setdefault("mock-extractor", {"type": "mock", "params_b": None, "large": False,
                                       "price_in": 0.0, "price_out": 0.0})
    per_1m = cfg.get("prices_per_1m_tokens", True)
    test_doc_ids = {s["doc_id"] for s in load_test_set(cfg)["doc_ids"]}

    done = completed_keys(runs_path)
    _write_manifest(cfg, runs_path, runners, records, shot_modes, variants, n_samples, pilot)

    print(f"Models: {[r.model_id for r in runners]}")
    print(f"Docs: {len(records)} | shot: {shot_modes} | variants: {variants} | "
          f"samples: {n_samples} | already done: {len(done)} cells")

    total_cost = 0.0
    cells = 0
    for runner in runners:
        try:
            runner.ensure_available()
        except Exception as exc:
            print(f"  [skip] {runner.model_id}: {exc}")
            continue
        print(f"\n=== {runner.model_id} ({runner.kind}) ===")

        for rec in records:
            schema = schema_for_record(cfg, rec)
            gold = align_gold(rec.get("gold", {}), schema)
            simple_json = from_dataset_record(rec)
            # Constrained JSON decoding applies to local (Ollama) models only.
            rf = json_schema_format(schema) if (structured and runner.kind == "local") else None

            for shot in shot_modes:
                examples = (
                    few_shot_pool(cfg, rec, test_doc_ids, few_shot_k)
                    if shot == "few_shot" else None
                )
                for variant in variants:
                    for s in range(n_samples):
                        row_key = (rec["dataset"], rec["doc_id"], runner.model_id, shot, variant, s)
                        if row_key in done:
                            continue
                        prompt = build_prompt(schema, simple_json, shot, variant,
                                              examples, max_lines=max_lines)
                        result = _run_with_retry(
                            runner, prompt,
                            attempts=cfg["run"]["retry_attempts"],
                            backoff=cfg["run"]["retry_backoff_s"],
                            response_format=rf,
                        )
                        pred, perr = parse_prediction(result.text, schema)
                        score = score_document(gold, pred, schema)
                        eff = efficiency_metrics(result, meta[runner.model_id], runner.kind, per_1m)

                        row = _build_row(rec, runner, shot, variant, s, pred, perr,
                                         gold, score, eff, result)
                        append_row(runs_path, row)
                        done.add(cell_key(row))
                        total_cost += eff["cost_usd"] or 0.0
                        cells += 1
                        _print_progress(row, total_cost)

        if hasattr(runner, "unload"):
            try:
                runner.unload()
            except Exception:
                pass

    print(f"\nDone. {cells} new cells. Cumulative API cost this run: ${total_cost:.4f}")
    print(f"Results: {runs_path}")
    return {"cells": cells, "cost_usd": round(total_cost, 4), "runs_path": str(runs_path)}


def _build_row(rec, runner, shot, variant, sample_idx, pred, perr, gold, score, eff, result):
    return {
        "dataset": rec["dataset"],
        "schema_name": rec.get("schema_name", rec["dataset"]),
        "doc_id": rec["doc_id"],
        "model_id": runner.model_id,
        "model_type": runner.kind,
        "shot_mode": shot,
        "input_variant": variant,
        "sample_idx": sample_idx,
        "predicted_json": pred,
        "gold_json": gold,
        "parse_error": perr,
        "precision": score["precision"],
        "recall": score["recall"],
        "f1": score["f1"],
        "exact_match_doc": score["exact_match_doc"],
        "counts": score["counts"],
        "per_field": score["per_field"],
        "latency_s": eff["latency_s"],
        "tokens_per_s": eff["tokens_per_s"],
        "peak_mem_mb": eff["peak_mem_mb"],
        "prompt_tokens": eff["prompt_tokens"],
        "completion_tokens": eff["completion_tokens"],
        "cost_usd": eff["cost_usd"],
        "is_local": eff["is_local"],
        "error": result.error,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _print_progress(row: dict[str, Any], total_cost: float) -> None:
    flag = "ERR" if row["error"] else ("PARSE!" if row["parse_error"] else "ok")
    print(
        f"  {row['model_id']:<14} {row['doc_id']:<20} {row['shot_mode']:<9} "
        f"{row['input_variant']:<13} F1={row['f1']:.2f} lat={row['latency_s']}s "
        f"${row['cost_usd']:.5f} [{flag}] cum=${total_cost:.4f}"
    )


def _write_manifest(cfg, runs_path, runners, records, shot_modes, variants, n_samples, pilot):
    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pilot": pilot,
        "seed": cfg.get("seed"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "models": [{"id": r.model_id, "kind": r.kind} for r in runners],
        "n_docs": len(records),
        "shot_modes": shot_modes,
        "input_variants": variants,
        "n_samples": n_samples,
        "api_provider": cfg.get("api", {}).get("provider"),
        "datasets": sorted({r["dataset"] for r in records}),
    }
    manifest_path = Path(runs_path).parent / "manifest.json"
    write_manifest(manifest_path, manifest)
