"""Operational efficiency: latency, throughput, peak memory, and cost.

API models: cost_usd = prompt_tokens*price_in + completion_tokens*price_out,
with prices given per 1,000,000 tokens in config. Local models: cost_usd = 0.0
(marked local) but latency and memory are still recorded.
"""
from __future__ import annotations

from typing import Any

from src.models.base import RunResult


def cost_usd(prompt_tokens: int | None, completion_tokens: int | None,
             price_in: float, price_out: float, per_1m: bool = True) -> float:
    if not price_in and not price_out:
        return 0.0
    p = prompt_tokens or 0
    c = completion_tokens or 0
    denom = 1_000_000 if per_1m else 1_000
    return round((p * price_in + c * price_out) / denom, 6)


def efficiency_metrics(result: RunResult, meta: dict[str, Any], kind: str,
                       per_1m: bool = True) -> dict[str, Any]:
    tps = None
    if result.completion_tokens and result.latency_s and result.latency_s > 0:
        tps = round(result.completion_tokens / result.latency_s, 2)

    if kind == "api":
        usd = cost_usd(
            result.prompt_tokens, result.completion_tokens,
            meta.get("price_in", 0.0), meta.get("price_out", 0.0), per_1m=per_1m,
        )
    else:
        usd = 0.0

    return {
        "latency_s": result.latency_s,
        "tokens_per_s": tps,
        "peak_mem_mb": result.peak_mem_mb,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "cost_usd": usd,
        "is_local": kind == "local",
    }
