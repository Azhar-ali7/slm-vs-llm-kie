"""Build the active model runners from config.

Adding a model is config-only: add an entry to `models:` (and, for an API model,
to `api.azure.models`). The runner loop never changes.
"""
from __future__ import annotations

from typing import Any

from src.models.api_runner import AzureOpenAIRunner
from src.models.base import ModelRunner
from src.models.ollama_runner import OllamaRunner


def _make_runner(cfg: dict[str, Any], model: dict[str, Any]) -> ModelRunner:
    mtype = model["type"]
    if mtype == "local":
        return OllamaRunner(model_id=model["id"], tag=model["tag"], ollama_cfg=cfg["ollama"])
    if mtype == "api":
        provider = cfg["api"].get("provider", "azure_openai")
        if provider != "azure_openai":
            raise NotImplementedError(
                f"api.provider '{provider}' not implemented; this build uses azure_openai."
            )
        return AzureOpenAIRunner(
            model_id=model["id"], azure_model_key=model["azure_model"], api_cfg=cfg["api"]
        )
    if mtype == "foundry":
        from src.models.foundry_runner import FoundryRunner
        return FoundryRunner(
            model_id=model["id"], foundry_model_key=model["foundry_model"], api_cfg=cfg["api"]
        )
    if mtype == "openrouter":
        from src.models.openrouter_runner import OpenRouterRunner
        return OpenRouterRunner(
            model_id=model["id"], openrouter_model_key=model["openrouter_model"], api_cfg=cfg["api"]
        )
    if mtype == "do_serverless":
        from src.models.do_runner import DigitalOceanRunner
        return DigitalOceanRunner(
            model_id=model["id"], do_model_key=model["do_model"], api_cfg=cfg["api"]
        )
    raise ValueError(f"Unknown model type {mtype!r} for {model['id']!r}")


def build_runners(
    cfg: dict[str, Any],
    only: list[str] | None = None,
    skip_large: bool = False,
) -> list[ModelRunner]:
    """Return runners in config order (smallest local -> largest -> API).

    `only` restricts to specific model ids; `skip_large` drops models flagged
    `large: true` (phi4-mini, mistral-7b) — useful on 8GB.
    """
    runners: list[ModelRunner] = []
    for model in cfg["models"]:
        if only and model["id"] not in only:
            continue
        if skip_large and model.get("large"):
            continue
        runners.append(_make_runner(cfg, model))
    return runners


def model_meta(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """id -> {type, params_b, large, price_in, price_out} for analysis/plots."""
    meta: dict[str, dict[str, Any]] = {}
    azure_models = cfg.get("api", {}).get("azure", {}).get("models", {})
    foundry_models = cfg.get("api", {}).get("foundry", {}).get("models", {})
    openrouter_models = cfg.get("api", {}).get("openrouter", {}).get("models", {})
    do_models = cfg.get("api", {}).get("digitalocean", {}).get("models", {})
    for model in cfg["models"]:
        entry = {
            "type": model["type"],
            "params_b": model.get("params_b"),
            "large": bool(model.get("large", False)),
            # Human-readable name for tables/plots; config ids are generic
            # (e.g. frontier-llm -> GPT-OSS-120B). Fall back to the id.
            "display": model.get("display", model["id"]),
            "price_in": 0.0,
            "price_out": 0.0,
        }
        if model["type"] == "api":
            am = azure_models.get(model.get("azure_model"), {})
            entry["price_in"] = am.get("price_in", 0.0)
            entry["price_out"] = am.get("price_out", 0.0)
        elif model["type"] == "foundry":
            fm = foundry_models.get(model.get("foundry_model"), {})
            entry["price_in"] = fm.get("price_in", 0.0)
            entry["price_out"] = fm.get("price_out", 0.0)
        elif model["type"] == "openrouter":
            om = openrouter_models.get(model.get("openrouter_model"), {})
            entry["price_in"] = om.get("price_in", 0.0)
            entry["price_out"] = om.get("price_out", 0.0)
        elif model["type"] == "do_serverless":
            dm = do_models.get(model.get("do_model"), {})
            entry["price_in"] = dm.get("price_in", 0.0)
            entry["price_out"] = dm.get("price_out", 0.0)
        # Price range for the accuracy-cost figure. Billed models have a single
        # real rate (low == high). Local models carry a *hypothetical* market
        # band (hosted_price) — what it would cost to rent them — since "$0"
        # misrepresents the economics even though on-device has no per-token bill.
        hp = model.get("hosted_price")
        if hp:
            entry["price_in_low"], entry["price_in_high"] = hp["in"]
            entry["price_out_low"], entry["price_out_high"] = hp["out"]
        else:
            entry["price_in_low"] = entry["price_in_high"] = entry["price_in"]
            entry["price_out_low"] = entry["price_out_high"] = entry["price_out"]
        meta[model["id"]] = entry
    return meta
