"""Azure AI Foundry serverless runner.

Foundry serverless models (Phi-4-mini, Ministral, Mistral, Llama-3.3-70B, …) are
reached via the Azure AI Inference SDK's ChatCompletionsClient — a single client
shape that works across every provider in the Foundry catalog. This is distinct
from Azure OpenAI (which only serves GPT models via the openai SDK).

Config (config.yaml api.foundry):
  endpoint_env / api_key_env  -> env vars holding the endpoint URL + key
  api_version                 -> optional
  models.<key>: {model, price_in, price_out, [endpoint_env, api_key_env]}
Per-model endpoint/key overrides support per-deployment serverless endpoints
(each model gets its own *.models.ai.azure.com URL); omit them to share one
unified *.services.ai.azure.com/models endpoint and select by `model` name.
"""
from __future__ import annotations

import os
import time
from typing import Any

from src.models.base import ModelRunner, RunResult


class FoundryRunner(ModelRunner):
    def __init__(self, model_id: str, foundry_model_key: str, api_cfg: dict[str, Any]):
        super().__init__(model_id=model_id, kind="api")  # billed like an API model
        foundry = api_cfg["foundry"]
        mc = foundry["models"][foundry_model_key]
        self.endpoint_env = mc.get("endpoint_env", foundry["endpoint_env"])
        self.api_key_env = mc.get("api_key_env", foundry["api_key_env"])
        self.api_version = mc.get("api_version", foundry.get("api_version"))
        self.model_name = mc.get("model")  # may be None for single-model endpoints
        self.temperature = foundry.get("temperature", 0.0)
        self.max_tokens = foundry.get("max_tokens", 512)
        self.timeout_s = foundry.get("request_timeout_s", 60)
        self.price_in = mc.get("price_in", 0.0)
        self.price_out = mc.get("price_out", 0.0)
        self._client = None

    def _missing_reasons(self) -> list[str]:
        reasons = []
        if not os.environ.get(self.endpoint_env):
            reasons.append(f"env {self.endpoint_env} not set")
        if not os.environ.get(self.api_key_env):
            reasons.append(f"env {self.api_key_env} not set")
        if self.model_name and self.model_name.startswith("<"):
            reasons.append(
                f"model for '{self.model_id}' is a placeholder ({self.model_name!r}); "
                f"set it in config.yaml api.foundry.models"
            )
        return reasons

    def ensure_available(self) -> None:
        reasons = self._missing_reasons()
        if reasons:
            raise RuntimeError(
                f"Foundry model '{self.model_id}' not configured: " + "; ".join(reasons)
            )

    def _get_client(self):
        if self._client is None:
            from azure.ai.inference import ChatCompletionsClient  # lazy import
            from azure.core.credentials import AzureKeyCredential

            kwargs: dict[str, Any] = {}
            if self.api_version:
                kwargs["api_version"] = self.api_version
            self._client = ChatCompletionsClient(
                endpoint=os.environ[self.endpoint_env],
                credential=AzureKeyCredential(os.environ[self.api_key_env]),
                **kwargs,
            )
        return self._client

    def run(self, prompt: str) -> RunResult:
        reasons = self._missing_reasons()
        if reasons:
            return RunResult(text="", latency_s=0.0, error="config: " + "; ".join(reasons))

        start = time.perf_counter()
        try:
            from azure.ai.inference.models import UserMessage

            client = self._get_client()
            kwargs: dict[str, Any] = {
                "messages": [UserMessage(content=prompt)],
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
            }
            if self.model_name:
                kwargs["model"] = self.model_name
            resp = client.complete(**kwargs)
            latency = time.perf_counter() - start
            text = resp.choices[0].message.content or ""
            usage = getattr(resp, "usage", None)
            return RunResult(
                text=text,
                latency_s=round(latency, 3),
                prompt_tokens=getattr(usage, "prompt_tokens", None),
                completion_tokens=getattr(usage, "completion_tokens", None),
                peak_mem_mb=None,  # remote
            )
        except Exception as exc:
            latency = time.perf_counter() - start
            return RunResult(
                text="", latency_s=round(latency, 3), error=f"{type(exc).__name__}: {exc}"
            )
