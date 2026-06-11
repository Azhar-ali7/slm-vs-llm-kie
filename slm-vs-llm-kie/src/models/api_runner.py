"""API model runner — Azure OpenAI (active provider).

Azure differs from generic OpenAI: it uses an endpoint + deployment name +
api-version and AZURE_OPENAI_API_KEY / AZURE_OPENAI_ENDPOINT. Latency and token
usage come from the API response; peak memory is N/A for a remote call.
"""
from __future__ import annotations

import os
import time
from typing import Any

from src.models.base import ModelRunner, RunResult


class AzureOpenAIRunner(ModelRunner):
    def __init__(self, model_id: str, azure_model_key: str, api_cfg: dict[str, Any]):
        super().__init__(model_id=model_id, kind="api")
        azure = api_cfg["azure"]
        self.endpoint_env = azure["endpoint_env"]
        self.api_key_env = azure["api_key_env"]
        self.api_version = azure["api_version"]
        self.temperature = azure.get("temperature", 0.0)
        self.max_tokens = azure.get("max_tokens", 512)
        self.timeout_s = azure.get("request_timeout_s", 60)

        model_cfg = azure["models"][azure_model_key]
        self.deployment = model_cfg["deployment"]
        self.price_in = model_cfg.get("price_in", 0.0)
        self.price_out = model_cfg.get("price_out", 0.0)
        self._client = None

    # -- pre-flight ---------------------------------------------------------
    def _missing_reasons(self) -> list[str]:
        reasons = []
        if not os.environ.get(self.endpoint_env):
            reasons.append(f"env {self.endpoint_env} not set")
        if not os.environ.get(self.api_key_env):
            reasons.append(f"env {self.api_key_env} not set")
        if not self.deployment or self.deployment.startswith("<"):
            reasons.append(
                f"deployment for '{self.model_id}' is a placeholder "
                f"({self.deployment!r}); set it in config.yaml api.azure.models"
            )
        return reasons

    def ensure_available(self) -> None:
        reasons = self._missing_reasons()
        if reasons:
            raise RuntimeError(
                f"Azure model '{self.model_id}' not configured: " + "; ".join(reasons)
            )

    def _get_client(self):
        if self._client is None:
            from openai import AzureOpenAI  # lazy import

            self._client = AzureOpenAI(
                azure_endpoint=os.environ[self.endpoint_env],
                api_key=os.environ[self.api_key_env],
                api_version=self.api_version,
            )
        return self._client

    # -- inference ----------------------------------------------------------
    def run(self, prompt: str) -> RunResult:
        reasons = self._missing_reasons()
        if reasons:
            return RunResult(text="", latency_s=0.0, error="config: " + "; ".join(reasons))

        start = time.perf_counter()
        try:
            client = self._get_client()
            resp = client.chat.completions.create(
                model=self.deployment,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=self.timeout_s,
            )
            latency = time.perf_counter() - start
            text = resp.choices[0].message.content or ""
            usage = resp.usage
            return RunResult(
                text=text,
                latency_s=round(latency, 3),
                prompt_tokens=getattr(usage, "prompt_tokens", None),
                completion_tokens=getattr(usage, "completion_tokens", None),
                peak_mem_mb=None,  # remote call
            )
        except Exception as exc:
            latency = time.perf_counter() - start
            return RunResult(
                text="", latency_s=round(latency, 3), error=f"{type(exc).__name__}: {exc}"
            )
