"""OpenRouter runner — one OpenAI-compatible API for many models.

OpenRouter (https://openrouter.ai/api/v1) aggregates Phi, Mistral, Llama, GPT,
Claude, etc. behind the OpenAI chat-completions API, so a single key + the openai
SDK runs every cloud model. Cheaper/simpler than per-provider Azure deployments
when you already have an OPENROUTER_API_KEY.
"""
from __future__ import annotations

import os
import time
from typing import Any

from src.models.base import ModelRunner, RunResult


class OpenRouterRunner(ModelRunner):
    def __init__(self, model_id: str, openrouter_model_key: str, api_cfg: dict[str, Any]):
        super().__init__(model_id=model_id, kind="api")
        orc = api_cfg["openrouter"]
        mc = orc["models"][openrouter_model_key]
        self.base_url = orc.get("base_url", "https://openrouter.ai/api/v1")
        self.api_key_env = orc["api_key_env"]
        self.model = mc["model"]
        self.temperature = orc.get("temperature", 0.0)
        self.max_tokens = orc.get("max_tokens", 512)
        self.timeout_s = orc.get("request_timeout_s", 60)
        self.price_in = mc.get("price_in", 0.0)
        self.price_out = mc.get("price_out", 0.0)
        # Optional OpenRouter ranking headers.
        self.referer = orc.get("referer", "https://github.com/Azhar-ali7")
        self.title = orc.get("title", "slm-vs-llm-kie")
        self._client = None

    def _missing_reasons(self) -> list[str]:
        reasons = []
        if not os.environ.get(self.api_key_env):
            reasons.append(f"env {self.api_key_env} not set")
        if not self.model or self.model.startswith("<"):
            reasons.append(
                f"model for '{self.model_id}' is a placeholder ({self.model!r}); "
                f"set it in config.yaml api.openrouter.models"
            )
        return reasons

    def ensure_available(self) -> None:
        reasons = self._missing_reasons()
        if reasons:
            raise RuntimeError(
                f"OpenRouter model '{self.model_id}' not configured: " + "; ".join(reasons)
            )

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI  # lazy import

            self._client = OpenAI(
                base_url=self.base_url, api_key=os.environ[self.api_key_env]
            )
        return self._client

    def run(self, prompt: str) -> RunResult:
        reasons = self._missing_reasons()
        if reasons:
            return RunResult(text="", latency_s=0.0, error="config: " + "; ".join(reasons))

        start = time.perf_counter()
        try:
            client = self._get_client()
            resp = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=self.timeout_s,
                extra_headers={"HTTP-Referer": self.referer, "X-Title": self.title},
            )
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
