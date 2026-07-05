"""DigitalOcean serverless inference runner — OpenAI-compatible.

DigitalOcean's GenAI Platform serverless inference (https://inference.do-ai.run/v1)
serves open models (Llama, Qwen, Mistral) *and* frontier models (OpenAI GPT,
Anthropic Claude) behind the OpenAI chat-completions API with a single
"model access key". It is pay-per-token (no idle/hourly charge), so it is the
safe way to run the large arm on the DigitalOcean $200 credit without a GPU
droplet to provision and remember to destroy.

Auth: Authorization: Bearer <model access key>  (env DO_INFERENCE_KEY).
"""
from __future__ import annotations

import os
import time
from typing import Any

from src.models.base import ModelRunner, RunResult


class DigitalOceanRunner(ModelRunner):
    def __init__(self, model_id: str, do_model_key: str, api_cfg: dict[str, Any]):
        super().__init__(model_id=model_id, kind="api")
        doc = api_cfg["digitalocean"]
        mc = doc["models"][do_model_key]
        self.base_url = doc.get("base_url", "https://inference.do-ai.run/v1")
        self.api_key_env = doc.get("api_key_env", "DO_INFERENCE_KEY")
        self.model = mc["model"]
        self.temperature = doc.get("temperature", 0.0)
        self.max_tokens = doc.get("max_tokens", 512)
        self.timeout_s = doc.get("request_timeout_s", 60)
        self.price_in = mc.get("price_in", 0.0)
        self.price_out = mc.get("price_out", 0.0)
        self._client = None

    def _missing_reasons(self) -> list[str]:
        reasons = []
        if not os.environ.get(self.api_key_env):
            reasons.append(f"env {self.api_key_env} not set")
        if not self.model or self.model.startswith("<"):
            reasons.append(
                f"model for '{self.model_id}' is a placeholder ({self.model!r}); "
                f"set it in config.yaml api.digitalocean.models"
            )
        return reasons

    def ensure_available(self) -> None:
        reasons = self._missing_reasons()
        if reasons:
            raise RuntimeError(
                f"DigitalOcean model '{self.model_id}' not configured: " + "; ".join(reasons)
            )

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI  # lazy import

            self._client = OpenAI(
                base_url=self.base_url, api_key=os.environ[self.api_key_env]
            )
        return self._client

    def run(self, prompt: str, response_format: dict | None = None) -> RunResult:
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
