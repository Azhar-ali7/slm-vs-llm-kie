"""Google Generative Language API runner — Gemma / Gemini via `generateContent`.

Used for the Gemma arm (Gemma isn't on AWS Bedrock). Talks to
`https://generativelanguage.googleapis.com/v1beta/models/<model>:generateContent`
with a simple API key (Google AI Studio -> Get API key). Gemma models via this API are
free within generous rate limits, so the cost figure is $0 (token counts still recorded).

Auth: env `GEMINI_API_KEY`. Uses httpx (already a dependency) — no extra SDK.
"""
from __future__ import annotations

import os
import time
from typing import Any

from src.models.base import ModelRunner, RunResult


class GoogleRunner(ModelRunner):
    def __init__(self, model_id: str, google_model_key: str, api_cfg: dict[str, Any]):
        super().__init__(model_id=model_id, kind="api")
        gc = api_cfg["google"]
        mc = gc["models"][google_model_key]
        self.base_url = gc.get("base_url", "https://generativelanguage.googleapis.com/v1beta")
        self.api_key_env = gc.get("api_key_env", "GEMINI_API_KEY")
        self.model = mc["model"]
        self.temperature = gc.get("temperature", 0.0)
        self.max_tokens = gc.get("max_tokens", 512)
        self.timeout_s = gc.get("request_timeout_s", 60)
        self.price_in = mc.get("price_in", 0.0)
        self.price_out = mc.get("price_out", 0.0)

    def _missing_reasons(self) -> list[str]:
        reasons = []
        if not os.environ.get(self.api_key_env):
            reasons.append(f"env {self.api_key_env} not set")
        if not self.model or self.model.startswith("<"):
            reasons.append(
                f"model for '{self.model_id}' is a placeholder ({self.model!r}); "
                f"set it in config.yaml api.google.models"
            )
        return reasons

    def ensure_available(self) -> None:
        reasons = self._missing_reasons()
        if reasons:
            raise RuntimeError(
                f"Google model '{self.model_id}' not configured: " + "; ".join(reasons)
            )

    def run(self, prompt: str, response_format: dict | None = None) -> RunResult:
        reasons = self._missing_reasons()
        if reasons:
            return RunResult(text="", latency_s=0.0, error="config: " + "; ".join(reasons))

        import httpx  # lazy import

        url = f"{self.base_url}/models/{self.model}:generateContent"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_tokens,
            },
        }
        start = time.perf_counter()
        try:
            resp = httpx.post(
                url,
                params={"key": os.environ[self.api_key_env]},
                json=payload,
                timeout=self.timeout_s,
            )
            resp.raise_for_status()
            data = resp.json()
            latency = time.perf_counter() - start
            candidates = data.get("candidates", [])
            parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
            text = "".join(p.get("text", "") for p in parts)
            usage = data.get("usageMetadata", {})
            return RunResult(
                text=text,
                latency_s=round(latency, 3),
                prompt_tokens=usage.get("promptTokenCount"),
                completion_tokens=usage.get("candidatesTokenCount"),
                peak_mem_mb=None,  # remote
            )
        except Exception as exc:
            latency = time.perf_counter() - start
            detail = f"{type(exc).__name__}: {exc}"
            if isinstance(exc, httpx.HTTPStatusError):
                detail += f" | body: {exc.response.text[:200]}"
            return RunResult(text="", latency_s=round(latency, 3), error=detail)
