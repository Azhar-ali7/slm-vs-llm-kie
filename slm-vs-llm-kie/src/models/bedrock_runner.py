"""AWS Bedrock runner — fully-managed serverless models via the Converse API.

Bedrock's Converse API (`bedrock-runtime.converse`) is a single uniform interface
across every serverless model (Meta Llama, Alibaba Qwen, OpenAI GPT-OSS, ...), so one
runner covers the whole large arm. All models used here are on-demand serverless — no
provisioned endpoint, no hourly charge, billed per token only.

Auth (Bedrock API key, preferred): set env `AWS_BEARER_TOKEN_BEDROCK` to a Bedrock API
key (Console -> Bedrock -> API keys). boto3 >= 1.40 picks it up automatically for the
bedrock-runtime client — no IAM keys needed. Standard IAM creds (AWS_ACCESS_KEY_ID /
AWS_PROFILE) also work as a fallback via the default credential chain.

Region note: the largest models (e.g. Llama-3.1-405B) are on-demand only in some
regions and via CROSS-REGION inference profiles. If a bare model id returns
"on-demand throughput isn't supported", use the profile id with a geo prefix
(e.g. `us.meta.llama3-1-405b-instruct-v1:0`) in config — see config.yaml api.bedrock.
"""
from __future__ import annotations

import os
import time
from typing import Any

from src.models.base import ModelRunner, RunResult


class BedrockRunner(ModelRunner):
    def __init__(self, model_id: str, bedrock_model_key: str, api_cfg: dict[str, Any]):
        super().__init__(model_id=model_id, kind="api")
        bc = api_cfg["bedrock"]
        mc = bc["models"][bedrock_model_key]
        self.region = bc.get("region", "us-west-2")
        self.api_key_env = bc.get("api_key_env", "AWS_BEARER_TOKEN_BEDROCK")
        self.model = mc["model"]
        self.temperature = bc.get("temperature", 0.0)
        self.max_tokens = bc.get("max_tokens", 512)
        self.timeout_s = bc.get("request_timeout_s", 90)
        self.price_in = mc.get("price_in", 0.0)
        self.price_out = mc.get("price_out", 0.0)
        self._client = None

    def _has_credentials(self) -> bool:
        return bool(
            os.environ.get(self.api_key_env)          # Bedrock API key (preferred)
            or os.environ.get("AWS_ACCESS_KEY_ID")    # IAM keys fallback
            or os.environ.get("AWS_PROFILE")
        )

    def _missing_reasons(self) -> list[str]:
        reasons = []
        if not self._has_credentials():
            reasons.append(
                f"no Bedrock credentials: set {self.api_key_env} (API key) "
                f"or AWS_ACCESS_KEY_ID / AWS_PROFILE"
            )
        if not self.model or self.model.startswith("<"):
            reasons.append(
                f"model for '{self.model_id}' is a placeholder ({self.model!r}); "
                f"set it in config.yaml api.bedrock.models"
            )
        return reasons

    def ensure_available(self) -> None:
        reasons = self._missing_reasons()
        if reasons:
            raise RuntimeError(
                f"Bedrock model '{self.model_id}' not configured: " + "; ".join(reasons)
            )

    def _get_client(self):
        if self._client is None:
            import boto3  # lazy import

            self._client = boto3.client("bedrock-runtime", region_name=self.region)
        return self._client

    def run(self, prompt: str, response_format: dict | None = None) -> RunResult:
        reasons = self._missing_reasons()
        if reasons:
            return RunResult(text="", latency_s=0.0, error="config: " + "; ".join(reasons))

        start = time.perf_counter()
        try:
            client = self._get_client()
            resp = client.converse(
                modelId=self.model,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={
                    "temperature": self.temperature,
                    "maxTokens": self.max_tokens,
                },
            )
            latency = time.perf_counter() - start
            parts = resp.get("output", {}).get("message", {}).get("content", [])
            text = "".join(p.get("text", "") for p in parts)
            usage = resp.get("usage", {})
            return RunResult(
                text=text,
                latency_s=round(latency, 3),
                prompt_tokens=usage.get("inputTokens"),
                completion_tokens=usage.get("outputTokens"),
                peak_mem_mb=None,  # remote
            )
        except Exception as exc:
            latency = time.perf_counter() - start
            return RunResult(
                text="", latency_s=round(latency, 3), error=f"{type(exc).__name__}: {exc}"
            )
