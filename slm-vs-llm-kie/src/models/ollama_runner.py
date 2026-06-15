"""Local model runner via the Ollama REST API.

M1 / 8GB notes:
 - peak memory is sampled from the *ollama server* process tree (psutil RSS)
   during the call — there is no NVIDIA GPU, so pynvml is not used.
 - keep_alive=0 unloads the model right after each call to free unified memory.
"""
from __future__ import annotations

import os
import threading
import time
from typing import Any

import requests

from src.models.base import ModelRunner, RunResult


class _OllamaMemSampler:
    """Background sampler of the ollama process-tree RSS; reports peak in MB."""

    def __init__(self, interval_s: float = 0.25):
        self.interval_s = interval_s
        self._peak_bytes = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        try:
            import psutil  # noqa: F401
            self._enabled = True
        except Exception:
            self._enabled = False

    @staticmethod
    def _ollama_rss_bytes() -> int:
        import psutil

        total = 0
        for proc in psutil.process_iter(["name", "cmdline"]):
            try:
                name = (proc.info.get("name") or "").lower()
                cmd = " ".join(proc.info.get("cmdline") or []).lower()
                if "ollama" in name or "ollama" in cmd:
                    total += proc.memory_info().rss
            except Exception:
                continue
        return total

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._peak_bytes = max(self._peak_bytes, self._ollama_rss_bytes())
            except Exception:
                pass
            self._stop.wait(self.interval_s)

    def start(self) -> None:
        if not self._enabled:
            return
        self._peak_bytes = self._ollama_rss_bytes()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> float | None:
        if not self._enabled:
            return None
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        return round(self._peak_bytes / (1024 * 1024), 1)


class OllamaRunner(ModelRunner):
    def __init__(self, model_id: str, tag: str, ollama_cfg: dict[str, Any]):
        super().__init__(model_id=model_id, kind="local")
        self.tag = tag
        # OLLAMA_HOST env wins over config so a remote droplet can be targeted
        # without editing (and dirtying) the committed config.yaml.
        self.host = os.environ.get("OLLAMA_HOST", ollama_cfg["host"]).rstrip("/")
        self.temperature = ollama_cfg.get("temperature", 0.0)
        self.num_predict = ollama_cfg.get("num_predict", 512)
        self.seed = ollama_cfg.get("seed", 42)
        self.keep_alive = ollama_cfg.get("keep_alive", 0)
        self.timeout_s = ollama_cfg.get("request_timeout_s", 600)
        self.mem_interval = ollama_cfg.get("mem_sample_interval_s", 0.25)

    def ensure_available(self) -> None:
        """Verify the tag is pulled; raise a helpful error otherwise."""
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=10)
            resp.raise_for_status()
        except Exception as exc:  # ollama not running / unreachable
            raise RuntimeError(
                f"Ollama not reachable at {self.host}. Start it with `ollama serve`. ({exc})"
            ) from exc
        tags = {m.get("name", "") for m in resp.json().get("models", [])}
        # tags look like 'llama3.2:1b'; allow a match on the base name too.
        if self.tag not in tags and not any(t.split(":")[0] == self.tag for t in tags):
            raise RuntimeError(
                f"Ollama model '{self.tag}' not found. Pull it: `ollama pull {self.tag}`."
            )

    def run(self, prompt: str, response_format: dict | None = None) -> RunResult:
        sampler = _OllamaMemSampler(self.mem_interval)
        payload = {
            "model": self.tag,
            "prompt": prompt,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.num_predict,
                "seed": self.seed,
            },
        }
        if response_format is not None:
            payload["format"] = response_format  # constrained JSON decoding
        sampler.start()
        start = time.perf_counter()
        try:
            resp = requests.post(
                f"{self.host}/api/generate", json=payload, timeout=self.timeout_s
            )
            resp.raise_for_status()
            data = resp.json()
            latency = time.perf_counter() - start
            peak = sampler.stop()
            return RunResult(
                text=data.get("response", ""),
                latency_s=round(latency, 3),
                prompt_tokens=data.get("prompt_eval_count"),
                completion_tokens=data.get("eval_count"),
                peak_mem_mb=peak,
            )
        except Exception as exc:
            latency = time.perf_counter() - start
            peak = sampler.stop()
            return RunResult(
                text="",
                latency_s=round(latency, 3),
                peak_mem_mb=peak,
                error=f"{type(exc).__name__}: {exc}",
            )
