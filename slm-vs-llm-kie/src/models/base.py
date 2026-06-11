"""Model runner interface + result record shared by local and API runners."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class RunResult:
    """One model call's raw output and operational measurements."""
    text: str
    latency_s: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    peak_mem_mb: float | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ModelRunner(ABC):
    """A model that turns a prompt string into a RunResult.

    `kind` is 'local' or 'api'; `cost_usd` for a result is computed downstream by
    eval/efficiency.py from token counts and the per-model price table.
    """

    def __init__(self, model_id: str, kind: str):
        self.model_id = model_id
        self.kind = kind

    @abstractmethod
    def run(self, prompt: str) -> RunResult:  # pragma: no cover - interface
        ...

    def ensure_available(self) -> None:
        """Optional pre-flight check (e.g. model pulled / endpoint reachable)."""
        return None

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"{self.__class__.__name__}(model_id={self.model_id!r}, kind={self.kind!r})"
