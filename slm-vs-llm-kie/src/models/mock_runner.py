"""Offline rule-based extractor.

Purpose: exercise the full harness (and seed demo --replay) without Ollama or
Azure. It is a genuine naive baseline — it sees only the prompt text, parses the
listed field names and the document LINES/KEY_VALUES, and applies simple regex
heuristics. Results are labelled model_id 'mock-extractor' / kind 'mock' so they
are never mistaken for real model results.
"""
from __future__ import annotations

import json
import re
import time

from src.models.base import ModelRunner, RunResult

_FIELD_RE = re.compile(r'^- "([^"]+)" \(([^,]+),', re.MULTILINE)
_DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2}|\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4})\b")
_NUM_RE = re.compile(r"£?\s*([0-9][0-9,]*\.?[0-9]*)")
_POSTCODE_RE = re.compile(r"\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\b")


def _section(prompt: str, header: str) -> str:
    """Return the text following 'HEADER:' up to the next blank line / header."""
    idx = prompt.rfind(header)
    if idx == -1:
        return ""
    rest = prompt[idx + len(header):]
    out_lines = []
    for line in rest.splitlines()[1:]:
        if line.strip() == "" or line.strip().endswith(":") or line.startswith("###"):
            break
        out_lines.append(line)
    return "\n".join(out_lines)


def _heuristic_extract(prompt: str) -> dict:
    fields = _FIELD_RE.findall(prompt)               # [(name, type), ...]
    lines_block = _section(prompt, "LINES:")
    doc_lines = [ln.strip() for ln in lines_block.splitlines() if ln.strip()]
    doc_text = "\n".join(doc_lines)
    first_line = doc_lines[0] if doc_lines else None

    out: dict[str, object] = {}
    for name, ftype in fields:
        name_l = name.lower()
        ftype = ftype.strip()
        value = None
        if ftype == "date":
            m = _DATE_RE.search(doc_text)
            value = m.group(1) if m else None
        elif ftype in ("currency", "number"):
            keyword = None
            if "income" in name_l:
                keyword = "income"
            elif "spend" in name_l or "expend" in name_l:
                keyword = "expenditure"
            elif "total" in name_l:
                keyword = "total"
            value = _find_number(doc_lines, keyword)
        elif "name" in name_l or name_l == "company":
            value = first_line
        elif "postcode" in name_l:
            m = _POSTCODE_RE.search(doc_text)
            value = m.group(0) if m else None
        # other string fields (address parts, number id) -> left null by the baseline
        out[name] = value
    return out


def _find_number(lines: list[str], keyword: str | None) -> str | None:
    candidates = lines
    if keyword:
        kw = [ln for ln in lines if keyword in ln.lower()]
        if kw:
            candidates = kw
    for ln in candidates:
        m = _NUM_RE.search(ln)
        if m:
            return m.group(1).replace(",", "")
    return None


class MockRunner(ModelRunner):
    def __init__(self, model_id: str = "mock-extractor"):
        super().__init__(model_id=model_id, kind="mock")

    def ensure_available(self) -> None:
        return None

    def run(self, prompt: str, response_format: dict | None = None) -> RunResult:
        start = time.perf_counter()
        extracted = _heuristic_extract(prompt)
        text = json.dumps(extracted, ensure_ascii=False)
        latency = time.perf_counter() - start
        approx_in = max(1, len(prompt.split()))
        approx_out = max(1, len(text.split()))
        return RunResult(
            text=text,
            latency_s=round(latency, 4),
            prompt_tokens=approx_in,
            completion_tokens=approx_out,
            peak_mem_mb=None,
        )
