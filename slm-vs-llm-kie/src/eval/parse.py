"""Robustly extract a JSON object from raw model output and validate it.

On failure we return a structured parse_error rather than raising, so a single
bad model output never crashes a run.
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional

from pydantic import create_model

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def _strip_fences(text: str) -> str:
    m = _FENCE_RE.search(text)
    return m.group(1) if m else text


def _first_json_object(s: str) -> str | None:
    """Return the first balanced {...} substring, respecting quoted strings."""
    start = s.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return s[start : i + 1]
    return None


def extract_json_text(text: str) -> str | None:
    """Best-effort: strip markdown fences, then grab the first balanced object."""
    if not text:
        return None
    return _first_json_object(_strip_fences(text))


def _schema_model(schema: dict[str, Any]):
    """Dynamic pydantic model: every schema field is an Optional[str]."""
    fields = {f["name"]: (Optional[str], None) for f in schema["fields"]}
    return create_model("Extraction", **fields)


def _coerce_scalar(v: Any) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, float)):
        return repr(v) if isinstance(v, float) else str(v)
    if isinstance(v, str):
        return v
    # lists/dicts/other -> JSON string so validation still succeeds
    return json.dumps(v, ensure_ascii=False)


def parse_prediction(text: str, schema: dict[str, Any]) -> tuple[dict[str, Any] | None, dict | None]:
    """Parse model output into {field: str|None} validated against the schema.

    Returns (prediction, None) on success or (None, parse_error) on failure.
    Extra keys not in the schema are dropped (and noted in `_extras`).
    """
    fields = [f["name"] for f in schema["fields"]]

    raw = extract_json_text(text)
    if raw is None:
        return None, {"type": "no_json", "detail": "no JSON object found", "raw": text[:300]}

    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, {"type": "json_decode", "detail": str(exc), "raw": raw[:300]}

    if not isinstance(obj, dict):
        return None, {"type": "not_object", "detail": f"top-level is {type(obj).__name__}", "raw": raw[:300]}

    coerced = {f: _coerce_scalar(obj.get(f)) for f in fields}
    try:
        validated = _schema_model(schema)(**coerced).model_dump()
    except Exception as exc:  # pragma: no cover - lenient model rarely fails
        return None, {"type": "schema_validation", "detail": str(exc), "raw": raw[:300]}

    extras = [k for k in obj if k not in fields]
    if extras:
        validated["_extras"] = extras
    return validated, None
