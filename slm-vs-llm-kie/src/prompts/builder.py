"""Build the model prompt from (a) the target schema and (b) the simple JSON.

Two config-controlled conditions:
  shot_mode:     zero_shot | few_shot  (few-shot examples come from the TRAIN
                 split only — never the 20-doc test set, to avoid leakage)
  input_variant: lines_only | lines_plus_kv

The instruction text lives in templates/instruction.txt so it is easy to inspect.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.data.preprocess import from_dataset_record

_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"


def _load_template(name: str) -> str:
    return (_TEMPLATE_DIR / name).read_text(encoding="utf-8")


def field_spec(schema: dict[str, Any]) -> str:
    rows = []
    for f in schema["fields"]:
        req = "required" if f.get("required") else "optional"
        rows.append(f'- "{f["name"]}" ({f.get("type", "string")}, {req})')
    return "\n".join(rows)


def render_input_block(simple_json: dict[str, Any], input_variant: str,
                       max_lines: int | None = None) -> str:
    """Render the document text fed to the model for a given input variant.

    `max_lines` caps the number of document lines included (bounds context and
    cost for long documents); None means no cap.
    """
    lines = simple_json.get("lines", [])
    if max_lines is not None and len(lines) > max_lines:
        lines = lines[:max_lines]
    block = "LINES:\n" + ("\n".join(lines) if lines else "(none)")
    if input_variant == "lines_plus_kv":
        kv = simple_json.get("key_values", {})
        kv_text = "\n".join(f"{k}: {v}" for k, v in kv.items()) if kv else "(none)"
        block += "\n\nKEY_VALUES:\n" + kv_text
    return block


def _example_gold(record: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    fields = [f["name"] for f in schema["fields"]]
    gold = record.get("gold", {})
    return {f: gold.get(f) for f in fields}


def _build_example(record: dict[str, Any], schema: dict[str, Any], input_variant: str,
                   max_lines: int | None = None) -> str:
    sj = from_dataset_record(record)
    inp = render_input_block(sj, input_variant, max_lines=max_lines)
    out = json.dumps(_example_gold(record, schema), ensure_ascii=False)
    return f"{inp}\n\nOutput:\n{out}"


def build_prompt(
    schema: dict[str, Any],
    simple_json: dict[str, Any],
    shot_mode: str = "zero_shot",
    input_variant: str = "lines_only",
    examples: list[dict[str, Any]] | None = None,
    max_lines: int | None = None,
) -> str:
    """Assemble the full prompt string.

    `examples` are TRAIN-split records (used only when shot_mode == 'few_shot').
    `max_lines` caps document lines (long-document context/cost control).
    """
    instruction = _load_template("instruction.txt").format(field_spec=field_spec(schema))
    parts = [instruction]

    if shot_mode == "few_shot" and examples:
        for i, ex in enumerate(examples, 1):
            parts.append(f"### Example {i}\n{_build_example(ex, schema, input_variant, max_lines)}")

    document = render_input_block(simple_json, input_variant, max_lines=max_lines)
    parts.append(f"### Document\n{document}\n\nOutput:")
    return "\n\n".join(parts)
