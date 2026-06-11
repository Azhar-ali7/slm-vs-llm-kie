"""Normalisation to the canonical *simple JSON*.

Two entry points must produce the SAME simple JSON for the same document:

    from_dataset_record(record)      # dataset OCR words+boxes (+ optional KV/tables)
    from_textract(blocks_json)       # raw AWS Textract Blocks response

simple JSON schema:
    {"doc_id": str, "page": int,
     "lines": ["text line 1", ...],
     "key_values": {"DETECTED_KEY": "detected value"},
     "tables": [[["r1c1","r1c2"], ["r2c1","r2c2"]]]}

Ordering is deterministic: reading order = top-to-bottom, then left-to-right.
Both converters reduce their input to a common list of positioned words and a
shared assembler builds the final structure, which is why the outputs are equal.
"""
from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Shared internal representation: a "word" = text + left/top/width/height.
# Coordinates may be pixels or normalised [0,1]; line grouping is scale-tolerant.
# ---------------------------------------------------------------------------


def _word(text: str, left: float, top: float, width: float, height: float) -> dict[str, Any]:
    return {
        "text": text.strip(),
        "left": float(left),
        "top": float(top),
        "width": float(width),
        "height": float(height),
    }


def _vcenter(w: dict[str, Any]) -> float:
    return w["top"] + w["height"] / 2.0


def _words_to_lines(words: list[dict[str, Any]]) -> list[str]:
    """Group positioned words into reading-order text lines.

    A new line starts when a word's vertical centre departs from the current
    line's running centre by more than a tolerance derived from word height,
    making the grouping robust to pixel vs normalised coordinate scales.
    """
    words = [w for w in words if w["text"]]
    if not words:
        return []

    heights = sorted(w["height"] for w in words)
    median_h = heights[len(heights) // 2] or 1.0
    tol = median_h * 0.6

    ordered = sorted(words, key=lambda w: (_vcenter(w), w["left"]))
    groups: list[list[dict[str, Any]]] = [[ordered[0]]]
    running_center = _vcenter(ordered[0])

    for w in ordered[1:]:
        if abs(_vcenter(w) - running_center) <= tol:
            groups[-1].append(w)
        else:
            groups.append([w])
        grp = groups[-1]
        running_center = sum(_vcenter(x) for x in grp) / len(grp)

    rows: list[tuple[float, float, str]] = []
    for grp in groups:
        grp_sorted = sorted(grp, key=lambda w: w["left"])
        text = " ".join(w["text"] for w in grp_sorted).strip()
        if not text:
            continue
        rows.append((min(w["top"] for w in grp), min(w["left"] for w in grp), text))

    rows.sort(key=lambda r: (r[0], r[1]))
    return [text for _, _, text in rows]


def _clean_kv(key_values: dict[str, Any] | None) -> dict[str, str]:
    if not key_values:
        return {}
    out = {}
    for k, v in key_values.items():
        key = str(k).strip()
        if key:
            out[key] = str(v).strip()
    # Sorted keys -> stable, deterministic serialisation.
    return {k: out[k] for k in sorted(out)}


def _clean_tables(tables: list | None) -> list[list[list[str]]]:
    if not tables:
        return []
    cleaned = []
    for table in tables:
        cleaned.append([[str(cell or "").strip() for cell in row] for row in table])
    return cleaned


def _assemble(
    doc_id: str,
    page: int,
    words: list[dict[str, Any]],
    key_values: dict[str, Any] | None,
    tables: list | None,
) -> dict[str, Any]:
    return {
        "doc_id": str(doc_id),
        "page": int(page),
        "lines": _words_to_lines(words),
        "key_values": _clean_kv(key_values),
        "tables": _clean_tables(tables),
    }


# ---------------------------------------------------------------------------
# Converter 1: dataset record
# ---------------------------------------------------------------------------


def from_dataset_record(record: dict[str, Any]) -> dict[str, Any]:
    """Convert a dataset OCR record into simple JSON.

    Expected record shape (produced by src/data/loaders.py):
        {
          "doc_id": str,
          "page": int,                                   # optional, default 1
          "words": [{"text": str, "bbox": [x0,y0,x1,y1]}, ...],
          "key_values": {key: value},                    # optional
          "tables": [[[cell, ...], ...], ...],           # optional
        }
    bbox is [left, top, right, bottom].
    """
    words: list[dict[str, Any]] = []
    for w in record.get("words", []):
        x0, y0, x1, y1 = w["bbox"]
        words.append(_word(w["text"], left=x0, top=y0, width=x1 - x0, height=y1 - y0))

    return _assemble(
        doc_id=record["doc_id"],
        page=record.get("page", 1),
        words=words,
        key_values=record.get("key_values"),
        tables=record.get("tables"),
    )


# ---------------------------------------------------------------------------
# Converter 2: AWS Textract Blocks response
# ---------------------------------------------------------------------------


def _index_blocks(blocks: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {b["Id"]: b for b in blocks}


def _rel_ids(block: dict[str, Any], rel_type: str) -> list[str]:
    for rel in block.get("Relationships", []) or []:
        if rel.get("Type") == rel_type:
            return list(rel.get("Ids", []))
    return []


def _word_text(block_map: dict[str, dict[str, Any]], ids: list[str]) -> str:
    parts = []
    for i in ids:
        b = block_map.get(i)
        if b and b.get("BlockType") == "WORD":
            parts.append(b.get("Text", ""))
        elif b and b.get("BlockType") == "SELECTION_ELEMENT" and b.get("SelectionStatus") == "SELECTED":
            parts.append("[X]")
    return " ".join(p for p in parts if p).strip()


def _bbox_to_word(text: str, bbox: dict[str, Any]) -> dict[str, Any]:
    return _word(
        text,
        left=bbox.get("Left", 0.0),
        top=bbox.get("Top", 0.0),
        width=bbox.get("Width", 0.0),
        height=bbox.get("Height", 0.0),
    )


def from_textract(blocks_json: dict[str, Any] | list[dict[str, Any]]) -> dict[str, Any]:
    """Convert a raw AWS Textract Blocks response into simple JSON.

    Accepts either the full response dict ({"Blocks": [...], "DocumentMetadata"...})
    or a bare list of Block dicts. Uses WORD blocks for lines (so grouping matches
    the dataset path), KEY_VALUE_SET blocks for key_values, and TABLE/CELL blocks
    for tables.
    """
    if isinstance(blocks_json, dict):
        blocks = blocks_json.get("Blocks", [])
        doc_id = blocks_json.get("doc_id") or blocks_json.get("DocumentId") or "textract_doc"
    else:
        blocks = blocks_json
        doc_id = "textract_doc"

    block_map = _index_blocks(blocks)

    # --- words (for lines) ---
    words: list[dict[str, Any]] = []
    page = 1
    for b in blocks:
        if b.get("BlockType") == "WORD":
            geom = b.get("Geometry", {}).get("BoundingBox", {})
            words.append(_bbox_to_word(b.get("Text", ""), geom))
            page = b.get("Page", page)

    # --- key_values ---
    key_values: dict[str, str] = {}
    for b in blocks:
        if b.get("BlockType") != "KEY_VALUE_SET":
            continue
        if "KEY" not in (b.get("EntityTypes") or []):
            continue
        key_text = _word_text(block_map, _rel_ids(b, "CHILD"))
        value_text = ""
        for vid in _rel_ids(b, "VALUE"):
            vblock = block_map.get(vid)
            if vblock:
                value_text = _word_text(block_map, _rel_ids(vblock, "CHILD"))
        if key_text:
            key_values[key_text] = value_text

    # --- tables ---
    tables: list[list[list[str]]] = []
    for b in blocks:
        if b.get("BlockType") != "TABLE":
            continue
        cells = []
        for cid in _rel_ids(b, "CHILD"):
            cell = block_map.get(cid)
            if cell and cell.get("BlockType") == "CELL":
                cells.append(cell)
        if not cells:
            continue
        n_rows = max(c.get("RowIndex", 1) for c in cells)
        n_cols = max(c.get("ColumnIndex", 1) for c in cells)
        grid = [["" for _ in range(n_cols)] for _ in range(n_rows)]
        for cell in cells:
            r = cell.get("RowIndex", 1) - 1
            col = cell.get("ColumnIndex", 1) - 1
            grid[r][col] = _word_text(block_map, _rel_ids(cell, "CHILD"))
        tables.append(grid)

    return _assemble(doc_id=doc_id, page=page, words=words, key_values=key_values, tables=tables)
