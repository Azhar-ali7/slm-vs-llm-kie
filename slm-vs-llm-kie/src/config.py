"""Central config + schema loading. No magic constants live in code."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml

# Repo root = parent of src/
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config" / "config.yaml"


def _load_dotenv(root: Path = ROOT) -> None:
    """Minimal .env loader (avoids a python-dotenv dependency).

    Lines of the form KEY=VALUE are exported into os.environ if not already set.
    """
    env_path = root / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load config.yaml and the .env file."""
    _load_dotenv()
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["_root"] = str(ROOT)
    return cfg


def resolve_path(cfg: dict[str, Any], rel: str | Path) -> Path:
    """Resolve a path from config relative to the repo root."""
    p = Path(rel)
    return p if p.is_absolute() else ROOT / p


def load_schema(cfg: dict[str, Any], dataset: str) -> dict[str, Any]:
    """Load a target-field schema for a dataset by name."""
    schema_rel = cfg["datasets"][dataset]["schema"]
    with open(resolve_path(cfg, schema_rel), "r", encoding="utf-8") as fh:
        return json.load(fh)


def schema_field_names(schema: dict[str, Any]) -> list[str]:
    return [f["name"] for f in schema["fields"]]
