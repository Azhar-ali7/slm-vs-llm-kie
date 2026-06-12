"""List OpenRouter models (id + price per 1M tokens) matching search terms.

Uses OPENROUTER_API_KEY from .env. Helps pick exact slugs + cheap-tier prices.

  python scripts/list_openrouter_models.py mistral phi llama-3.3 gpt-4o-mini
  python scripts/list_openrouter_models.py :free          # free models only
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests  # noqa: E402

from src.config import load_config  # noqa: E402


def main() -> None:
    load_config()  # loads .env
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        print("OPENROUTER_API_KEY not set (.env). Add it, then re-run.")
        sys.exit(1)
    terms = [t.lower() for t in sys.argv[1:]] or ["mistral", "phi", "llama", "gpt-4o-mini"]

    resp = requests.get(
        "https://openrouter.ai/api/v1/models",
        headers={"Authorization": f"Bearer {key}"},
        timeout=30,
    )
    resp.raise_for_status()
    models = resp.json().get("data", [])

    def per_m(p):  # price string per-token -> per 1M
        try:
            return round(float(p) * 1_000_000, 4)
        except (TypeError, ValueError):
            return None

    rows = []
    for m in models:
        mid = m.get("id", "")
        if any(t in mid.lower() for t in terms):
            pr = m.get("pricing", {})
            rows.append((mid, per_m(pr.get("prompt")), per_m(pr.get("completion")),
                         m.get("context_length")))
    rows.sort(key=lambda r: (r[1] if r[1] is not None else 1e9))

    print(f"{len(rows)} matches (price USD / 1M tokens):\n")
    print(f"{'id':<48} {'in':>8} {'out':>8}  ctx")
    for mid, pin, pout, ctx in rows:
        print(f"{mid:<48} {str(pin):>8} {str(pout):>8}  {ctx}")


if __name__ == "__main__":
    main()
