"""CLI entry point for the evaluation run.

Examples:
  python scripts/run_eval.py --pilot                 # 3 docs x 2 models (cheap)
  python scripts/run_eval.py --skip-large            # full run, drop phi4/mistral
  python scripts/run_eval.py --models smollm2-360m frontier-llm
  python scripts/run_eval.py --datasets sample
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.runner.run import run_eval  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the SLM-vs-LLM KIE evaluation.")
    ap.add_argument("--pilot", action="store_true",
                    help="Tiny subset (config run.pilot_docs/pilot_models) to validate the pipeline.")
    ap.add_argument("--skip-large", action="store_true",
                    help="Skip models flagged large (phi4-mini, mistral-7b) — recommended on 8GB.")
    ap.add_argument("--models", nargs="*", default=None,
                    help="Restrict to these model ids.")
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="Restrict to these datasets.")
    ap.add_argument("--config", default=None, help="Path to config.yaml.")
    ap.add_argument("--mock", action="store_true",
                    help="Offline rule-based baseline only (no Ollama/Azure) — seeds demo --replay.")
    args = ap.parse_args()

    run_eval(
        config_path=args.config,
        pilot=args.pilot,
        skip_large=args.skip_large,
        only_models=args.models,
        only_datasets=args.datasets,
        use_mock=args.mock,
    )


if __name__ == "__main__":
    main()
