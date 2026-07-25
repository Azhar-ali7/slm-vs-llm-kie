#!/usr/bin/env bash
#
# import_finetuned.sh — register a Kaggle-trained GGUF as an Ollama model, so the
# fine-tuned model becomes a normal type:local entry the eval harness can run.
#
#   bash scripts/import_finetuned.sh <model.gguf> [Modelfile]
#
# Example (after unzipping the Kaggle artifacts into ./artifacts/):
#   bash scripts/import_finetuned.sh artifacts/phi4-mini-ft.Q4_K_M.gguf artifacts/Modelfile
#
# If no Modelfile is given, one is generated that inherits the base model's chat
# TEMPLATE/PARAMETERS (so /api/generate wrapping at eval time matches training).
set -euo pipefail
cd "$(dirname "$0")/.."

GGUF="${1:-}"
MODELFILE="${2:-}"
TAG="${TAG:-phi4-mini-ft}"     # must match the tag in config/config.yaml
BASE="${BASE:-phi4-mini}"      # baseline model to copy the chat template from

if [ -z "$GGUF" ] || [ ! -f "$GGUF" ]; then
  echo "ERROR: pass the path to the downloaded GGUF."
  echo "  bash scripts/import_finetuned.sh artifacts/phi4-mini-ft.Q4_K_M.gguf [Modelfile]"
  exit 1
fi

if ! command -v ollama >/dev/null 2>&1; then
  echo "ERROR: ollama not found. Install from https://ollama.com and run 'ollama serve'."
  exit 1
fi

# If no Modelfile provided, synthesize one from the base model's template.
if [ -z "$MODELFILE" ]; then
  MODELFILE="$(mktemp -t "${TAG}.Modelfile.XXXX")"
  echo "==> No Modelfile given; deriving TEMPLATE from '$BASE' via 'ollama show'."
  {
    echo "FROM ./$(basename "$GGUF")"
    # Copy the base chat template so the fine-tune is prompted identically.
    if ollama show "$BASE" --modelfile >/tmp/_base_modelfile 2>/dev/null; then
      grep -E '^(TEMPLATE|PARAMETER|SYSTEM)' -A0 /tmp/_base_modelfile || true
      # Preserve multi-line TEMPLATE blocks verbatim.
      awk '/^TEMPLATE/{p=1} p{print} /^"""/{if(p>1){p=0}} /^TEMPLATE.*"""/{}' /tmp/_base_modelfile >/dev/null 2>&1 || true
    else
      echo "# (base template unavailable — Ollama will apply the GGUF's built-in template)"
    fi
  } > "$MODELFILE"
  # The GGUF must sit next to the Modelfile for the relative FROM to resolve.
  cp "$GGUF" "$(dirname "$MODELFILE")/$(basename "$GGUF")"
  echo "    wrote $MODELFILE"
fi

echo "==> Creating Ollama model '$TAG' from $GGUF"
ollama create "$TAG" -f "$MODELFILE"

echo "==> Smoke test: one extraction prompt"
PROMPT='Return ONLY a JSON object with keys "company","date","address","total".
LINES:
MAPLE CAFE
Date: 2021-03-04
12 Oak Street, Leeds
TOTAL 18.40

Output:'
ollama run "$TAG" "$PROMPT" || {
  echo "Smoke test failed — check the Modelfile template."; exit 1;
}

cat <<EOF

✓ '$TAG' created. Next:
    python scripts/run_eval.py --models $TAG     # appends 80 cells to results/runs.jsonl
    python scripts/make_plots.py                 # before->after delta in results/REPORT.md
EOF
