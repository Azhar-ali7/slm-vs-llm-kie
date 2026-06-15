#!/usr/bin/env bash
#
# run_large_arm.sh — run the LARGE arm (GPT-4o + Claude Haiku 4.5) on DigitalOcean
# serverless inference, then refresh the report artifacts.
#
# Prereq (one-time, manual): create a "model access key" in the DigitalOcean
# Control Panel (GenAI Platform → serverless inference), then:
#     export DO_INFERENCE_KEY=<your-key>
#     bash scripts/run_large_arm.sh
#
# Pay-per-token on the $200 credit; whole large arm ≈ $1. No droplet, no idle billing.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -z "${DO_INFERENCE_KEY:-}" ]; then
  echo "ERROR: DO_INFERENCE_KEY is not set."
  echo "  Create a model access key in the DigitalOcean Control Panel, then:"
  echo "    export DO_INFERENCE_KEY=<your-key>"
  exit 1
fi

echo "==> [1/3] Verifying the configured model IDs exist in your DO catalog"
AVAIL="$(curl -fsS https://inference.do-ai.run/v1/models \
  -H "Authorization: Bearer $DO_INFERENCE_KEY" | python3 -c 'import sys,json; print("\n".join(m["id"] for m in json.load(sys.stdin)["data"]))')"
for want in openai-gpt-4o anthropic-claude-haiku-4.5; do
  if echo "$AVAIL" | grep -qx "$want"; then
    echo "    OK  $want"
  else
    echo "    !!  '$want' not found in your catalog. Available (grep gpt/claude):"
    echo "$AVAIL" | grep -iE 'gpt|claude' | sed 's/^/        /'
    echo "    Fix the IDs in config.yaml -> api.digitalocean.models, then re-run."
    exit 1
  fi
done

echo "==> [2/3] Smoke test: one direct completion (no side effects on runs.jsonl)"
curl -fsS https://inference.do-ai.run/v1/chat/completions \
  -H "Authorization: Bearer $DO_INFERENCE_KEY" -H "Content-Type: application/json" \
  -d '{"model":"openai-gpt-4o","messages":[{"role":"user","content":"reply with the single word: ok"}],"max_tokens":5}' \
  | python3 -c 'import sys,json; print("    reply:", repr(json.load(sys.stdin)["choices"][0]["message"]["content"]))' || {
  echo "Smoke test failed — check the key/catalog before the full run."; exit 1; }

echo "==> [3/3] Full large arm (frontier-llm + large-open-llm)"
python3 scripts/run_eval.py --models frontier-llm large-open-llm 2>&1 | tail -10
python3 scripts/make_plots.py

echo
echo "Done. Large arm complete. Review results/REPORT.md, then snapshot with:"
echo "  cp results/runs.jsonl results/summary.csv results/REPORT.md results/plot_*.png \\"
echo "     results/snapshots/<date>-phase2-with-large/"
