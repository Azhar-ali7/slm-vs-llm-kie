#!/usr/bin/env bash
#
# run_large_arm.sh — run the LARGE arm on AWS Bedrock (GPT-OSS-120B, Qwen3-235B,
# Llama-3.3-70B, Llama-4-Maverick) + Google (Gemma-4-31B), then refresh the report
# artifacts.
#
# Prereqs (one-time, manual):
#   1. AWS: create a Bedrock API key (Console -> Bedrock -> API keys). No model-access
#      opt-in needed — serverless models (GPT-OSS/Qwen3/Llama) auto-enable on first call
#      since 2025-09-29 (Model access page retired).
#   2. Google: get an API key at https://aistudio.google.com/apikey
#   3. Put both in .env (see .env.example): AWS_BEARER_TOKEN_BEDROCK, GEMINI_API_KEY.
#   4. pip install -r requirements.txt  (boto3>=1.40 for API-key auth).
#
# COST: these are LARGE billed models (Llama-4-Maverick especially). ALWAYS dry-run first:
#     python scripts/run_eval.py --mock                      # no API, wiring check
#     python scripts/run_eval.py --pilot --models frontier-llm   # 3 docs, tiny cost
# then run this for the full arm and watch cost_usd in results/runs.jsonl.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] && set -a && . ./.env && set +a || true

MODELS="gemma3-27b gemma4-31b frontier-llm qwen3-235b llama-70b llama4-maverick"

if [ -z "${AWS_BEARER_TOKEN_BEDROCK:-}" ] && [ -z "${AWS_ACCESS_KEY_ID:-}" ] && [ -z "${AWS_PROFILE:-}" ]; then
  echo "ERROR: no Bedrock credentials. Set AWS_BEARER_TOKEN_BEDROCK (API key) in .env,"
  echo "       or AWS_ACCESS_KEY_ID/AWS_PROFILE. See .env.example."
  exit 1
fi
if [ -z "${GEMINI_API_KEY:-}" ]; then
  echo "WARN: GEMINI_API_KEY not set — gemma3-27b will be skipped (Bedrock models still run)."
fi

echo "==> [1/3] Wiring preflight (no API calls, writes nothing)"
python3 - <<'PY'
from src.config import load_config
from src.models.registry import build_runners, model_meta
cfg = load_config(); meta = model_meta(cfg)
for r in build_runners(cfg, only="gemma3-27b gemma4-31b frontier-llm qwen3-235b llama-70b llama4-maverick".split()):
    assert r.model_id in meta, r.model_id
print("    OK: config + 6 large runners build")
PY

echo "==> [2/3] Smoke test: one real Bedrock call (frontier-llm)"
python3 - <<'PY'
from src.config import load_config
from src.models.registry import build_runners
r = build_runners(load_config(), only=["frontier-llm"])[0]
r.ensure_available()
res = r.run("Reply with the single word: ok")
print("    reply:", repr(res.text[:60]), "| err:", res.error)
raise SystemExit(1 if res.error else 0)
PY
echo "    smoke test passed."

echo "==> [3/3] Full large arm: $MODELS"
python3 scripts/run_eval.py --models $MODELS 2>&1 | tail -12
python3 scripts/make_plots.py

echo
echo "Done. Review results/REPORT.md + the summed cost_usd in results/runs.jsonl."
echo "NOTE: if old DigitalOcean rows (gemma4-31b / large-open-llm / llama-405b) linger, remove them"
echo "      from runs.jsonl for a clean current arm — the old data stays in results/snapshots/."
