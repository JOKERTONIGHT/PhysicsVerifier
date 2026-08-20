#!/usr/bin/env bash
# Generate experience-code symbolic checks for the canonical 900-scale unified catalog.
#
# Usage:
#   # LLM translation (v2-style, resume-safe; ~1225 rules):
#   bash scripts/generate_symbolic_checks_0900.sh --repair --resume --refresh-fallback
#
#   # Full optimization pass (loose-pass + failed rules, API):
#   bash scripts/optimize_symbolic_checks_0900.sh
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
source "$ROOT/evaluation/experiments/catalog_defaults.sh"

PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
ENV_FILE="${ENV_FILE:-$ROOT/.env}"

if [[ -f "$ENV_FILE" ]]; then
  eval "$(
    "$PYTHON" - "$ENV_FILE" <<'PY'
import os, shlex, sys
from pathlib import Path
try:
    from dotenv import load_dotenv
    load_dotenv(Path(sys.argv[1]), override=True)
except ImportError:
    pass
for key in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_API_BASE"):
    val = os.environ.get(key)
    if val:
        print(f"export {key}={shlex.quote(val)}")
PY
  )"
fi

MODEL="${SYMBOLIC_MODEL:-$MAIN_SYMBOLIC_MODEL}"
INPUT="${SYMBOLIC_INPUT:-$MAIN_UNIFIED_CATALOG}"
MODULE="${SYMBOLIC_OUTPUT_MODULE:-symbolic/generated_experience_checks_0900.py}"
MANIFEST="${SYMBOLIC_OUTPUT_MANIFEST:-$MAIN_EXPERIENCE_MANIFEST}"
REPORT="${SYMBOLIC_OUTPUT_REPORT:-results/experience_symbolic_translation_report_0900.json}"

exec "$PYTHON" scripts/generate_symbolic_checks.py \
  --input "$INPUT" \
  --model "$MODEL" \
  --output-module "$MODULE" \
  --output-manifest "$MANIFEST" \
  --report "$REPORT" \
  --save-every 25 \
  "$@"
