#!/usr/bin/env bash
# Optimize 900-scale symbolic checks: re-translate loose-pass + failed rules via API.
#
# Usage (detach-safe):
#   cd /home/jinjianhan/PhysicsVerifier
#   nohup bash scripts/optimize_symbolic_checks_0900.sh \
#     > results/_symbolic_0900_optimize_batch.log 2>&1 &
#   echo $! > results/_symbolic_0900_optimize_batch.pid
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
source "$ROOT/evaluation/experiments/catalog_defaults.sh"

PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
export PYTHONUNBUFFERED=1

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

log "phase 0: audit current symbolic module"
loose_grep=$(grep -cE "逻辑基本正确|公式结构正确|检查通过|if has_formula:" symbolic/generated_experience_checks_0900.py 2>/dev/null || echo 0)
failed_n=$("$PYTHON" -c "import json; r=json.load(open('results/experience_symbolic_translation_report_0900.json')); print(sum(1 for x in r.get('report',[]) if x.get('status')=='failed'))")
log "  loose_pattern_hits=${loose_grep:-0} prior_failed=${failed_n}"

log "phase 1: LLM re-translate loose-pass + failed rules (model=${MAIN_SYMBOLIC_MODEL})"
bash "$ROOT/evaluation/experiments/generate_symbolic_checks_0900.sh" \
  --repair --resume \
  --refresh-loose-pass \
  --refresh-fallback \
  --retry-failed \
  2>&1 | tee "$ROOT/results/_symbolic_0900_optimize_upgrade.log"

log "phase 2: verify manifest"
SYMBOLIC_MIN_CHECKS="${SYMBOLIC_MIN_CHECKS:-1150}" SYMBOLIC_MIN_LLM_OK="${SYMBOLIC_MIN_LLM_OK:-1100}" "$PYTHON" - <<'PY'
import json, os, sys
from pathlib import Path

min_checks = int(os.environ.get("SYMBOLIC_MIN_CHECKS", "1150"))
manifest = json.loads(Path("results/experience_symbolic_program_manifest_0900.json").read_text())
checks = manifest.get("checks") or []
if len(checks) < min_checks:
    print(f"[error] manifest too small: {len(checks)} < {min_checks}", file=sys.stderr)
    sys.exit(5)
report = json.loads(Path("results/experience_symbolic_translation_report_0900.json").read_text())
from collections import Counter
status = Counter(r.get("status") for r in report.get("report") or [])
print(f"[ok] manifest={len(checks)} status={dict(status)}")
PY

log "symbolic optimization batch complete"
