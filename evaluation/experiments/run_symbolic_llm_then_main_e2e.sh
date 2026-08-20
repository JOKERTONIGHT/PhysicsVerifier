#!/usr/bin/env bash
# Phase 1: LLM symbolic translation (v2-style) for 900-scale catalog.
# Phase 2: Error-level main e2e experiments (30B API, 4B local, 235B API).
#
# Usage (detach-safe):
#   cd /home/jinjianhan/PhysicsVerifier
#   nohup bash scripts/run_symbolic_llm_then_main_e2e.sh \
#     > results/_symbolic_llm_then_e2e_batch.log 2>&1 &
#   echo $! > results/_symbolic_llm_then_e2e_batch.pid
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
source "$ROOT/evaluation/experiments/catalog_defaults.sh"

PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
export PYTHONUNBUFFERED=1

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

stop_running_experiments() {
  pkill -TERM -f 'run_main_e2e_experiments.sh' 2>/dev/null || true
  pkill -TERM -f 'run_verifier.py.*e2e_0900' 2>/dev/null || true
  sleep 2
}

verify_llm_symbolic() {
  local min_checks="${SYMBOLIC_MIN_CHECKS:-1150}"
  local min_llm_ok="${SYMBOLIC_MIN_LLM_OK:-1100}"
  SYMBOLIC_MIN_CHECKS="$min_checks" SYMBOLIC_MIN_LLM_OK="$min_llm_ok" "$PYTHON" - <<'PY'
import json, os, sys
from pathlib import Path

min_checks = int(os.environ.get("SYMBOLIC_MIN_CHECKS", "1150"))
min_llm_ok = int(os.environ.get("SYMBOLIC_MIN_LLM_OK", "1100"))

manifest = json.loads(Path("results/experience_symbolic_program_manifest_0900.json").read_text())
report_path = Path("results/experience_symbolic_translation_report_0900.json")
checks = manifest.get("checks") or []
if len(checks) < min_checks:
    print(f"[error] manifest too small: {len(checks)} < {min_checks}", file=sys.stderr)
    sys.exit(5)

if not report_path.exists():
    print("[error] missing translation report", file=sys.stderr)
    sys.exit(5)

report = json.loads(report_path.read_text()).get("report") or []
from collections import Counter
status = Counter(r.get("status") for r in report)
llm_ok = status.get("translated", 0) + status.get("repaired", 0)
fallback = status.get("fallback", 0)
failed = status.get("failed", 0)
print(f"[ok] manifest={len(checks)} translated={status.get('translated',0)} repaired={status.get('repaired',0)} fallback={fallback} failed={failed}")
if llm_ok < min_llm_ok:
    print(f"[error] LLM-translated checks too few: {llm_ok} < {min_llm_ok}", file=sys.stderr)
    sys.exit(5)
if failed > 100:
    print(f"[warn] {failed} rules failed translation (continuing)", file=sys.stderr)
PY
}

# ---- phase 1: LLM symbolic translation ----
SKIP_SYMBOLIC_LLM="${SKIP_SYMBOLIC_LLM:-0}"

log "phase 1/2: stopping any in-flight e2e jobs"
stop_running_experiments

if [[ "$SKIP_SYMBOLIC_LLM" == "1" ]]; then
  log "phase 1/2: skip LLM translation (SKIP_SYMBOLIC_LLM=1)"
else
  log "phase 1/2: LLM symbolic translation for 900-scale catalog (~1225 rules)"
  log "  model=${MAIN_SYMBOLIC_MODEL} output=${MAIN_EXPERIENCE_MODULE}"

  bash "$ROOT/evaluation/experiments/generate_symbolic_checks_0900.sh" --repair --resume --retranslate-all \
    2>&1 | tee "$ROOT/results/_symbolic_0900_llm_upgrade.log"
fi

log "phase 1/2: verifying LLM symbolic module (min_checks=${SYMBOLIC_MIN_CHECKS})"
verify_llm_symbolic

# ---- phase 2: main e2e experiments ----
log "phase 2/2: starting error-level main e2e experiments (post LLM symbolic)"
export START_SYMBOLIC_LLM_UPGRADE=0
exec bash "$ROOT/evaluation/experiments/run_main_e2e_experiments.sh"
