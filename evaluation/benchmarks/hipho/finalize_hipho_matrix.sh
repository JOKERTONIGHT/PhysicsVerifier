#!/usr/bin/env bash
# Finalize HiPhO baseline matrix: score base, wait for step5/10, summarize.
set -euo pipefail

ROOT="${PHYSICS_ROOT:-/home/jinjianhan/PhysicsVerifier}"
MATRIX_DIR="${MATRIX_DIR:-${ROOT}/results/hipho_baseline_matrix_30b}"
PYTHON="${PYTHON:-/data1/jinjianhan/venv/openrlhf_train/bin/python}"
LOG="${LOG:-${MATRIX_DIR}/finalize.log}"

mkdir -p "${MATRIX_DIR}"
exec >>"${LOG}" 2>&1
echo "[finalize] start $(date -Iseconds)"

BASE_PRED="${MATRIX_DIR}/base_30b/predictions.jsonl"
BASE_SCORE="${MATRIX_DIR}/base_30b/scores.json"
while [[ ! -s "${BASE_PRED}" ]] || [[ $(wc -l <"${BASE_PRED}") -lt 150 ]]; do
  n=$(wc -l <"${BASE_PRED}" 2>/dev/null || echo 0)
  echo "[finalize] waiting base predictions (${n}/150)"
  sleep 120
done
if [[ ! -f "${BASE_SCORE}" ]]; then
  "${ROOT}/.venv/bin/python" "${ROOT}/evaluation/benchmarks/hipho/score_hipho_predictions.py" \
    --predictions "${BASE_PRED}" --output "${BASE_SCORE}" --no-use-verifier
fi

for label in global_step5 global_step10; do
  score="${MATRIX_DIR}/${label}/scores.json"
  while [[ ! -f "${score}" ]]; do
    pred="${MATRIX_DIR}/${label}/predictions.jsonl"
    n=$(wc -l <"${pred}" 2>/dev/null || echo 0)
    echo "[finalize] waiting ${label} (${n}/150 scored=$(test -f "${score}" && echo yes || echo no))"
    sleep 180
  done
done

"${ROOT}/.venv/bin/python" "${ROOT}/evaluation/benchmarks/hipho/summarize_hipho_baseline.py" \
  --matrix-dir "${MATRIX_DIR}" \
  --base-label base_30b \
  --output "${MATRIX_DIR}/summary.json"

echo "[finalize] done $(date -Iseconds)"
