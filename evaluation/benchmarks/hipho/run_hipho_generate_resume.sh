#!/usr/bin/env bash
# Resume HiPhO prediction generation for one matrix label (daemon-friendly).
set -euo pipefail

LABEL="${1:?usage: $0 <base_30b|global_step5|global_step10>}"
ROOT="${PHYSICS_ROOT:-/home/jinjianhan/PhysicsVerifier}"
MATRIX_DIR="${MATRIX_DIR:-${ROOT}/results/hipho_baseline_matrix_30b}"
PYTHON="${PYTHON:-/data1/jinjianhan/venv/openrlhf_train/bin/python}"
INPUT="${INPUT:-/slow_share/jinjianhan/workspace/benchmarks/hipho/hipho_text_only.jsonl}"
MODEL="${MODEL:-qwen3-30b-a3b-instruct-2507}"
TEMPERATURE="${TEMPERATURE:-0.2}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
TARGET_LINES="${TARGET_LINES:-150}"

case "${LABEL}" in
  base_30b) PORT="${PORT:-8766}" ;;
  global_step5) PORT="${PORT:-8767}" ;;
  global_step10) PORT="${PORT:-8768}" ;;
  *) echo "[error] unknown label ${LABEL}" >&2; exit 2 ;;
esac

OUT_SUB="${MATRIX_DIR}/${LABEL}"
PRED_OUT="${OUT_SUB}/predictions.jsonl"
SCORE_OUT="${OUT_SUB}/scores.json"
LOG="${OUT_SUB}/generate_daemon.log"
mkdir -p "${OUT_SUB}"

exec >>"${LOG}" 2>&1
echo "[daemon] ${LABEL} start $(date -Iseconds) port=${PORT}"

for _ in $(seq 1 360); do
  if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then
    break
  fi
  echo "[daemon] waiting vLLM :${PORT}"
  sleep 30
done
curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null || {
  echo "[error] vLLM not ready on :${PORT}"
  exit 1
}

done_lines=0
if [[ -f "${PRED_OUT}" ]]; then
  done_lines=$(wc -l <"${PRED_OUT}")
fi
echo "[daemon] resume from line ${done_lines}/${TARGET_LINES}"

if [[ "${done_lines}" -lt "${TARGET_LINES}" ]]; then
  "${PYTHON}" "${ROOT}/evaluation/benchmarks/hipho/generate_hipho_predictions.py" \
    --input "${INPUT}" \
    --output "${PRED_OUT}" \
    --base-url "http://127.0.0.1:${PORT}/v1" \
    --model "${MODEL}" \
    --temperature "${TEMPERATURE}" \
    --max-tokens "${MAX_TOKENS}" \
    --resume
fi

"${ROOT}/.venv/bin/python" "${ROOT}/evaluation/benchmarks/hipho/score_hipho_predictions.py" \
  --predictions "${PRED_OUT}" \
  --output "${SCORE_OUT}" \
  --no-use-verifier

echo "[daemon] ${LABEL} done $(date -Iseconds) -> ${SCORE_OUT}"
