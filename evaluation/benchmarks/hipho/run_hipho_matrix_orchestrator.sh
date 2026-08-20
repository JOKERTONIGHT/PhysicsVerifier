#!/usr/bin/env bash
# Wait for HF exports, then finish HiPhO baseline matrix (base/step5/step10).
set -euo pipefail

ROOT="${PHYSICS_ROOT:-/home/jinjianhan/PhysicsVerifier}"
CKPT_ROOT="${CKPT_ROOT:-/slow_share/jinjianhan/ckpt/qwen3-30b-physics-openrlhf/ckpt}"
MATRIX_DIR="${MATRIX_DIR:-${ROOT}/results/hipho_baseline_matrix_30b}"
CUDA_DEVICE="${CUDA_DEVICE:-4}"
PORT="${PORT:-8766}"
LOG="${LOG:-${MATRIX_DIR}/orchestrator.log}"

mkdir -p "${MATRIX_DIR}"
exec >>"${LOG}" 2>&1
echo "[orchestrator] start $(date -Iseconds)"

for tag in global_step5 global_step10; do
  hf_dir="${CKPT_ROOT}/${tag}_hf"
  while [[ ! -f "${hf_dir}/model.safetensors.index.json" && ! -f "${hf_dir}/model.safetensors" ]]; do
    echo "[orchestrator] waiting for ${hf_dir}"
    sleep 120
    if [[ ! -f "${CKPT_ROOT}/_actor/${tag}/latest" && ! -d "${CKPT_ROOT}/_actor/${tag}" ]]; then
      TAG="${tag}" OUT_DIR="${hf_dir}" bash "${ROOT}/training/openrlhf/convert_openrlhf_ckpt_to_hf.sh" || true
    fi
  done
  echo "[orchestrator] ready ${hf_dir}"
done

CUDA_DEVICE="${CUDA_DEVICE}" PORT="${PORT}" MATRIX_DIR="${MATRIX_DIR}" \
  bash "${ROOT}/evaluation/benchmarks/hipho/run_hipho_baseline_matrix.sh"

echo "[orchestrator] done $(date -Iseconds)"
