#!/usr/bin/env bash
# Restart HiPhO matrix + 8B download + finalize under nohup (survives SSH/Cursor disconnect).
set -euo pipefail

ROOT="${PHYSICS_ROOT:-/home/jinjianhan/PhysicsVerifier}"
WORKSPACE="${WORKSPACE_ROOT:-/slow_share/jinjianhan/workspace}"
MATRIX_DIR="${MATRIX_DIR:-${ROOT}/results/hipho_baseline_matrix_30b}"
DAEMON_DIR="${MATRIX_DIR}/daemon"
PID_DIR="${DAEMON_DIR}/pids"
LOG_DIR="${DAEMON_DIR}/logs"
PYTHON="${PYTHON:-/data1/jinjianhan/venv/openrlhf_train/bin/python}"

mkdir -p "${PID_DIR}" "${LOG_DIR}" "${MATRIX_DIR}"/{base_30b,global_step5,global_step10}

stop_if_running() {
  local name="$1"
  local pid_file="${PID_DIR}/${name}.pid"
  if [[ -f "${pid_file}" ]]; then
    local pid
    pid="$(cat "${pid_file}" 2>/dev/null || true)"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      echo "[stop] ${name} pid=${pid}"
      kill -TERM "${pid}" 2>/dev/null || true
      sleep 2
      kill -9 "${pid}" 2>/dev/null || true
    fi
    rm -f "${pid_file}"
  fi
}

# Stop Cursor-tied wrappers (keep nohup vLLM on :8766/8767/8768)
pkill -f 'generate_hipho_predictions.py' 2>/dev/null || true
pkill -f 'finalize_hipho_matrix.sh' 2>/dev/null || true
pkill -f 'eval_hipho_checkpoint.sh global_step10' 2>/dev/null || true
pkill -f 'download_qwen3_8b.sh' 2>/dev/null || true
pkill -f 'run_four_gpu_pilot.sh' 2>/dev/null || true
sleep 2

# Ensure vLLM services (nohup via manage_eval_vllm)
ensure_vllm() {
  local label="$1" port="$2" gpu="$3" model_dir="$4"
  if curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null; then
    echo "[ok] vLLM already up :${port} (${label})"
    return 0
  fi
  echo "[start] vLLM ${label} GPU${gpu} :${port}"
  VLLM_READY_SECS=3600 RUN_ID="${label}" PORT="${port}" CUDA_DEVICE="${gpu}" MODEL_DIR="${model_dir}" \
    nohup bash "${ROOT}/evaluation/benchmarks/hipho/manage_eval_vllm.sh" start \
    >>"${LOG_DIR}/vllm_${label}.log" 2>&1 &
  echo $! >"${PID_DIR}/vllm_${label}.pid"
}

CKPT="/slow_share/jinjianhan/ckpt/qwen3-30b-physics-openrlhf/ckpt"
BASE_MODEL="/slow_share/jinjianhan/models/Qwen3-30B-A3B-Instruct-2507"

ensure_vllm base_30b 8766 4 "${BASE_MODEL}"
ensure_vllm global_step5 8767 5 "${CKPT}/global_step5_hf"
ensure_vllm global_step10 8768 6 "${CKPT}/global_step10_hf"

launch() {
  local name="$1"
  shift
  stop_if_running "${name}"
  nohup "$@" >>"${LOG_DIR}/${name}.log" 2>&1 &
  echo $! >"${PID_DIR}/${name}.pid"
  echo "[launch] ${name} pid=$(cat "${PID_DIR}/${name}.pid") log=${LOG_DIR}/${name}.log"
}

launch hipho_base \
  bash "${ROOT}/evaluation/benchmarks/hipho/run_hipho_generate_resume.sh" base_30b

launch hipho_step5 \
  bash "${ROOT}/evaluation/benchmarks/hipho/run_hipho_generate_resume.sh" global_step5

launch hipho_step10 \
  bash "${ROOT}/evaluation/benchmarks/hipho/run_hipho_generate_resume.sh" global_step10

launch finalize \
  bash "${ROOT}/evaluation/benchmarks/hipho/finalize_hipho_matrix.sh"

launch download_8b \
  bash "${ROOT}/training/openrlhf/download_qwen3_8b.sh"

launch pilot_8b bash -c "
  while [[ \$(ls /slow_share/jinjianhan/models/Qwen3-8B/model*.safetensors 2>/dev/null | wc -l) -lt 1 ]]; do
    sleep 60
  done
  source ${WORKSPACE}/openrlhf_rl/env.sh
  CUDA_VISIBLE_DEVICES=0,1,2,3 PHYSICS_REWARD_MODE=answer_only \
    bash ${ROOT}/training/openrlhf/run_four_gpu_pilot.sh
"

cat >"${DAEMON_DIR}/status.txt" <<EOF
restarted_at=$(date -Iseconds)
pid_dir=${PID_DIR}
log_dir=${LOG_DIR}
predictions:
$(wc -l "${MATRIX_DIR}"/*/predictions.jsonl 2>/dev/null || true)
EOF

echo "[ok] all tasks relaunched under nohup"
echo "status: ${DAEMON_DIR}/status.txt"
echo "monitor: tail -f ${LOG_DIR}/hipho_base.log ${LOG_DIR}/finalize.log"
