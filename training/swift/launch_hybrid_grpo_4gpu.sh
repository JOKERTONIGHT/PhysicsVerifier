#!/usr/bin/env bash
# N-GPU Swift GRPO. Default: outcome-gated hybrid (DeepSeek process + local acc).
# Set PHYSICS_REWARD_MODE=outcome_only to skip the judge and use acc+boxed only.
# Paths and GPU count come from training/swift/train_env.sh (see train_env.example.sh).
# Does not start local judges. Do not resume from the failed llm_step onset ckpt.
set -euo pipefail
ulimit -f unlimited 2>/dev/null || true
# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_load_train_env.sh"
ROOT="${PHYSICS_ROOT}"
export PHYSICS_ROOT="${ROOT}"
export TMPDIR="${TMPDIR:-${SLOW_TMP_ROOT}/swift}"
export TEMP="${TEMP:-${TMPDIR}}"
export TMP="${TMP:-${TMPDIR}}"
mkdir -p "${TMPDIR}"
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  unset CUDA_VISIBLE_DEVICES
fi
if [[ -f "${ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT}/.env"
  set +a
fi
WORKSPACE="${WORKSPACE_ROOT}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs}"
MODE="${MODE:-pilot}"  # smoke | pilot | full
PHYSICS_REWARD_MODE="${PHYSICS_REWARD_MODE:-hybrid_llm_outcome}"
USER_OVERLONG_FILTER="${OVERLONG_FILTER-}"
SFT_CKPT="${QWEN8B_SFT_CKPT}"
if [[ "${MODE}" == "smoke" ]]; then
  CKPT="${QWEN8B_HYBRID_SMOKE_CKPT}"
  MAX_STEPS="${MAX_STEPS:-2}"
  SAVE_STEPS="${SAVE_STEPS:-2}"
  NUM_GENERATIONS="${NUM_GENERATIONS:-6}"
  MAX_COMPLETION_LEN="${MAX_COMPLETION_LEN:-2048}"
  MAX_LENGTH="${MAX_LENGTH:-4096}"
  BETA="${BETA:-0.04}"
  OVERLONG_FILTER="${OVERLONG_FILTER:-false}"
elif [[ "${MODE}" == "full" ]]; then
  CKPT="${QWEN8B_HYBRID_CKPT}"
  MAX_STEPS="${MAX_STEPS:-30}"
  SAVE_STEPS="${SAVE_STEPS:-10}"
  NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
  MAX_COMPLETION_LEN="${MAX_COMPLETION_LEN:-3072}"
  MAX_LENGTH="${MAX_LENGTH:-5120}"
  BETA="${BETA:-0.04}"
  OVERLONG_FILTER="${OVERLONG_FILTER:-true}"
else
  CKPT="${QWEN8B_HYBRID_PILOT_CKPT}"
  MAX_STEPS="${MAX_STEPS:-10}"
  SAVE_STEPS="${SAVE_STEPS:-5}"
  NUM_GENERATIONS="${NUM_GENERATIONS:-6}"
  MAX_COMPLETION_LEN="${MAX_COMPLETION_LEN:-2048}"
  MAX_LENGTH="${MAX_LENGTH:-4096}"
  BETA="${BETA:-0.04}"
  OVERLONG_FILTER="${OVERLONG_FILTER:-false}"
fi
if [[ "${PHYSICS_REWARD_MODE}" == "outcome_only" ]]; then
  SKIP_CALIBRATION=1
  SKIP_SMOKE_GATE=1
  # Truncated unboxed rollouts must stay in the loss (reward 0) to punish loops.
  if [[ -z "${USER_OVERLONG_FILTER}" ]]; then
    OVERLONG_FILTER=false
  else
    OVERLONG_FILTER="${USER_OVERLONG_FILTER}"
  fi
  if [[ "${MODE}" == "full" ]]; then
    CKPT="${QWEN8B_OUTCOME_CKPT}"
  elif [[ "${MODE}" == "smoke" ]]; then
    CKPT="${QWEN8B_OUTCOME_SMOKE_CKPT}"
  else
    CKPT="${QWEN8B_OUTCOME_PILOT_CKPT}"
  fi
  SWIFT_PID_FILE="${SWIFT_PID_FILE:-${LOG_DIR}/swift_outcome_grpo_${MODE}.pid}"
  REWARD_PID_FILE="${REWARD_PID_FILE:-${LOG_DIR}/physics_reward_server_outcome_${MODE}.pid}"
else
  SWIFT_PID_FILE="${SWIFT_PID_FILE:-${LOG_DIR}/swift_hybrid_grpo_${MODE}.pid}"
  REWARD_PID_FILE="${REWARD_PID_FILE:-${LOG_DIR}/physics_reward_server_hybrid_${MODE}.pid}"
fi
PID_FILE="${SWIFT_PID_FILE}"
LOG_FILE="${LOG_FILE:-${CKPT}/swift_grpo.log}"
REPORT="${REPORT:-${CKPT}/swift_launch_report.json}"
SMOKE_CKPT="${QWEN8B_HYBRID_SMOKE_CKPT}"
MODEL_DIR="${QWEN8B_MODEL_DIR:-}"
if [[ -z "${MODEL_DIR}" ]]; then
  if [[ "${MODE}" == "full" && "${PHYSICS_REWARD_MODE}" != "outcome_only" ]]; then
    if [[ -f "${SFT_CKPT}/config.json" ]]; then
      MODEL_DIR="${SFT_CKPT}"
    else
      MODEL_DIR="$(ls -d "${SFT_CKPT}"/v*-*/checkpoint-* 2>/dev/null | tail -1 || true)"
    fi
  fi
  if [[ -z "${MODEL_DIR}" || ! -f "${MODEL_DIR}/config.json" ]]; then
    MODEL_DIR="${QWEN8B_MODEL_DIR}"
  fi
fi
PROMPT_DATA="${PROMPT_DATA:-}"
if [[ -z "${PROMPT_DATA}" ]]; then
  if [[ "${PHYSICS_REWARD_MODE}" == "outcome_only" && -s "${ROOT}/data/rl/swift_prompts_hybrid_band.jsonl" ]]; then
    PROMPT_DATA="${ROOT}/data/rl/swift_prompts_hybrid_band.jsonl"
  else
    PROMPT_DATA="${ROOT}/data/rl/swift_prompts_max2048.jsonl"
  fi
fi
PLUGIN="${PLUGIN:-${ROOT}/training/swift/hybrid_reward_plugin.py}"
FREE_MIB="${FREE_MIB:-75000}"
UTIL_MAX="${UTIL_MAX:-5}"
PER_DEVICE_TRAIN_BS="${PER_DEVICE_TRAIN_BS:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-3}"
NPROC="${NPROC_PER_NODE:-${N_TRAIN_GPUS:-4}}"
VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.30}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-4}"
MAX_RESAMPLE_TIMES="${MAX_RESAMPLE_TIMES:-2}"
SEED="${SEED:-42}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-12}"
global_bs=$(( NPROC * PER_DEVICE_TRAIN_BS * GRAD_ACCUM ))
if (( NUM_GENERATIONS <= 0 || global_bs % NUM_GENERATIONS != 0 )); then
  suggest=1
  while (( suggest < 128 )) && (( (NPROC * PER_DEVICE_TRAIN_BS * suggest) % NUM_GENERATIONS != 0 )); do
    suggest=$((suggest + 1))
  done
  echo "[error] global batch ${global_bs} (= ${NPROC}*${PER_DEVICE_TRAIN_BS}*${GRAD_ACCUM}) is not divisible by NUM_GENERATIONS=${NUM_GENERATIONS}." >&2
  echo "        Set GRAD_ACCUM=${suggest} (e.g. 2-GPU G=8 → GRAD_ACCUM=4)." >&2
  exit 2
fi
if [[ "${MODE}" == "full" ]]; then
  REWARD_PORT="${REWARD_PORT:-8773}"
elif [[ "${MODE}" == "smoke" ]]; then
  REWARD_PORT="${REWARD_PORT:-8774}"
else
  REWARD_PORT="${REWARD_PORT:-8772}"
fi
CALIB_REPORT="${CALIB_REPORT:-${ROOT}/logs/llm_step_judge_calibration.json}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-${ROOT}/data/rl/train_manifest.json}"
HIPHO_JSONL="${HIPHO_JSONL:-${WORKSPACE}/benchmarks/hipho/hipho_text_only.jsonl}"
HELDOUT="${HELDOUT:-${ROOT}/data/rl/heldout_eval.jsonl}"
SKIP_CALIBRATION="${SKIP_CALIBRATION:-1}"
if [[ "${MODE}" == "full" ]]; then
  SKIP_SMOKE_GATE="${SKIP_SMOKE_GATE:-0}"
else
  SKIP_SMOKE_GATE="${SKIP_SMOKE_GATE:-1}"
fi
# Never continue the failed 100-step llm_step onset run.
if [[ "${MODEL_DIR}" == *"deepseek-v4-flash-grpo-onset"* ]]; then
  echo "[refuse] refusing to start from failed llm_step onset ckpt ${MODEL_DIR}" >&2
  exit 2
fi

mkdir -p "${LOG_DIR}" "${CKPT}/plots" "${CKPT}/runs"

refuse() {
  local reason="$1"
  mkdir -p "$(dirname "${REPORT}")"
  "${VENV_PY}" -c 'import json,datetime,os,sys; print(json.dumps({"ok":False,"phase":"refused","reason":sys.argv[1],"at":datetime.datetime.utcnow().isoformat()+"Z","ckpt":os.environ.get("CKPT","")},ensure_ascii=False,indent=2))' "${reason}" >"${REPORT}"
  echo "[refuse] ${reason}" >&2
  exit 2
}

alive_pid_file() {
  local file="$1"
  [[ -f "${file}" ]] || return 1
  local pid
  pid="$(cat "${file}" 2>/dev/null || true)"
  [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null
}

[[ -x "${SWIFT_VENV}/bin/swift" ]] || refuse "missing swift binary"
[[ -f "${MODEL_DIR}/config.json" ]] || refuse "missing base model ${MODEL_DIR}"
[[ -s "${PROMPT_DATA}" ]] || refuse "missing prompt data ${PROMPT_DATA}"
[[ -f "${PLUGIN}" ]] || refuse "missing plugin ${PLUGIN}"

if [[ "${MODE}" == "pilot" ]]; then
  echo "[warn] MODEL_DIR is ${MODEL_DIR}; hybrid pilot starts from base Qwen3-8B unless overridden"
elif [[ "${MODE}" != "full" && "${MODEL_DIR}" != *"/Qwen3-8B" && "${MODEL_DIR}" != *"/Qwen3-8B/" ]]; then
  echo "[warn] MODEL_DIR is ${MODEL_DIR}; this experiment should start from base Qwen3-8B"
fi

avail_kb="$(df -Pk "${CKPT}" | awk 'NR==2{print $4}')"
if [[ -n "${avail_kb}" && "${avail_kb}" -lt 50000000 ]]; then
  refuse "insufficient disk on $(dirname "${CKPT}"): ${avail_kb} KiB"
fi

if alive_pid_file "${SWIFT_PID_FILE}" || alive_pid_file "${CKPT}/swift_train.pid"; then
  refuse "stale_or_live_pid: hybrid GRPO still running"
fi

LOCK_FILE="${CKPT}/launch.lock"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  refuse "another launch holds ${LOCK_FILE}"
fi
if pgrep -f "${CKPT}/run_swift.sh" >/dev/null 2>&1; then
  refuse "leftover run_swift.sh is still running under ${CKPT}; not starting a second copy"
fi

pick_master_port() {
  local p
  for p in 29511 29512 29513 29514 29515 29516 29517 29518 29611; do
    if ! ss -ltn 2>/dev/null | grep -qE ":${p}[[:space:]]"; then
      echo "${p}"
      return 0
    fi
  done
  return 1
}
MASTER_PORT="${MASTER_PORT:-$(pick_master_port)}" || refuse "no free MASTER_PORT"
export MASTER_PORT

bash "${ROOT}/training/openrlhf/ensure_cuda_ready.sh" || refuse "CUDA not ready"

"${VENV_PY}" "${ROOT}/training/rl_data/audit_eval_leakage.py" \
  --train "${PROMPT_DATA}" \
  --heldout "${HELDOUT}" \
  --hipho "${HIPHO_JSONL}" \
  --manifest "${TRAIN_MANIFEST}" \
  --fail-on-exact || refuse "eval leakage audit failed"

if [[ "${SKIP_CALIBRATION}" != "1" && "${MODE}" == "full" ]]; then
  if [[ ! -f "${CALIB_REPORT}" ]]; then
    refuse "calibration report missing; run training/swift/calibrate_llm_step_judge.py first"
  fi
  "${VENV_PY}" - "${CALIB_REPORT}" <<'PY' || refuse "calibration gate failed"
import json, sys
rep = json.loads(open(sys.argv[1], encoding="utf-8").read())
if not rep.get("ok"):
    sys.exit(2)
PY
fi

if [[ "${MODE}" == "full" && "${SKIP_SMOKE_GATE:-0}" != "1" ]]; then
  SMOKE_CKPT="${SMOKE_CKPT}" "${VENV_PY}" - <<'PY' || refuse "2-step smoke has not passed; rerun MODE=smoke first"
import json, os, sys
from pathlib import Path
root = Path(os.environ["SMOKE_CKPT"])
logs = sorted(root.glob("v*/logging.jsonl"), key=lambda p: p.stat().st_mtime)
if not logs:
    sys.exit(2)
steps = set()
for line in logs[-1].read_text(encoding="utf-8", errors="replace").splitlines():
    line = line.strip()
    if not line:
        continue
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        continue
    gs = str(obj.get("global_step/max_steps") or "")
    if "/" in gs:
        steps.add(int(gs.split("/")[0]))
if 2 not in steps:
    sys.exit(2)
PY
fi

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a _vis <<< "${CUDA_VISIBLE_DEVICES}"
  if [[ ${#_vis[@]} -lt "${NPROC}" ]]; then
    refuse "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} has ${#_vis[@]} ids, need ${NPROC}"
  fi
  train_gpus="${CUDA_VISIBLE_DEVICES}"
  echo "{\"ok\": true, \"train_gpus\": [$(printf '%s,' "${_vis[@]}" | sed 's/,$//')], \"source\": \"CUDA_VISIBLE_DEVICES\"}" >"${CKPT}/gpu_selection.json"
else
  probe="$("${ORHF_PYTHON}" "${ROOT}/training/openrlhf/gpu_bundle_utils.py" probe --train-only --n-train "${NPROC}" --free-mib "${FREE_MIB}" --util-max "${UTIL_MAX}")"
  echo "${probe}" >"${CKPT}/gpu_selection.json"
  ok="$("${VENV_PY}" -c 'import json,sys; print(int(json.loads(sys.stdin.read()).get("ok", False)))' <<<"${probe}")"
  if [[ "${ok}" != "1" ]]; then
    reason="$("${VENV_PY}" -c 'import json,sys; print(json.loads(sys.stdin.read()).get("reason",""))' <<<"${probe}")"
    refuse "need_${NPROC}_idle_train_gpus: ${reason}"
  fi
  train_gpus="$("${VENV_PY}" -c 'import json,sys; d=json.loads(sys.stdin.read()); print(",".join(str(x) for x in d["train_gpus"]))' <<<"${probe}")"
fi

export PHYSICS_REWARD_MODE
export PHYSICS_REWARD_W_ANSWER="${PHYSICS_REWARD_W_ANSWER:-1.0}"
export PHYSICS_REWARD_W_FORMAT="${PHYSICS_REWARD_W_FORMAT:-0.05}"
export PHYSICS_REWARD_W_PROCESS="${PHYSICS_REWARD_W_PROCESS:-0.3}"
export PHYSICS_REWARD_PROCESS_ALPHA="${PHYSICS_REWARD_PROCESS_ALPHA:-0.2}"
if [[ "${PHYSICS_REWARD_MODE}" == "outcome_only" ]]; then
  export PHYSICS_REWARD_W_PROCESS=0
  export PHYSICS_REWARD_PROCESS_ALPHA=0
  export PHYSICS_REWARD_CONCURRENCY="${PHYSICS_REWARD_CONCURRENCY:-16}"
else
  export PHYSICSVERIFIER_LLM_MODEL=deepseek-v4-flash
  export LLM_STEP_JUDGE_TIMEOUT=300
  export LLM_STEP_JUDGE_MAX_TOKENS="${LLM_STEP_JUDGE_MAX_TOKENS:-4096}"
  export LLM_STEP_JUDGE_MAX_RETRIES="${LLM_STEP_JUDGE_MAX_RETRIES:-6}"
  export LLM_STEP_JUDGE_CONCURRENCY="${LLM_STEP_JUDGE_CONCURRENCY:-32}"
  export PHYSICS_REWARD_CONCURRENCY="${PHYSICS_REWARD_CONCURRENCY:-32}"
fi
export PHYSICS_REWARD_HTTP_RETRIES="${PHYSICS_REWARD_HTTP_RETRIES:-5}"
export PHYSICS_REWARD_CACHE_SIZE="${PHYSICS_REWARD_CACHE_SIZE:-4096}"
export PHYSICS_REWARD_METRICS_LOG="${PHYSICS_REWARD_METRICS_LOG:-${CKPT}/physics_reward_metrics.jsonl}"
export HOST=127.0.0.1
export PORT="${REWARD_PORT}"
export PID_FILE="${REWARD_PID_FILE}"
export LOG="${CKPT}/physics_reward_server.log"
export VENV="${VENV:-${ROOT}/.venv}"
if [[ ! -x "${VENV}/bin/python" ]]; then
  export VENV="$(cd "$(dirname "${ORHF_PYTHON}")/.." && pwd)"
fi
bash "${ROOT}/training/reward_server/start_reward_server.sh" || refuse "reward server failed"

export PHYSICS_REWARD_URL="http://127.0.0.1:${REWARD_PORT}/get_reward"
export PHYSICS_REWARD_TIMEOUT="${PHYSICS_REWARD_TIMEOUT:-3600}"
export MASTER_ADDR=127.0.0.1
export CUDA_VISIBLE_DEVICES="${train_gpus}"
export NPROC_PER_NODE="${NPROC}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export CUDA_HOME="${CUDA_HOME:-}"
if [[ -n "${CUDA_HOME}" ]]; then
  export PATH="${CUDA_HOME}/bin:${PATH}"
fi
export DS_SKIP_CUDA_CHECK="${DS_SKIP_CUDA_CHECK:-1}"
export TRL_EXPERIMENTAL_SILENCE=1
# shellcheck disable=SC1091
source "${ROOT}/training/openrlhf/setup_slow_share_tmp.sh" || true
SWIFT_TMP="${SWIFT_TMP:-${SLOW_TMP_ROOT}/swift}"
mkdir -p "${SWIFT_TMP}" "${SWIFT_TMP}/hf_datasets" "${SWIFT_TMP}/hf_home"
export TMPDIR="${SWIFT_TMP}" TEMP="${SWIFT_TMP}" TMP="${SWIFT_TMP}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${SWIFT_TMP}/hf_datasets}"
export HF_HOME="${HF_HOME:-${SWIFT_TMP}/hf_home}"

echo "[launch] ${PHYSICS_REWARD_MODE} GRPO mode=${MODE} gpus=${train_gpus} steps=${MAX_STEPS} ckpt=${CKPT} model=${MODEL_DIR}"
echo "[wrapper] $(date -u +%Y-%m-%dT%H:%M:%SZ) starting swift rlhf mode=${MODE} gpus=${train_gpus} steps=${MAX_STEPS} reward=${PHYSICS_REWARD_MODE}" >>"${LOG_FILE}"
export LOG_FILE
export RESUME_FROM="${RESUME_FROM:-}"
export PHYSICS_REWARD_HTTP_RETRIES="${PHYSICS_REWARD_HTTP_RETRIES:-5}"
export PHYSICS_REWARD_FUNC="${PHYSICS_REWARD_FUNC:-${PHYSICS_REWARD_MODE}}"
export BETA
export OVERLONG_FILTER
RUN_SH="${CKPT}/run_swift.sh"
"${VENV_PY}" - "${RUN_SH}" "${SWIFT_PID_FILE}" "${CKPT}/swift_train.pid" "${train_gpus}" "${NPROC}" "${ROOT}" "${PHYSICS_REWARD_URL}" "${PHYSICS_REWARD_TIMEOUT}" "${CUDA_HOME}" "${PATH}" "${DS_SKIP_CUDA_CHECK}" "${TMPDIR}" "${HF_DATASETS_CACHE}" "${HF_HOME}" "${SWIFT_VENV}" "${MODEL_DIR}" "${PLUGIN}" "${VLLM_GPU_UTIL}" "${MAX_LENGTH}" "${VLLM_MAX_NUM_SEQS}" "${PROMPT_DATA}" "${MAX_COMPLETION_LEN}" "${PER_DEVICE_TRAIN_BS}" "${GRAD_ACCUM}" "${NUM_GENERATIONS}" "${SEED}" "${MAX_RESAMPLE_TIMES}" "${SAVE_STEPS}" "${SAVE_TOTAL_LIMIT}" "${CKPT}" "${MAX_STEPS}" <<'PY'
import os, sys, textwrap, pathlib
out = pathlib.Path(sys.argv[1])
pid_file, train_pid = sys.argv[2], sys.argv[3]
vals = sys.argv[4:]
keys = [
    "train_gpus","nproc","root","reward_url","reward_timeout","cuda_home","path",
    "ds_skip","tmpdir","hf_datasets","hf_home","swift_venv","model_dir","plugin",
    "vllm_util","max_length","vllm_seqs","prompt_data","max_comp","per_device_bs",
    "grad_accum","num_gen","seed","max_resample","save_steps","save_total","ckpt","max_steps",
]
env = dict(zip(keys, vals))
log_file = os.environ["LOG_FILE"]
http_retries = os.environ.get("PHYSICS_REWARD_HTTP_RETRIES", "5")
master_port = os.environ.get("MASTER_PORT", "29511")
beta = os.environ.get("BETA", "0.04")
overlong = os.environ.get("OVERLONG_FILTER", "false")
reward_funcs = os.environ.get("PHYSICS_REWARD_FUNC", "hybrid_llm_outcome")
resume_from = os.environ.get("RESUME_FROM", "").strip()
resume_flags = ""
if resume_from:
    resume_flags = (
        f'    --resume_from_checkpoint "{resume_from}" \\\n'
        "    --resume_only_model true \\\n"
        "    --load_args false \\\n"
    )
script = f'''#!/usr/bin/env bash
exec >>"{log_file}" 2>&1
if [[ -s "{pid_file}" ]]; then
  old="$(cat "{pid_file}" 2>/dev/null || true)"
  if [[ -n "${{old}}" && "${{old}}" != "$$" ]] && kill -0 "${{old}}" 2>/dev/null; then
    echo "[wrapper] pid=$$ refusing to start; live wrapper ${{old}} owns {pid_file}"
    exit 0
  fi
fi
echo $$ >"{pid_file}"
echo $$ >"{train_pid}"
echo "[wrapper] pid=$$ starting /usr/bin/env swift $(date -u +%Y-%m-%dT%H:%M:%SZ)"
trap '' HUP
/usr/bin/env \\
  CUDA_VISIBLE_DEVICES="{env["train_gpus"]}" \\
  NPROC_PER_NODE="{env["nproc"]}" \\
  MASTER_ADDR=127.0.0.1 \\
  MASTER_PORT="{master_port}" \\
  PYTHONPATH="{env["root"]}:${{PYTHONPATH:-}}" \\
  PHYSICS_REWARD_URL="{env["reward_url"]}" \\
  PHYSICS_REWARD_TIMEOUT="{env["reward_timeout"]}" \\
  PHYSICS_REWARD_HTTP_RETRIES="{http_retries}" \\
  CUDA_HOME="{env["cuda_home"]}" \\
  PATH="{env["path"]}" \\
  DS_SKIP_CUDA_CHECK="{env["ds_skip"]}" \\
  TRL_EXPERIMENTAL_SILENCE=1 \\
  PYTHONUNBUFFERED=1 \\
  PYTHONFAULTHANDLER=1 \\
  TOKENIZERS_PARALLELISM=false \\
  CUDA_DEVICE_MAX_CONNECTIONS=1 \\
  NCCL_CUMEM_ENABLE=0 \\
  PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.8" \\
  TMPDIR="{env["tmpdir"]}" TEMP="{env["tmpdir"]}" TMP="{env["tmpdir"]}" \\
  HF_DATASETS_CACHE="{env["hf_datasets"]}" HF_HOME="{env["hf_home"]}" \\
  "{env["swift_venv"]}/bin/swift" rlhf \\
    --rlhf_type grpo \\
    --model "{env["model_dir"]}" \\
    --external_plugins "{env["plugin"]}" \\
    --reward_funcs {reward_funcs} \\
    --use_vllm true \\
    --vllm_mode colocate \\
    --vllm_gpu_memory_utilization "{env["vllm_util"]}" \\
    --vllm_tensor_parallel_size 1 \\
    --vllm_max_model_len "{env["max_length"]}" \\
    --vllm_max_num_seqs "{env["vllm_seqs"]}" \\
    --vllm_enforce_eager true \\
    --vllm_enable_prefix_caching true \\
    --sleep_level 0 \\
    --offload_model true \\
    --offload_optimizer true \\
    --tuner_type full \\
    --torch_dtype bfloat16 \\
    --attn_impl sdpa \\
    --dataset "{env["prompt_data"]}" \\
    --max_completion_length "{env["max_comp"]}" \\
    --max_length "{env["max_length"]}" \\
    --num_train_epochs 1 \\
    --per_device_train_batch_size "{env["per_device_bs"]}" \\
    --gradient_accumulation_steps "{env["grad_accum"]}" \\
    --learning_rate 1e-6 \\
    --epsilon 0.2 \\
    --beta {beta} \\
    --temperature 1.0 \\
    --num_generations "{env["num_gen"]}" \\
    --seed "{env["seed"]}" \\
    --dynamic_sample true \\
    --max_resample_times "{env["max_resample"]}" \\
    --eval_strategy no \\
    --save_steps "{env["save_steps"]}" \\
    --save_only_model true \\
    --save_total_limit "{env["save_total"]}" \\
    --logging_steps 1 \\
    --gradient_checkpointing true \\
    --deepspeed zero3 \\
    --report_to tensorboard \\
    --logging_dir "{env["ckpt"]}/runs" \\
    --output_dir "{env["ckpt"]}" \\
    --log_completions true \\
    --dataloader_num_workers 0 \\
    --use_hf true \\
    --overlong_filter {overlong} \\
{resume_flags}    --max_steps "{env["max_steps"]}"
status=$?
echo "[wrapper] swift exited status=${{status}} $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit ${{status}}
'''
out.write_text(script)
os.chmod(out, 0o755)
print(f"[ok] wrote {out}")
PY
UNIT_DIR="${XDG_CONFIG_HOME:-${HOME}/.config}/systemd/user"
mkdir -p "${UNIT_DIR}"
UNIT_NAME="hybrid-grpo-${MODE}.service"
cat >"${UNIT_DIR}/${UNIT_NAME}" <<EOF
[Unit]
Description=PhysicsVerifier hybrid_llm_outcome GRPO (${MODE})
After=default.target

[Service]
Type=simple
KillMode=none
RemainAfterExit=yes
TasksMax=infinity
LimitNOFILE=1048576
WorkingDirectory=${ROOT}
Environment=HOME=${HOME}
Environment=USER=${USER}
Environment=LANG=C.UTF-8
ExecStart=/bin/bash ${RUN_SH}
Restart=no
EOF
systemctl --user daemon-reload
if systemctl --user is-active --quiet "${UNIT_NAME}"; then
  refuse "systemd unit ${UNIT_NAME} is already active"
fi
systemctl --user reset-failed "${UNIT_NAME}" 2>/dev/null || true
systemctl --user start "${UNIT_NAME}"
echo "[launch] detached via systemctl --user start ${UNIT_NAME} master_port=${MASTER_PORT}"
for _ in $(seq 1 40); do
  if [[ -s "${SWIFT_PID_FILE}" ]] && kill -0 "$(cat "${SWIFT_PID_FILE}" 2>/dev/null)" 2>/dev/null; then
    break
  fi
  sleep 0.25
done
sleep 8
if [[ ! -s "${SWIFT_PID_FILE}" ]] || ! kill -0 "$(cat "${SWIFT_PID_FILE}" 2>/dev/null)" 2>/dev/null; then
  refuse "swift died during startup; see ${LOG_FILE}"
fi

export CKPT TRAIN_GPUS="${train_gpus}" SWIFT_PID_FILE LOG_FILE REPORT MAX_STEPS MODE MODEL_DIR PROMPT_DATA SEED PER_DEVICE_TRAIN_BS GRAD_ACCUM SAVE_STEPS SAVE_TOTAL_LIMIT
export NUM_GENERATIONS MAX_COMPLETION_LEN MAX_LENGTH BETA OVERLONG_FILTER
"${VENV_PY}" - <<'PY' >"${REPORT}"
import hashlib, json, datetime, os, subprocess
from pathlib import Path

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

git = ""
try:
    git = subprocess.check_output(["git", "-C", os.environ.get("PHYSICS_ROOT", "."), "rev-parse", "HEAD"], text=True).strip()
except Exception:
    pass
print(json.dumps({
  "ok": True,
  "phase": "launched",
  "reason": "hybrid_llm_outcome_grpo_4gpu",
  "mode": os.environ.get("MODE"),
  "at": datetime.datetime.utcnow().isoformat() + "Z",
  "ckpt": os.environ["CKPT"],
  "pid": int(open(os.environ["SWIFT_PID_FILE"]).read().strip()),
  "log": os.environ["LOG_FILE"],
  "cuda_visible_devices": os.environ["TRAIN_GPUS"],
  "judge_gpus": [],
  "max_steps": int(os.environ["MAX_STEPS"]),
  "save_steps": int(os.environ.get("SAVE_STEPS", "10")),
  "save_total_limit": int(os.environ.get("SAVE_TOTAL_LIMIT", "12")),
  "model_dir": os.environ.get("MODEL_DIR"),
  "prompt_data": os.environ.get("PROMPT_DATA"),
  "prompt_sha256": sha256(os.environ["PROMPT_DATA"]) if Path(os.environ["PROMPT_DATA"]).is_file() else "",
  "git_commit": git,
  "seed": int(os.environ.get("SEED", "42")),
  "num_generations": int(os.environ.get("NUM_GENERATIONS", "6")),
  "max_completion_length": int(os.environ.get("MAX_COMPLETION_LEN", "2048")),
  "max_length": int(os.environ.get("MAX_LENGTH", "4096")),
  "per_device_train_batch_size": int(os.environ.get("PER_DEVICE_TRAIN_BS", "2")),
  "gradient_accumulation_steps": int(os.environ.get("GRAD_ACCUM", "3")),
  "learning_rate": 1e-6,
  "epsilon": 0.2,
  "beta": float(os.environ.get("BETA", "0.04")),
  "overlong_filter": os.environ.get("OVERLONG_FILTER", "false"),
  "reward_mode": os.environ.get("PHYSICS_REWARD_MODE", "hybrid_llm_outcome"),
  "reward_funcs": os.environ.get("PHYSICS_REWARD_FUNC", os.environ.get("PHYSICS_REWARD_MODE", "hybrid_llm_outcome")),
  "judge_model": "" if os.environ.get("PHYSICS_REWARD_MODE") == "outcome_only" else "deepseek-v4-flash",
  "prompt_version": "" if os.environ.get("PHYSICS_REWARD_MODE") == "outcome_only" else "llm_step_v1",
  "w_answer": float(os.environ.get("PHYSICS_REWARD_W_ANSWER", "1.0")),
  "w_format": float(os.environ.get("PHYSICS_REWARD_W_FORMAT", "0.05")),
  "w_process": float(os.environ.get("PHYSICS_REWARD_W_PROCESS", "0.3")),
  "process_alpha": float(os.environ.get("PHYSICS_REWARD_PROCESS_ALPHA", "0.2")),
}, ensure_ascii=False, indent=2))
PY

if [[ "${PHYSICS_REWARD_MODE}" == "outcome_only" ]]; then
  nohup "${VENV_PY}" "${ROOT}/training/swift/monitor_process_reward.py" \
    --metrics "${PHYSICS_REWARD_METRICS_LOG:-${CKPT}/physics_reward_metrics.jsonl}" \
    --train-log "${LOG_FILE}" \
    --pid-file "${SWIFT_PID_FILE}" \
    >>"${CKPT}/monitor.log" 2>&1 &
else
  nohup "${VENV_PY}" "${ROOT}/training/swift/monitor_llm_step_reward.py" \
    --metrics "${PHYSICS_REWARD_METRICS_LOG:-${CKPT}/physics_reward_metrics.jsonl}" \
    --train-log "${LOG_FILE}" \
    --pid-file "${SWIFT_PID_FILE}" \
    >>"${CKPT}/monitor.log" 2>&1 &
fi
echo $! >"${CKPT}/monitor.pid"

echo "[launch] pid=$(cat "${SWIFT_PID_FILE}") log=${LOG_FILE} train=${train_gpus} report=${REPORT}"
