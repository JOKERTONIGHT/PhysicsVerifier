#!/usr/bin/env bash
# Rewrite existing SFT rows with deepseek-v4-flash long-chain targets.
# Drops DISAGREE gold ids first. Seeds remaining ids from base rollouts, then
# fills/replaces via API. Does not overwrite sft_solutions.jsonl.
set -euo pipefail
ROOT="${PHYSICS_ROOT:-/home/jinjianhan/PhysicsVerifier}"
PYTHON="${PYTHON:-${ROOT}/.venv/bin/python}"
SRC="${SRC:-${ROOT}/data/rl/sft_solutions.jsonl}"
DISAGREE="${DISAGREE:-${ROOT}/data/rl/sft_disagree.jsonl}"
PROMPTS="${PROMPTS:-${ROOT}/data/rl/sft_longchain_prompts.jsonl}"
OUT="${OUT:-${ROOT}/data/rl/sft_solutions_longchain.jsonl}"
UNSOLVED="${UNSOLVED:-${ROOT}/data/rl/sft_longchain_unsolved.jsonl}"
REPORT="${REPORT:-${ROOT}/data/rl/sft_longchain_gen_report.json}"
FEWSHOT="${FEWSHOT:-${ROOT}/training/rl_data/sft_fewshot_disabled.json}"

[[ -s "${SRC}" ]] || { echo "[error] missing ${SRC}" >&2; exit 2; }

"${PYTHON}" - <<PY
import json
from pathlib import Path
import sys
sys.path.insert(0, "${ROOT}")
from training.rl_data.screen_training_data import sample_id
src = Path("${SRC}")
disagree_path = Path("${DISAGREE}")
out = Path("${PROMPTS}")
drop = set()
if disagree_path.is_file():
    for line in disagree_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        sid = sample_id(json.loads(line))
        if sid:
            drop.add(sid)
n_in = n_drop = n_out = 0
out.parent.mkdir(parents=True, exist_ok=True)
with src.open(encoding="utf-8") as fin, out.open("w", encoding="utf-8") as fout:
    for line in fin:
        if not line.strip():
            continue
        n_in += 1
        row = json.loads(line)
        sid = sample_id(row)
        if sid and sid in drop:
            n_drop += 1
            continue
        fout.write(json.dumps(row, ensure_ascii=False) + "\n")
        n_out += 1
print(json.dumps({"n_in": n_in, "n_disagree_dropped": n_drop, "n_out": n_out, "n_disagree_ids": len(drop), "prompts": str(out)}))
PY

"${PYTHON}" "${ROOT}/training/rl_data/drop_disagree_prompts.py" \
  --disagree "${DISAGREE}" \
  --pools "${ROOT}/data/rl/rl_prompts.jsonl" \
        "${ROOT}/data/rl/swift_prompts.jsonl" \
        "${ROOT}/data/rl/swift_prompts_max2048.jsonl" \
  --report "${ROOT}/data/rl/disagree_prompt_filter.json" || true

"${PYTHON}" "${ROOT}/training/rl_data/seed_longchain_from_rollouts.py" \
  --prompts "${PROMPTS}" \
  --rollouts "${ROOT}/data/rl/base_pass_rates_sft554.jsonl" \
  --output "${OUT}" \
  --report "${ROOT}/data/rl/sft_longchain_anchor_report.json"

cd "${ROOT}"
api=()
for attempt in $(seq 1 "${SFT_API_WAIT_TRIES:-60}"); do
  if mapfile -t api < <(SFT_REQUIRE_SMOKE=1 "${PYTHON}" "${ROOT}/training/rl_data/pick_sft_api.py" "${ROOT}"); then
    if [[ ${#api[@]} -ge 3 && -n "${api[0]:-}" ]]; then
      break
    fi
  fi
  echo "[longchain] pick_sft_api attempt ${attempt} failed (need smoke chat); retry in 60s" >&2
  api=()
  sleep 60
done
if [[ ${#api[@]} -lt 3 || -z "${api[0]:-}" ]]; then
  echo "[warn] no smoke-ok API pair; keeping rollout anchors in ${OUT}" >&2
  exit 0
fi
export OPENAI_BASE_URL="${api[0]}"
export OPENAI_API_KEY="${api[1]}"
API_MODEL="${SFT_API_MODEL:-${api[2]:-deepseek-v4-flash}}"
echo "[longchain] pair_model=${API_MODEL} prompts=$(wc -l < "${PROMPTS}") seeded=$(wc -l < "${OUT}")"

SFT_API_MODEL="${API_MODEL}" SFT_HINT_GOLD=1 SFT_MIN_CHARS=2500 \
  "${PYTHON}" "${ROOT}/training/rl_data/generate_sft_solutions.py" \
    --prompts "${PROMPTS}" \
    --heldout "${ROOT}/data/rl/heldout_eval.jsonl" \
    --output "${OUT}" \
    --unsolved "${UNSOLVED}" \
    --report "${REPORT}" \
    --api-only \
    --hint-gold \
    --api-model "${API_MODEL}" \
    --k "${SFT_API_K:-2}" \
    --api-k "${SFT_API_K:-2}" \
    --min-chars 2500 \
    --max-tokens "${SFT_MAX_TOKENS:-4096}" \
    --fewshot "${FEWSHOT}" \
    --concurrency "${SFT_API_CONCURRENCY:-4}" \
    --timeout "${SFT_TIMEOUT:-180}"
echo "[longchain] output=${OUT} report=${REPORT}"
