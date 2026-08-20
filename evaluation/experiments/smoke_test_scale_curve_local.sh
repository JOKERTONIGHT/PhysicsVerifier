#!/usr/bin/env bash
# Smoke test: verify local 30B pipeline components without full 3000-sample run.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8766}"
MODEL="${MODEL:-qwen3-30b-a3b-instruct-2507}"
DATA_DIR="${DATA_DIR:-data/derived/leak_free_scale_seed20260508}"
SMOKE_DIR="${SMOKE_DIR:-results/scale_curve_error_smoke}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-local-vllm}"
export OPENAI_BASE_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1"
export OPENAI_API_BASE="$OPENAI_BASE_URL"

pass=0
fail=0
ok() { echo "[PASS] $*"; pass=$((pass + 1)); }
bad() { echo "[FAIL] $*"; fail=$((fail + 1)); }

echo "=== smoke test: local scale-curve pipeline ==="

# 1) vLLM
if curl -sf "http://${VLLM_HOST}:${VLLM_PORT}/v1/models" >/dev/null; then
  ok "vLLM API http://${VLLM_HOST}:${VLLM_PORT}/v1"
else
  bad "vLLM API not reachable"
fi

# 2) Python.h / gcc wrapper
if [[ -f "$ROOT/.local/deb-root/usr/include/python3.10/Python.h" ]]; then
  ok "local Python.h present"
else
  bad "local Python.h missing (run scripts/setup_venv_python_dev.sh)"
fi

# 3) data split audit
if "$PYTHON" - <<PY
import json, sys
m = json.load(open("$DATA_DIR/split_manifest.json"))
if not m.get("overlap_audit", {}).get("passes"):
    sys.exit(1)
print("holdout audit passes")
PY
then
  ok "leak-free split manifest"
else
  bad "split manifest audit failed"
fi

# 4) LLM chat (1 call)
if "$PYTHON" - <<PY
import os, sys
from openai import OpenAI
c = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=os.environ["OPENAI_BASE_URL"])
r = c.chat.completions.create(
    model="$MODEL",
    messages=[{"role": "user", "content": "Reply with exactly: OK"}],
    max_tokens=16,
    temperature=0,
)
text = (r.choices[0].message.content or "").strip()
print("llm_reply:", text[:80])
if not text:
    sys.exit(1)
PY
then
  ok "local LLM chat completion"
else
  bad "local LLM chat completion"
fi

mkdir -p "$SMOKE_DIR"
SMOKE_INPUT="$SMOKE_DIR/expansion_sample_3.json"
SMOKE_SEM="$SMOKE_DIR/semantic_experience.json"
SMOKE_DIST="$SMOKE_DIR/semantic_experience_distilled.json"
SMOKE_CATALOG="$SMOKE_DIR/rules_unified_smoke.json"
ERROR_DATASET="data/derived/combined_language_dual_chain_seed20260508_test200/annotated_chain/error_eval_dataset_100.json"
MANIFEST="${EXPERIENCE_CODE_MANIFEST:-results/experience_symbolic_program_manifest_v2_unified.json}"
MODULE="${EXPERIENCE_CODE_MODULE:-symbolic.generated_experience_checks_v2_unified}"

# 5) semantic experience (3 samples)
"$PYTHON" - <<PY
import json
pool = json.load(open("$DATA_DIR/expansion_pool.json"))
json.dump(pool[:3], open("$SMOKE_INPUT", "w"), ensure_ascii=False, indent=2)
print("wrote", len(pool[:3]), "smoke expansion samples")
PY

if "$PYTHON" scripts/generate_experience_rules.py \
  --input "$SMOKE_INPUT" \
  --rules-catalog catalogs/rules_catalog_top_down.json \
  --model "$MODEL" \
  --output "$SMOKE_SEM" \
  --distilled-output "$SMOKE_DIST" \
  --max-rules-per-sample 1 \
  --min-rule-count 1; then
  ok "semantic experience (3 samples)"
else
  bad "semantic experience (3 samples)"
fi

# 6) build catalog (use proven 300-sample distilled library)
if "$PYTHON" scripts/build_unified_catalog.py \
  --experience-distilled catalogs/semantic_experience_distilled_300.json \
  --scenario-cluster-blueprints catalogs/scenario_cluster_blueprints.json \
  --output "$SMOKE_CATALOG"; then
  ok "build_unified_catalog"
else
  bad "build_unified_catalog"
fi

# 6b) subset + aggregate/plot helpers
if "$PYTHON" scripts/subset_semantic_experience_for_scale.py \
  --semantic-input "$SMOKE_SEM" \
  --expansion-input "$DATA_DIR/expansion_pool.json" \
  --expansion-size 3 \
  --semantic-output "$SMOKE_DIR/subset_sem_3.json" \
  --distilled-output "$SMOKE_DIR/subset_dist_3.json" >/dev/null; then
  ok "subset_semantic_experience_for_scale"
else
  bad "subset_semantic_experience_for_scale"
fi

mkdir -p "$SMOKE_DIR/scale_0300"
cp -f "$SMOKE_DIR/error_metrics.json" "$SMOKE_DIR/scale_0300/error_metrics.json" 2>/dev/null || true
if "$PYTHON" scripts/aggregate_error_scale_curve.py \
  --metrics-glob "$SMOKE_DIR/scale_*/error_metrics.json" \
  --output-csv "$SMOKE_DIR/curve_metrics.csv" \
  --output-json "$SMOKE_DIR/curve_metrics.json" >/dev/null \
  && "$PYTHON" scripts/plot_error_scale_curve.py \
  --input-csv "$SMOKE_DIR/curve_metrics.csv" \
  --output "$SMOKE_DIR/curve_plot.png" >/dev/null; then
  ok "aggregate + plot_error_scale_curve"
else
  bad "aggregate + plot_error_scale_curve"
fi

# 7) error-level verifier (2 samples)
SMOKE_EVAL="$SMOKE_DIR/error_eval_2.json"
"$PYTHON" - <<PY
import json
rows = json.load(open("$ERROR_DATASET"))
json.dump(rows[:2], open("$SMOKE_EVAL", "w"), ensure_ascii=False, indent=2)
PY

if "$PYTHON" scripts/run_verifier.py \
  --input "$SMOKE_EVAL" \
  --output "$SMOKE_DIR/error_verifier_results.json" \
  --symbolic-output "$SMOKE_DIR/error_symbolic_audit.json" \
  --model "$MODEL" \
  --unified-catalog "$SMOKE_CATALOG" \
  --experience-code-manifest "$MANIFEST" \
  --experience-code-module "$MODULE" \
  --max-per-sample 4 \
  --max-per-paragraph 1 \
  --no-symbolic-check; then
  ok "run_verifier (2 error-eval samples)"
else
  bad "run_verifier (2 error-eval samples)"
fi

if "$PYTHON" scripts/evaluate_physics_eval_sets.py \
  --dataset "$SMOKE_EVAL" \
  --results "$SMOKE_DIR/error_verifier_results.json" \
  --audit "$SMOKE_DIR/error_symbolic_audit.json" \
  --output "$SMOKE_DIR/error_metrics.json" \
  --match-mode location; then
  ok "evaluate_physics_eval_sets"
else
  bad "evaluate_physics_eval_sets"
fi

echo ""
echo "=== summary: ${pass} passed, ${fail} failed ==="
echo "artifacts: $SMOKE_DIR"
[[ "$fail" -eq 0 ]]
