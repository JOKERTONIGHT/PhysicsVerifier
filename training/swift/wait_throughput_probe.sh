#!/usr/bin/env bash
# Wait for a 2-step smoke GRPO to finish and record step_time.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_load_train_env.sh"
ROOT="${PHYSICS_ROOT}"
CKPT="${QWEN8B_OUTCOME_SMOKE_CKPT}"
LOG="${CKPT}/swift_grpo.log"
REPORT="${ROOT}/logs/throughput_probe.json"
DEADLINE="${DEADLINE_SECS:-2400}"
start="$(date +%s)"
bash "${ROOT}/training/swift/run_throughput_probe.sh"
while true; do
  now="$(date +%s)"
  if grep -qE 'swift exited status=|train_loss|global_step/max_steps.: .2/' "${LOG}" 2>/dev/null; then
    break
  fi
  if (( now - start >= DEADLINE )); then
    echo "[error] probe timeout ${DEADLINE}s; see ${LOG}" >&2
    exit 2
  fi
  sleep 15
done
"${VENV_PY}" - <<PY
import json, re, statistics, time
from pathlib import Path
ckpt = Path("${CKPT}")
log = Path("${LOG}")
text = log.read_text(encoding="utf-8", errors="replace") if log.is_file() else ""
times = []
for m in re.finditer(r"'train_runtime': ([0-9.]+)", text):
    times.append(float(m.group(1)))
step_times = []
for m in re.finditer(r"step_time['\"]?:\s*([0-9.]+)", text):
    step_times.append(float(m.group(1)))
# ms-swift logging.jsonl
rows = []
for p in sorted(ckpt.glob("v*/logging.jsonl")) + sorted(ckpt.glob("logging.jsonl")):
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line=line.strip()
        if not line: continue
        try:
            obj=json.loads(line)
        except json.JSONDecodeError:
            continue
        if "loss" in obj or "reward" in obj:
            rows.append(obj)
clipped=[]
for row in rows:
    for k in ("completions/clipped_ratio","clipped_ratio"):
        if row.get(k) is not None:
            clipped.append(float(row[k])); break
report={
    "ckpt": str(ckpt),
    "n_log_rows": len(rows),
    "step_times": step_times[-10:],
    "mean_step_time": (sum(step_times)/len(step_times)) if step_times else None,
    "train_runtime": times[-1] if times else None,
    "clipped_ratio_mean": (sum(clipped)/len(clipped)) if clipped else None,
    "target_step_time": 150,
    "ok": bool(step_times and (sum(step_times)/len(step_times)) <= 150) or bool(times and times[-1] <= 400),
}
Path("${REPORT}").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
PY
