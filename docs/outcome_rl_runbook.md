# Outcome-only GRPO runbook

Restart GRPO with **pure outcome reward** (`answer=1.0`, `format=0.05`) on the 233-prompt difficulty band. Process reward stays deferred. Collapse on SFT/RFT LoRA was **repetition loops hitting the 8192-token cap**, not a missing `\boxed`. `outcome_only` therefore defaults `--overlong_filter false` so truncated unboxed rollouts stay in the loss with reward 0.

## Dependencies

Install into two venvs (or one, if you prefer):

| Role | Packages |
|---|---|
| Training (`SWIFT_VENV`) | `ms-swift==4.5.2`, `trl`, `deepspeed` |
| vLLM / OpenRLHF (`ORHF_PYTHON`) | `vllm==0.8.5` |
| Project `.venv` (`VENV_PY`) | `sympy`, `fastapi`, plus this repo |

`training/compat/math_grading.py` (imported by the reward server and heldout scorer) needs **sympy**. Gate scripts must use `VENV_PY`, not system `python3`.

## Machine paths

```bash
cp training/swift/train_env.example.sh training/swift/train_env.sh
# edit SWIFT_VENV, ORHF_PYTHON, QWEN8B_MODEL_DIR, CKPT_ROOT, SLOW_TMP_ROOT
```

`train_env.sh` is gitignored. Every launch/eval script sources `training/swift/_load_train_env.sh`.

On this original host, `data`, `results`, and `.venv` are **symlinks** to `/slow_share/...` and `/data1/...`. On a new machine they will be broken. Either:

1. Copy the 1.1 MB data list below into `data/` and `results/` as real directories, and create a real `.venv`, or
2. Recreate the same layout and point `train_env.sh` at the new mounts.

Do **not** start GRPO from `deepseek-v4-flash-grpo-onset` or from a collapsed SFT/RFT LoRA. Actor is **base Qwen3-8B**.

## Minimal data (1.1 MB)

| File | Why |
|---|---|
| `data/rl/swift_prompts_hybrid_band.jsonl` | 233 GRPO prompts |
| `data/rl/heldout_eval_trusted.jsonl` | 65-question gate set |
| `data/rl/heldout_eval.jsonl` | 88-question heldout (optional fallback) |
| `results/hipho_baseline_matrix_8b/base_8b_h88/heldout_scores.json` | base gate baseline |

Trusted-set baseline (avg@4, 8192 tokens): `part_avg=0.2516`, `degrade=0.0192`, `no_boxed=0.0154`. Gate: **≥0.252 / ≤0.05 / ≤0.05**.

Optional (SFT rewrite, not required to train): `data/rl/sft_solutions_longchain.jsonl` → clean with `training/rl_data/clean_longchain_anchors.py`.

## GPU count

`N_TRAIN_GPUS` (default 4) drives both idle-GPU probe and `NPROC_PER_NODE`.

Global batch `N_TRAIN_GPUS × PER_DEVICE_TRAIN_BS × GRAD_ACCUM` must be divisible by `NUM_GENERATIONS` (full mode G=8, `PER_DEVICE_TRAIN_BS=2`). Launch refuses otherwise and prints a suggested `GRAD_ACCUM`.

| GPUs | working example |
|---|---|
| 4 | `GRAD_ACCUM=3` → 24 / 8 = 3 problems/step |
| 2 | `N_TRAIN_GPUS=2 GRAD_ACCUM=4` → 16 / 8 = 2 problems/step |

Pin cards with `CUDA_VISIBLE_DEVICES` to skip the idle probe.

## Command sequence

```bash
# 0. Preflight on the 233-prompt band (one GPU)
CUDA_DEVICE=0 bash training/swift/run_preflight_band.sh
# pass if mixed >= 0.4, trunc <= 0.10, boxed_complete >= 0.95
# this host: mixed 0.785 / trunc 0.054 / boxed_complete 0.994

# 1. Outcome-only GRPO from base Qwen3-8B
N_TRAIN_GPUS=4 MODE=full PHYSICS_REWARD_MODE=outcome_only \
  bash training/swift/wait_and_launch_hybrid_grpo.sh
# or skip the wait:
CUDA_VISIBLE_DEVICES=0,1,2,3 N_TRAIN_GPUS=4 MODE=full \
  PHYSICS_REWARD_MODE=outcome_only \
  bash training/swift/launch_hybrid_grpo_4gpu.sh

# 2. Heldout gate (part_avg >= 0.252 AND degrade <= 0.05 AND no_boxed <= 0.05)
bash training/swift/run_outcome_grpo_gate.sh
```

`OVERLONG_FILTER` defaults to `false` in `outcome_only`. Override with `OVERLONG_FILTER=true` only for an ablation. `monitor_process_reward.py` still hard-stops at 1.8× length explosion and warns on a rising truncation rate.

## Pack for another host

```bash
bash training/swift/pack_outcome_rl.sh /tmp/outcome_rl_bundle.tgz
```

On the new host: unpack, copy `train_env.example.sh` → `train_env.sh`, install the two venvs, place Qwen3-8B, then run the sequence above.
