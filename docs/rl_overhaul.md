# RL 全链路改造说明

训练奖励与 heldout 评测共用 `training/compat/part_scoring.py`：`r_answer = hits / n_parts`，`acc_item` 只记日志。不要再实现第二套 boxed 匹配。

## 先跑这些门禁

```bash
# sympy（两个 venv 都要有）
.venv/bin/python -c "import sympy; print(sympy.__version__)"
/data1/jinjianhan/venv/openrlhf_train/bin/python -c "import sympy; print(sympy.__version__)"

python -m unittest \
  training.tests.test_part_frac_reward \
  training.tests.test_reward_server \
  training.tests.test_paragraph_process \
  training.tests.test_outcome_only_tools \
  training.tests.test_verifier_auc \
  evaluation.tests.test_score_hipho_predictions
```

`test_rft_solutions_median_part_frac` 要求 RFT 正解的 `r_answer` 中位数 ≥0.9、打乱金标 ≤0.1。不过则不要开训。

## 数据

```bash
python training/rl_data/build_val_split.py
bash training/swift/download_external_assets.sh   # PHYSICS + UGPhysics + Intern-S1-mini
python training/rl_data/build_physics_tiers.py
python training/rl_data/audit_eval_leakage.py \
  --train data/rl/tiers/easy_prompts.jsonl \
  --hipho /slow_share/jinjianhan/workspace/benchmarks/hipho/hipho_text_only.jsonl \
  --heldout data/rl/heldout_eval_trusted.jsonl
```

easy 档验收：base `part_avg@8` 落在 0.4–0.7。高于 0.85 的档只作 smoke。测评分辨率：`data/rl/eval_power300.jsonl`（n=300）+ val `data/rl/val_same_dist.jsonl`（n=180）；k=8，bootstrap CI 写在 `part_avg_at_k_ci`。

```bash
python training/rl_data/build_val_split.py
python training/rl_data/build_eval_power.py
bash training/swift/run_tier_ladder.sh   # 出 results/tier_ladder/ladder.json
```

## 训练

```bash
# 吞吐探针（2 步，目标 ≤150s/step）
CUDA_VISIBLE_DEVICES=0,1,2,3 bash training/swift/run_throughput_probe.sh

# 400 步 outcome-only（easy/mid 优先）
CUDA_VISIBLE_DEVICES=0,1,2,3 bash training/swift/run_outcome_grpo_full.sh
# 结束后对每个 ckpt 打 val 曲线
bash training/swift/eval_saved_checkpoints.sh /slow_share/jinjianhan/ckpt/qwen3-8b-outcome-only-rl
```

前 5 步不满足 `clipped_ratio<0.15`、组内 `reward_std>0.02`、`r_answer>0.05` 会硬停。不要用 train reward 上升当成功；每 50 步在 `val_same_dist.jsonl` 上 k=8 打分，看 `part_avg_at_k` 与 `part_pass_minus_avg`。65 题 olympiad heldout 只做最终外部确认。

## 过程检查

1. 用带正确性标签的 rollout 跑 `training/swift/eval_verifier_auc.py`，AUC≥0.65 才进奖励。
2. 段落过程分已改为指数密度 + 组内 rank 归一化（`PHYSICS_PROCESS_RANK_NORM=1`）+ 无错误时的完成度 `r_first`。离线 `effective_group_rate` 已到 100%。
3. **实测（2026-09-13）**：error_eval 上过程分 vs 答案正确性 **AUC=0.45**，不过 0.65 门禁；`w_process=0`，禁止 hybrid 消融。见 `logs/process_reward_gate.json`、`logs/p6_ablation.json`。
4. 消融脚本：`ARM=outcome` / `ARM=hybrid` 跑 `training/swift/run_process_ablation.sh`（hybrid 会读 AUC 门禁，失败即 refuse）。

GPU 空闲后跑全流程：

```bash
bash training/swift/run_overhaul_pipeline.sh   # wait GPU → ladder → 2-step probe → 400-step
```

## 基座

先零样本，不要直接换着训：

```bash
python training/swift/check_intern_s1_compat.py
HELDOUT=data/rl/tiers/eval_easy.jsonl bash training/swift/run_base_compare.sh
```

Intern-S1-mini 比 Qwen3-8B 在 easy 档高 ≥5pp（超过 2-SE）才值得把 400 步迁过去。

**实测：** 本机 vLLM 0.8.5 **没有** `InternS1` 架构（只有 InternLM / InternVLChat）。transformers 4.57.1 能读 config。在升级 vLLM 或走 transformers 生成之前不要换基座训练。PHYSICS HF 卡是 `desimfj/PHYSICS`（当前仅 test 2k）。UGPhysics 英文 5520 条已纳入 mid/hard/easy（Knowledge Recall / Practical Application → easy）。
