# PhysicsVerifier

物理竞赛题模型回答规则检查框架。对给定题目和模型作答，逐条匹配规则并调用 LLM 进行语义检查，可选用经验代码执行确定性符号核查。

> 详细结构说明见 [`docs/整体结构.md`](docs/整体结构.md)；快速参考见 [`docs/概览.md`](docs/概览.md)。

---

## 目录结构

```text
.
├── core/
│   ├── physics_rule_verifier.py    # 主检查流程（匹配→语义→符号→合并）
│   ├── semantic_rule_checker.py    # LLM + SRD 语义规则检查引擎
│   └── rule_catalog_retrieval.py   # 层次化目录上的主题/规则检索与打分
├── rule_framework/
│   ├── builder.py                  # 从经验规则集合从头构建层次化规则库
│   ├── maintenance.py              # 增量添加/删除/重聚类/符号绑定
│   ├── validation.py               # 规则库结构校验
│   ├── normalization.py            # 规范化与关键词提取工具
│   ├── models.py                   # 数据模型（RulePath、BuildConfig 等）
│   └── io.py                       # 统一 JSON 读写
├── rules/
│   ├── base.py                     # RuleContext、RuleRuntime、RulePlugin 接口
│   ├── llm_rules.py                # LLM 规则插件定义
│   ├── symbolic_checks.py          # 符号检查规格与执行器
│   └── graph_consistency.py        # 图一致性规则
├── symbolic/
│   ├── experience_code_engine.py   # manifest → rule_id → Python 检查（主路径）
│   ├── generated_experience_checks.py  # 自动生成的确定性检查函数（勿手动编辑）
│   ├── symbolic_system.py          # SymPy LaTeX 解析与公式工具
│   ├── match_utils.py              # 符号规范化与别名映射
│   ├── symbolic_catalog.py         # 符号检查目录加载
│   ├── spec_synthesis.py           # 规则文本 → 符号规格（辅助路径）
│   └── experience_bank.py          # 经验沉淀与晋升（辅助路径）
├── scripts/
│   ├── run_verifier.py             # 主入口：批量运行 PhysicsRuleVerifier
│   ├── manage_rule_library.py      # 规则库管理 CLI（build/add/remove/recluster/validate/bind-symbolic）
│   ├── generate_experience_rules.py # 经验规则生成（语义蒸馏）
│   ├── generate_symbolic_checks.py  # 经验规则 → 符号检查代码
│   ├── analyze_rule_matching.py    # 离线规则匹配质量分析
│   ├── build_physics_eval_sets.py  # 构建错误级/题目级测评集
│   ├── run_physics_eval_pipeline.py # 端到端测评流水线
│   ├── evaluate_physics_eval_sets.py  # 错误级指标计算
│   ├── evaluate_question_level_sets.py # 题目级指标计算
│   ├── compute_strict_eval_metrics.py  # 严格口径 P/R/F1
│   ├── audit_eval_set_quality.py   # 测评集质量审计
│   └── run_llm_checker_baseline.py # 纯 LLM 基线（对比实验用）
├── tests/
│   ├── test_rule_framework.py      # rule_framework 包单元测试
│   ├── test_unified_rules_v2.py    # PhysicsRuleVerifier + 统一规则库集成测试
│   ├── test_symbolic_pipeline.py   # 符号管道测试
│   └── test_symbolic_rules.py      # ExperienceCodeEngine 回归测试
├── catalogs/
│   ├── rules_catalog_top_down.json # 知识规则骨架（领域→主题→规则）
│   ├── symbolic_catalog.json       # 符号检查目录
│   └── unified_rule_library_*.json # 构建产物：统一层次化规则库
├── data/                           # 评测输入数据集
├── docs/                           # 文档
└── results/                        # 运行产物（结果、日志、审计）
```

---

## 快速上手

### 运行规则检查

```bash
./.venv/bin/python scripts/run_verifier.py \
  --input data/evaluation_sample_30.json \
  --output results/result.json \
  --symbolic-output results/symbolic_audit.json \
  --model qwen3-30b-a3b \
  --unified-catalog catalogs/unified_rule_library_20260324.json
```

### 规则库管理

```bash
# 从经验规则集合从头构建层次化规则库
./.venv/bin/python scripts/manage_rule_library.py build \
  --experience results/semantic_experience_distilled_300.json \
  --output catalogs/unified_rule_library.json

# 增量添加新规则
./.venv/bin/python scripts/manage_rule_library.py add \
  --catalog catalogs/unified_rule_library.json \
  --experience results/new_rules.json

# 校验规则库结构
./.venv/bin/python scripts/manage_rule_library.py validate \
  --catalog catalogs/unified_rule_library.json
```

### 经验规则生成与符号代码翻译

```bash
# 语义蒸馏生成经验规则
./.venv/bin/python scripts/generate_experience_rules.py \
  --input data/evaluation_sample_300.json \
  --output results/semantic_experience_distilled_300.json \
  --model qwen3-30b-a3b

# 翻译为可执行符号检查代码
./.venv/bin/python scripts/generate_symbolic_checks.py \
  --input results/semantic_experience_distilled_300.json \
  --manifest results/experience_symbolic_program_manifest.json \
  --output symbolic/generated_experience_checks.py \
  --model qwen3-30b-a3b
```

### 测评流水线

```bash
# 构建测评集
./.venv/bin/python scripts/build_physics_eval_sets.py \
  --input data/physics_rubric_data_1000.json \
  --recall-output data/evaluation_recall_20.json \
  --precision-output data/evaluation_precision_20.json \
  --recall-size 20 --precision-size 20 \
  --model qwen3-30b-a3b

# 端到端测评流水线
./.venv/bin/python scripts/run_physics_eval_pipeline.py \
  --recall-input data/evaluation_recall_20.json \
  --precision-input data/evaluation_precision_20.json \
  --output-dir results/eval_run \
  --model qwen3-30b-a3b \
  --unified-catalog catalogs/unified_rule_library_20260324.json

# 严格口径指标计算
./.venv/bin/python scripts/compute_strict_eval_metrics.py \
  --predictions results/eval_run/error_verifier_results.json \
  --audit results/eval_run/error_symbolic_audit.json \
  --rubric-meta data/rubric_eval_100_meta.json \
  --output results/eval_run/strict_metrics.json
```

### 规模曲线实验

```bash
# 准备分检查点数据
./.venv/bin/python scripts/prepare_error_expansion_samples.py \
  --input data/combined_language_only.json \
  --output data/evaluation_sample_1000_expansion.json \
  --target-size 1000 --seed 20260331

./.venv/bin/python scripts/prepare_scale_checkpoints.py \
  --input data/evaluation_sample_1000_expansion.json \
  --output-dir data/checkpoints --step 200 \
  --manifest results/scale_curve/checkpoint_manifest.json

    # 生成运行手册（内含各检查点的 generate_experience_rules / generate_symbolic_checks / run_verifier 命令）
./.venv/bin/python scripts/generate_scale_runbook.py \
  --manifest results/scale_curve/checkpoint_manifest.json \
  --output-md docs/SCALE_EXPERIMENT_RUNBOOK.md \
  --output-sh scripts/run_scale_checkpoints.sh \
  --model qwen3-30b-a3b

# 聚合结果并绘图
./.venv/bin/python scripts/aggregate_scale_curve.py \
  --metrics-glob 'results/scale_curve/ckpt_*/strict_metrics.json' \
  --output-csv results/scale_curve/curve_metrics.csv

./.venv/bin/python scripts/plot_scale_curve.py \
  --input-csv results/scale_curve/curve_metrics.csv \
  --output results/scale_curve/scale_curve.png
```

---

## 测试

```bash
./.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -q
```

---

## 符号检查路径说明

当前主路径为**经验代码确定性检查**：每条经验规则经 `generate_symbolic_checks.py` 翻译为 Python 函数，写入 `symbolic/generated_experience_checks.py`，在运行时由 `ExperienceCodeEngine`（`symbolic/experience_code_engine.py`）按 `rule_id` 查找并执行。

`symbolic/spec_synthesis.py` 和 `symbolic/experience_bank.py` 为辅助路径，保留代码稳定性，非默认执行路径。

---

## 许可证

MIT
