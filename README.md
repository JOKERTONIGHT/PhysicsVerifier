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
│   ├── experience_code_engine.py   # manifest → rule_id → Python 检查（唯一符号核查路径）
│   ├── generated_experience_checks.py  # 自动生成的确定性检查函数（勿手动编辑）
│   ├── symbolic_system.py          # SymPy LaTeX 解析与公式工具（保留，供脚本使用）
│   ├── match_utils.py              # 符号规范化与别名映射
│   ├── symbolic_catalog.py         # （已弃用）旧 primitive+spec 目录
│   ├── spec_synthesis.py           # （已弃用）旧规则→符号规格合成
│   └── experience_bank.py          # （已弃用）旧经验沉淀模块
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
  --unified-catalog catalogs/unified_rule_library_20260324.json \
  --experience-code-manifest results/experience_symbolic_program_manifest_v2_unified.json \
  --experience-code-module symbolic.generated_experience_checks_v2_unified
```

> 默认会执行确定性符号核查（experience-code）。若要关闭，加 `--no-symbolic-check`。
> `--no-agentic` / `--agentic-max` / `--experience` / `--experience-rules` 仍然兼容老脚本，但已是 no-op。

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

# 端到端测评流水线（默认带符号核查）
./.venv/bin/python scripts/run_physics_eval_pipeline.py \
  --recall-input data/evaluation_recall_20.json \
  --precision-input data/evaluation_precision_20.json \
  --output-dir results/eval_run \
  --check-model qwen3-30b-a3b \
  --unified-catalog catalogs/unified_rule_library_20260324.json \
  --experience-code-manifest results/experience_symbolic_program_manifest_v2_unified.json

# 在 screen 中一键跑现有双链路测评集 + 经验代码符号核查：
bash scripts/run_e2e_with_experience_symbolic.sh

# 说明：
# 1. 流水线默认会向 run_verifier 传入 `--max-per-sample 12`、`--max-per-paragraph 2` 以降低过度诊断；
#    若要完全关闭限额以便与早期实验严格可比：`--max-per-sample 0 --max-per-paragraph 0`。
# 2. 符号核查默认开启，使用 `symbolic/generated_experience_checks.py` 中由
#    `scripts/generate_symbolic_checks.py` 翻译的 Python 函数；可通过
#    `--no-symbolic-check` 关闭，或修改 `--experience-code-manifest` /
#    `--experience-code-module` 切换不同 manifest。

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

## 符号核查路径说明

**当前唯一符号核查路径**：经验代码确定性检查。每条经验规则经
`scripts/generate_symbolic_checks.py` 翻译为 Python 函数，写入
`symbolic/generated_experience_checks_v2_unified.py`，并在
`results/experience_symbolic_program_manifest_v2_unified.json`
中按 `rule_id → function_name` 注册（v2 版基于 `unified_rule_library_v2_distilled300_20260503.json`（默认主流程，无 LLM 元数据增强）；历史增强版见 `unified_rule_library_v2_llm_enhanced_20260504.json`
的 514 条 `exp_*` 规则直接生成，覆盖率 482 / 514 ≈ 94%）。运行时 `ExperienceCodeEngine`
（`symbolic/experience_code_engine.py`）按 `rule_id` 查找并执行对应函数。

`PhysicsRuleVerifier.verify()` 中的核查流程：

1. **Top-down**：对 LLM 给出的每条诊断 `d`，若其 `rule_id` 在 manifest 中，
   则运行对应函数得到 `pass / fail / inconclusive`。
   - `fail` → `symbolic_reconciliation.status = "supported"`，保留诊断；
   - `pass` → 抑制诊断（写入 `agentic.suppressed_diagnostics`），但若诊断引用片段
     与规则 `required_symbols` 重叠 ≥0.5 则降级为 `quote_overlap` 保留；
   - `inconclusive` → 标记为 `inconclusive` 并保留。
2. **Bottom-up**：对每个被检索到的主题，遍历该主题在 manifest 中其余的经验
   规则代码，运行后若返回 `fail` 则发出新的诊断（写入 `experience_post_diagnostics`
   并合入最终 `diagnostics`）。每个主题最多运行 `--symbolic-topic-check-limit`
   条（默认 40）。

> 已移除：`SymbolicCatalog`（primitive+spec）、`RuleSymbolicSpecSynthesizer`、
> `SymbolicExperienceBank`、agentic LLM-driven spec 合成、关键词触发的
> `enable_experience_pipeline`。这些路径在新版本中均不再执行；相关 CLI 参数
> （`--no-agentic` / `--agentic-max` / `--experience` / `--experience-rules`）
> 作为 no-op 保留以兼容老脚本。
> 历史模块（`symbolic/symbolic_catalog.py` 等）仍在仓库中，仅供离线分析或回放。

---

## 许可证

MIT
