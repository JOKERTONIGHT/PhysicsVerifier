# 形式化符号验证设计文档

本目录包含 PhysicsVerifier 符号检查模块优化与重构的设计文档。当前阶段**不**对 1500 规模规则库做完整符号测评（尚未生成对应符号检查代码），而是：

1. **近期**：基于纯语义检查结果，筛选可符号化错误类型，在小样本上做针对性实验；
2. **长期**：将学生作答抽象为 Physics Reasoning Trace (PRT)，按后端路由形式化验证。

## 文档索引

| 文档 | 内容 |
|------|------|
| [01_semantic_result_analysis.md](01_semantic_result_analysis.md) | 纯语义/规则语义结果分析、失败类型、符号化边界 |
| [02_small_sample_experiment_plan.md](02_small_sample_experiment_plan.md) | 小样本选择、primitives、指标、消融与 Go/No-Go |
| [03_physics_reasoning_trace.md](03_physics_reasoning_trace.md) | PRT 中间表示、抽取流程、与 SymbolGraph 关系 |
| [04_verification_backends.md](04_verification_backends.md) | SymPy、量纲、SMT/Z3、Lean 职责边界 |
| [05_integration_with_current_pipeline.md](05_integration_with_current_pipeline.md) | 与现有 SemanticRuleChecker / PhysicsRuleVerifier 衔接 |
| [06_reconciliation_policy.md](06_reconciliation_policy.md) | 证据分层、自动抑制限制、inconclusive 策略 |
| [07_risks_and_roadmap.md](07_risks_and_roadmap.md) | 风险、阶段路线、决策标准 |

## 数据与脚本

| 路径 | 说明 |
|------|------|
| `scripts/analyze_symbolic_candidate_errors.py` | 从语义评测结果生成可符号化候选清单 |
| `data/derived/symbolic_small_sample_experiment_v1/candidate_errors.json` | 405 条 GT 的分类与排序结果 |
| `data/derived/symbolic_small_sample_experiment_v1/experiment_manifest.json` | 首批 6 样本、18 项实验配置 |

## 设计原则

- **符号层不重复语义擅长任务**：建模选择、题意理解、概念适用性仍由 LLM 语义检查承担。
- **符号层提供硬证据**：公式等价、量纲齐次、约束违例、幂次/符号错误等可判定片段。
- **保守合并**：首轮实验只输出 audit evidence，不启用自动 refute/suppress。
- **可失败、可解释**：解析失败 → `no_signal`；不将 inconclusive 当作负面信号。

## 相关代码

- 现有符号执行：[`symbolic/experience_code_engine.py`](../../symbolic/experience_code_engine.py)
- 主验证器：[`core/physics_rule_verifier.py`](../../core/physics_rule_verifier.py)
- SymPy 预留：[`symbolic/symbolic_system.py`](../../symbolic/symbolic_system.py)
- 遗留 primitive：[`rules/symbolic_checks.py`](../../rules/symbolic_checks.py)

## 前置约束

1500 规则库（`norm_*`）目前**没有**对应的符号检查 manifest。`scale_1500_cleaned` 的 symbolic audit 为空是预期状态，不是接线 bug。完整符号测评需待规则库符号代码生成后再进行。
