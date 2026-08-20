# Evaluation

测评层调用远程仓库提供的 verifier，不修改 `core/` 或规则库。

## 正式流程

- 通用测评：`scripts/run_physics_eval_pipeline.py`
- 错误级指标：`scripts/evaluate_physics_eval_sets.py`
- 题目级指标：`scripts/evaluate_question_level_sets.py`
- HiPhO：`evaluation/benchmarks/hipho/`

默认使用：

```text
catalogs/rules_unified_3000_runtime_backfilled.json
retrieval mode: semantic
symbolic: off
```

3000 规则库使用 `norm_*` Rule ID，不能与历史 `exp_*` symbolic manifest 混用。需要符号实验时，必须显式提供相互匹配的 catalog、manifest 和 Python module。

## 历史资产

- `experiments/`：保留的本地规模曲线、审计和消融脚本，属于历史实验入口。
- `generated/`：历史 0900 规则库对应的生成检查代码，不是当前默认。
- `docs/legacy/`：早期报告和运行手册，仅用于复现。

大型输入和输出继续通过根目录 `data`、`results` 链接保存在 slow-share。
