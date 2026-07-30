# PhysicsVerifier

PhysicsVerifier 用于检查物理竞赛题的模型解答。系统根据题目背景检索适用规则，再对待检查答案进行语义和可选符号核查。

## 当前主流程

```text
题目与上下文
  → 背景分析与 Domain
  → Topic
  → Scenario Cluster
  → Rule
  → 语义检查
  → 可选符号核查
  → 最终诊断
```

正式检索使用 API 语义导航。题目背景负责规则适用性，`prediction` 负责答案证据，参考答案不进入 verifier。

## 快速入口

- 运行检查：`scripts/run_verifier.py`
- 语义导航：`core/unified_semantic_matcher.py`
- 检查主流程：`core/physics_rule_verifier.py`
- 当前运行规则库：`catalogs/rules_unified_3000_runtime_backfilled.json`
- 共享文档：[文档索引](docs/文档索引.md)

所有项目命令必须先进入对应的 conda 环境。

```bash
conda env create -f environment.yml
conda activate physicsverifier

python scripts/run_verifier.py --help
```

只运行语义检索：

```bash
python scripts/run_verifier.py \
  --retrieval-only \
  --continue-on-semantic-error \
  --unified-retrieval-mode semantic \
  --input data/input.json \
  --output results/background_retrieval/semantic_tree_results.json \
  --unified-catalog catalogs/rules_unified_3000_runtime_backfilled.json \
  --model qwen3-30b-a3b-instruct-2507 \
  --semantic-output-adapter forced_tool_call \
  --semantic-json-attempts 3 \
  --unified-rule-top-n 6
```

## 测试

```bash
python -m unittest \
  tests.test_unified_semantic_matcher \
  tests.test_verifier_semantic_trace_cli
```

项目状态和下一步见[项目进展](docs/项目进展.md)。
