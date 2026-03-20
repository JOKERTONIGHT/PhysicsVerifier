# PhysicsVerifier

PhysicsVerifier 是一个面向物理解题诊断的混合框架，核心目标是把：
- 自顶向下规则检查（Top-Down SRD）
- 自底向上符号核查（Bottom-Up Symbolic）

做成可迭代、可审计、可经验沉淀的统一流程。

## 当前状态（已完成优化）

### 1. 符号检查覆盖增强
- 扩展了符号关系抽取，支持不等式（`<`, `>`, `<=`, `>=`, `\leq`, `\geq`）进入公式图。
- 新增原语：`inequality_consistency`，用于 `v < c` 这类安全边界检查。
- 新增原语：`formula_pattern`，用于向量积分等难解析公式的保守文本模式校验（如 Faraday 积分形式）。
- `equation_equivalence` 增加了解析失败时的 pattern fallback，降低 `canonical_unparseable` 造成的大规模 `inconclusive`。

### 2. Top-Down 与 Bottom-Up 联动优化
- 增加 `symbolic/spec_synthesis.py`：从 top-down 规则文本自动合成 deterministic symbolic specs。
- 诊断后核查改为三层来源合并：
  1. curated catalog (`catalogs/symbolic_catalog.json`)
  2. promoted experience specs (`results/rule_experience_bank.json`)
  3. rule text synthesized specs（本轮即时合成）
- 只有在上述检查全部无效或仅 `inconclusive` 时，才触发 agentic 生成。

### 3. 错题经验沉淀机制
- 增加 `symbolic/experience_bank.py`，将 agentic 提案先沉淀到经验池，避免直接污染 curated catalog。
- 当同一 `(domain, topic, rule_id)` 的提案重复出现达到阈值后，可被自动晋升为可复用的 bottom-up spec。
- 主流程已经接入 experience 记录，支持后续离线审查和人工回灌 catalog。

### 4. 匹配安全性优化
- `symbolic/symbolic_catalog.py` 的 `find_applicable` 增加 rule-id 对齐约束。
- 有 `match_rule_ids` 的 spec 必须命中同 rule，减少跨 topic/跨规则漂移误匹配。

## 优化过程记录

本轮优化步骤详见：
- `docs/OPTIMIZATION_LOG.md`
- `docs/FINAL_REPORT.md`
- `docs/DIRECTORY_GUIDE.md`

## 目录结构（更新后）

```text
.
├── catalogs/
│   ├── rules_catalog_top_down.json
│   └── symbolic_catalog.json
├── core/
│   ├── top_down_verifier.py
│   └── rule_based_verifier.py
├── rules/
│   ├── llm_rules.py
│   └── symbolic_checks.py
├── symbolic/
│   ├── symbolic_system.py
│   ├── symbolic_catalog.py
│   ├── spec_synthesis.py
│   └── experience_bank.py
├── scripts/
│   ├── run_top_down.py
│   ├── analyze_symbolic_catalog.py
│   └── analyze_symbolic_audit.py
├── tests/
│   ├── test_symbolic_pipeline.py
│   └── test_symbolic_rules.py
├── docs/
│   ├── OPTIMIZATION_LOG.md
│   ├── FINAL_REPORT.md
│   └── DIRECTORY_GUIDE.md
└── results/
```

## 运行与验证

### 1. 批量评估

```bash
uv run python scripts/run_top_down.py \
  --input data/evaluation_sample_30.json \
  --output results/final_result.json \
  --symbolic-output results/symbolic_audit.json \
  --model qwen3-30b-a3b
```

### 2. 分析脚本

```bash
python scripts/analyze_symbolic_catalog.py --catalog catalogs/symbolic_catalog.json --outdir results/symbolic_catalog_analysis_after
python scripts/analyze_symbolic_audit.py --audit results/symbolic_audit_100.json --outdir results/symbolic_audit_100_analysis
```

### 3. 测试

当前新增回归测试可直接运行：

```bash
python -m unittest tests.test_symbolic_pipeline
```

## 许可证

MIT
