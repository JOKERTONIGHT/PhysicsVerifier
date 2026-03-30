# PhysicsVerifier

PhysicsVerifier 是一个面向物理解题诊断的混合框架，核心目标是把：
- 自顶向下规则检查（Top-Down SRD）
- 自底向上符号核查（Bottom-Up Symbolic）

做成可迭代、可审计、可经验沉淀的统一流程。

## 当前状态（已完成优化）

### 0. 统一规则/符号框架（新增）
- 已将 top-down 规则与 experience 规则统一到同一主题框架视图（`unified_rule_frame`）。

- 每个样本在分类后会输出该主题下：top-down 规则数量/ID 与 experience 规则数量/ID。
- 这使后续检查、审计与迭代都可以在统一结构上进行，减少规则分散管理。

### 1. 符号检查覆盖增强
- 扩展了符号关系抽取，支持不等式（`<`, `>`, `<=`, `>=`, `\leq`, `\geq`）进入公式图。
- 新增原语：`inequality_consistency`，用于 `v < c` 这类安全边界检查。
- 新增原语：`formula_pattern`，用于向量积分等难解析公式的保守文本模式校验（如 Faraday 积分形式）。
- `equation_equivalence` 增加了解析失败时的 pattern fallback，降低 `canonical_unparseable` 造成的大规模 `inconclusive`。
- `required_symbols` 从“硬性全匹配”升级为“覆盖率软门槛”（`required_symbol_min_ratio`），减少符号命名差异导致的误拒绝。
- 新增统一符号匹配工具：`symbolic/match_utils.py`，集中处理规范化、别名映射与覆盖率计算。

### 2. Top-Down 与 Bottom-Up 联动优化
- 增加 `symbolic/spec_synthesis.py`：从 top-down 规则文本自动合成 deterministic symbolic specs。
- 诊断后核查改为三层来源合并：
  1. curated catalog (`catalogs/symbolic_catalog.json`)
  2. promoted experience specs (`results/rule_experience_bank.json`)
  3. rule text synthesized specs（本轮即时合成）
- 只有在上述检查全部无效或仅 `inconclusive` 时，才触发 agentic 生成。
- experience symbolic spec 默认注入软匹配阈值，优先走确定性检查，避免过多 fallback 分支。

### 3. 错题经验沉淀机制

- 增加 `symbolic/experience_bank.py`，将 agentic 提案先沉淀到经验池，避免直接污染 curated catalog。
- 当同一 `(domain, topic, rule_id)` 的提案重复出现达到阈值后，可被自动晋升为可复用的 bottom-up spec。
- 主流程已经接入 experience 记录，支持后续离线审查和人工回灌 catalog。

### 4. 匹配安全性优化
- `symbolic/symbolic_catalog.py` 的 `find_applicable` 增加 rule-id 对齐约束。
- 有 `match_rule_ids` 的 spec 必须命中同 rule，减少跨 topic/跨规则漂移误匹配。
- `SymbolicCatalog` 增加 topic 级索引与文件缓存（mtime 失效），优化检索性能和一致性。

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
│   ├── match_utils.py
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

### 0. 本轮新增产物（2026-03-24）

- 300样例经验输出：`results/semantic_experience_300_20260324.json`
- 300样例经验蒸馏：`results/semantic_experience_distilled_300_20260324.json`
- 新统一规则库：`catalogs/unified_rule_library_20260324.json`
- 30样例评测结果：`results/top_down_results_experience_30_20260324_unified.json`
- 30样例符号审计：`results/symbolic_audit_experience_30_20260324_unified.json`

300样例蒸馏统计（本轮）：
- total_distilled_rules: 529
- topic_buckets: 84

### 1. 批量评估

```bash
uv run python scripts/run_top_down.py \
  --input data/evaluation_sample_30.json \
  --output results/final_result.json \
  --symbolic-output results/symbolic_audit.json \
  --model qwen3-30b-a3b
```

使用新蒸馏库跑 sample_30：

```bash
./.venv/bin/python scripts/run_top_down.py \
  --input data/evaluation_sample_30.json \
  --output results/top_down_results_experience_30_20260324_unified.json \
  --symbolic-output results/symbolic_audit_experience_30_20260324_unified.json \
  --model qwen3-30b-a3b \
  --experience \
  --experience-rules results/semantic_experience_distilled_300_20260324.json
```

### 2. 分析脚本

```bash
python scripts/analyze_symbolic_catalog.py --catalog catalogs/symbolic_catalog.json --outdir results/symbolic_catalog_analysis_after
python scripts/analyze_symbolic_audit.py --audit results/symbolic_audit_100.json --outdir results/symbolic_audit_100_analysis
```

构建统一规则库：

```bash
./.venv/bin/python scripts/build_unified_rule_library.py \
  --experience-distilled results/semantic_experience_distilled_300_20260324.json \
  --output catalogs/unified_rule_library_20260324.json
```

### 3. 测试

当前新增回归测试可直接运行：

```bash
python -m unittest tests.test_symbolic_pipeline
```

推荐补充回归：

```bash
python -m unittest tests.test_symbolic_rules
```

建议在项目虚拟环境中执行完整回归：

```bash
./.venv/bin/python -m unittest tests.test_symbolic_pipeline tests.test_symbolic_rules
```

最近一次回归结果（2026-03-24）：
- Ran 7 tests in 0.665s
- OK

## 许可证

MIT
