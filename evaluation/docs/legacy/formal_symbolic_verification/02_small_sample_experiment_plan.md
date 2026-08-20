# 02 小样本符号实验计划

## 1. 目标

在 **6 个样本、约 20 条目标 GT** 上验证：符号 verifier 能否为语义 checker 提供**纯语义无法稳定给出的硬证据**（公式结构、量纲、约束），而不启动 1500 规则库符号代码生成或 release gate 改造。

## 2. 样本选择

### 2.1 首批 curated 样本

| sample_id | 主题 | GT 总数 | 目标 GT | 跳过 GT | 优先级 |
|-----------|------|---------|---------|---------|--------|
| cl_188_110801 | 全反射/倏逝波 | 2 | e1 | e2（无依据假设 λ） | 1 |
| cl_104_24899 | 双星/弹弓能量 | 9 | e1,e5,e6,e7,e8 | e9（逻辑非 sequitur） | 1 |
| cl_209_132531 | 涡流/磁矩 | 11 | e3,e5,e8 | e1,e2,e4,e6（建模） | 1 |
| cl_172_95214 | LHC 相对论加速 | 6 | e2,e6 | e5（概念矛盾） | 1 |
| cl_110_31637 | 切伦科夫分辨 | 7 | e3,e4 | e1,e7（题意） | 2 |
| cl_132_53961 | 逆流热交换 | 8 | e1,e2 | e8（方向叙事） | 2 |

清单见 [`experiment_manifest.json`](../../data/derived/symbolic_small_sample_experiment_v1/experiment_manifest.json)。

### 2.2 选择标准

1. `category == symbolizable`（见 candidate_errors.json）
2. 错误可写成 1–2 条 canonical 式或约束
3. 优先 semantic_gap 或高价值 near_miss（formula_algebra / orbital_gravity）
4. 排除跨题误触发、纯建模链、标注噪声

## 3. 符号 Primitives

| primitive | 输入 | 输出 evidence 类型 | backend |
|-----------|------|-------------------|---------|
| `equation_equivalence_sympy` | student expr, canonical expr | formula_mismatch / supports_error | sympy |
| `dimension_check` | expression, variable dimensions | violates_dimension | dimension |
| `power_exponent_check` | expr, var, expected_exp | formula_mismatch | sympy |
| `sign_check` | expr, expected_sign | sign_error | sympy |
| `component_relation_check` | expr, sin/cos pattern | component_mismatch | sympy |
| `inequality_constraint` | expr, constraint (v≤c) | constraint_violation | constraint |
| `formula_pattern` | text, canonical substring | pattern_mismatch | pattern |

### 3.1 输出 schema

```json
{
  "result": "supports_error|refutes_error|no_signal",
  "backend": "sympy|dimension|constraint|pattern",
  "experiment_id": "exp_orbital_velocity_factor",
  "sample_id": "cl_104_24899",
  "target_error_id": "cl_104_24899_e1",
  "evidence": "v_bin = sqrt(GM/(2a)) differs from canonical sqrt(GM/(4a))",
  "matched_span": {"start_char": 0, "end_char": 0, "formula_text": "..."},
  "details": {"sympy_diff": "...", "parse_ok": true}
}
```

**首轮约定**：仅使用 `supports_error` 和 `no_signal`；**不启用** `refutes_error` 自动抑制。

## 4. 实验配置示例

### cl_188_110801 — 波矢 z 分量

```yaml
experiment_id: exp_evanescent_kz_component
primitive: component_relation_check
canonical: "k_{1z} = k_1 \\cos\\theta_1"
wrong_patterns:
  - "k_{1z} = k_1 \\sin\\theta_1"
target_error_id: cl_188_110801_e1
```

### cl_104_24899 — 轨道速度因子

```yaml
experiment_id: exp_orbital_velocity_factor
primitive: equation_equivalence_sympy
canonical: "v = sqrt(G*M/(4*a))"
wrong_patterns:
  - "sqrt(G*M/(2*a))"
target_error_id: cl_104_24899_e1
```

### cl_209_132531 — 电阻量纲

```yaml
experiment_id: exp_R_shell_dimension
primitive: dimension_check
expression: "R_shell = rho/(2*pi*R*D)"
expected_dimension: resistance
target_error_id: cl_209_132531_e5
```

## 5. 评价指标

| 指标 | 定义 | 目标（Go 阈值） |
|------|------|----------------|
| evidence_precision | supports_error 中对应真实 GT 的比例 | ≥ 0.70 |
| hard_evidence_rate | 可解析且返回 supports_error 的比例 | ≥ 0.50 |
| semantic_gap_recovery | 原 semantic_gap 的 GT 被符号补中的比例 | ≥ 0.30 |
| near_miss_localization_gain | near_miss 通过公式 span 对齐 GT 的比例 | 记录，不设硬阈值 |
| no_signal_rate | parse 失败/无公式/不可形式化 | 可解释即可 |
| suppression_risk | refute 误伤真错误（首轮应为 0） | 0 |

## 6. 消融设置

| 标签 | 说明 |
|------|------|
| semantic_only | local_30b checker，无符号 |
| semantic_plus_sympy | 仅 SymPy primitives |
| semantic_plus_dimension | 仅量纲检查 |
| semantic_plus_hybrid | SymPy + 量纲 + 约束 |

基线结果路径见 experiment_manifest.json `baseline` 字段。

## 7. 执行步骤（暂不接入主流程）

1. 从 prediction 抽取 LaTeX/行内公式（复用 `SemanticRuleChecker._extract_symbols_and_formulas`）
2. 按 manifest 逐条运行 declarative spec（不依赖 1500 generated checks）
3. 将 evidence 写入独立 JSON：`results/symbolic_small_sample_v1/audit.json`
4. 人工或脚本对照 target_error_id 计算 precision / gap recovery
5. 根据 Go/No-Go 决定是否进入 PRT 工程化

## 8. Go / No-Go 标准

### 继续推进

- hard evidence precision ≥ 0.70
- 至少一类 semantic_gap 被稳定补中
- no_signal 主要来自不可形式化 GT，而非解析器脆弱
- 不需要每题手写大量专属规则

### 暂缓推进

- evidence 仅重复语义诊断、无 span/硬证据提升
- parse_failure_rate > 0.50
- 自动 refute 带来误抑制

## 9. 相关文件

- 实验 manifest：[`data/derived/symbolic_small_sample_experiment_v1/experiment_manifest.json`](../../data/derived/symbolic_small_sample_experiment_v1/experiment_manifest.json)
- 候选分析：[`data/derived/symbolic_small_sample_experiment_v1/candidate_errors.json`](../../data/derived/symbolic_small_sample_experiment_v1/candidate_errors.json)
- 分析脚本：[`scripts/analyze_symbolic_candidate_errors.py`](../../scripts/analyze_symbolic_candidate_errors.py)
