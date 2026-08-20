# 06 证据分层与 Reconciliation 策略

## 1. 现状问题

当前 [`PhysicsRuleVerifier.verify`](../../core/physics_rule_verifier.py) 中符号 reconcile 逻辑：

| 符号结果 | 对 diagnostic 的影响 |
|----------|---------------------|
| `fail` | `symbolic_reconciliation.status = supported`，保留 |
| `pass`（direct spec） | **抑制** diagnostic |
| `inconclusive` | 中性，不标记 |
| topic-bridge `pass` | 降为 inconclusive |

问题：

1. **pass 抑制风险**：e2e 实验中出现符号 pass 误 suppress 真错误（如 TIR 动量规则 pass 与 GT 的 sin/cos 错误无关）
2. **弱 pass**：`ExperienceCodeEngine._normalize_symbolic_result` 已将含「基本正确」「已考虑」等降级为 inconclusive，但仍不足
3. **heuristic pass 与 CAS pass 等价**：token 覆盖即可 refute，精度风险高
4. **inconclusive 故意不惩罚语义**（L1443–1449 注释）：合理，但 pass 门槛应更严

## 2. 证据分层模型（目标）

将三值 `pass|fail|inconclusive` 扩展为证据类型：

| 证据类型 | 含义 | 对 diagnostic 的影响（Phase 1+） |
|----------|------|--------------------------------|
| `supports_error` | 硬证据支持 LLM 诊断 | 提高排序；可选 mark supported |
| `refutes_error` | 硬证据证明诊断不成立 | **仅高置信时** suppress |
| `supports_correctness` | 规范式正确存在 | 不单独 suppress |
| `violates_dimension` | 量纲不齐次 | 强 supports_error |
| `formula_mismatch` | 表达式结构错误 | supports_error |
| `constraint_violation` | 约束不满足 | supports_error |
| `no_signal` | 无法判定 | 中性，仅统计 |

### 2.1 refute 条件（全部满足才可 suppress）

1. **direct rule**：spec 来自 diagnostic 同 rule_id，非 topic-bridge
2. **局部验证**：证据 span 与 diagnostic quote 重叠，或验证目标 equation 在 quote 内
3. **强验证**：backend 为 sympy/dimension/z3，且 parse_ok；**非** pattern/heuristic
4. **quote_overlap 保护**：现有 `_quote_symbol_overlap >= 0.5` 时禁止 refute（保留 diagnostic）

## 3. Phase 0 策略（小样本实验）

- **仅输出 audit evidence**，不修改 diagnostics 列表
- **不启用** refute / suppress
- 指标：`suppression_risk` 应为 0

## 4. Phase 1 策略（接入主流程）

### 4.1 supported 侧

- `supports_error` / `violates_dimension` / `formula_mismatch` → 等同现有 `fail`
- 写入 `symbolic_reconciliation.status = supported`
- release gate 可提高 `rule_score` 权重

### 4.2 refute 侧

- 仅 `refutes_error` + 强验证 + direct + 无 quote_overlap → 等同现有 `pass` suppress
- heuristic/pattern 的 pass → 映射为 `supports_correctness` 或 `no_signal`，**不 suppress**

### 4.3 topic-bridge

- 仅 `supports_error` 可 corroborate
- bridge 永不 refute（与现有 L1342–1345 一致）

## 5. release gate 衔接

[`_diagnostic_release_gate`](../../core/physics_rule_verifier.py) 当前读：

- `symbolic_reconciliation.status`
- `symbolic_policy`（`suppress_on_pass`, `require_fail`, `suppress_on_inconclusive`）
- `quote_symbol_ratio`

规划扩展：

```python
release_gate = {
    "symbolic_status": "supported|refuted|none|quote_overlap",
    "evidence_types": ["formula_mismatch"],
    "evidence_backend": "sympy",
    "evidence_strength": "hard|soft|none",
    ...
}
```

- `evidence_strength=hard` + supported → 放宽 `min_publish_score`
- `evidence_strength=soft` → 不影响 publish
- refuted 仅当 `evidence_strength=hard` 且 policy 允许

## 6. 排序权重

[`DiagnosticAggregator._symbolic_rank`](../../core/diagnostic_aggregator.py) 规划：

```
supported (hard) > supported (soft) > none > inconclusive
```

再按 `rule_score`, `quote_symbol_ratio`。

## 7. 与 semantic_policy 分工

| 任务 | 负责层 |
|------|--------|
| 建模/题意/概念 | SemanticRuleChecker |
| 公式/量纲/约束硬证据 | VerificationRouter |
| 是否发布 diagnostic | release_gate（综合） |
| FP 过滤（broad/irrelevant） | DiagnosticValidator |

符号层**不应**替代 Validator 做 broad/irrelevant 过滤。

## 8. inconclusive 策略（保持不变）

来自现有代码注释的合理策略：

> inconclusive 不标 negative，避免 uneven coverage 导致误 suppress

扩展：

- `no_signal` 等同 inconclusive
- parse_error 写入 audit，不惩罚语义 diagnostic
- strict 模式下 `require_fail` 对 inconclusive 的抑制需单独评估，避免误杀

## 9. 迁移路径

1. Phase 0：audit-only，无 reconcile 变更
2. Phase 1：新后端输出 evidence_types；reconcile 仅认 hard supports
3. Phase 2：refute 门槛收紧；heuristic pass 不再 suppress
4. Phase 3：按规则 `symbolic_policy` 配置 evidence 权重

## 10. 反例与防护

| 场景 | 错误行为 | 防护 |
|------|----------|------|
| cl_188 TIR 动量 pass | suppress sin/cos GT | refute 需 direct + 强验证 + quote 局部 |
| norm_bfd804513bfd48ee FP | 转动惯量 token 误触 | heuristic 不 suppress |
| bridge pass | 不同 rule 误 refute | bridge pass → inconclusive（已有） |
| 公式对、解释错 | 弱 pass | 禁止仅凭 canonical 出现 pass |
