# 04 验证后端职责边界

## 1. 设计原则

- **按可判定性路由**，不追求整题形式化
- **LLM 负责抽取与标注**，符号后端负责**最终可判定判断**
- **解析/抽取失败 → no_signal**，不降级为 heuristic pass/fail
- **保守输出**：fail 仅作 corroboration；pass/refute 需高置信等价

## 2. 后端对照表

| 后端 | 适合验证 | 不适合 | 现有代码 | 阶段 |
|------|----------|--------|----------|------|
| **SymPy** | 代数等价、导数、代入、幂律指数 | 适用条件、建模选择 | `symbolic/symbolic_system.py` | Phase 1 |
| **量纲系统** | 加减齐次、最终答案单位、非法组合 | 无量纲纯数比值 | 无（待建） | Phase 2 |
| **SMT/Z3** | 不等式、区间、正根、分段条件 | 非线性超越方程完整理论 | 无（待建） | Phase 3 |
| **Pattern/Heuristic** | trigger 关键词、canonical 子串 | 鲁棒等价 | `rules/symbolic_checks.py`, generated checks | 保留 fallback |
| **Lean** | 模板化守恒、矢量恒等式、单位类型 | 开放竞赛题全覆盖 | 无 | Phase 4 研究 |

## 3. SymPy 后端

### 3.1 能力

- LaTeX → SymPy（`FormulaParser.parse`）
- 移项后 `simplify(lhs - rhs) == 0`
- 比例关系：两式之比为常数
- 幂律：`RelationAnalyzer.check_power_relationship`
- 导数：对 `dβ/dθ` 类表达式符号求导比对

### 3.2 verify_kind 映射

```json
{
  "verify_kind": "sympy_equiv",
  "canonical_sympy": "Eq(v, sqrt(G*M/(4*a)))",
  "student_scope": "step|equation|global",
  "tolerance": "symbolic"
}
```

### 3.3 失败模式

- LaTeX 非标准 → parse_error → no_signal
- 多解/参数未绑定 → inconclusive
- 等价但写法差异大 → 需 normalize 或 ratio test

## 4. 量纲后端

### 4.1 能力

- 变量 dimension registry：`v: L/T`, `P: M/(L*T^2)`, `rho: M/L^3`
- 等式两边维度传播
- 加减项必须同维

### 4.2 参考工作

- Lean 量纲形式化（dimensional analysis in Lean 4）
- Lean4Physics unit system 扩展
- Python 层可用 SymPy `units` 或自建 `[M,L,T,Q,...]` 向量

### 4.3 verify_kind 映射

```json
{
  "verify_kind": "dimensional_homogeneity",
  "unit_signature": {"R": "m", "rho": "kg/m^3", "D": "m"},
  "expression": "R_shell = rho/(2*pi*R*D)"
}
```

### 4.4 输出

- `violates_dimension`：强 supports_error
- `dimension_ok`：弱证据，**不单独用于 refute**

## 5. SMT/Z3 后端

### 5.1 能力

- `v <= c` 在题设 ΔE 下是否可满足
- `t > 0`, `m > 0`, 取正根
- 线性方程组一致性
- 有限域/simple nonlinear 约束

### 5.2 verify_kind 映射

```json
{
  "verify_kind": "z3_constraint",
  "variables": {"v": "Real", "c": "Real"},
  "constraints": ["v >= 0", "v <= c", "DeltaE > 0"],
  "claim": "v = sqrt(2*DeltaE/m_p)"
}
```

### 5.3 输出

- `constraint_violation`：claim 与 constraints 不可同时满足 → supports_error
- `sat`：claim 在约束下可满足 → 可能 refute（需 direct + 高置信）

## 6. Pattern / Heuristic 后端

### 6.1 现状

[`rules/symbolic_checks.py`](../../rules/symbolic_checks.py) 的 primitive：

- `formula_pattern`, `equation_equivalence`（实为子串）
- `required_symbols`, `power_law`, `compliance_phrase`

[`ExperienceCodeEngine`](../../symbolic/experience_code_engine.py) 执行 LLM 生成的 regex/关键词 Python。

### 6.2 定位

- **fallback**：SymPy/量纲解析失败时
- **trigger 检测**：场景关键词 gate
- **逐步迁移**：高价值规则迁至声明式 SymPy spec

## 7. Lean 后端（长期）

### 7.1 适用场景

- 已模板化的守恒律推导
- 矢量点积/分解恒等式
- 单位系统的类型化 quantity（参考 lean-units）

### 7.2 不适用

- 开放建模、题意选择
- 快速迭代的 1500 规则库
- 需要 LLM 自由形式化的步骤

### 7.3 集成方式

- 仅 `verify_kind=lean` 且模板匹配成功时调用
- LLM 填证明骨架，不自由生成 Lean
- 失败 → no_signal，回退 SymPy

## 8. VerificationRouter 接口（规划）

```python
class VerificationRouter:
    def verify(
        self,
        spec: Dict[str, Any],
        prt: Optional[Dict[str, Any]],
        sample: Dict[str, Any],
        diagnostic: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        kind = spec.get("verify_kind") or spec.get("primitive")
        if kind in ("sympy_equiv", "equation_equivalence_sympy"):
            return self._sympy_backend.verify(spec, sample, prt)
        if kind == "dimensional_homogeneity":
            return self._dimension_backend.verify(spec, sample, prt)
        if kind == "z3_constraint":
            return self._z3_backend.verify(spec, sample, prt)
        return self._heuristic_backend.verify(spec, sample, prt)
```

返回统一 evidence schema（见 02_small_sample_experiment_plan.md）。

## 9. 后端选型决策树

```
错误是否可写成封闭表达式？
├─ 否 → SemanticRuleChecker（语义）
└─ 是 → 是否需要单位/量纲？
    ├─ 是 → 量纲后端
    └─ 否 → 是否为不等式/区间约束？
        ├─ 是 → Z3
        └─ 否 → SymPy
            └─ 解析失败 → Pattern fallback → no_signal
```

## 10. 相关文件

- SymPy：[`symbolic/symbolic_system.py`](../../symbolic/symbolic_system.py)
- Primitive：[`rules/symbolic_checks.py`](../../rules/symbolic_checks.py)
- 经验代码：[`symbolic/experience_code_engine.py`](../../symbolic/experience_code_engine.py)
- 符号生成：[`scripts/generate_symbolic_checks.py`](../../scripts/generate_symbolic_checks.py)
