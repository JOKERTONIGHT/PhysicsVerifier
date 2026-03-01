# PhysicsVerifier

一个面向物理解题的**自顶向下（Top-Down）规则检查框架**，集成了**Agentic 符号系统**以提升诊断的准确性。

## 核心特性

- **Top-Down 验证**：自动将题目分类至特定 Topic，并加载对应的物理规则（SRD）进行检查。
- **Agentic 符号后核查**：引入本地 SymPy 引擎对 LLM 的诊断进行“数学/物理一致性”二次核验。
  - **自动纠错**：如果符号检查（如量纲、公式等价性、幂次关系）证明解题过程正确，系统会自动**剔除（Suppress）** LLM 的错误诊断。
  - **知识库沉淀**：符号检查逻辑按 Domain/Topic 维护在 Catalog 中，支持复用与自动生成。
- **清晰输出**：最终诊断结果与符号核查审计日志分文件输出，便于分析。

## 项目结构

详细目录结构如下：

```
.
├── catalogs/                       # 核心逻辑库
│   ├── rules_catalog_top_down.json    # 物理规则库（SRD定义）
│   └── symbolic_catalog.json          # 符号检查逻辑库（按Topic索引的可执行Spec）
├── core/                           # 核心引擎
│   ├── top_down_verifier.py           # 主流程：分类 -> 规则检查 -> 符号核查 -> 结果对齐
│   └── rule_based_verifier.py         # 基于 LLM 的规则检查器实现
├── symbolic/                       # 符号系统基础设施
│   └── symbolic_system.py             # LaTeX -> SymPy 解析、符号图构建
├── rules/                          # 检查逻辑实现
│   └── symbolic_checks.py             # 符号检查执行器 (Primitive: equation_equivalence, power_law等)
├── scripts/                        # 运行脚本
│   └── run_top_down.py                # 批量评估入口
├── tests/                          # 测试用例
└── results/                        # 结果输出目录
```

## 核心流程

1.  **Topic 分类**：LLM 分析题目，将其映射到 `rules_catalog_top_down.json` 中的具体主题。
2.  **规则检查**：注入该主题的规则集，LLM 产出初步的诊断（Diagnostics）。
3.  **符号后核查（Agentic Workflow）**：
    *   系统检测诊断中的数学/物理断言，在 `symbolic_catalog.json` 中检索匹配的符号检查 Spec。
    *   若无匹配，Agent 可尝试生成新的安全 Spec（Data-only）。
    *   **本地执行**：调用 `rules/symbolic_checks.py` 中的原语（Primitives）执行严格的数学验证。
4.  **结果对齐**：若符号检查通过，则撤销原诊断；最终只保留经过验证的“干净”结果。

## 深入了解：Agentic 符号检查系统

本系统核心在于引入“硬”数学计算来防范“软”LLM 的幻觉误判（False Positives）。

### 1. 符号构建 (Symbol Construction)
系统利用 `symbolic/symbolic_system.py` 对解题步骤进行深度解析：
- **LaTeX 解析**：使用 `antlr4` 高精度解析器，将非结构化的 LaTeX 文本（如 `\frac{GM}{r^2}`）转换为结构化的 SymPy 表达式树。
- **增强符号图 (Enriched Symbol Graph)**：构建变量依赖网络，自动识别并规范化物理量符号（如统一 $T$, $Period$, $T_{orb}$）。
- **关系提取**：自动推导变量间的隐含代数关系（例如：从 $E = mc^2$ 推导出 $m \propto E$），为一致性检查提供数学基础。

### 2. 检查流程 (Workflow)
当 LLM 抛出一个诊断（如有疑问）时，系统按以下步骤介入：
1.  **检索 (Retrieval)**：根据当前 Topic 和诊断信息，在 `catalogs/symbolic_catalog.json` 中查找匹配的检查逻辑（Spec）。
2.  **生成 (Generation - 仅 Agentic 模式)**：若无匹配，Agent 分析 SRD 规则，动态生成一个“安全检查规格”（例如：“检查该公式是否等价于 $F=ma$”）。
3.  **执行 (Execution)**：调用 `rules/symbolic_checks.py` 中的原语（Primitives）在符号图上运行验证。
4.  **决策 (Reconciliation)**：
    *   **Fail**: 符号计算确认公式错误 → **保留** LLM 诊断（Error Confirmed）。
    *   **Pass**: 符号计算确认公式数学上正确（即使写法不同） → **Suppress (剔除)** LLM 诊断（False Positive）。
    *   **Inconclusive**: 无法验证（如缺乏足够数学上下文） → 保守保留原诊断。

### 3. 核心原语 (Primitives)
目前支持以下基础数学一致性检查，覆盖常见的代数与幂次律验证：
- **`equation_equivalence`**（代数等价性）：
    - *功能*：验证学生方程是否与标准物理方程（Canonical Form）代数等价。
    - *机制*：计算 `simplify(UserEq - CanonicalEq)` 是否为 0（可选允许常数倍差或常数偏移）。
    - *场景*：能量守恒、牛顿定律、相对论变换等。
- **`power_law`**（幂次律依赖）：
    - *功能*：验证两个变量间是否存在符合物理定律的指数关系。
    - *机制*：支持显式/隐式形式（如 $T^2 \sim r^3$），通过参数 `dependent_power` 精确控制。
    - *场景*：开普勒定律、量纲分析、比例关系验证。
- **`multi_power_law`**（多变量幂次律依赖）：
    - *功能*：验证 $y^k$ 是否满足多变量幂次乘积形式（如 $y^2 \sim r^3 M^{-1}$）。
    - *机制*：对多个自变量的幂指数进行一致性检测，适合复杂比例关系。
    - *场景*：轨道力学、量纲推断与多变量标度律。

### 4. 实例说明
**场景**：学生写出开普勒第三定律 $T^2 = \frac{4\pi^2}{GM} r^3$。
- **原始诊断**：LLM 错误地认为“公式形式应为 $T \propto r$”，判定为 Error。
- **符号核查介入**：
    1.  检测到关键词 "Kepler"，命中 `catalogs/symbolic_catalog.json` 中的 `kepler_check_01`。
    2.  Spec 要求：验证 $T$ 与 $r$ 满足 $T^2 \sim r^3$。
    3.  **执行结果**：SymPy 提取图中关系，确认指数匹配。
    4.  **最终处理**：系统判定“LLM 误报”，自动 Suppress 该条错误。
    5.  **输出结果**：`final_result.json` 中不再显示此错误；在 `symbolic_audit.json` 中记录 `status: supported (suppressed original)`.

## 运行方法

推荐使用 `uv` 或标准 `python` (>=3.10) 运行。

### 1. 执行批量验证

```bash
uv run python scripts/run_top_down.py \
    --input data/evaluation_sample_30.json \
    --output results/final_result.json \
    --symbolic-output results/symbolic_audit.json \
    --model qwen3-30b-a3b
```

**参数说明：**
*   `--output`：**主结果文件**。只包含最终有效的诊断，已自动剔除被符号系统反驳的误报。
*   `--symbolic-output`：**审计日志**。包含所有触发符号检查的记录、执行结果以及被 Suppress 的诊断详情。

### 2. 回归测试

当前版本暂无独立的回归测试脚本（历史测试已清理）。如需添加，请在 tests/ 下新增基于 Symbolic primitives 的测试用例。

## 许可证

MIT
