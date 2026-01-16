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

### 2. 运行回归测试

测试符号系统核心功能（如 Kepler 定律检查、LaTeX 解析等）：

```bash
uv run python tests/test_symbolic_rules.py
```

## 许可证

MIT
