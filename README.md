# PhysicsVerifier

一个基于 LLM 的物理问题解答规则检查器。

本工具的核心思想是采用一个两阶段的流程来验证一份解答（例如，学生或AI模型的作答）是否符合一系列预定义的物理和逻辑规则：

1.  **规则翻译 (离线)**: 使用 `translate_rules.py` 脚本，将用 Python 定义的规则插件（位于 `rules/` 目录）通过 LLM 翻译成一种清晰、结构化的“符号化规则定义” (Symbolic Rule Definition, SRD)。这些 SRD 文本存储在 `rule_translations.json` 文件中，作为检查的“规则手册”。

2.  **规则检查 (在线)**: 主程序 `rule_based_verifier.py` 在运行时加载 `rule_translations.json`。对于每一个待检查的样本，它会：
    a.  从解答文本中提取符号、公式等结构化信息，构建一个临时的符号图。
    b.  将样本的结构化信息摘要和 SRD 文本组合成一个 Prompt。
    c.  请求 LLM 根据 SRD 规则来检查样本，并以结构化的 JSON 格式返回发现的违规项。

这个架构将“规则定义”和“规则执行”解耦，使得规则本身变得透明、可审计，同时将复杂的逻辑判断任务完全交给强大的 LLM，极大地简化了本地代码的复杂性。

## 主要文件

- `rule_based_verifier.py`: **核心检查器和命令行入口**。用于加载样本、执行规则检查并生成报告。
- `translate_rules.py`: **规则翻译器**。运行此脚本以（重新）生成 `rule_translations.json`。
- `rule_translations.json`: 由翻译器生成的**符号化规则定义手册**。
- `rules/`: **规则插件目录**。你可以在此添加或修改规则。

## 快速开始

### 1. 安装依赖

建议使用 Python 3.10+。

```bash
# 创建并激活虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 安装核心依赖
pip install -U python-dotenv pandas numpy openpyxl sympy rich openai
```

如果需要使用 LLM 功能（规则翻译或检查），请在项目根目录创建一个 `.env` 文件并填入你的 API Key：

```
OPENAI_API_KEY=sk-xxxx
```

### 2. 准备输入数据

检查器 `rule_based_verifier.py` 需要一个 JSON 文件作为输入，该文件是一个包含多个样本对象的数组。每个对象应至少包含以下字段：

- `id`: 样本的唯一标识符。
- `prediction`: 需要被检查的完整解答文本。
- `question` (可选): 问题的描述文本。
- `answer` (可选): 标准答案。

`data/evaluation_input.json` 是一个符合此格式的示例文件。你可以参考 `scripts/convert_combined_to_eval_format.py` 脚本来转换你自己的数据。

### 3. 翻译规则

在第一次运行或修改了 `rules/` 目录下的规则后，你需要生成规则手册：

```bash
python3 translate_rules.py
```

此命令会调用 LLM 将 `rules/` 下的每个规则翻译成 SRD，并覆盖写入 `rule_translations.json`。

### 4. 运行检查

使用 `rule_based_verifier.py` 对输入文件进行检查：

```bash
# 使用默认配置运行检查
python3 rule_based_verifier.py --input data/evaluation_input.json --output evaluation_results/my_report.json

# 禁用 LLM 缓存
python3 rule_based_verifier.py --input data/evaluation_input.json --no-cache

# 指定只检查部分规则
python3 rule_based_verifier.py --input data/evaluation_input.json --rules var_const_consistency formula_correctness

# 完全禁用 LLM，只进行基础的符号提取（此时无法进行规则检查）
python3 rule_based_verifier.py --input data/evaluation_input.json --no-llm
```

检查报告将保存在 `--output` 参数指定的路径下。报告中会包含每个样本的诊断信息和评分。

## 扩展规则

要添加新规则，你可以在 `rules/` 目录下创建一个新的 Python 文件，并实现一个继承自 `rules.base.RulePlugin` 的类。你只需要提供规则的自然语言描述 (`description`) 和一些元信息。之后，运行 `scripts/translate_rules.py`，LLM 会自动为你的新规则生成 SRD，使其在检查流程中生效。

## 故障排查

- **LLM 调用失败**: 确认 `.env` 文件中的 API Key 是否正确且有效，并检查网络连接。
- **找不到输入文件**: 确认 `--input` 参数提供的路径是否正确。
- **规则检查无效**: 确认你已经运行了 `scripts/translate_rules.py` 来生成 `rule_translations.json` 文件。

## 许可证

本项目基于 MIT 许可证发布，详见 `LICENSE`。
