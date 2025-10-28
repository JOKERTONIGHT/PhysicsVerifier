# PhysicsVerifier

一个通用的“物理竞赛题自动评测”工具集，支持：
- 统一读取模型推理结果（JSON 数组）并进行评分（粗粒度答案匹配 + 可选细粒度 Judge）
- 多次运行结果统计（均值、方差、题目维度统计与汇总表）
- 新增 变量/常量混淆 检查器（基于符号图 + 规则，支持可选 LLM 辅助）

主要文件：
- `eval_physics.py`：命令行评测入口
- `universal_physics_evaluator.py`：评测器实现（数据加载、并行评分、统计/导出）
- `variable_constant_verifier.py`：变量/常量检查器

## 快速开始

1) 安装依赖（建议 Python 3.10+）：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U python-dotenv pandas numpy openpyxl sympy rich
# HuggingFace datasets（VLMEvalKit 的部分模块会间接依赖）
pip install -U datasets
# 安装 VLMEvalKit（包名为 vlmeval，来自 GitHub 仓库）
pip install "git+https://github.com/open-compass/VLMEvalKit.git"
# 若使用 uv：
# uv pip install -U python-dotenv pandas numpy openpyxl sympy rich datasets
# uv pip install "git+https://github.com/open-compass/VLMEvalKit.git"
```

如需使用 Judge（细粒度）或变量/常量检查的 LLM 辅助，请准备对应 API Key，并写入 `.env`：

```
OPENAI_API_KEY=sk-xxxx
```

如果无法访问 GitHub，可尝试：

- 使用压缩包直链安装：
	- `pip install https://codeload.github.com/open-compass/VLMEvalKit/zip/refs/heads/main`
- 或从旧仓库子目录安装（备选）：
	- `pip install "git+https://github.com/OpenGVLab/Ask-Anything.git#subdirectory=VLMEvalKit"`
- 实在网络受限：手动下载/拷贝仓库后 `pip install -e VLMEvalKit`

2) 准备推理结果（输入）

- 目录结构：`results_reasoning/<Dataset>/<Model>/*.json`
- 文件为 JSON 数组，每个元素代表一道题。最小字段：
	- `prediction`：模型作答全文（可包含推导与 \boxed{} 最终答案）
	- `answer`：标准答案（字符串或字符串数组，允许写成 JSON 数组字符串）
	- 可选：`points/point`（对应每个答案的分值，单答案可为数值）、`marking`（细粒度评分细则，支持多套）

示例：

```json
[
	{
		"id": "APhO-2025-Q1",
		"question": "…",
		"prediction": "推导… 最终 \boxed{2.50 m/s}。",
		"answer": ["2.50 m/s"],
		"points": [7]
	}
]
```

3) 运行评测

评测单个数据集：

```bash
# 在仓库根目录（PhysicsVerifier/）运行
python3 eval_physics.py --dataset PanPhO_2024/Physics-235B-0929

# 或使用 uv，从任何工作目录运行（脚本会自动将相对路径锚定到脚本目录）
uv run python3 /home/jinjianhan/PhysicsVerifier/eval_physics.py --dataset PanPhO_2024/Physics-235B-0929
```

评测所有可用数据集（会自动发现 `results_reasoning/<Dataset>/<Model>` 且尚未有输出目录的组合）：

```bash
python3 eval_physics.py
# 或
uv run python3 /home/jinjianhan/PhysicsVerifier/eval_physics.py
```

启用细粒度 Judge（如 gpt-4o）：

```bash
python3 eval_physics.py --dataset PanPhO_2024/Physics-235B-0929 \
	--judge-model gpt-4o --api-key YOUR_KEY
```

多次运行统计：

```bash
python3 eval_physics.py --dataset PanPhO_2024/Physics-235B-0929 --multi-runs
```

输出默认在 `evaluation_results/<Dataset>/<Model>/` 下，包含：
- `*_score.json` 汇总
- `*_detailed_results.json` 明细
- `*_detailed.xlsx` 表格
- 多次运行时：`multi_run_statistics.json`、题目统计/运行汇总 xlsx

## 变量/常量混淆检查器

启用检查器：

```bash
python3 eval_physics.py --dataset PanPhO_2024/Physics-235B-0929 --varconst-check
```

可选 LLM 辅助：

```bash
python3 eval_physics.py --dataset PanPhO_2024/Physics-235B-0929 \
	--varconst-check --varconst-llm-model gpt-4o --varconst-max-llm-calls 5 --api-key YOUR_KEY
```

报告输出在 `evaluation_results/varconst_reports/`，包含：
- `summary`：样本数、总分/均分
- `results[*].diagnostics`：逐题诊断（规则名、严重度、证据）
- `symbol_nodes`：符号图（每个符号的 kind、来源、公式等）

## 输入字段对齐规则（关键）

- `answer` 和 `points`（或 `point`）一一对应，长度不一致时会截断到最短长度。
- `marking` 支持：
	- 单套：`["……", "……"]`
	- 多套：`[["……"], ["……"]]`，系统会择优取分最高的一套
- 无 judge 或无 marking 时，仅进行答案匹配（粗粒度）。

## 故障排查

- 运行时报路径/字段错误：检查 `--dataset` 是否为 `Dataset/Model` 形式；确认 JSON 中包含 `prediction` 与 `answer`。
- 细粒度评分无效：确认已设置 API Key，且 `--judge-model` 可正常调用（网络/配额）。
- 多次运行未被识别：确保同一目录下有 2 个及以上 `*.json` 推理结果文件。

## 许可证

本项目基于 MIT 许可证发布，详见 `LICENSE`。