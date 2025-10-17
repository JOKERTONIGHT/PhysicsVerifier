"""
变量/常量混淆检查器

目标：在评估过程中，对模型作答（prediction）进行基于符号的静态一致性检查，必要时可用LLM辅助判断物理公式等价性与适用性。

设计概览：
- 符号图（SymbolGraph）：节点是符号（量或常数），包含属性：
  - name: 符号名（如 v, a, g, m, k, I）
  - kind: "variable" | "constant" | "unknown"
  - sources: 来源（题目条件/推导步骤/常识库），字符串列表
  - formulas: 涉及该符号的公式字符串列表
  - parents: 由哪些符号（边）推导而来
  - meta: 任意键值
- 有限状态：按“出现→赋义→使用→定值/变值→终态”的视角对符号进行检查。

检查规则（静态）：
1) 常量库匹配：如 g, c, h, e, k_B, R（气体常数）等，如被标记为 variable 且在后续被当作可变量反复赋不同值，记为疑似混淆。
2) 题目条件显式常量：识别“取常量”“保持不变”“为常数”等关键词，如之后被重复赋不同值或显著依赖变量变化，记为混淆。
3) 变量被当常量：出现“设 x 为常数/固定值”且后续又随时间/位置变化（通过文本线索：随 t、x、y 变化），记为混淆。
4) 公式适用性：通过关键字匹配（如匀加速、理想流体、绝热/等温等），若上下文不一致，提出提醒（需要 LLM 时可更准确）。

LLM 辅助（可选）：
- 当 max_llm_calls > 0 且提供 llm_model 时，对存在歧义的等价性/适用性做一次问询，以提升准确率（此处仅保留接口与占位实现）。

输出：
diagnostics: List[{
  id, severity("error"|"warning"|"info"), message, symbol, rule, evidence
}]
并返回 summary 分数：建议简单映射，错误-1，警告-0.5，信息-0；总分越低代表问题越多。
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import re
import json
import math
from datetime import datetime

# 可复用项目内 Judge 构建与 Key 检测
try:
	from vlmeval.dataset.utils import build_judge  # type: ignore
	from vlmeval.smp import gpt_key_set  # type: ignore
except Exception:  # 允许在无依赖环境下继续工作（仅规则检查）
	build_judge = None  # type: ignore
	def gpt_key_set():  # type: ignore
		return False


@dataclass
class SymbolNode:
	name: str
	kind: str = "unknown"  # variable|constant|unknown
	sources: List[str] = field(default_factory=list)
	formulas: List[str] = field(default_factory=list)
	parents: List[str] = field(default_factory=list)
	meta: Dict[str, Any] = field(default_factory=dict)


class SymbolGraph:
	def __init__(self) -> None:
		self.nodes: Dict[str, SymbolNode] = {}

	def get(self, name: str) -> SymbolNode:
		if name not in self.nodes:
			self.nodes[name] = SymbolNode(name=name)
		return self.nodes[name]

	def ensure_kind(self, name: str, kind: str, source: Optional[str] = None):
		node = self.get(name)
		if node.kind == "unknown":
			node.kind = kind
		elif node.kind != kind:
			# 记录冲突
			node.meta.setdefault("kind_conflicts", []).append({"from": node.kind, "to": kind, "source": source})
			# 不强制覆盖，保留首次判定
		if source:
			node.sources.append(source)
		return node

	def add_formula(self, formula: str, symbols: List[str], parents: Optional[List[str]] = None):
		for s in symbols:
			node = self.get(s)
			node.formulas.append(formula)
			if parents:
				node.parents.extend(parents)


DEFAULT_PHYSICAL_CONSTANTS = {
	"g": "standard gravity (≈9.8 m/s^2)",
	"c": "speed of light",
	"h": "Planck constant",
	"e": "elementary charge",
	"k_B": "Boltzmann constant",
	"R": "gas constant",
	"G": "gravitational constant",
	"mu0": "magnetic constant",
	"epsilon0": "electric constant",
}


class VariableConstantVerifier:
	def __init__(self, llm_model: Optional[str] = None, max_llm_calls: int = 0, logger=None):
		self.llm_model = llm_model
		self.max_llm_calls = max_llm_calls
		self.logger = logger
		self._llm = None
		self._llm_calls_used = 0
		# 尝试初始化 LLM（若可用）
		if self.llm_model and self.max_llm_calls > 0 and build_judge and gpt_key_set():
			try:
				self._llm = build_judge(model=self.llm_model, timeout=180, retry=1, temperature=0.2, max_output_tokens=4096, verbose=False)
				self._log(f"LLM 辅助已启用: {self.llm_model}")
			except Exception as e:
				self._llm = None
				self._log(f"LLM 初始化失败，回退为规则检查: {e}")

	def _log(self, *args):
		if self.logger is not None:
			try:
				self.logger.info(" ".join(map(str, args)))
			except Exception:
				pass

	# 粗略提取可能的符号：基于 latex/变量命名的启发式
	_symbol_regex = re.compile(r"\\?([a-zA-Z][a-zA-Z0-9_]*)")

	def _extract_symbols_and_formulas(self, text: str) -> Dict[str, Any]:
		symbols = set()
		formulas = []
		for line in (text or "").splitlines():
			line = line.strip()
			# 粗略识别公式：包含 =、~、≈、∝ 等且含变量
			if any(op in line for op in ["=", "≈", "~", "∝"]):
				formulas.append(line)
			for m in self._symbol_regex.finditer(line):
				sym = m.group(1)
				# 排除常见非物理词
				if sym.lower() in {"final", "answer", "boxed", "ans", "unit", "units"}:
					continue
				symbols.add(sym)
		return {"symbols": list(symbols), "formulas": formulas}

	def _infer_symbol_kinds(self, graph: SymbolGraph, question: str, context: str, prediction: str):
		text = "\n".join([t for t in [question, context, prediction] if t])
		# 常量库判定：若出现与常量名匹配的符号，优先设为 constant
		for const_name in DEFAULT_PHYSICAL_CONSTANTS.keys():
			if re.search(rf"\b{re.escape(const_name)}\b", text):
				graph.ensure_kind(const_name, "constant", source="constant_library")

		# 关键字判定："设 X 为常量/固定" -> constant；"令 X 随 t" -> variable
		for name, node in list(graph.nodes.items()):
			pass  # 将在后续统一处理

		# 从文本中再扫描“设/定义/为常量/固定/保持不变”等
		for m in re.finditer(r"(?:设|定义|令)\s*([a-zA-Z][a-zA-Z0-9_]*)\s*(?:为)?\s*(常量|常数|固定|不变)", text):
			sym = m.group(1)
			graph.ensure_kind(sym, "constant", source="explicit_decl")
		for m in re.finditer(r"(?:随|关于)\s*([a-zA-Z][a-zA-Z0-9_]*)\s*(?:变化|改变)", text):
			sym = m.group(1)
			graph.ensure_kind(sym, "variable", source="explicit_var_change")

		# 按常见物理量推断（弱启发式）
		for name in list(graph.nodes.keys()):
			if name in DEFAULT_PHYSICAL_CONSTANTS:
				graph.ensure_kind(name, "constant", source="constant_library")

	def _static_checks(self, graph: SymbolGraph, question: str, context: str, prediction: str) -> List[Dict[str, Any]]:
		diagnostics: List[Dict[str, Any]] = []
		text = "\n".join([t for t in [question, context, prediction] if t])

		# 规则1：常量被多次赋不同值（启发式：出现 "g = 9.8" 与 "g = 10" 等多处不同数字）
		for const in DEFAULT_PHYSICAL_CONSTANTS.keys():
			assigns = re.findall(rf"\b{const}\s*=\s*([-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)", text, flags=re.IGNORECASE)
			uniq = set(assigns)
			if len(uniq) > 1:
				diagnostics.append({
					"severity": "error",
					"rule": "constant-multiple-assignment",
					"symbol": const,
					"message": f"常量 {const} 在推导中被赋予多个不同的数值：{sorted(uniq)}",
					"evidence": assigns,
				})

		# 规则2：显式声明为常量的符号，后续又出现 "随 t/时间/位置/角度 变化"
		for name, node in graph.nodes.items():
			if node.kind == "constant":
				if re.search(rf"\b{name}\b\s*(?:随|关于).*(?:t|x|y|z|r|θ|phi|时间|位置|角度).*(?:变化|改变)", text):
					diagnostics.append({
						"severity": "error",
						"rule": "declared-constant-varies",
						"symbol": name,
						"message": f"符号 {name} 被声明为常量，但文本中出现其随其它变量变化的描述",
					})

		# 规则3：显式“设为常量/固定”，又出现不同赋值
		for name, node in graph.nodes.items():
			if node.kind == "constant":
				assigns = re.findall(rf"\b{name}\s*=\s*([-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)", text, flags=re.IGNORECASE)
				if len(set(assigns)) > 1:
					diagnostics.append({
						"severity": "warning",
						"rule": "declared-constant-multivalue",
						"symbol": name,
						"message": f"符号 {name} 作为常量出现了多个取值：{sorted(set(assigns))}",
						"evidence": assigns,
					})

		# 规则4：变量被当常量（粗略）：出现“设 x 为常量/固定”，后续又和 t 的导数/函数同现
		for name, node in graph.nodes.items():
			if node.kind == "constant":
				# 如果明确导数形式出现，如 dx/dt, v(t), a(t)
				if re.search(rf"(?:d\s*{re.escape(name)}\s*/\s*dt)|\b{re.escape(name)}\s*\(t\)|\b{re.escape(name)}\s*\(x\)", text):
					diagnostics.append({
						"severity": "error",
						"rule": "constant-has-derivative",
						"symbol": name,
						"message": f"符号 {name} 被视为常量却出现了随时间/位置变化的函数或导数",
					})

		# 规则5：公式适用性关键词不一致（仅提示）
		if re.search(r"匀加速|恒加速度", text) and re.search(r"空气阻力|阻尼|粘滞|变化的力", text):
			diagnostics.append({
				"severity": "info",
				"rule": "kinematics-assumption-mismatch",
				"symbol": None,
				"message": "同时出现匀加速假设与阻力等非恒定力的描述，需检查适用性",
			})

		return diagnostics

	# ------------------------ LLM 辅助模块 ------------------------
	def _llm_available(self) -> bool:
		return self._llm is not None and self._llm_calls_used < self.max_llm_calls

	def _llm_json(self, system_prompt: str, user_prompt: str, fallback=None) -> Any:
		if not self._llm_available():
			return fallback
		try:
			prompt = system_prompt.strip() + "\n\n" + user_prompt.strip()
			resp = self._llm.generate(prompt).strip()
			self._llm_calls_used += 1
			# 尝试解析为 JSON（宽松提取）
			data = self._safe_parse_json_from_text(resp)
			if data is None:
				return fallback
			return data
		except Exception as e:
			self._log(f"LLM JSON 解析失败: {e}")
			return fallback

	@staticmethod
	def _safe_parse_json_from_text(text: str):
		text = text.strip()
		# 尝试直接解析
		try:
			return json.loads(text)
		except Exception:
			pass
		# 通过提取第一个 JSON 对象或数组
		m = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
		if m:
			try:
				return json.loads(m.group(1))
			except Exception:
				return None
		return None

	def _llm_initial_semantic_judgment(self, symbols: List[str], text: str) -> Dict[str, Dict[str, Any]]:
		"""使用 LLM 对候选符号进行初始语义分类（变量/常量/参数），返回映射。"""
		if not symbols:
			return {}
		system = (
			"你是物理竞赛评分助理。请仅输出 JSON，不要解释。"
		)
		user = (
			"根据以下题目与解答文本，从候选符号列表中判断每个符号的语义类型，并给出理由。\n"
			"请严格输出 JSON 对象，键是符号名，值包含: {type: 'variable'|'constant'|'parameter', reason: string, unit: string|null, known_constant: bool}.\n"
			f"候选符号: {symbols}\n\n"
			f"文本:\n{text}\n\n"
			"注意：常见物理常量如 g, c, h, e, k_B, R, G 等更可能为常量；具体以上下文为准。"
		)
		result = self._llm_json(system, user, fallback={})
		return result if isinstance(result, dict) else {}

	def _llm_parse_formulas(self, formulas: List[str]) -> List[Dict[str, Any]]:
		if not formulas:
			return []
		system = "你是物理解题助手。请仅输出 JSON。"
		user = (
			"解析下面的公式列表，输出一个 JSON 数组，每个元素包含:"
			"{ index: number, formula: string, symbols: string[], assumptions: string[], law: string|null }.\n"
			f"公式列表: {formulas}"
		)
		data = self._llm_json(system, user, fallback=[])
		return data if isinstance(data, list) else []

	def _llm_check_applicability(self, question: str, context: str, formula_infos: List[Dict[str, Any]]):
		if not formula_infos:
			return []
		text = "\n".join([t for t in [question, context] if t])
		system = "你是物理竞赛评分助理。请仅输出 JSON。"
		user = (
			"基于题目与上下文，检查这些公式的适用性，返回 JSON 数组："
			"[{ index, formula, applicable: true|false, reason: string }].\n"
			f"题目与上下文:\n{text}\n\n"
			f"公式信息: {formula_infos}"
		)
		data = self._llm_json(system, user, fallback=[])
		return data if isinstance(data, list) else []

	def analyze(self, sample: Dict[str, Any], dataset_key: Optional[str] = None) -> Dict[str, Any]:
		q = sample.get("question") or ""
		c = sample.get("context") or ""
		p = sample.get("prediction") or ""

		# 构建符号图
		graph = SymbolGraph()
		parsed = self._extract_symbols_and_formulas("\n".join([q, c, p]))
		for sym in parsed["symbols"]:
			graph.get(sym)
		for f in parsed["formulas"]:
			# 这里简单将公式中的变量名都计入
			syms_in_f = [m.group(1) for m in self._symbol_regex.finditer(f)]
			graph.add_formula(f, syms_in_f)

		# 推断符号类别
		self._infer_symbol_kinds(graph, q, c, p)

		# 静态检查（基础规则）
		diagnostics = self._static_checks(graph, q, c, p)

		# LLM 动态检查：符号分类 + 公式解析 + 适用性
		text_all = "\n".join([q, c, p])
		if self._llm_available():
			# 1) 符号初判
			llm_sym_map = self._llm_initial_semantic_judgment(list(graph.nodes.keys()), text_all)
			for name, info in llm_sym_map.items():
				t = str(info.get("type", "unknown")).lower()
				if t in {"variable", "constant"}:
					graph.ensure_kind(name, t, source="llm_initial")
				node = graph.get(name)
				node.meta.setdefault("llm", {}).update(info)

			# 2) 公式解析
			f_infos = self._llm_parse_formulas(parsed["formulas"])
			# 合并回 graph 的 meta
			for fi in f_infos:
				try:
					idx = int(fi.get("index"))
				except Exception:
					idx = None
				f_str = fi.get("formula")
				syms = fi.get("symbols", []) or []
				if f_str and syms:
					graph.add_formula(str(f_str), [str(s) for s in syms])
				# 缓存到全局 meta
			# 3) 适用性检查
			app_checks = self._llm_check_applicability(q, c, f_infos)
			for ac in app_checks:
				applicable = bool(ac.get("applicable", True))
				if not applicable:
					diagnostics.append({
						"severity": "warning",
						"rule": "formula-inapplicable-llm",
						"symbol": None,
						"message": f"公式可能不适用: {ac.get('formula', '')}",
						"evidence": ac.get("reason", ""),
					})

		# 汇总分数
		score = 0.0
		for d in diagnostics:
			if d.get("severity") == "error":
				score -= 1.0
			elif d.get("severity") == "warning":
				score -= 0.5
		return {
			"id": sample.get("id"),
			"dataset": dataset_key,
			"diagnostics": diagnostics,
			"symbol_nodes": {k: vars(v) for k, v in graph.nodes.items()},
			"score": score,
		}

	def analyze_batch(self, samples: List[Dict[str, Any]], dataset_key: Optional[str] = None) -> Dict[str, Any]:
		results = [self.analyze(s, dataset_key=dataset_key) for s in samples or []]
		total = sum(r.get("score", 0.0) for r in results)
		summary = {
			"dataset": dataset_key,
			"num_samples": len(results),
			"total_score": total,
			"avg_score": (total / len(results)) if results else 0.0,
			"created_at": __import__("datetime").datetime.now().isoformat(),
		}
		return {"summary": summary, "results": results}

