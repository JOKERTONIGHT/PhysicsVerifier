"""
变量/常量混淆检查器（纯 LLM 判定）

目标：完全去除规则与符号推理的依赖，统一由 LLM 直接给出诊断；
仅保留轻量的符号容器，用于写入 LLM 返回的符号元信息（若提供）。

实现：单次（或极少次）LLM 调用，输出 diagnostics 与可选 symbols 概览；
本地不再执行任何规则兜底或公式适用性检查。
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import re
import json
import math
from datetime import datetime
import hashlib
from pathlib import Path

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
	"""极简符号存储：仅承载 LLM 元信息。"""
	def __init__(self) -> None:
		self.nodes: Dict[str, SymbolNode] = {}

	def get(self, name: str) -> SymbolNode:
		if name not in self.nodes:
			self.nodes[name] = SymbolNode(name=name)
		return self.nodes[name]


class VariableConstantVerifier:
	def __init__(self, llm_model: Optional[str] = None, max_llm_calls: int = 0, logger=None,
				 llm_conf_threshold: float = 0.6, enable_cache: bool = True,
				 llm_temperature: float = 0.2, llm_max_output_tokens: int = 4096):
		"""LLM 优先的统一检查流程（单路径，无多模式分支）。"""
		self.llm_model = llm_model
		self.max_llm_calls = max_llm_calls
		self.logger = logger
		self._llm = None
		self._llm_calls_used = 0
		self.llm_conf_threshold = float(llm_conf_threshold)
		self.llm_temperature = float(llm_temperature)
		self.llm_max_output_tokens = int(llm_max_output_tokens)
		# 简易缓存，避免重复 LLM 调用
		self.enable_cache = bool(enable_cache)
		self._cache: Dict[str, Any] = {}
		self._cache_path = (Path(__file__).parent / ".cache" / "varconst_llm_cache.json").resolve()
		if self.enable_cache:
			try:
				self._cache_path.parent.mkdir(parents=True, exist_ok=True)
				if self._cache_path.exists():
					self._cache = json.loads(self._cache_path.read_text(encoding="utf-8"))
			except Exception:
				self._cache = {}
		# 尝试初始化 LLM（若可用）
		if self.llm_model and self.max_llm_calls > 0 and build_judge and gpt_key_set():
			try:
				self._llm = build_judge(model=self.llm_model, timeout=180, retry=1,
										temperature=self.llm_temperature, max_output_tokens=self.llm_max_output_tokens,
										verbose=False)
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

	_common_units = {
		# SI base and derived units (common)
		"m","s","kg","A","K","mol","cd","N","Pa","J","W","C","V","F","Ω","ohm","S","H","Hz","T","lm","lx","Bq","Gy","Sv","Wb",
		# variants and common abbreviations
		"mm","cm","dm","km","ms","us","μs","ns","deg","rad","sr","eV","MeV","GeV","keV","u","L","ml","mL","bar","atm",
	}

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
				# 排除常见单位，减小噪声
				if sym in self._common_units:
					continue
				symbols.add(sym)
		return {"symbols": list(symbols), "formulas": formulas}

	# 取消符号种类本地推断，完全交给 LLM 主调用

	# 无任何本地规则兜底

	# ------------------------ LLM 主导的综合判定 ------------------------
	def _llm_varconst_assess(self, text: str, symbols: List[str], formulas: List[str]) -> Dict[str, Any]:
		"""调用 LLM 进行主判定，期望返回 {diagnostics:[], symbols:{...}}。包含缓存。"""
		payload = {"text": text, "symbols": sorted(symbols or []), "formulas": formulas, "model": self.llm_model}
		cached = self._cache_get("llm_assess", payload)
		if cached is not None:
			return cached if isinstance(cached, dict) else {"diagnostics": [], "symbols": {}}
		system = "你是物理竞赛评分助理。请仅输出 JSON。"
		user = (
			"基于下述题目/上下文/作答，判断是否存在变量/常量使用混淆。\n"
			"请输出 JSON 对象 { diagnostics: [], symbols: {} }：\n"
			"- diagnostics: JSON 数组，每个元素包含 {severity: 'error'|'warning'|'info', rule: string, symbol: string|null, message: string, evidence: any}\n"
			"- symbols: 可选的 JSON 对象，键为符号名，值为 { type: 'variable'|'constant'|'parameter'|'unknown', confidence: number(0..1), unit: string|null, known_constant: bool, ambiguous: bool, overloaded: bool, aliases: string[], canonical: string, reason: string }\n"
			"若无法确定，symbols 可为空对象。\n\n"
			f"候选符号: {symbols}\n"
			f"公式(若有): {formulas}\n"
			f"文本:\n{text}"
		)
		data = self._llm_json(system, user, fallback={"diagnostics": [], "symbols": {}})
		if not isinstance(data, dict):
			data = {"diagnostics": data if isinstance(data, list) else [], "symbols": {}}
		self._cache_set("llm_assess", payload, data)
		return data

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

	# ------------------------ 缓存工具 ------------------------
	def _cache_key(self, payload: Any) -> str:
		try:
			blob = json.dumps(payload, ensure_ascii=False, sort_keys=True)
		except Exception:
			blob = str(payload)
		return hashlib.sha256(blob.encode("utf-8")).hexdigest()

	def _cache_get(self, namespace: str, payload: Any) -> Optional[Any]:
		if not self.enable_cache:
			return None
		key = f"{namespace}:{self._cache_key(payload)}"
		return self._cache.get(key)

	def _cache_set(self, namespace: str, payload: Any, value: Any) -> None:
		if not self.enable_cache:
			return
		key = f"{namespace}:{self._cache_key(payload)}"
		self._cache[key] = value
		try:
			self._cache_path.write_text(json.dumps(self._cache, ensure_ascii=False), encoding="utf-8")
		except Exception:
			pass

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

	# 删除符号分型/聚合与公式适用性附加 LLM 调用，统一纳入主调用输出

	def analyze(self, sample: Dict[str, Any], dataset_key: Optional[str] = None) -> Dict[str, Any]:
		q = sample.get("question") or ""
		c = sample.get("context") or ""
		p = sample.get("prediction") or ""

		# 构建符号容器（仅用于存储 LLM 元信息）
		graph = SymbolGraph()
		parsed = self._extract_symbols_and_formulas("\n".join([q, c, p]))
		for sym in parsed["symbols"]:
			graph.get(sym)

		# 1) LLM 主诊断（期望含 diagnostics 与 symbols）
		text_all = "\n".join([q, c, p])
		llm_out = self._llm_varconst_assess(text_all, parsed["symbols"], parsed["formulas"])
		diagnostics: List[Dict[str, Any]] = []
		diagnostics.extend(llm_out.get("diagnostics", []) if isinstance(llm_out, dict) else [])

		# 写入每符号元信息（若 LLM 提供）
		llm_syms = llm_out.get("symbols", {}) if isinstance(llm_out, dict) else {}
		if isinstance(llm_syms, dict):
			for name, info in llm_syms.items():
				n = graph.get(str(name))
				n.meta.setdefault("llm", {}).update(info if isinstance(info, dict) else {"raw": info})

		# 去重（按 rule+symbol+message）
		seen = set()
		unique_diags: List[Dict[str, Any]] = []
		for d in diagnostics:
			rule = str(d.get("rule"))
			symbol = d.get("symbol")
			msg = str(d.get("message"))
			key = (rule, symbol, msg)
			if key in seen:
				continue
			seen.add(key)
			unique_diags.append(d)
		diagnostics = unique_diags

		# 2) 无规则兜底，完全依赖 LLM 输出

		# 汇总分数（info 不计分，warning -0.5，error -1.0）
		score = 0.0
		for d in diagnostics:
			sev = d.get("severity")
			if sev == "error":
				score -= 1.0
			elif sev == "warning":
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

