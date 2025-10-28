"""
物理规则检查器（LLM 驱动 + 轻量符号提取）

目标：将“规则”结构化后交由 LLM 评估，避免本地复杂规则/图推理；
同时用极简的符号/公式提取与上下文聚合，帮助 LLM 更稳地做出判断。

特性：
- 单一或多条规则，逐条转成结构化 Prompt 提交给 LLM；
- 统一的 JSON 输出协议：diagnostics 列表，含 severity/rule/symbol/message/evidence；
- 轻量的符号推理：符号/公式抽取、出现频次、赋值提取、局部上下文片段；
- 缓存与调用上限控制，复用项目内的 build_judge/gpt_key_set。

备注：默认内置一个样例规则 var_const_consistency，将“变量/常量混淆检查”改写成结构化规则 Prompt。
可在 RULES_REGISTRY 中按相同格式新增更多规则。
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import re
import json
import hashlib

# 复用评测框架内的 LLM 构造与 Key 检测（若不可用则退化为本地空结果）
try:
    from vlmeval.dataset.utils import build_judge  # type: ignore
    from vlmeval.smp import gpt_key_set  # type: ignore
except Exception:  # 允许在无依赖环境下继续工作
    build_judge = None  # type: ignore
    def gpt_key_set():  # type: ignore
        return False


# ------------------------- 轻量符号容器 -------------------------
@dataclass
class SymbolNode:
    name: str
    kind: str = "unknown"
    meta: Dict[str, Any] = field(default_factory=dict)


class SymbolGraph:
    """极简符号存储：仅承载用于 LLM 的元信息容器。"""
    def __init__(self) -> None:
        self.nodes: Dict[str, SymbolNode] = {}

    def get(self, name: str) -> SymbolNode:
        if name not in self.nodes:
            self.nodes[name] = SymbolNode(name=name)
        return self.nodes[name]


# ------------------------- 规则定义与注册 -------------------------
def rule_var_const_consistency_payload(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """将变量/常量混淆检查转成结构化规则输入。
    inputs 包含：
      - text: 完整文本(question+context+prediction)
      - symbols: 提取到的符号列表
      - formulas: 提取到的公式列表
      - sym_stats: 符号统计（出现频次、赋值值域等）
      - snippets: 每个符号的上下文片段列表
    返回给 LLM 的 JSON 对象字段仅做轻加工，避免 LLM 无谓的文本清洗开销。
    """
    return {
        "id": "var_const_consistency",
        "title": "变量/常量使用一致性检查",
        "description": (
            "判断解答中是否存在变量/常量混淆，包括但不限于："
            "(1) 常量被多次赋不同值；(2) 把变量当常量使用或反之；"
            "(3) 同一符号被重载为不同物理量导致歧义；(4) 单位或物理含义不一致。"
        ),
        "expected_output": {
            "type": "json_array",
            "schema": {
                "severity": "error|warning|info",
                "rule": "string",
                "symbol": "string|null",
                "message": "string",
                "evidence": "any"
            }
        },
        "inputs": {
            "text": inputs.get("text", ""),
            "symbols": inputs.get("symbols", []),
            "formulas": inputs.get("formulas", []),
            "sym_stats": inputs.get("sym_stats", {}),
            "snippets": inputs.get("snippets", {}),
        }
    }


RULES_REGISTRY = {
    "var_const_consistency": rule_var_const_consistency_payload,
}


# ------------------------- 主检查器实现 -------------------------
class RuleBasedVerifier:
    def __init__(self, llm_model: Optional[str] = None, max_llm_calls: int = 0, logger=None,
                 enable_cache: bool = True, llm_temperature: float = 0.2,
                 llm_max_output_tokens: int = 4096,
                 rules: Optional[List[str]] = None) -> None:
        """LLM 驱动的通用规则检查器。

        - llm_model/max_llm_calls: 与 VariableConstantVerifier 对齐；
        - rules: 要运行的规则 id 列表；若为空则默认只运行 var_const_consistency 作为样例；
        """
        self.llm_model = llm_model
        self.max_llm_calls = int(max_llm_calls)
        self.logger = logger
        self._llm = None
        self._llm_calls_used = 0
        self.llm_temperature = float(llm_temperature)
        self.llm_max_output_tokens = int(llm_max_output_tokens)
        self.rules = rules or ["var_const_consistency"]

        # 缓存
        self.enable_cache = bool(enable_cache)
        self._cache: Dict[str, Any] = {}
        self._cache_path = (Path(__file__).parent / ".cache" / "rule_based_llm_cache.json").resolve()
        if self.enable_cache:
            try:
                self._cache_path.parent.mkdir(parents=True, exist_ok=True)
                if self._cache_path.exists():
                    self._cache = json.loads(self._cache_path.read_text(encoding="utf-8"))
            except Exception:
                self._cache = {}

        # LLM 初始化
        if self.llm_model and self.max_llm_calls > 0 and build_judge and gpt_key_set():
            try:
                self._llm = build_judge(model=self.llm_model, timeout=180, retry=1,
                                        temperature=self.llm_temperature,
                                        max_output_tokens=self.llm_max_output_tokens,
                                        verbose=False)
                self._log(f"LLM 辅助已启用: {self.llm_model}")
            except Exception as e:
                self._llm = None
                self._log(f"LLM 初始化失败: {e}")

    # ------------------------- 日志/缓存/LLM 工具 -------------------------
    def _log(self, *args):
        if self.logger is not None:
            try:
                self.logger.info(" ".join(map(str, args)))
            except Exception:
                pass

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

    def _llm_available(self) -> bool:
        return self._llm is not None and self._llm_calls_used < self.max_llm_calls

    @staticmethod
    def _safe_parse_json_from_text(text: str):
        text = (text or "").strip()
        try:
            return json.loads(text)
        except Exception:
            pass
        m = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                return None
        return None

    def _llm_json(self, system_prompt: str, user_prompt: str, fallback=None) -> Any:
        if not self._llm_available():
            return fallback
        try:
            prompt = system_prompt.strip() + "\n\n" + user_prompt.strip()
            resp = self._llm.generate(prompt).strip()
            self._llm_calls_used += 1
            data = self._safe_parse_json_from_text(resp)
            if data is None:
                return fallback
            return data
        except Exception as e:
            self._log(f"LLM JSON 解析失败: {e}")
            return fallback

    # ------------------------- 轻量符号/公式提取 -------------------------
    _symbol_regex = re.compile(r"\\?([a-zA-Z][a-zA-Z0-9_]*)")
    _common_units = {
        "m","s","kg","A","K","mol","cd","N","Pa","J","W","C","V","F","Ω","ohm","S","H","Hz","T","lm","lx","Bq","Gy","Sv","Wb",
        "mm","cm","dm","km","ms","us","μs","ns","deg","rad","sr","eV","MeV","GeV","keV","u","L","ml","mL","bar","atm",
    }

    def _extract_symbols_and_formulas(self, text: str) -> Dict[str, Any]:
        symbols = []
        symbol_set = set()
        formulas = []
        lines = (text or "").splitlines()
        for li, raw in enumerate(lines):
            line = raw.strip()
            if any(op in line for op in ["=", "≈", "~", "∝"]):
                formulas.append(line)
            for m in self._symbol_regex.finditer(line):
                sym = m.group(1)
                if sym.lower() in {"final", "answer", "boxed", "ans", "unit", "units"}:
                    continue
                if sym in self._common_units:
                    continue
                if sym not in symbol_set:
                    symbol_set.add(sym)
                    symbols.append(sym)
        return {"symbols": symbols, "formulas": formulas, "lines": lines}

    def _build_symbol_contexts(self, lines: List[str], symbols: List[str], window: int = 1) -> Dict[str, List[str]]:
        """为每个符号截取上下文片段，帮助 LLM 判断语义/重载。"""
        snippets: Dict[str, List[str]] = {s: [] for s in symbols}
        for i, raw in enumerate(lines):
            line = raw.strip()
            for s in symbols:
                if re.search(rf"\b{re.escape(s)}\b", line):
                    seg = lines[max(0, i-window): min(len(lines), i+window+1)]
                    snippets[s].append(" ".join(x.strip() for x in seg))
        return snippets

    def _collect_symbol_stats(self, text: str, symbols: List[str]) -> Dict[str, Any]:
        stats: Dict[str, Any] = {}
        for s in symbols:
            assigns = re.findall(rf"\b{s}\s*=\s*([-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)", text, flags=re.IGNORECASE)
            stats[s] = {
                "frequency": len(re.findall(rf"\b{re.escape(s)}\b", text)),
                "assigned_values": sorted(set(assigns)),
            }
        return stats

    # ------------------------- 规则执行（逐条 LLM 调用） -------------------------
    def _run_rule(self, rule_id: str, inputs: Dict[str, Any]) -> List[Dict[str, Any]]:
        if rule_id not in RULES_REGISTRY:
            return []
        rule_payload = RULES_REGISTRY[rule_id](inputs)
        system = "你是物理竞赛评分助理。请仅输出 JSON。"
        user = (
            "现在你将评估一个规则：给定 inputs，请返回 diagnostics（JSON 数组）。\n"
            "每个 diagnostic 需包含 {severity, rule, symbol, message, evidence}。\n"
            "注意：\n"
            "- 严格遵循 JSON 输出，不要输出多余文本；\n"
            "- 若无问题，返回空数组 [];\n"
            "- rule 字段请使用规则 id。\n\n"
            f"rule: {json.dumps({k: v for k, v in rule_payload.items() if k != 'inputs'}, ensure_ascii=False)}\n"
            f"inputs: {json.dumps(rule_payload.get('inputs', {}), ensure_ascii=False)}"
        )
        payload = {"rule": rule_id, "inputs": inputs, "model": self.llm_model}
        cached = self._cache_get("rule", payload)
        if cached is not None:
            return cached if isinstance(cached, list) else []
        data = self._llm_json(system, user, fallback=[])
        if not isinstance(data, list):
            data = []
        # 规范化：补齐必要字段
        norm = []
        for d in data:
            if not isinstance(d, dict):
                continue
            sev = str(d.get("severity", "info")).lower()
            if sev not in {"error", "warning", "info"}:
                sev = "info"
            norm.append({
                "severity": sev,
                "rule": rule_id,
                "symbol": d.get("symbol"),
                "message": d.get("message", ""),
                "evidence": d.get("evidence"),
            })
        self._cache_set("rule", payload, norm)
        return norm

    # ------------------------- 公共 API -------------------------
    def analyze(self, sample: Dict[str, Any], dataset_key: Optional[str] = None) -> Dict[str, Any]:
        q = sample.get("question") or ""
        c = sample.get("context") or ""
        p = sample.get("prediction") or ""
        text_all = "\n".join([q, c, p])

        # 轻量符号推理上下文
        graph = SymbolGraph()
        parsed = self._extract_symbols_and_formulas(text_all)
        for s in parsed["symbols"]:
            graph.get(s)
        snippets = self._build_symbol_contexts(parsed["lines"], parsed["symbols"]) if parsed.get("lines") else {}
        sym_stats = self._collect_symbol_stats(text_all, parsed["symbols"]) if parsed.get("symbols") else {}

        # 逐条规则执行
        diagnostics: List[Dict[str, Any]] = []
        for rid in self.rules:
            inputs = {
                "text": text_all,
                "symbols": parsed["symbols"],
                "formulas": parsed["formulas"],
                "sym_stats": sym_stats,
                "snippets": snippets,
            }
            diagnostics.extend(self._run_rule(rid, inputs))

        # 去重（rule+symbol+message）
        seen = set()
        uniq: List[Dict[str, Any]] = []
        for d in diagnostics:
            key = (str(d.get("rule")), d.get("symbol"), str(d.get("message")))
            if key in seen:
                continue
            seen.add(key)
            uniq.append(d)
        diagnostics = uniq

        # 计分：error -1.0，warning -0.5，info 0
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
