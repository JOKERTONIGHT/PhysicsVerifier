"""
优化后的物理规则检查器（LLM 驱动 + 符号节点网络）
包含对你原始实现的多项改进：
- 更鲁棒的公式解析（统一 relation 抽取、边界判断）
- 更精确的符号匹配（避免 \b 导致的误判）
- 改进的定义检测（lhs 中提取符号视为定义）
- 更健壮的公式行识别（避免把描述性文本当公式）
- LLM 调用与 JSON 解析更坚固的回退逻辑
- 缓存写入采取原子写入（temp -> rename），避免部分损坏
- 若外部 build_judge 不可用，仍能在本地运行并返回合理结构
- 若需要可继续加入文件锁等机制（此处使用原子写法以减少依赖）
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set
from pathlib import Path
import re
import json
import hashlib
import os
import tempfile
import datetime

# 复用评测框架内的 LLM 构造与 Key 检测（若不可用则退化为本地空结果）
try:
    from vlmeval.dataset.utils import build_judge  # type: ignore
    from vlmeval.smp import gpt_key_set  # type: ignore
except Exception:  # 允许在无依赖环境下继续工作
    build_judge = None  # type: ignore
    def gpt_key_set():  # type: ignore
        return False


# ------------------------- 符号节点网络 -------------------------
@dataclass
class SymbolNode:
    name: str
    kind: str = "unknown"  # variable|constant|parameter|unknown
    occurrences: List[Dict[str, Any]] = field(default_factory=list)  # {line, context}
    defined_by: List[str] = field(default_factory=list)  # formula ids
    used_in: List[str] = field(default_factory=list)     # formula ids
    meta: Dict[str, Any] = field(default_factory=dict)   # {unit, assigned_values, llm: {...}}


@dataclass
class FormulaNode:
    fid: str
    raw: str
    relation: str  # '=', '≈', '~', '∝', 'unknown'
    lhs: Optional[str]
    rhs: Optional[str]
    symbols: List[str]
    line_index: int


class SymbolGraph:
    """轻量符号图：包含符号与公式节点及依赖关系。"""
    def __init__(self) -> None:
        self.symbols: Dict[str, SymbolNode] = {}
        self.formulas: Dict[str, FormulaNode] = {}

    def sym(self, name: str) -> SymbolNode:
        if name not in self.symbols:
            self.symbols[name] = SymbolNode(name=name)
        return self.symbols[name]

    def add_occurrence(self, name: str, line: int, context: str):
        node = self.sym(name)
        node.occurrences.append({"line": line, "context": context})

    def add_formula(self, fid: str, node: FormulaNode):
        self.formulas[fid] = node
        # link symbols
        for s in node.symbols:
            self.sym(s).used_in.append(fid)
        # 增强定义检测：尝试从 lhs 中抽取第一个符号作为定义
        if node.relation in {"=", "≈", "~"} and node.lhs:
            # 从 lhs 抽取第一个可能的符号 token
            m = re.search(r'([A-Za-z][A-Za-z0-9_]*)', node.lhs)
            if m:
                lhs_sym = m.group(1)
                # 只在该符号在公式符号列表中时认为是定义
                if lhs_sym in node.symbols:
                    self.sym(lhs_sym).defined_by.append(fid)


# ------------------------- 规则定义与注册 -------------------------
def rule_var_const_consistency_payload(inputs: Dict[str, Any]) -> Dict[str, Any]:
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


def rule_formula_correctness_payload(inputs: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": "formula_correctness",
        "title": "公式正确性检查",
        "description": (
            "基于给定的解析公式，检查是否存在物理上不正确或适用性错误的公式，"
            "包括但不限于：量纲不一致、单位不一致、定律误用、同一变量被相互矛盾的关系定义。"
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
            "formulas_structured": inputs.get("formulas_structured", []),
            "symbols": inputs.get("symbols", []),
            "sym_stats": inputs.get("sym_stats", {}),
            "snippets": inputs.get("snippets", {}),
        }
    }


def rule_precondition_consistency_payload(inputs: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": "precondition_consistency",
        "title": "前提条件一致性检查",
        "description": (
            "检查作答中使用的公式/近似是否与所声明或文本隐含的前提条件一致，"
            "例如小角近似、忽略空气阻力、稳态/准静态、理想气体等。"
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
            "formulas_structured": inputs.get("formulas_structured", []),
            "precondition_cues": inputs.get("precondition_cues", []),
            "symbols": inputs.get("symbols", []),
            "snippets": inputs.get("snippets", {}),
        }
    }


RULES_REGISTRY = {
    "var_const_consistency": rule_var_const_consistency_payload,
    "formula_correctness": rule_formula_correctness_payload,
    "precondition_consistency": rule_precondition_consistency_payload,
}


# ------------------------- 主检查器实现 -------------------------
class RuleBasedVerifier:
    def __init__(self, llm_model: Optional[str] = None, max_llm_calls: int = 0, logger=None,
                 enable_cache: bool = True, llm_temperature: float = 0.2,
                 llm_max_output_tokens: int = 4096,
                 rules: Optional[List[str]] = None) -> None:
        """LLM 驱动的通用规则检查器。"""
        self.llm_model = llm_model
        self.max_llm_calls = int(max_llm_calls)
        self.logger = logger
        self._llm = None
        self._llm_calls_used = 0
        self.llm_temperature = float(llm_temperature)
        self.llm_max_output_tokens = int(llm_max_output_tokens)
        # 默认启用三类规则
        self.rules = rules or ["var_const_consistency", "formula_correctness", "precondition_consistency"]

        # 缓存
        self.enable_cache = bool(enable_cache)
        self._cache: Dict[str, Any] = {}
        # 放在脚本同目录下的 .cache 子目录
        try:
            base_dir = Path(__file__).parent
        except Exception:
            base_dir = Path(".").resolve()
        self._cache_path = (base_dir / ".cache" / "rule_based_llm_cache.json").resolve()
        if self.enable_cache:
            try:
                self._cache_path.parent.mkdir(parents=True, exist_ok=True)
                if self._cache_path.exists():
                    # 仅在文件可读且合法 JSON 时加载
                    try:
                        self._cache = json.loads(self._cache_path.read_text(encoding="utf-8"))
                    except Exception:
                        self._cache = {}
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
        # 原子写入缓存文件（写入临时文件然后替换）
        try:
            tmp_fd, tmp_path = tempfile.mkstemp(dir=str(self._cache_path.parent))
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, ensure_ascii=False, indent=None)
            os.replace(tmp_path, str(self._cache_path))
        except Exception:
            # 忽略写失败
            pass

    def _llm_available(self) -> bool:
        return self._llm is not None and self._llm_calls_used < self.max_llm_calls

    @staticmethod
    def _safe_parse_json_from_text(text: str):
        text = (text or "").strip()
        if not text:
            return None
        # 先尝试直接解析
        try:
            return json.loads(text)
        except Exception:
            pass
        # 尝试在文本里提取 JSON 结构（[] 或 {}）
        m = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                return None
        return None

    def _llm_json(self, system_prompt: str, user_prompt: str, fallback=None) -> Any:
        if fallback is None:
            fallback = []
        if not self._llm_available():
            return fallback
        try:
            prompt = system_prompt.strip() + "\n\n" + user_prompt.strip()
            # 不同 judge 接口可能返回不同类型，尽量兼容
            resp_obj = self._llm.generate(prompt)
            # 若返回对象带 text 属性
            if hasattr(resp_obj, "text"):
                resp = resp_obj.text.strip()
            else:
                # 可能直接返回字符串
                resp = str(resp_obj).strip()
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
    # 常见 LaTeX 控制词与英文停用词，避免被当作符号
    _latex_words: Set[str] = {
        "frac","dfrac","tfrac","cfrac","sqrt","text","mathrm","mathbf","mathit","mathbb","operatorname",
        "cdot","times","left","right","begin","end","over","hat","bar","dot","ddot","vec","cal",
        "partial","nabla","sum","int","lim","log","ln","sin","cos","tan","arcsin","arccos","arctan",
        "sinh","cosh","tanh","exp","min","max","arg","det","Tr","trace","sgn","deg","infty","infin",
    }
    _stop_words: Set[str] = {
        # English common
        "the","of","to","in","for","and","is","we","this","that","with","on","by","as","be","are","or","an","at","from","it",
        "our","can","will","not","have","has","but","if","then","else","than","which","these","those","their","its","into","using","use",
        "problem","solution","thus","hence","therefore","let","consider","assume","since","because","also","so","such","where","when","while",
        "between","within","about","approx","approximate","approximately","given","known","unknown","constant","planck","mass","electron","proton",
    }

    # 少量常见物理常数/变量白名单（可扩充）
    _common_physics_names = {"g", "c", "h", "k", "G", "m", "M", "q", "e", "k_B", "R", "T", "P", "V", "E", "U", "F", "v", "u", "t"}

    def _looks_like_symbol(self, tok: str) -> bool:
        """启发式过滤：保留更可能是物理符号的 token。"""
        if not tok:
            return False
        low = tok.lower()
        if low in self._latex_words or low in self._stop_words:
            return False
        if tok in self._common_units:
            return False
        # 白名单保留
        if tok in self._common_physics_names:
            return True
        # 全小写且长度 >=4 的英文单词，保守排除
        if re.fullmatch(r"[a-z]+", tok) and len(tok) >= 4:
            return False
        # 带下划线或数字通常是变量名称
        if ("_" in tok) or re.search(r"\d", tok):
            return True
        # 短 token（长度<=3）通常可能是符号
        if len(tok) <= 3:
            return True
        # 其余：保守过滤
        return False

    def _is_likely_formula_line(self, line: str) -> bool:
        """更严格判断一行是否为公式表达式，避免将描述性句子误判为公式。"""
        if not line or len(line) > 500:
            return False
        # 必须包含关系符并且左右至少有一个字母/数字/括号
        if not re.search(r'([A-Za-z0-9_\)\]\}]\s*(=|≈|∝|~)\s*[-+A-Za-z0-9_\(\[\{])', line):
            return False
        # 排除纯自然语言句子（小写单词过多）——启发式
        tokens = re.findall(r"[A-Za-z]+", line)
        if tokens and sum(1 for t in tokens if len(t) >= 4) > max(3, len(tokens) // 2):
            return False
        return True

    def _extract_symbols_and_formulas(self, text: str) -> Dict[str, Any]:
        symbols = []
        symbol_set = set()
        formulas = []
        # 先统一换行与空行处理
        lines = (text or "").splitlines()
        for li, raw in enumerate(lines):
            line = raw.strip()
            # 更严格地认为是公式才加入 formulas
            if self._is_likely_formula_line(line):
                formulas.append(line)
            for m in self._symbol_regex.finditer(line):
                sym = m.group(1)
                if sym.lower() in {"final", "answer", "boxed", "ans", "unit", "units"}:
                    continue
                if not self._looks_like_symbol(sym):
                    continue
                if sym not in symbol_set:
                    symbol_set.add(sym)
                    symbols.append(sym)
        return {"symbols": symbols, "formulas": formulas, "lines": lines}

    def _parse_formula_line(self, raw: str, line_idx: int) -> FormulaNode:
        raw = raw.strip()
        # 找关系符（取第一个出现的）
        m = re.search(r'(=|≈|∝|~)', raw)
        if m:
            relation = m.group(1)
            parts = [p.strip() for p in raw.split(relation, 1)]
            lhs = parts[0] if len(parts) == 2 else None
            rhs = parts[1] if len(parts) == 2 else None
        else:
            relation, lhs, rhs = "unknown", None, raw

        # 抽取符号（忽略纯数字）
        syms: List[str] = []
        seen: Set[str] = set()
        for t in self._symbol_regex.findall(raw):
            if re.fullmatch(r"[-+]?\d+(\.\d+)?([eE][-+]?\d+)?", t):
                continue
            if not self._looks_like_symbol(t):
                continue
            if t not in seen:
                seen.add(t)
                syms.append(t)

        fid = f"F{line_idx:04d}"
        return FormulaNode(fid=fid, raw=raw, relation=relation, lhs=(lhs or None), rhs=(rhs or None), symbols=syms, line_index=line_idx)

    def _build_symbol_graph(self, lines: List[str], symbols: List[str], formulas: List[str]) -> SymbolGraph:
        graph = SymbolGraph()
        # 符号出现上下文（使用更精确的 token 边界匹配）
        for i, raw in enumerate(lines):
            for s in symbols:
                pattern = re.compile(rf'(?<![A-Za-z0-9_]){re.escape(s)}(?![A-Za-z0-9_])')
                if pattern.search(raw):
                    seg = lines[max(0, i-1): min(len(lines), i+2)]
                    graph.add_occurrence(s, i, " ".join(x.strip() for x in seg))
        # 解析公式并接入
        for i, f_raw in enumerate(formulas):
            fn = self._parse_formula_line(f_raw, i)
            graph.add_formula(fn.fid, fn)
        return graph

    def _build_symbol_contexts(self, lines: List[str], symbols: List[str], window: int = 1) -> Dict[str, List[str]]:
        """为每个符号截取上下文片段，帮助 LLM 判断语义/重载。"""
        snippets: Dict[str, List[str]] = {s: [] for s in symbols}
        for i, raw in enumerate(lines):
            line = raw.strip()
            for s in symbols:
                pattern = re.compile(rf'(?<![A-Za-z0-9_]){re.escape(s)}(?![A-Za-z0-9_])')
                if pattern.search(line):
                    seg = lines[max(0, i-window): min(len(lines), i+window+1)]
                    snippets[s].append(" ".join(x.strip() for x in seg))
        return snippets

    def _collect_symbol_stats(self, text: str, symbols: List[str]) -> Dict[str, Any]:
        stats: Dict[str, Any] = {}
        for s in symbols:
            # 捕获常见数字格式（整数、浮点、科学计数）
            assigns = re.findall(rf"\b{re.escape(s)}\s*=\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", text, flags=re.IGNORECASE)
            stats[s] = {
                "frequency": len(re.findall(rf"(?<![A-Za-z0-9_]){re.escape(s)}(?![A-Za-z0-9_])", text)),
                "assigned_values": sorted(set(assigns)),
            }
        return stats

    def _extract_precondition_cues(self, text: str) -> List[str]:
        cues = []
        # 中英文常见前提/近似关键词（可继续扩充）
        keywords = [
            # 中文
            "小角近似", "忽略空气阻力", "忽略摩擦", "理想气体", "稳态", "准静态", "绝热", "等温", "非相对论", "低速近似", "小振幅",
            # 英文
            "small angle", "neglect air resistance", "ignore friction", "ideal gas", "steady state", "quasi-static",
            "adiabatic", "isothermal", "non-relativistic", "low speed", "small amplitude",
        ]
        low = text.lower()
        for k in keywords:
            if k in text or k in low:
                cues.append(k)
        return sorted(set(cues))

    # ------------------------- 内置：图一致性检查 -------------------------
    def _builtin_graph_checks(self, graph: SymbolGraph) -> List[Dict[str, Any]]:
        diags: List[Dict[str, Any]] = []

        # 1) 自引用：x = f(x)（在等号/近似关系下）
        for f in graph.formulas.values():
            if f.relation in {"=", "≈", "~"} and f.lhs:
                # 如果 LHS 的符号也出现在同一表达的 symbols 中，且该公式含多于 1 个符号，警告
                m = re.search(r'([A-Za-z][A-Za-z0-9_]*)', f.lhs or "")
                if m:
                    lhs_sym = m.group(1)
                    if lhs_sym in f.symbols and len(f.symbols) > 1:
                        diags.append({
                            "severity": "warning",
                            "rule": "graph_consistency",
                            "symbol": lhs_sym,
                            "message": "公式出现在自引用形式（左边变量也出现在右边），需确认是否合理。",
                            "evidence": {"formula": f.raw, "fid": f.fid}
                        })

        # 2) 未定义即使用：被频繁使用但没有任何定义/赋值痕迹
        for name, s in graph.symbols.items():
            likely_var = len(name) <= 3 or ("_" in name)
            if not likely_var:
                continue
            if not s.defined_by and len(s.used_in) >= 2 and len(s.occurrences) >= 2:
                diags.append({
                    "severity": "info",
                    "rule": "graph_consistency",
                    "symbol": name,
                    "message": "符号被多次使用但未在等式左边被明确定义，可能需要给出定义或说明。",
                    "evidence": {"used_in": s.used_in[:5], "occurrences": s.occurrences[:3]}
                })

        # 3) 重复且冲突的定义：同一符号多次被不同公式定义（启发式：不同 RHS 文本）
        rhs_map: Dict[str, Set[str]] = {}
        for f in graph.formulas.values():
            if f.lhs and f.relation in {"=", "≈", "~"} and f.rhs:
                # 记录原始 rhs 文本的启发式签名
                rhs_map.setdefault(re.search(r'([A-Za-z][A-Za-z0-9_]*)', f.lhs).group(1) if re.search(r'([A-Za-z][A-Za-z0-9_]*)', f.lhs or "") else f.lhs or f.fid, set()).add(f.rhs)
        for name, rhs_set in rhs_map.items():
            if len(rhs_set) >= 2:
                diags.append({
                    "severity": "warning",
                    "rule": "graph_consistency",
                    "symbol": name,
                    "message": "同一符号在文本中由不同表达式定义，可能存在冲突或重载。",
                    "evidence": {"rhs_variants": list(sorted(rhs_set))[:5]}
                })

        return diags

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
        # 强制保证输出为 list
        if not isinstance(data, list):
            data = []

        # 规范化：补齐必要字段并修正 severity
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
                "message": d.get("message", "") or d.get("detail", ""),
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

        # 轻量符号推理上下文 + 符号节点网络
        parsed = self._extract_symbols_and_formulas(text_all)
        graph = self._build_symbol_graph(parsed.get("lines", []), parsed["symbols"], parsed["formulas"])

        snippets = self._build_symbol_contexts(parsed["lines"], parsed["symbols"]) if parsed.get("lines") else {}
        sym_stats = self._collect_symbol_stats(text_all, parsed["symbols"]) if parsed.get("symbols") else {}
        precondition_cues = self._extract_precondition_cues(text_all)

        # 内置图一致性检查
        diagnostics: List[Dict[str, Any]] = []
        diagnostics.extend(self._builtin_graph_checks(graph))

        # 逐条规则执行
        for rid in self.rules:
            inputs = {
                "text": text_all,
                "symbols": parsed["symbols"],
                "formulas": parsed["formulas"],
                # 提供结构化公式与前提 cues
                "formulas_structured": [
                    {
                        "fid": f.fid,
                        "raw": f.raw,
                        "relation": f.relation,
                        "lhs": f.lhs,
                        "rhs": f.rhs,
                        "symbols": f.symbols,
                        "line_index": f.line_index,
                    } for f in graph.formulas.values()
                ],
                "precondition_cues": precondition_cues,
                "sym_stats": sym_stats,
                "snippets": snippets,
            }
            try:
                diagnostics.extend(self._run_rule(rid, inputs) or [])
            except Exception as e:
                self._log(f"规则 {rid} 执行出错: {e}")

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

        # 输出序列化 symbol_nodes / formula_nodes（使用 vars）
        symbol_nodes = {k: vars(v) for k, v in graph.symbols.items()}
        formula_nodes = {k: vars(v) for k, v in graph.formulas.items()}

        return {
            "id": sample.get("id"),
            "dataset": dataset_key,
            "diagnostics": diagnostics,
            "symbol_nodes": symbol_nodes,
            "formula_nodes": formula_nodes,
            "precondition_cues": precondition_cues,
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
            "created_at": datetime.datetime.now().isoformat(),
        }
        return {"summary": summary, "results": results}


# ------------------------- 若作为脚本运行时的简单示例 -------------------------
if __name__ == "__main__":
    # 简单示例，演示 analyzer 在无 LLM 环境下也能运行
    rv = RuleBasedVerifier(llm_model=None, max_llm_calls=0, enable_cache=True)
    sample = {
        "id": "demo1",
        "question": "A block slides with initial speed v. Show that v^2 = u^2 + 2as.",
        "context": "Assume constant acceleration a. Let u be initial speed.",
        "prediction": "Using v = u + at and s = ut + 1/2 at^2, we get v^2 = u^2 + 2as. Also assume v = v0."
    }
    out = rv.analyze(sample)
    print(json.dumps(out, ensure_ascii=False, indent=2))
