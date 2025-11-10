from __future__ import annotations
import json
from typing import List, Dict, Any

from .base import RulePlugin, RuleContext, RuleRuntime


def _normalize_llm_array(data: Any) -> List[Dict[str, Any]]:
    if not isinstance(data, list):
        return []
    out: List[Dict[str, Any]] = []
    for d in data:
        if not isinstance(d, dict):
            continue
        sev = str(d.get("severity", "info")).lower()
        if sev not in {"error", "warning", "info"}:
            sev = "info"
        out.append({
            "severity": sev,
            "rule": d.get("rule"),  # 可被调用方覆盖
            "symbol": d.get("symbol"),
            "message": d.get("message", "") or d.get("detail", ""),
            "evidence": d.get("evidence"),
        })
    return out


class _BaseLLMRule:
    id: str = ""
    title: str = ""
    description: str = ""

    def _build_inputs(self, ctx: RuleContext) -> Dict[str, Any]:
        """子类可覆盖：构造传给 LLM 的 inputs。"""
        return {
            "text": ctx.text_all,
            "symbols": ctx.symbols,
            "formulas": ctx.formulas_raw,
            "formulas_structured": [
                {
                    "fid": f.fid,
                    "raw": f.raw,
                    "relation": f.relation,
                    "lhs": f.lhs,
                    "rhs": f.rhs,
                    "symbols": f.symbols,
                    "line_index": f.line_index,
                } for f in ctx.graph.formulas.values()
            ],
            "precondition_cues": ctx.precondition_cues,
            "sym_stats": ctx.sym_stats,
            "snippets": ctx.snippets,
        }

    def _system_prompt(self) -> str:
        return "你是物理竞赛评分助理。请仅输出 JSON。"

    def _user_prompt(self, inputs: Dict[str, Any]) -> str:
        rule_meta = {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "expected_output": {
                "type": "json_array",
                "schema": {
                    "severity": "error|warning|info",
                    "rule": "string",
                    "symbol": "string|null",
                    "message": "string",
                    "evidence": "any",
                },
            },
        }
        return (
            "现在你将评估一个规则：给定 inputs，请返回 diagnostics（JSON 数组）。\n"
            "每个 diagnostic 需包含 {severity, rule, symbol, message, evidence}。\n"
            "注意：\n- 严格遵循 JSON 输出，不要输出多余文本；\n- 若无问题，返回空数组 [];\n- rule 字段请使用规则 id。\n\n"
            f"rule: {json.dumps(rule_meta, ensure_ascii=False)}\n"
            f"inputs: {json.dumps(inputs, ensure_ascii=False)}"
        )

    def run(self, ctx: RuleContext, rt: RuleRuntime) -> List[Dict[str, Any]]:  # type: ignore[override]
        inputs = self._build_inputs(ctx)
        payload = {"rule": self.id, "inputs": inputs, "model": rt.llm_model}
        cached = rt.cache_get("rule", payload) if callable(rt.cache_get) else None
        if cached is not None:
            data = cached
        else:
            data = rt.llm_json(self._system_prompt(), self._user_prompt(inputs), fallback=[])
            if callable(rt.cache_set):
                rt.cache_set("rule", payload, data)
        out = _normalize_llm_array(data)
        for d in out:
            d["rule"] = self.id
        return out


class VarConstConsistencyRule(_BaseLLMRule):
    id = "var_const_consistency"
    title = "变量/常量使用一致性检查"
    description = (
        "判断解答中是否存在变量/常量混淆，包括但不限于："
        "(1) 常量被多次赋不同值；(2) 把变量当常量使用或反之；"
        "(3) 同一符号被重载为不同物理量导致歧义；(4) 单位或物理含义不一致。"
    )


class FormulaCorrectnessRule(_BaseLLMRule):
    id = "formula_correctness"
    title = "公式正确性检查"
    description = (
        "基于解析得到的公式，检查是否存在物理上不正确或适用性错误的公式，"
        "包括但不限于：量纲不一致、单位不一致、定律误用、相互矛盾的定义等。"
    )


class PreconditionConsistencyRule(_BaseLLMRule):
    id = "precondition_consistency"
    title = "前提条件一致性检查"
    description = (
        "检查作答中使用的公式/近似是否与所声明或文本隐含的前提条件一致，"
        "例如小角近似、忽略空气阻力、稳态/准静态、理想气体等。"
    )
