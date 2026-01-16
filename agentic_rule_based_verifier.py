"""Agentic variant of the physics rules checker.

This module keeps the same public interface as `rule_based_verifier.RuleBasedVerifier`
(`analyze`, `analyze_batch`, constructor args) but introduces an agent-style
loop where the LLM can actively request tooling such as building symbol graphs
or querying precise snippets before finalising diagnostics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import json

from rule_based_verifier import RuleBasedVerifier
from rules.llm_rules import _normalize_llm_array


@dataclass
class AgentToolResult:
    """Container for tool call side-effects and renderable summaries."""
    action: str
    arguments: Dict[str, Any]
    output: str


class AgenticRuleBasedVerifier(RuleBasedVerifier):
    """Agent-driven checker that lets the LLM pull auxiliary symbol context on demand."""

    def __init__(
        self,
        *args,
        agent_max_turns: int = 5,
        agent_force_symbol_graph: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.agent_max_turns = int(agent_max_turns)
        self.agent_force_symbol_graph = bool(agent_force_symbol_graph)

    # ---------------- Agent loop helpers -----------------
    def _agent_system_prompt(self) -> str:
        return (
            "You are an expert physics rule checker acting as a tool-using agent. "
            "At each turn you may CHOOSE one of the allowed actions below. "
            "When you have enough evidence with high confidence, choose 'submit_diagnostics' "
            "with the final JSON diagnostic array (same schema as the legacy checker)."
        )

    def _available_tools_description(self) -> str:
        return """
Available actions:
1. build_symbol_graph {}
   - Build (or rebuild) the symbol graph for the student's text. Returns counts and the first few symbols/formulas.
2. describe_symbol {"symbol": "<name>"}
   - Requires a graph. Gives detailed occurrences and formulas mentioning the symbol.
3. inspect_formula {"fid": "F0001"}
   - Requires a graph. Returns the raw equation and the symbols involved.
4. quote_text {"line_start": int, "line_end": int}
   - Returns the raw lines (1-indexed) from the student text to cite as evidence.
5. submit_diagnostics {"diagnostics": [ ... ]}
   - Finish the analysis. Diagnostics must follow the usual schema.
6. stop {}
   - Abort if you believe the rule does not trigger.
Always respond with a JSON object: {"action": ..., "arguments": {...}, "commentary": "optional thoughts"}.
"""

    def _format_agent_user_prompt(
        self,
        sample: Dict[str, Any],
        rule_id: str,
        srd: str,
        text_all: str,
        tool_log: List[AgentToolResult],
    ) -> str:
        log_lines = [f"Turn {idx+1}: {log.action} -> {log.output}" for idx, log in enumerate(tool_log)]
        rendered_log = "\n".join(log_lines) if log_lines else "(no tools used yet)"
        return f"""
Rule to enforce: {rule_id}
SRD / Description:
{srd}

Student submission (question + answer):
---
{text_all}
---

Tool log so far:
{rendered_log}

{self._available_tools_description()}
"""

    def _ensure_parsed(self, cache: Dict[str, Any], text_all: str) -> Dict[str, Any]:
        if "parsed" not in cache:
            cache["parsed"] = self._extract_symbols_and_formulas(text_all)
        return cache["parsed"]

    def _ensure_graph(self, cache: Dict[str, Any], parsed: Dict[str, Any]) -> Any:
        if "graph" not in cache:
            graph = self._build_symbol_graph(parsed["lines"], parsed["symbols"], parsed["formulas"])
            cache["graph"] = graph
        return cache["graph"]

    def _tool_build_symbol_graph(self, cache: Dict[str, Any], text_all: str) -> AgentToolResult:
        parsed = self._ensure_parsed(cache, text_all)
        graph = self._ensure_graph(cache, parsed)
        symbol_preview = list(graph.symbols.keys())[:8]
        formula_preview = [f.raw for f in list(graph.formulas.values())[:5]]
        summary = {
            "total_symbols": len(graph.symbols),
            "symbol_preview": symbol_preview,
            "total_formulas": len(graph.formulas),
            "formula_preview": formula_preview,
        }
        return AgentToolResult("build_symbol_graph", {}, json.dumps(summary, ensure_ascii=False))

    def _tool_describe_symbol(self, cache: Dict[str, Any], text_all: str, symbol: str) -> AgentToolResult:
        parsed = self._ensure_parsed(cache, text_all)
        graph = self._ensure_graph(cache, parsed)
        node = graph.symbols.get(symbol)
        if not node:
            payload = {"error": f"Symbol '{symbol}' not found."}
        else:
            payload = {
                "occurrences": node.occurrences[:10],
                "defined_by": node.defined_by[:5],
                "used_in": node.used_in[:5],
            }
        return AgentToolResult("describe_symbol", {"symbol": symbol}, json.dumps(payload, ensure_ascii=False))

    def _tool_inspect_formula(self, cache: Dict[str, Any], text_all: str, fid: str) -> AgentToolResult:
        parsed = self._ensure_parsed(cache, text_all)
        graph = self._ensure_graph(cache, parsed)
        formula = graph.formulas.get(fid)
        if not formula:
            payload = {"error": f"Formula '{fid}' not found."}
        else:
            payload = {
                "fid": formula.fid,
                "raw": formula.raw,
                "relation": formula.relation,
                "lhs": formula.lhs,
                "rhs": formula.rhs,
                "symbols": formula.symbols,
            }
        return AgentToolResult("inspect_formula", {"fid": fid}, json.dumps(payload, ensure_ascii=False))

    def _tool_quote_text(
        self,
        cache: Dict[str, Any],
        text_all: str,
        line_start: int,
        line_end: int,
    ) -> AgentToolResult:
        parsed = self._ensure_parsed(cache, text_all)
        lines = parsed["lines"]
        s = max(1, line_start)
        e = min(len(lines), line_end)
        snippet = "\n".join(f"{idx+1}: {lines[idx]}" for idx in range(s - 1, e))
        return AgentToolResult(
            "quote_text",
            {"line_start": s, "line_end": e},
            snippet or "(empty snippet)",
        )

    def _dispatch_tool(
        self,
        action: str,
        arguments: Dict[str, Any],
        cache: Dict[str, Any],
        text_all: str,
    ) -> AgentToolResult:
        if action == "build_symbol_graph":
            return self._tool_build_symbol_graph(cache, text_all)
        if action == "describe_symbol":
            return self._tool_describe_symbol(cache, text_all, arguments.get("symbol", ""))
        if action == "inspect_formula":
            return self._tool_inspect_formula(cache, text_all, arguments.get("fid", ""))
        if action == "quote_text":
            start = int(arguments.get("line_start", 1))
            end = int(arguments.get("line_end", start))
            return self._tool_quote_text(cache, text_all, start, end)
        return AgentToolResult(action, arguments, "Unsupported tool")

    def _run_agent_for_rule(
        self,
        sample: Dict[str, Any],
        rule_id: str,
        srd: str,
        text_all: str,
    ) -> List[Dict[str, Any]]:
        if not self._llm_available():
            return []

        cache: Dict[str, Any] = {}
        tool_log: List[AgentToolResult] = []
        diagnostics: List[Dict[str, Any]] = []

        for turn in range(self.agent_max_turns):
            user_prompt = self._format_agent_user_prompt(sample, rule_id, srd, text_all, tool_log)
            response = self._llm_json(self._agent_system_prompt(), user_prompt, fallback={})
            if not isinstance(response, dict):
                break
            action = response.get("action", "stop")
            arguments = response.get("arguments", {}) or {}

            if action == "submit_diagnostics":
                raw_diags = response.get("diagnostics", [])
                diagnostics = _normalize_llm_array(raw_diags)
                break
            if action == "stop":
                diagnostics = []
                break
            tool_result = self._dispatch_tool(action, arguments, cache, text_all)
            tool_log.append(tool_result)

        return diagnostics

    # ---------------- Public API overrides -----------------
    def analyze(self, sample: Dict[str, Any], dataset_key: Optional[str] = None, export_graph: bool = False) -> Dict[str, Any]:
        text_all = "\n".join([
            sample.get("question", ""),
            sample.get("context", ""),
            sample.get("prediction", ""),
        ])

        answer_correct = self._answer_matches(sample)
        all_diagnostics: List[Dict[str, Any]] = []

        if (not answer_correct) and self.rule_translations:
            for rule_id in self.rules_to_check:
                rule_info = self.rule_translations.get(rule_id)
                if not rule_info:
                    continue
                srd = rule_info.get("srd") or rule_info.get("description")
                if not srd:
                    continue
                diags = self._run_agent_for_rule(sample, rule_id, srd, text_all)
                all_diagnostics.extend(diags)

        seen = set()
        unique_diagnostics: List[Dict[str, Any]] = []
        for d in all_diagnostics:
            key = (d.get("rule"), d.get("symbol"), d.get("message"))
            if key in seen:
                continue
            unique_diagnostics.append(d)
            seen.add(key)

        score = sum(
            -1.0 if d.get("severity") == "error" else -0.5
            for d in unique_diagnostics
            if d.get("severity") in {"error", "warning"}
        )

        return {
            "id": sample.get("id"),
            "dataset": dataset_key,
            "diagnostics": unique_diagnostics,
            "score": score,
            "answer_correct": answer_correct,
        }

    def analyze_batch(self, samples: List[Dict[str, Any]], dataset_key: Optional[str] = None, export_graph: bool = False) -> Dict[str, Any]:
        results = [self.analyze(sample, dataset_key=dataset_key, export_graph=export_graph) for sample in samples or []]
        total_score = sum(result.get("score", 0.0) for result in results)
        return {
            "summary": {
                "dataset": dataset_key,
                "num_samples": len(results),
                "total_score": total_score,
                "avg_score": (total_score / len(results)) if results else 0.0,
            },
            "results": results,
        }
