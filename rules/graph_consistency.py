from __future__ import annotations
from typing import List, Dict, Any

from .base import RulePlugin, RuleContext, RuleRuntime


class GraphConsistencyRule:
    id = "graph_consistency"
    title = "符号图一致性检查"
    description = (
        "使用符号节点网络进行静态一致性检查：自引用、未定义多次使用、重复且冲突的定义等。"
    )

    def run(self, ctx: RuleContext, rt: RuleRuntime) -> List[Dict[str, Any]]:  # type: ignore[override]
        diags: List[Dict[str, Any]] = []

        graph = ctx.graph

        # 1) 自引用：x = f(x)
        for f in graph.formulas.values():
            if f.relation in {"=", "≈", "~"} and f.lhs:
                # 若 LHS 的 token 也出现在 symbols（且公式含多个符号），提示警告
                import re
                m = re.search(r'([A-Za-z][A-Za-z0-9_]*)', f.lhs or "")
                if m:
                    lhs_sym = m.group(1)
                    if lhs_sym in f.symbols and len(f.symbols) > 1:
                        diags.append({
                            "severity": "warning",
                            "rule": self.id,
                            "symbol": lhs_sym,
                            "message": "公式存在自引用（左侧变量也出现在右侧），需确认是否合理。",
                            "evidence": {"formula": f.raw, "fid": f.fid},
                        })

        # 2) 未定义即使用：可能变量（短名/带下划线）被频繁使用但从未定义
        for name, s in graph.symbols.items():
            likely_var = len(name) <= 3 or ("_" in name)
            if not likely_var:
                continue
            if not s.defined_by and len(s.used_in) >= 2 and len(s.occurrences) >= 2:
                diags.append({
                    "severity": "info",
                    "rule": self.id,
                    "symbol": name,
                    "message": "符号被多次使用但未在等式左侧明确定义，可能需要给出定义或说明。",
                    "evidence": {"used_in": s.used_in[:5], "occurrences": s.occurrences[:3]},
                })

        # 3) 重复且冲突的定义：同一符号在多条等式中拥有不同 RHS 表达
        rhs_map: Dict[str, set] = {}
        import re
        for f in graph.formulas.values():
            if f.lhs and f.relation in {"=", "≈", "~"} and f.rhs:
                m = re.search(r'([A-Za-z][A-Za-z0-9_]*)', f.lhs or "")
                lhs_key = m.group(1) if m else (f.lhs or f.fid)
                rhs_map.setdefault(lhs_key, set()).add(f.rhs)
        for name, rhs_set in rhs_map.items():
            if len(rhs_set) >= 2:
                diags.append({
                    "severity": "warning",
                    "rule": self.id,
                    "symbol": name,
                    "message": "同一符号在文本中由不同表达式定义，可能存在冲突或重载。",
                    "evidence": {"rhs_variants": sorted(rhs_set)[:5]},
                })

        return diags
