from __future__ import annotations
import json
from typing import List, Dict, Any

from .base import RuleContext, RuleRuntime


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

    def run(self, ctx: RuleContext, rt: RuleRuntime) -> List[Dict[str, Any]]:
        # This is a placeholder now. The main logic is in SemanticRuleChecker,
        # which uses the rule's SRD translation to perform checks.
        # This plugin-based execution can be kept for non-LLM/hybrid rules
        # or completely removed if all rules become LLM-driven via SRDs.
        rt.log(f"Skipping legacy execution for LLM-driven rule: {self.id}")
        return []


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


class DimensionalHomogeneityRule(_BaseLLMRule):
    id = "dimensional_homogeneity"
    title = "量纲齐次性检查"
    description = (
        "检查关键等式是否量纲齐次：等式左侧和右侧的物理量在 M、L、T 等基本量纲上的指数必须完全一致。"
        "典型错误包括将时间与长度、速度与力、能量与功率等不同量纲的量直接相等或相加。"
    )


class SmallAngleApproxRule(_BaseLLMRule):
    id = "small_angle_approx"
    title = "小角近似适用性检查"
    description = (
        "检查解答中使用 sinθ≈θ、cosθ≈1 等小角近似时，是否明确说明 θ 远小于 1 rad，"
        "并验证该近似在题目给出的角度范围内合理。若角度较大却仍直接套用小角近似，应视为违规。"
    )


class EnergyConservationContextRule(_BaseLLMRule):
    id = "energy_conservation_context"
    title = "能量守恒适用场景检查"
    description = (
        "当解答使用机械能守恒或总能量守恒时，检查是否忽略了明显存在的非保守力做功（如摩擦或外部驱动力）。"
        "若情境中存在显著非保守作用却仍直接写出能量守恒而无额外说明，应判为错误。"
    )


class MomentumConservationContextRule(_BaseLLMRule):
    id = "momentum_conservation_context"
    title = "动量守恒适用场景检查"
    description = (
        "当解答使用线动量守恒时，需确认“系统所受外力合为零”或外力只能成对抵消。"
        "若在显然存在净外力（如持续推力、地面支持力不对消）的情况下仍直接写出 p_initial = p_final，应视为误用动量守恒。"
    )


class GivenDataUseRule(_BaseLLMRule):
    id = "given_data_use"
    title = "已知量使用与伪造数据检查"
    description = (
        "检查解答是否合理使用题目给出的数值与物理量，"
        "包括：对关键给定量完全忽略不使用，或在题目未提供的情况下随意伪造并代入具体数值。"
    )


class NonEmptySolutionRule(_BaseLLMRule):
    id = "non_empty_solution"
    title = "解答内容非空检查"
    description = (
        "检查学生给出的解答是否本质上为空白或只包含与物理无关的噪声文本。"
        "如果没有任何与物理推导、公式、数值计算或明确结论相关的内容，应判定为“空解答”。"
    )


class OrderOfMagnitudeRule(_BaseLLMRule):
    id = "order_of_magnitude"
    title = "数量级合理性检查"
    description = (
        "检查最终答案的数量级是否与题目背景的典型物理量一致。"
        "例如人体运动速度通常在 10 m/s 量级、地球轨道速度在 10^4 m/s 量级，若出现明显不合理的量级应报警。"
    )


class SafeDivideRule(_BaseLLMRule):
    id = "safe_divide"
    title = "除零/极小量防护检查"
    description = (
        "当推导中出现除以某个变量（如速度、sinθ、差值等）时，"
        "需要说明该量不为零或给出其保持为非零/非奇异的理由。若直接除以可能为零的量，应判为违规。"
    )


class FunctionDomainRule(_BaseLLMRule):
    id = "function_domain_guard"
    title = "函数定义域检查"
    description = (
        "检查对 log、sqrt、arcsin 等函数的使用，其输入是否落在合法定义域内。"
        "例如 log(负数)、sqrt(负数而未说明复数)、arcsin(|x|>1) 等都应判定为错误。"
    )

