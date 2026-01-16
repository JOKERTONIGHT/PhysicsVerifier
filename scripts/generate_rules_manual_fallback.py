import json
from pathlib import Path

def main():
    output_path = Path("PhysicsVerifier/results/generated_rules_from_errors.json")
    
    # Manually curated rules based on the errors in pure_llm_eval_results_100.json
    # since the API quota is exhausted.
    rules = [
        {
            "id": "rule_seismic_geometry_01",
            "title": "地震波传播路径几何检查",
            "description": "检查在涉及震源深度的地震波传播问题中，是否正确使用了斜距（直角三角形斜边）而非仅使用震中距（地面水平距离）。",
            "srd": "FOR EACH calculation: IF context.is_seismic_wave AND context.has_depth AND context.has_epicentral_distance THEN CHECK formula.distance == sqrt(depth^2 + epicentral_distance^2); REPORT violation 'seismic_geometry_error' IF formula.distance == epicentral_distance.",
            "source_error_id": "6557",
            "source_error_type": "modeling"
        },
        {
            "id": "rule_forced_vibration_01",
            "title": "受迫振动频率一致性检查",
            "description": "检查受迫振动问题中，系统的振荡频率是否被设定为驱动力的频率，而不是系统的固有频率（除非明确提及共振）。",
            "srd": "FOR EACH oscillation_system: IF system.is_driven AND system.driving_frequency IS defined THEN CHECK system.oscillation_frequency == system.driving_frequency; REPORT violation 'forced_vibration_frequency_mismatch' IF system.oscillation_frequency == system.natural_frequency AND NOT context.is_resonance.",
            "source_error_id": "72098",
            "source_error_type": "concept"
        },
        {
            "id": "rule_circuit_time_varying_01",
            "title": "时变电路电压一致性检查",
            "description": "在含电感或电容的电路中，若电流随时间变化（如线性增长），检查施加电压是否被错误假设为恒定值（DC）。",
            "srd": "FOR EACH circuit_node: IF current.is_time_varying AND component.is_inductor THEN CHECK voltage.is_time_varying; REPORT violation 'constant_voltage_assumption_invalid' IF voltage.is_constant.",
            "source_error_id": "36580",
            "source_error_type": "concept"
        },
        {
            "id": "rule_gradient_sign_01",
            "title": "函数单调性与导数符号检查",
            "description": "检查对物理量随坐标变化的描述（如“随高度增加”）是否与数学导数的符号一致（增加对应导数大于零）。",
            "srd": "FOR EACH function f(z): IF text.states 'f increases with z' THEN CHECK df/dz > 0; REPORT violation 'gradient_sign_error' IF df/dz < 0.",
            "source_error_id": "24562",
            "source_error_type": "logic"
        },
        {
            "id": "rule_transmission_resistance_01",
            "title": "输电线路电阻解释规则",
            "description": "在输电问题中，除非明确说明“单根导线电阻”，否则给定的“线路电阻 r”应被视为回路总电阻，不应再乘以 2。",
            "srd": "FOR EACH transmission_line: IF parameter.resistance_r IS given AND text.does_not_contain 'per wire' OR 'single conductor' THEN CHECK power_loss_formula == I^2 * r; REPORT violation 'double_counting_resistance' IF power_loss_formula == 2 * I^2 * r.",
            "source_error_id": "52125",
            "source_error_type": "modeling"
        },
        {
            "id": "rule_lift_force_direction_01",
            "title": "升力方向定义检查",
            "description": "检查升力（Lift Force）的方向是否被正确定义为垂直于流体相对速度的方向。在水平运动中，升力应为竖直方向。",
            "srd": "FOR EACH force_lift: CHECK force_lift.direction IS_PERPENDICULAR_TO velocity.relative_direction; REPORT violation 'lift_direction_error' IF force_lift.direction IS_PARALLEL_TO velocity.relative_direction.",
            "source_error_id": "110786",
            "source_error_type": "concept"
        },
        {
            "id": "rule_quasiparticle_dynamics_01",
            "title": "准粒子动力学动量关系检查",
            "description": "在准粒子（如电子在晶格中）问题中，检查是否错误使用了 p=mv 关系。应使用群速度 v = dK/dp。",
            "srd": "FOR EACH quasiparticle: IF context.is_solid_state_physics OR context.has_dispersion_relation THEN CHECK velocity_formula == dK/dp; REPORT violation 'quasiparticle_momentum_error' IF velocity_formula == p/m.",
            "source_error_id": "89195",
            "source_error_type": "concept"
        },
        {
            "id": "rule_relativistic_velocity_addition_01",
            "title": "相对论速度叠加检查",
            "description": "在涉及相对论速度（接近光速）的问题中，检查是否错误使用了伽利略速度叠加（v = v1 + v2）。",
            "srd": "FOR EACH velocity_addition: IF context.is_relativistic OR velocity > 0.1c THEN CHECK formula.uses_lorentz_transformation; REPORT violation 'galilean_addition_in_relativity' IF formula == v1 + v2.",
            "source_error_id": "40759",
            "source_error_type": "concept"
        }
    ]

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(rules, f, ensure_ascii=False, indent=2)
        
    print(f"Successfully generated {len(rules)} rules (manually curated due to API limit). Saved to {output_path}")

if __name__ == "__main__":
    main()
