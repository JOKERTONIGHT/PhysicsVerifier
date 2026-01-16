"""简单评估脚本：

跑 4 组配置：
- 仅原始回答 + gpt-5
- 仅原始回答 + gpt-4o
- 原始回答 + 符号图 + gpt-5
- 原始回答 + 符号图 + gpt-4o

输入：data/evaluation_input.json
输出：results/eval_<rule>_<mode>_<model>.json，其中 <rule> 为 dir 或 srd
可通过 --limit 仅抽样部分数据，或通过 --max-llm-calls 控制每次运行的 LLM 调用上限
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

from rule_based_verifier import RuleBasedVerifier, _BUILTIN_RULES_MAP
from agentic_rule_based_verifier import AgenticRuleBasedVerifier


def run_one_setting(samples, llm_model: str, use_symbol_graph: bool, rule_mode: str,
                    output_path: Path, max_llm_calls: int,
                    verifier_cls: type[RuleBasedVerifier]):
    print(f"\n=== Running setting: model={llm_model}, use_symbol_graph={use_symbol_graph}, rule_mode={rule_mode}, samples={len(samples)} ===")
    verifier = verifier_cls(
        llm_model=llm_model,
        rules=list(_BUILTIN_RULES_MAP.keys()),
        enable_cache=True,
        rule_mode=rule_mode,
        use_symbol_graph=use_symbol_graph,
        max_llm_calls=max_llm_calls,
    )
    full_report = verifier.analyze_batch(samples, dataset_key="evaluation_input", export_graph=False)

    # 只保留存在错误的样本，仿照 main 中 errors_only 的输出结构
    errors_only = []
    for i, result in enumerate(full_report["results"]):
        if result.get("diagnostics"):
            original_sample = samples[i]
            errors_only.append({
                "id": original_sample.get("id"),
                "question": original_sample.get("question"),
                "prediction": original_sample.get("prediction"),
                "answer": original_sample.get("answer"),
                "diagnostics": result["diagnostics"],
            })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(errors_only, f, ensure_ascii=False, indent=2)
    print(f"Saved report to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate RuleBasedVerifier under different context/model settings.")
    parser.add_argument("--input", "-i", type=str, default="data/evaluation_input.json")
    parser.add_argument("--out-dir", type=str, default="results/eval_runs",
                        help="Directory to save evaluation reports.")
    parser.add_argument("--rule-mode", type=str, choices=["srd", "direct", "both"], default="both",
                        help="Use SRD translations, direct natural-language rules, or run both.")
    parser.add_argument("--use-symbol-graph", type=str, choices=["both", "true", "false"], default="false",
                        help="Control whether to build and use the symbol graph: 'true', 'false', or 'both' (run both settings, default).")
    parser.add_argument("--limit", type=int, default=0,
                        help="Only evaluate the first N samples (0 means use the entire file).")
    parser.add_argument("--max-llm-calls", type=int, default=0,
                        help="Total LLM call budget per verifier instance (0 means unlimited).")
    parser.add_argument("--verifier-type", type=str, choices=["standard", "agentic"], default="standard",
                        help="Choose the checker implementation: 'standard' RuleBasedVerifier or 'agentic' AgenticRuleBasedVerifier.")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as f:
        samples = json.load(f)

    total_loaded = len(samples)
    if args.limit and args.limit > 0:
        samples = samples[:args.limit]
        print(f"Loaded {total_loaded} samples; limiting to the first {len(samples)} for this run.")
    else:
        print(f"Loaded {total_loaded} samples for evaluation.")

    out_dir = Path(args.out_dir)

    # 根据命令行参数决定要跑哪些规则模式
    rule_modes = ["direct", "srd"] if args.rule_mode == "both" else [args.rule_mode]

    # 根据命令行参数决定是否评估单一符号图设置或两种都跑
    base_settings = [
        # ("gpt-5", False),
        ("gpt-4o", False),
        # ("gpt-5", True),
        ("gpt-4o", True),
    ]

    def graph_filter(use_graph: bool) -> bool:
        if args.use_symbol_graph == "both":
            return True
        if args.use_symbol_graph == "true":
            return use_graph
        if args.use_symbol_graph == "false":
            return not use_graph
        return True

    verifier_cls = RuleBasedVerifier if args.verifier_type == "standard" else AgenticRuleBasedVerifier

    for rule_mode in rule_modes:
        rule_tag = "dir" if rule_mode == "direct" else "srd"
        for model, use_graph in base_settings:
            if not graph_filter(use_graph):
                continue
            suffix = "no_graph" if not use_graph else "with_graph"
            out_path = out_dir / f"eval_{rule_tag}_{suffix}_{model}.json"
            run_one_setting(samples, llm_model=model, use_symbol_graph=use_graph,
                            rule_mode=rule_mode, output_path=out_path,
                            max_llm_calls=args.max_llm_calls,
                            verifier_cls=verifier_cls)


if __name__ == "__main__":
    main()
