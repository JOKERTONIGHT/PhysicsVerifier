# Scale Checkpoint Runbook

下列命令按检查点从小到大执行。此文档只生成步骤，不自动执行。

## ckpt_0200 (200 samples)

```bash
mkdir -p results/checkpoints/ckpt_0200 results/scale_curve/ckpt_0200 catalogs/checkpoints
./.venv/bin/python scripts/run_semantic_experience.py --input data/checkpoints/evaluation_sample_200.json --rules-catalog catalogs/rules_catalog_top_down.json --model qwen3-30b-a3b --output results/checkpoints/ckpt_0200/semantic_experience.json --distilled-output results/checkpoints/ckpt_0200/semantic_experience_distilled.json
./.venv/bin/python scripts/translate_experience_to_symbolic.py --input results/checkpoints/ckpt_0200/semantic_experience_distilled.json --model gpt-4.1-mini --output-module symbolic/generated_experience_checks.py --output-manifest results/checkpoints/ckpt_0200/experience_symbolic_program_manifest.json --report results/checkpoints/ckpt_0200/experience_symbolic_translation_report.json --repair
./.venv/bin/python scripts/build_unified_rule_library.py --experience-distilled results/checkpoints/ckpt_0200/semantic_experience_distilled.json --output catalogs/checkpoints/unified_rule_library_ckpt_0200.json --rule-source experience-only
./.venv/bin/python scripts/run_top_down.py --input data/evaluation_rubric_100.json --output results/checkpoints/ckpt_0200/top_down_results_eval100.json --symbolic-output results/checkpoints/ckpt_0200/symbolic_audit_eval100.json --model qwen3-30b-a3b --unified-catalog catalogs/checkpoints/unified_rule_library_ckpt_0200.json --experience-code-manifest results/checkpoints/ckpt_0200/experience_symbolic_program_manifest.json --experience-code-module symbolic.generated_experience_checks --no-agentic
./.venv/bin/python scripts/compute_strict_eval_metrics.py --predictions results/checkpoints/ckpt_0200/top_down_results_eval100.json --audit results/checkpoints/ckpt_0200/symbolic_audit_eval100.json --rubric-meta data/rubric_eval_100_meta.json --output results/scale_curve/ckpt_0200/strict_metrics.json --checkpoint-size 200
```

## ckpt_0400 (400 samples)

```bash
mkdir -p results/checkpoints/ckpt_0400 results/scale_curve/ckpt_0400 catalogs/checkpoints
./.venv/bin/python scripts/run_semantic_experience.py --input data/checkpoints/evaluation_sample_400.json --rules-catalog catalogs/rules_catalog_top_down.json --model qwen3-30b-a3b --output results/checkpoints/ckpt_0400/semantic_experience.json --distilled-output results/checkpoints/ckpt_0400/semantic_experience_distilled.json
./.venv/bin/python scripts/translate_experience_to_symbolic.py --input results/checkpoints/ckpt_0400/semantic_experience_distilled.json --model gpt-4.1-mini --output-module symbolic/generated_experience_checks.py --output-manifest results/checkpoints/ckpt_0400/experience_symbolic_program_manifest.json --report results/checkpoints/ckpt_0400/experience_symbolic_translation_report.json --repair
./.venv/bin/python scripts/build_unified_rule_library.py --experience-distilled results/checkpoints/ckpt_0400/semantic_experience_distilled.json --output catalogs/checkpoints/unified_rule_library_ckpt_0400.json --rule-source experience-only
./.venv/bin/python scripts/run_top_down.py --input data/evaluation_rubric_100.json --output results/checkpoints/ckpt_0400/top_down_results_eval100.json --symbolic-output results/checkpoints/ckpt_0400/symbolic_audit_eval100.json --model qwen3-30b-a3b --unified-catalog catalogs/checkpoints/unified_rule_library_ckpt_0400.json --experience-code-manifest results/checkpoints/ckpt_0400/experience_symbolic_program_manifest.json --experience-code-module symbolic.generated_experience_checks --no-agentic
./.venv/bin/python scripts/compute_strict_eval_metrics.py --predictions results/checkpoints/ckpt_0400/top_down_results_eval100.json --audit results/checkpoints/ckpt_0400/symbolic_audit_eval100.json --rubric-meta data/rubric_eval_100_meta.json --output results/scale_curve/ckpt_0400/strict_metrics.json --checkpoint-size 400
```

## ckpt_0600 (600 samples)

```bash
mkdir -p results/checkpoints/ckpt_0600 results/scale_curve/ckpt_0600 catalogs/checkpoints
./.venv/bin/python scripts/run_semantic_experience.py --input data/checkpoints/evaluation_sample_600.json --rules-catalog catalogs/rules_catalog_top_down.json --model qwen3-30b-a3b --output results/checkpoints/ckpt_0600/semantic_experience.json --distilled-output results/checkpoints/ckpt_0600/semantic_experience_distilled.json
./.venv/bin/python scripts/translate_experience_to_symbolic.py --input results/checkpoints/ckpt_0600/semantic_experience_distilled.json --model gpt-4.1-mini --output-module symbolic/generated_experience_checks.py --output-manifest results/checkpoints/ckpt_0600/experience_symbolic_program_manifest.json --report results/checkpoints/ckpt_0600/experience_symbolic_translation_report.json --repair
./.venv/bin/python scripts/build_unified_rule_library.py --experience-distilled results/checkpoints/ckpt_0600/semantic_experience_distilled.json --output catalogs/checkpoints/unified_rule_library_ckpt_0600.json --rule-source experience-only
./.venv/bin/python scripts/run_top_down.py --input data/evaluation_rubric_100.json --output results/checkpoints/ckpt_0600/top_down_results_eval100.json --symbolic-output results/checkpoints/ckpt_0600/symbolic_audit_eval100.json --model qwen3-30b-a3b --unified-catalog catalogs/checkpoints/unified_rule_library_ckpt_0600.json --experience-code-manifest results/checkpoints/ckpt_0600/experience_symbolic_program_manifest.json --experience-code-module symbolic.generated_experience_checks --no-agentic
./.venv/bin/python scripts/compute_strict_eval_metrics.py --predictions results/checkpoints/ckpt_0600/top_down_results_eval100.json --audit results/checkpoints/ckpt_0600/symbolic_audit_eval100.json --rubric-meta data/rubric_eval_100_meta.json --output results/scale_curve/ckpt_0600/strict_metrics.json --checkpoint-size 600
```

## ckpt_0800 (800 samples)

```bash
mkdir -p results/checkpoints/ckpt_0800 results/scale_curve/ckpt_0800 catalogs/checkpoints
./.venv/bin/python scripts/run_semantic_experience.py --input data/checkpoints/evaluation_sample_800.json --rules-catalog catalogs/rules_catalog_top_down.json --model qwen3-30b-a3b --output results/checkpoints/ckpt_0800/semantic_experience.json --distilled-output results/checkpoints/ckpt_0800/semantic_experience_distilled.json
./.venv/bin/python scripts/translate_experience_to_symbolic.py --input results/checkpoints/ckpt_0800/semantic_experience_distilled.json --model gpt-4.1-mini --output-module symbolic/generated_experience_checks.py --output-manifest results/checkpoints/ckpt_0800/experience_symbolic_program_manifest.json --report results/checkpoints/ckpt_0800/experience_symbolic_translation_report.json --repair
./.venv/bin/python scripts/build_unified_rule_library.py --experience-distilled results/checkpoints/ckpt_0800/semantic_experience_distilled.json --output catalogs/checkpoints/unified_rule_library_ckpt_0800.json --rule-source experience-only
./.venv/bin/python scripts/run_top_down.py --input data/evaluation_rubric_100.json --output results/checkpoints/ckpt_0800/top_down_results_eval100.json --symbolic-output results/checkpoints/ckpt_0800/symbolic_audit_eval100.json --model qwen3-30b-a3b --unified-catalog catalogs/checkpoints/unified_rule_library_ckpt_0800.json --experience-code-manifest results/checkpoints/ckpt_0800/experience_symbolic_program_manifest.json --experience-code-module symbolic.generated_experience_checks --no-agentic
./.venv/bin/python scripts/compute_strict_eval_metrics.py --predictions results/checkpoints/ckpt_0800/top_down_results_eval100.json --audit results/checkpoints/ckpt_0800/symbolic_audit_eval100.json --rubric-meta data/rubric_eval_100_meta.json --output results/scale_curve/ckpt_0800/strict_metrics.json --checkpoint-size 800
```

## ckpt_1000 (1000 samples)

```bash
mkdir -p results/checkpoints/ckpt_1000 results/scale_curve/ckpt_1000 catalogs/checkpoints
./.venv/bin/python scripts/run_semantic_experience.py --input data/checkpoints/evaluation_sample_1000.json --rules-catalog catalogs/rules_catalog_top_down.json --model qwen3-30b-a3b --output results/checkpoints/ckpt_1000/semantic_experience.json --distilled-output results/checkpoints/ckpt_1000/semantic_experience_distilled.json
./.venv/bin/python scripts/translate_experience_to_symbolic.py --input results/checkpoints/ckpt_1000/semantic_experience_distilled.json --model gpt-4.1-mini --output-module symbolic/generated_experience_checks.py --output-manifest results/checkpoints/ckpt_1000/experience_symbolic_program_manifest.json --report results/checkpoints/ckpt_1000/experience_symbolic_translation_report.json --repair
./.venv/bin/python scripts/build_unified_rule_library.py --experience-distilled results/checkpoints/ckpt_1000/semantic_experience_distilled.json --output catalogs/checkpoints/unified_rule_library_ckpt_1000.json --rule-source experience-only
./.venv/bin/python scripts/run_top_down.py --input data/evaluation_rubric_100.json --output results/checkpoints/ckpt_1000/top_down_results_eval100.json --symbolic-output results/checkpoints/ckpt_1000/symbolic_audit_eval100.json --model qwen3-30b-a3b --unified-catalog catalogs/checkpoints/unified_rule_library_ckpt_1000.json --experience-code-manifest results/checkpoints/ckpt_1000/experience_symbolic_program_manifest.json --experience-code-module symbolic.generated_experience_checks --no-agentic
./.venv/bin/python scripts/compute_strict_eval_metrics.py --predictions results/checkpoints/ckpt_1000/top_down_results_eval100.json --audit results/checkpoints/ckpt_1000/symbolic_audit_eval100.json --rubric-meta data/rubric_eval_100_meta.json --output results/scale_curve/ckpt_1000/strict_metrics.json --checkpoint-size 1000
```

## Aggregate Curve

```bash
./.venv/bin/python scripts/aggregate_scale_curve.py --metrics-glob 'results/scale_curve/ckpt_*/strict_metrics.json' --output-csv results/scale_curve/curve_metrics.csv --output-json results/scale_curve/curve_metrics.json
./.venv/bin/python scripts/plot_scale_curve.py --input-csv results/scale_curve/curve_metrics.csv --output results/scale_curve/scale_curve.png
```