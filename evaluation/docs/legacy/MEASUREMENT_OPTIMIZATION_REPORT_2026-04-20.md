# 测评流程优化报告（2026-04-20）

## 1. 目标与约束
根据 docs/测评优化说明.md，本轮优化目标是：
1. 测评集作为 GT 同时用于 recall 与 precision。
2. GT 错误条目追求完整、准确，不人为限制数量。
3. 位置标注从“仅字符级”扩展为“段落级可匹配”，便于稳定评测。
4. 增加必要检查流程：先小规模（10）验证质量，不达标则继续迭代，再跑 20 全流程。

## 2. 代码层优化

### 2.1 GT 生成优化（scripts/build_physics_eval_sets.py）
1. 提示词升级为“exhaustive”模式：
- --max-errors 支持 0（默认），表示不设硬上限，要求尽量穷举错误。
- 输出 schema 扩展 paragraph_index。

2. 新增段落级定位能力：
- GT 除 start_char/end_char 外，新增 paragraph_index、paragraph_start_char、paragraph_end_char、paragraph_valid、paragraph_source。
- locatable_valid 定义为 span_valid 或 paragraph_valid。

3. 新增质量过滤流程（必要检查）：
- 去重：error/quote/paragraph 去重 + 同段/同 quote 的近重复过滤。
- 伪错误过滤：剔除 reference-answer 伪比较、低置信措辞（如 not an error / minor omission / might be）。
- 同 quote 限流：同一 quote 最多保留 1 条错误，减少冗余。

4. 保留并增强质量门控：
- 样本必须至少 min_valid_gt_per_sample 条可定位 GT（当前按 locatable_valid 统计）。
- 若样本不达标，自动跳过并补扫下一个候选，直到凑齐目标规模。

### 2.2 评测器优化（scripts/evaluate_physics_eval_sets.py）
1. 位置匹配升级为“双层匹配”：
- 第一层：字符 span IoU/coverage（原有）。
- 第二层：段落 index 匹配兜底（span 缺失或不命中时）。

2. 预测位置补全增强：
- 先用 span 推导 paragraph。
- span 缺失时，用 quote 回填 span，再推导 paragraph。
- 统一 locatable_valid（span 或 paragraph）。

3. 指标增强：
- 新增 location_paragraph_match_pairs，明确段落级命中数量。
- unmatched 统计改为基于 locatable_valid。

### 2.3 检查器输出优化（core/semantic_rule_checker.py）
1. 提示词 schema 扩展 paragraph_index。
2. 结果归一化新增 paragraph 字段补全：
- model 提供 paragraph_index 时校验并补齐 paragraph_start/end。
- 已有 span 时自动推导 paragraph。
- 输出 locatable_valid。

### 2.4 流水线与质量审计
1. 新增脚本 scripts/audit_eval_set_quality.py：
- 质量门槛检查：locatable_ratio、avg_errors、generic_ratio、quote_ratio、zero-error 样本。
- 输出 recall_quality_audit.json。

2. 增强脚本 scripts/run_physics_eval_pipeline.py：
- 支持 recall/precision 输入分离。
- 支持 exhaustive 参数、质量审计与质量门槛中断。
- 支持 location/hybrid/semantic 评测模式切换。
- 支持 --disable-agentic 以提高推理稳定性。

## 3. 小规模验证（10条）

### 3.1 验证流程
执行了多轮 10 条试跑，逐轮修正：
1. v1：实现段落级匹配与质量审计。
2. v2/v3：增加去重和伪错误过滤。
3. v4：增加低置信措辞过滤，最终作为通过版本。

### 3.2 v4 质量审计结论（通过）
- quality_gate_passed: true
- sample_count: 10
- total_errors: 62
- avg_errors_per_sample: 6.2
- quote_non_empty_ratio: 0.806
- span_valid_ratio: 0.371
- paragraph_valid_ratio: 1.000
- locatable_valid_ratio: 1.000
- generic_error_ratio: 0.000
- samples_with_zero_errors: 0
- samples_with_no_locatable_errors: 0

结论：
- 满足“完整且可匹配”要求（locatable=1.0）。
- 错误条目具备具体性（generic=0），并通过低置信过滤。

## 4. 全量流程（20条）

### 4.1 运行结果
输出目录：results/eval_pipeline_v0420_optimized_20

构建阶段：
- recall_collected_size: 20 / 20
- gt_total_errors: 127
- gt_locatable_errors: 127
- gt_span_locatable_errors: 57
- gt_paragraph_locatable_errors: 127
- gt_location_valid_ratio: 1.0

质量审计：
- quality_gate_passed: true
- avg_errors_per_sample: 6.35
- quote_non_empty_ratio: 0.929
- locatable_valid_ratio: 1.0
- generic_error_ratio: 0.0

评测指标（location）：
- recall_location_error_level: 0.2520
- recall_error_level: 0.2520
- recall_sample_location_hit_ratio: 0.7000
- pred_location_coverage: 0.8804
- location_match_pairs: 32
- location_paragraph_match_pairs: 31
- precision_proxy: 0.8500

样本级诊断摘要：
- recall 未触发诊断样本：1（238_279）
- recall 位置命中样本：14/20
- precision 假阳性样本：3（167, 145, 881）
- artifact 扫描（reference answer/not an error/minor omission）：0

## 5. 与此前口径对比（20条基线）
对比文件：results/eval_pipeline_v0416_qwen30b_a3b_location20/metrics_qwen30b_a3b_instruct_location.json

关键变化：
1. recall_location_error_level: 0.0370 -> 0.2520（显著提升）
2. gt_location_valid_ratio: 0.6750 -> 1.0000（位置完整性显著提升）
3. recall_sample_location_hit_ratio: 0.1000 -> 0.7000（样本层命中显著提升）
4. precision_proxy: 0.8500 -> 0.8500（保持不变）

## 6. 结论
本轮已完成“生成-检查-评测”全链路优化，并通过“10条先验收 + 20条全流程”验证：
1. GT 从字符级拓展到段落级后，位置可用性达到 100%。
2. 通过质量过滤与门控，GT 条目更可用于稳定测评（完整性与可匹配性提升明显）。
3. recall 主指标显著提升，同时 precision 侧保持稳定。

## 7. 产物清单
1. 小规模最终验证（v4）
- results/eval_pipeline_v0420_optimized_10_v4/evaluation_recall_20.json
- results/eval_pipeline_v0420_optimized_10_v4/recall_quality_audit.json
- results/eval_pipeline_v0420_optimized_10_v4/metrics.json

2. 全量 20 条
- results/eval_pipeline_v0420_optimized_20/evaluation_recall_20.json
- results/eval_pipeline_v0420_optimized_20/evaluation_precision_20.json
- results/eval_pipeline_v0420_optimized_20/recall_quality_audit.json
- results/eval_pipeline_v0420_optimized_20/recall_top_down_results.json
- results/eval_pipeline_v0420_optimized_20/recall_symbolic_audit.json
- results/eval_pipeline_v0420_optimized_20/precision_top_down_results.json
- results/eval_pipeline_v0420_optimized_20/precision_symbolic_audit.json
- results/eval_pipeline_v0420_optimized_20/metrics.json
