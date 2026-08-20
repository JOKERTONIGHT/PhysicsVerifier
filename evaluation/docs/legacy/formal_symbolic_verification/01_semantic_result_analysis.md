# 01 纯语义结果分析

## 1. 评测数据与实验对照

基于 `error_eval_dataset_100`（99 题、405 条 GT）的 matching sensitivity 报告（[`results/recall_attribution_v1/final_report/matching_sensitivity_report.json`](../../results/recall_attribution_v1/final_report/matching_sensitivity_report.json)）：

| 实验 | strict recall | paragraph recall | semantic recall | group recall | precision |
|------|---------------|------------------|-----------------|--------------|-----------|
| rules_baseline (1500 规则 + 语义) | 11.6% | 6.7% | **61.5%** | 63.6% | 18.4% |
| local_30b_checker (纯语义) | 20.5% | 12.4% | **90.6%** | 87.0% | 55.7% |
| gemini_flash (纯语义) | 23.7% | 16.3% | 66.7% | 63.2% | 71.6% |

**结论**：

- 纯语义 checker（local 30b）已能识别约 **90%** 的根因错误（semantic detection recall）。
- 规则语义链的主要瓶颈是 **定位**（strict vs semantic 差距约 43–50 个百分点），而非完全不理解错误。
- 符号层若重复做“是否违反物理概念”的判断，价值有限；应聚焦语义仍不稳定且可形式化的片段。

## 2. 漏检归因（unmatched GT）

来自 [`recall_cause_diagnosis_report.json`](../../results/recall_attribution_v1/final_report/recall_cause_diagnosis_report.json)：

| 原因 | rules_baseline | local_30b_checker |
|------|----------------|-------------------|
| semantic_near_miss | 222 (65.5%) | 279 |
| semantic_gap | 130 (34.5%) | **20** |
| no_detection | 3 | 12 |

- **rules_baseline**：大量 near_miss 来自规则触发宽泛、段落/span 不对齐。
- **local_30b_checker**：仅 20 条 semantic_gap，多数未命中为 near_miss（看到了相近问题但未对齐 GT 粒度/位置）。

## 3. 误检归因（FP）

来自 [`failure_analysis.json`](../../results/scale_curve_error_v2_local_30b/scale_1500_cleaned/failure_analysis.json)：

| 原因 | 数量 |
|------|------|
| rule_too_broad | 32 |
| semantic_near_miss | 21 |
| irrelevant_trigger | 7 |

符号 pass 若基于宽泛 token/关键词，可能**放大误抑制**风险，而非提升 precision。

## 4. 按主题漏检分布

来自 [`failure_analysis_by_rule.json`](../../results/scale_curve_error_v2_local_30b/scale_1500_cleaned/failure_analysis_by_rule.json) 的 `top_missed_gt_themes`：

| 主题 | FN 数 | semantic_gap | semantic_near_miss | 符号化潜力 |
|------|-------|--------------|-------------------|------------|
| other | 173 | 74 | 95 | 低（建模/跨题） |
| mechanics | 98 | 30 | 67 | 低–中 |
| thermo_fluid | 38 | 7 | 30 | 中（公式+量纲） |
| formula_algebra | 12 | **9** | 3 | **高** |
| orbital_gravity | 9 | 3 | 6 | **高** |
| electromagnetism | 13 | 3 | 10 | 中–高 |
| relativity_optics | 13 | 2 | 11 | 中 |

**formula_algebra** 主题中 semantic_gap 占比 **75%**，是规则语义最弱、且错误常可写成表达式的类别。

## 5. 自动分类结果

运行 `scripts/analyze_symbolic_candidate_errors.py` 对 405 条 GT 的启发式分类：

| 类别 | 数量 |
|------|------|
| symbolizable | 185 |
| uncertain | ~120 |
| non_symbolizable | ~100 |
| symbolizable + semantic_gap（纯语义漏检） | **10** |

说明：纯语义已覆盖绝大多数可符号化错误；符号层的近期价值是提供**硬证据**和补 **semantic_gap** 子集，而非整体 recall 提升。

## 6. 不适合符号化的错误类型

| 类型 | 示例 | 原因 |
|------|------|------|
| 跨题/跨 SRD 误触发 | 针孔相机题触发转动惯量规则 | 无稳定 canonical，需 trigger 修复 |
| 多步建模链 | 光行时→表观长度→Taylor 近似 | 每步依赖前序错误假设 |
| 题意/方向/叙事 | 应答 λ 倍数却给 nm 数值 | 需题意理解 |
| 缺失型规则 | “No mention of X” | 符号难以证伪未表达 |
| 标注噪声 | duplicate、not_error（~13.4%） | 评测口径问题 |

## 7. 适合符号化的错误类型

| 类型 | 示例样本 | 建议 primitive |
|------|----------|----------------|
| 公式因子/系数错误 | cl_104_24899 轨道速度少 √2 | equation_equivalence_sympy |
| 幂次/标度错误 | cl_209_132531 最终 R⁵ vs R⁴ | power_exponent_check |
| 量纲不一致 | cl_209_132531_e5 R_shell 量纲 | dimension_check |
| sin/cos 分量混用 | cl_188_110801_e1 k_z = k sin θ | component_relation_check |
| 导数/代数错误 | cl_110_31637 dβ/dθ | equation_equivalence_sympy |
| 相对论约束违例 | cl_172_95214 v > c | inequality_constraint |
| 符号/正负号 | cl_104_24899 dE/dt 为正 | sign_check |

## 8. 对符号层设计的含义

1. **不要**与语义竞争概念/建模判断。
2. **优先** formula_algebra、orbital_gravity、局部 EM/热学公式错误。
3. **首轮**只输出 structured evidence，不进入 release suppression。
4. **1500 规则库符号代码未生成**，当前分析基于 semantic-only 与 rules-semantic 对照，不依赖 symbolic audit。

## 9. 数据文件

- 完整候选清单：[`data/derived/symbolic_small_sample_experiment_v1/candidate_errors.json`](../../data/derived/symbolic_small_sample_experiment_v1/candidate_errors.json)
- 重新生成：`python3 scripts/analyze_symbolic_candidate_errors.py`
