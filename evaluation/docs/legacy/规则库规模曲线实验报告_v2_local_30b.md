# 规则库规模曲线实验报告（本地 30B，错误级测评）

> 生成时间：2026-06-18 02:56 UTC  
> 实验批次 STAMP：`20260617_190326`

## 实验设置

- **推理后端**：本地 vLLM（Qwen3-30B-A3B-Instruct-2507 AWQ 4-bit）
- **测评集**：`error_eval_dataset_100.json`（100 样本；与规则库按 question 去重后零重叠）
- **备用 holdout**：`eval_holdout_200.json`（200 样本，不参与规则挖掘）
- **规则挖掘数据**：`evaluation_sample_3000_expansion.json` 的非 blocked 子集（约 2691 样本；复用 `catalogs/semantic_experience.json`）
- **规则库构建**：每 scale 点完整 unified pipeline（prepare-cluster，无 baseline seed → embedding 聚类 → LLM cluster 标注 → blueprint 生成 → rebuild）
- **语义抽取**：跳过（使用已有 `catalogs/semantic_experience.json`）
- **测评集**：dual-chain 原版 `error_eval_dataset_100`（prediction 与 GT span 对齐）
- **评测模式**：错误级 location match，`--no-symbolic-check`
- **规模梯度**：300 → 2700（步长 300；若连续 2 档 F1 增益 < 0.5% 则提前停止）

### 数据泄露防护

- 规则池 vs blocked/holdout 重叠：0 条
- error_eval vs 规则池 question 重叠：0 条
- error_eval expansion IDs ⊆ holdout_200：True
- 审计通过：`True`

## 指标汇总

| 扩充样本数 | 规则数 | Recall | Precision | F1 | 触发率 | 定位命中率 |
|---:|---:|---:|---:|---:|---:|---:|
| 300 | 390 | 7.0% | 19.4% | 10.2% | 84.8% | 26.3% |
| 600 | 833 | 10.4% | 21.2% | 14.0% | 96.0% | 38.4% |
| 900 | 1225 | 10.4% | 19.4% | 13.6% | 98.0% | 37.4% |
| 1200 | 1669 | 11.5% | 21.3% | 15.0% | 97.0% | 39.4% |
| 1500 | 2129 | 11.7% | 20.7% | 15.0% | 97.0% | 39.4% |
| 1800 | 2537 | 11.7% | 19.4% | 14.6% | 99.0% | 41.4% |

## 曲线图

![错误级指标随规则库规模变化](results/scale_curve_error_v2_local_30b/error_scale_curve.png)

## 原始产物

- CSV：`results/scale_curve_error_v2_local_30b/curve_metrics.csv`
- JSON：`results/scale_curve_error_v2_local_30b/curve_metrics.json`
- 各档结果目录：`results/scale_curve_error_v2_local_30b/scale_*`

