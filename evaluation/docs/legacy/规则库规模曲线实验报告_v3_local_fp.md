# 规则库规模曲线实验报告（本地 30B，错误级测评）

> 生成时间：2026-06-13 22:36 UTC  
> 实验批次 STAMP：`20260613_162639`

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
| 300 | 390 | 13.4% | 54.4% | 21.5% | 83.0% | 62.0% |
| 600 | 833 | 17.3% | 50.9% | 25.8% | 97.0% | 71.0% |
| 900 | 1225 | 18.8% | 53.1% | 27.7% | 98.0% | 77.0% |
| 1200 | 1669 | 18.0% | 49.8% | 26.5% | 95.0% | 72.0% |
| 1500 | 2129 | 20.2% | 51.9% | 29.1% | 98.0% | 80.0% |

## 曲线图

![错误级指标随规则库规模变化](results/scale_curve_error_v3_local_fp/error_scale_curve.png)

## 原始产物

- CSV：`results/scale_curve_error_v3_local_fp/curve_metrics.csv`
- JSON：`results/scale_curve_error_v3_local_fp/curve_metrics.json`
- 各档结果目录：`results/scale_curve_error_v3_local_fp/scale_*`

