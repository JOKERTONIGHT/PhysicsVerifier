# unified_rules 当前文件整理

## 正式入口

统一从这里进入，不再直接组合多个脚本和一串后缀文件：

```text
scripts/unified_rules_pipeline.py
```

固定命令：

- `paths`
- `preflight`
- `server-command`
- `analyze-extraction`
- `prepare-cluster`
- `embedding-command`
- `analyze-embedding`
- `cluster-command`
- `build-blueprints`
- `validate-blueprints-command`
- `rebuild-command`

底层脚本仍可保留作为实现模块和调试入口，但日常推进不要再直接产出新的根目录后缀文件。

## 正式产物目录

3000 实验结果统一放在：

```text
results/unified_rules_3000/
```

当前正式文件：

- `semantic_experience.json`
- `semantic_experience_distilled.json`
- `semantic_experience_distilled_for_cluster.json`
- `rule_embedding_input.json`
- `rule_embedding_clusters.json`
- `rule_embedding_cluster_report.json`
- `cluster_proposals.json`
- `cluster_blueprints_validation.json`
- `extraction_report.json`
- `precluster_report.json`

正式 3000 catalog：

- `catalogs/rules_unified_3000.json`
- `catalogs/scenario_cluster_blueprints_generated_3000.json`

下一步补 cluster 的输入固定为：

- `catalogs/rules_unified_3000.json`
- `results/unified_rules_3000/rule_embedding_input.json`
- `results/unified_rules_3000/rule_embedding_clusters.json`

## 必须保留

### 代码

- `scripts/unified_rules_pipeline.py`
  - unified_rules 用户侧唯一正式流程入口。
- `scripts/run_semantic_experience.py`
  - API 抽取逐题 semantic audit、经验规则和 auxiliary。
- `scripts/analyze_semantic_experience_run.py`
  - 验收服务器抽取输出。
- `scripts/prepare_rules_for_cluster.py`
  - 确定性规范化、catalog 重建、precluster 报告。
- `scripts/build_unified_catalog.py`
  - 从知识骨架、distilled 规则、cluster blueprint 构建 minimal runtime catalog。
- `scripts/generate_cluster_proposals.py`
  - 基于 embedding 聚类结果，让强模型给 cluster 打标签和 summary。
- `scripts/refine_cluster_blueprints.py`
  - 将 labeled proposal 转换为 builder-ready blueprint；必要时也可做额外 refinement。
- `scripts/run_rule_embedding_clustering.py`
  - 调用 embedding API，在 topic 内对 rule 做语义聚类。
- `scripts/analyze_rule_embedding_clusters.py`
  - 验收 embedding 聚类结果，统计 cluster 覆盖和 residual 比例。
- `scripts/validate_cluster_blueprints.py`
  - cluster blueprint 验证。
- `core/unified_semantic_matcher.py`
- `core/top_down_verifier.py`
- `core/rule_based_verifier.py`

### 数据和 catalog

- `data/evaluation_sample_3000_expansion.json`
- `data/combined_language_only.json`
- `catalogs/rules_catalog_top_down.json`
- `catalogs/rules_300_tagged.json`
- `catalogs/semantic_experience_distilled_300.json`
- `catalogs/scenario_cluster_blueprints.json`
- `catalogs/scenario_cluster_blueprints_generated_3000.json`
- `catalogs/rules_unified.json`
- `catalogs/rules_unified_3000.json`

## 不再使用的命名方式

不要继续在 `results/` 根目录生成：

- `semantic_experience_3000.json`
- `semantic_experience_distilled_3000.json`
- `semantic_experience_distilled_3000_for_cluster.json`
- `semantic_experience_3000_report.json`
- `rules_unified_3000_comparison.json`
- `*_normalized.json`
- `*_sample10.json`
- `*_full.json`

这些信息现在统一进入 `results/unified_rules_3000/` 的固定文件。

## 当前状态

- 3000 semantic 输出已回传并迁移到 `results/unified_rules_3000/`。
- `prepare-cluster` 已完成。
- `catalogs/rules_unified_3000.json` 已是 cluster 前置候选 catalog。
- 当前下一步是第 3 步 rule embedding 聚类，需要调用 embedding API，应先运行：

```powershell
D:\conda_envs\physicsverifier\python.exe scripts\unified_rules_pipeline.py embedding-command --dataset 3000
```

生成 `rule_embedding_clusters.json` 后，再运行强模型标签与 summary：

```powershell
D:\conda_envs\physicsverifier\python.exe scripts\unified_rules_pipeline.py analyze-embedding --dataset 3000
```

确认聚类结果可用后运行：

```powershell
D:\conda_envs\physicsverifier\python.exe scripts\unified_rules_pipeline.py cluster-command --dataset 3000
```
