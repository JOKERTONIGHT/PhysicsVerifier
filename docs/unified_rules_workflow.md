# unified_rules Workflow

本文件记录当前唯一正式流程。旧的单步脚本只作为实现模块或调试入口；日常推进优先使用 `scripts/unified_rules_pipeline.py`。

## Canonical Entry

```powershell
D:\conda_envs\physicsverifier\python.exe scripts\unified_rules_pipeline.py <command>
```

## Canonical 3000 Outputs

所有 3000 经验规则运行结果统一放在：

```text
results/unified_rules_3000/
```

目录内固定文件：

- `semantic_experience.json`
- `semantic_experience_distilled.json`
- `semantic_experience_distilled_for_cluster.json`
- `rule_embedding_input.json`
- `rule_embedding_clusters.json`
- `rule_embedding_cluster_report.json`
- `extraction_report.json`
- `server_preflight.json`
- `precluster_report.json`
- `cluster_proposals.json`
- `cluster_blueprints_validation.json`

正式 3000 catalog 固定为：

```text
catalogs/rules_unified_3000.json
```

不再新增 `*_normalized.json`、`*_sample10.json`、`*_comparison.json` 这类过程后缀文件。需要报告时写入 `results/unified_rules_3000/` 中的固定报告文件。

## Commands

查看路径：

```powershell
D:\conda_envs\physicsverifier\python.exe scripts\unified_rules_pipeline.py paths --dataset 3000
```

服务器运行前预检：

```powershell
D:\conda_envs\physicsverifier\python.exe scripts\unified_rules_pipeline.py preflight --dataset 3000
```

打印服务器抽取命令：

```powershell
D:\conda_envs\physicsverifier\python.exe scripts\unified_rules_pipeline.py server-command --dataset 3000
```

服务器回传结果后验收：

```powershell
D:\conda_envs\physicsverifier\python.exe scripts\unified_rules_pipeline.py analyze-extraction --dataset 3000 --expected-samples 3000
```

生成 cluster 前置规则库：

```powershell
D:\conda_envs\physicsverifier\python.exe scripts\unified_rules_pipeline.py prepare-cluster --dataset 3000
```

打印 rule embedding 聚类命令：

```powershell
D:\conda_envs\physicsverifier\python.exe scripts\unified_rules_pipeline.py embedding-command --dataset 3000
```

该命令会调用 embedding API，完成 topic 内 rule embedding 聚类，输出：

```text
results/unified_rules_3000/rule_embedding_clusters.json
```

当前推荐阈值为 `--similarity-threshold 0.74`。如果服务器已有 `rule_embedding_cache.json`，重跑该步骤会复用缓存，主要只是重新聚类。

服务器回传 embedding 聚类结果后，先验收聚类覆盖：

```powershell
D:\conda_envs\physicsverifier\python.exe scripts\unified_rules_pipeline.py analyze-embedding --dataset 3000
```

embedding 聚类完成后，再进入强模型 cluster 标签和 summary 生成：

```powershell
D:\conda_envs\physicsverifier\python.exe scripts\unified_rules_pipeline.py cluster-command --dataset 3000
```

该命令使用 `rule_embedding_clusters.json` 和 `rule_embedding_input.json`，只让强模型给 embedding 簇命名、写 summary 和导航辅助线索；不再让模型直接阅读整个 topic 的所有规则并自行分簇。
该步骤会增量写入 `cluster_proposals.json`，并默认带 `--resume`，中途失败后可直接重跑命令继续。

强模型输出回传后，生成 builder 可读的 scenario cluster blueprints：

```powershell
D:\conda_envs\physicsverifier\python.exe scripts\unified_rules_pipeline.py build-blueprints --dataset 3000
```

验证 blueprints：

```powershell
D:\conda_envs\physicsverifier\python.exe scripts\unified_rules_pipeline.py validate-blueprints-command --dataset 3000
```

重建 3000 catalog：

```powershell
D:\conda_envs\physicsverifier\python.exe scripts\unified_rules_pipeline.py rebuild-command --dataset 3000
```

## Current Handoff

当前下一步补 cluster 的输入固定为：

```text
catalogs/rules_unified_3000.json
results/unified_rules_3000/semantic_experience_distilled_for_cluster.json
results/unified_rules_3000/rule_embedding_input.json
```

当前第 3 步 embedding 聚类会调用 API。需要执行时先用 `embedding-command` 打印命令，再去服务器运行。生成 `rule_embedding_clusters.json` 后，才进入第 4 步 cluster 标签和 summary。
