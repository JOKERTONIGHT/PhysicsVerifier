# unified_rules 进度同步

## 2026-05-26

### 当前完成的事

1. 精简了 unified_rules 的运行时树结构。
   - 当前 catalog 使用 `semantic_navigation_tree_minimal`。
   - 树结构保持为 `Domain -> Topic -> Scenario Cluster -> Rule Group -> Rule`。
   - 最终运行时 catalog 不再存储 `includes/excludes/applicability/negative_cues` 等旧字段。

2. 完成了 3000 条样本的经验规则抽取与回传整理。
   - 原始抽取结果统一放到 `results/unified_rules_3000/`。
   - 当前 distilled 规则数为 `4361`。
   - 有效覆盖 topic 数为 `93`。
   - 仍有 `168` 条样本为空规则或 Unknown topic。

3. 完成了 cluster 前的规则规范化。
   - 统一 topic 命名，处理模型输出中的重复 domain 前缀。
   - 基于 NaviRAG 思路在 rule 层生成 `summary`，优先使用抽取阶段得到的 `auxiliary.node_summary`。
   - 清洗 rule 字段并生成稳定 rule_id。
   - 生成 embedding 聚类输入：`results/unified_rules_3000/rule_embedding_input.json`。
   - 检查 exact duplicate，结果为 `0` 组。
   - 检查 near duplicate，同标题近重复为 `41` 组。
   - 规范化后规则数仍为 `4361`。

4. 重建了 3000 规则候选库。
   - 输出文件：`catalogs/rules_unified_3000.json`
   - schema：`semantic_navigation_tree_minimal`
   - executable rules：`4361`
   - topics with rules：`93`
   - scenario clusters：`80`

5. 整合了 unified_rules 的脚本和结果结构。
   - 新增统一入口：`scripts/unified_rules_pipeline.py`
   - 新增流程说明：`docs/unified_rules_workflow.md`
   - 3000 结果统一放在：`results/unified_rules_3000/`
   - 不再继续使用 `*_normalized.json`、`*_comparison.json`、`*_sample10.json` 这类不断叠后缀的过程文件。

6. 调整了 cluster 生成路线。
   - 放弃“强模型直接读取整个 topic 全部 rules 并自行分 cluster”的路径。
   - 当前路线改为：topic 内 rule embedding 聚类 -> 强模型只给已有簇命名和生成 summary -> deterministic 生成 scenario cluster blueprints。
   - `cluster-command` 已改为读取 `rule_embedding_clusters.json`，降低 prompt 长度和模型截断风险。

7. 补齐了 embedding 聚类验收。
   - 新增 `analyze-embedding`，用于统计 cluster 覆盖率、residual 比例、高 residual topic。
   - 服务器回传 `rule_embedding_clusters.json` 后，先验收再进入强模型打标签。

8. 第一次 embedding 聚类验收结果。
   - 使用 `similarity_threshold=0.78`，总规则 `4361`。
   - 聚入 cluster 的规则为 `1137`，覆盖率 `26.07%`。
   - residual 规则为 `3224`，覆盖偏低。
   - 结论：0.78 阈值偏保守，不建议直接进入强模型标签；应在服务器复用 embedding cache，用较低阈值重聚类。

9. 修复 cluster 标签生成的落盘机制。
   - 之前脚本在所有 topic 完成后才写 `cluster_proposals.json`，如果最后一个 topic 报错会丢失前面结果。
   - 现在改为每完成一个 topic 就增量写入，并支持 `--resume` 跳过已完成 topic。
   - 如果再次失败，可保留已有结果后继续补跑。

### 当前结论

3000 条数据已经显著扩大了规则覆盖，但当前规则仍偏“逐题经验点”，还不是完全聚合后的稳定规则库。由于没有 exact duplicate，单纯本地确定性规则很难继续压缩；如果后续要进一步提高规则质量，需要做语义聚合，这一步会调用模型 API。

### 下一步

现在已经推进到第 3 步：topic 内 rule embedding 聚类。该步骤需要调用 embedding API，需放到服务器执行。

当前 embedding 输入为：

- `results/unified_rules_3000/rule_embedding_input.json`

推荐重聚类阈值：

- `similarity_threshold=0.74`

执行前先运行：

```powershell
D:\conda_envs\physicsverifier\python.exe scripts\unified_rules_pipeline.py embedding-command --dataset 3000
```

embedding 聚类产出后，再进入强模型给 cluster 打标签和 summary。

后续固定顺序为：

1. `embedding-command`：生成 `rule_embedding_clusters.json`
2. `analyze-embedding`：验收 embedding 聚类覆盖
3. `cluster-command`：强模型给 embedding clusters 打标签和 summary
4. `build-blueprints`：生成 `catalogs/scenario_cluster_blueprints_generated_3000.json`
5. `validate-blueprints-command`：校验 generated blueprints
6. `rebuild-command`：重建 `catalogs/rules_unified_3000.json`

### 验证情况

已运行 unified_rules 相关测试：

```text
Ran 37 tests in 0.938s
OK
```
