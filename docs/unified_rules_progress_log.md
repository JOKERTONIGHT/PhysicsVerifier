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

10. 放宽 cluster 标签的英文硬限制。
   - 3000 规则本身包含中文经验描述，强模型可能把中文物理短语带入 summary/cues。
   - 这不应阻断流程；现在改为记录 `contains_cjk_generated_text` 和 warning 计数，不再抛错。
   - 已下载的 partial `cluster_proposals.json` 包含 18 个成功 topic 和 1 个失败记录，可在服务器 resume 继续补跑。

11. 继续增强 cluster 标签容错。
   - 个别 topic 仍可能返回非法或不完整 JSON。
   - 统一 `cluster-command` 现在默认带 `--continue-on-error`，单 topic 失败会记录到 `failures`，不再阻断后续 topic。
   - 后续对少量失败 topic 单独补跑或人工审查，不影响已成功 topic 进入 blueprint 生成。

12. 完成 3000 规则库的 cluster blueprint 生成与重建。
   - `cluster_proposals.json` 当前成功 `37` 个 topic，失败记录 `3` 条。
   - 已生成 `catalogs/scenario_cluster_blueprints_generated_3000.json`。
   - blueprint 校验通过：无未知 topic、未知 rule、重复 rule 分配。
   - 已重建 `catalogs/rules_unified_3000.json`。
   - 重建后 executable rules 为 `4361`，scenario clusters 为 `219`，clustered topics 为 `43`。
   - runtime schema 仍是 `semantic_navigation_tree_minimal`，未恢复旧的 `includes/excludes/applicability/negative_cues` 字段。

13. 完成第一轮规则库质量评估与本地优化。
   - 新增 `scripts/evaluate_unified_rules_quality.py` 和 `quality-report` 统一命令。
   - 首轮质量评分为 `67`，问题集中在 cluster summary 过长和未聚类 topic 过多。
   - 修复 builder 投影逻辑：cluster runtime `summary` 优先取 rule group 的 concise summary，不再直接取长 description。
   - 调整中文 summary 质量判定，避免把有效的中文短摘要误判为过短。
   - 重新评估后质量评分提升到 `86`，状态为 `usable_with_known_gaps`。
   - 当前剩余主要问题：50 个有规则 topic 无 scenario cluster，general_reasoning rule 占比约 `30.06%`，3 条 cluster proposal 失败记录。

14. 补齐端到端 runtime 评估入口。
   - 新增 `scripts/evaluate_top_down_runtime.py`，用于批量运行 `TopDownVerifier` 并统计 topic/cluster/rule 选择率、空规则选择率和诊断数量。
   - `scripts/unified_rules_pipeline.py` 新增 `runtime-eval-command`，统一生成服务器运行命令。
   - 该步骤会调用语义匹配 API，应在服务器 conda 环境运行。

15. 修复 runtime 评估暴露的 semantic matcher JSON 解析问题。
   - 首次 30 条 runtime 评估中，`semantic_tree_selection_count=0`，30/30 都报 `Semantic matcher must return a JSON object`。
   - 原因是 `core/unified_semantic_matcher.py` 只直接 `json.loads(content)`，无法处理模型返回 Markdown fenced JSON 或截断预览。
   - 已新增 JSON object 提取兜底，支持 ` ```json ... ``` ` 和 loose JSON object。
   - 需要服务器重新跑 runtime 评估，验证 topic/cluster/rule 真实选择率。

16. 修复 runtime smoke1 中模型直接返回数组的问题。
   - 服务器重新跑 1 条后，错误变为模型返回 JSON array，而不是 `{ "topics": [...] }` object。
   - 已将 semantic matcher 改为按选择阶段包装数组响应：`domains/topics/clusters/rules`。
   - 最新 `top_down_runtime_eval_smoke1_fixed.json` 已成功进入 `semantic_tree_selection`。
   - smoke1 结果：topic 命中 1 个，cluster 命中 1 个，rule 命中 3 条，semantic error 为 0。
   - 下一步应在服务器继续跑 30 条 runtime eval，观察整体 topic/cluster/rule 选择率和空规则率。

17. 完成 30 条端到端 runtime 评估。
   - 最新文件：`results/unified_rules_3000/top_down_runtime_eval_30_fixed.json`。
   - 30/30 均进入 `semantic_tree_selection`，`semantic_error_count=0`。
   - topic 命中率 `100%`，cluster 命中率 `100%`，rule 命中率 `96.67%`。
   - 总共选择 84 条规则，平均每题 2.8 条；20/30 题产生 diagnostic。
   - 唯一空规则样本为 `170364`：heat-transfer 冷却曲线积分题，topic/cluster 命中正确，但对应 cluster 只有 2 条规则，未覆盖“按 P(T) 图像积分冷却时间”的经验规则。
   - 少数跨学科样本存在过宽选择，例如 `157816` 选 8 条 rule、`142965` 选 3 个 topic 和 4 个 cluster。
   - 已增强 `scripts/evaluate_top_down_runtime.py`，后续 summary 会直接列出空规则、高规则数、过宽 topic/cluster 样本，减少手工排查。

18. 将 runtime 评估并入规则库质量报告。
   - `scripts/evaluate_unified_rules_quality.py` 现在支持读取 `--runtime-eval`。
   - 质量报告不再只看静态 catalog，也会纳入端到端语义树检索问题：
     - `runtime_empty_rules`
     - `runtime_overbroad_selection`
     - `runtime_semantic_errors`
   - 当前综合质量分为 `77`，状态仍是 `usable_with_known_gaps`。
   - 分数低于此前静态结构评估的 `86`，原因是新增了真实 runtime 问题作为扣分项；这个口径更适合判断是否能进入全流程。
   - 当前必须优先处理的实证问题：
     - `170364`：正确 topic/cluster 下无 rule，属于覆盖缺口。
     - `157816`、`142965`、`147128`：选择过宽，属于检索噪声控制问题。

19. 修复旧 300 规则在 3000 扩展库中的覆盖回归。
   - 排查 `170364` 后发现：旧 `catalogs/rules_unified.json` 中已有精确规则，例如“变功率冷却时间积分规则”“图表信息提取完整性校验”。
   - 当前 3000 构建链路只把旧 300 作为 tagged reference/对比输入，没有把旧 300 的 executable rules 合入 3000 distilled，因此出现覆盖回归。
   - 已修改 `scripts/prepare_rules_for_cluster.py`：默认将 baseline catalog 的 executable rules 作为 seed coverage 合入规范化规则集。
   - 重新生成后：
     - `distilled_input_rules=4361`
     - `baseline_seed_rules=514`
     - `total_executable_rules=4875`
     - `total_scenario_clusters=220`
   - `170364` 对应的 heat-transfer 规则已重新进入 `rules_unified_3000.json`，并被放入 `heating_cooling_and_capacity_model` cluster。
   - 同时修复 `build_unified_catalog.py` 中重复生成 `general_reasoning` cluster 的问题：如果 blueprint 已有 general fallback，剩余规则会合并进去，不再生成重复 cluster。
   - `validate_cluster_blueprints.py` 的 subset 模式已修正：允许未覆盖规则由 builder fallback 承接，只把未知 topic/rule、重复分配等结构错误判为 invalid。
   - 当前 30 条 runtime 报告已被标记为 stale，因为 catalog 已重建；需要在服务器重新跑 30 条验证，确认 `170364` 是否已消除空规则。

20. 统一 pipeline 质量报告口径。
   - `scripts/unified_rules_pipeline.py quality-report` 已接入 canonical runtime eval 文件：`results/unified_rules_3000/top_down_runtime_eval.json`。
   - 后续运行 `quality-report` 会自动判断 runtime eval 是否早于当前 catalog。
   - 当前报告中 `runtime_eval.stale=true`，说明必须重跑 runtime eval 后才能把 runtime 结果作为当前 catalog 的有效证据。
   - 推荐后续不再手写 `evaluate_unified_rules_quality.py` 参数，统一使用：
     - `python scripts/unified_rules_pipeline.py runtime-eval-command --dataset 3000 --limit 30`
     - `python scripts/unified_rules_pipeline.py quality-report --dataset 3000`

21. 增加 3000 catalog 覆盖回归测试。
   - 新增 `tests/test_unified_3000_catalog_regression.py`。
   - 固定检查两类问题：
     - `170364` 对应的 heat-transfer 旧规则必须存在于当前 3000 catalog。
     - 任意 topic 下不得出现重复 `scenario_cluster.id`。
   - 这可以防止后续重建 catalog 时再次丢掉旧 300 的高价值规则，或再次生成重复 `general_reasoning` cluster。

### 当前结论

3000 条数据已经显著扩大了规则覆盖，但当前规则仍偏“逐题经验点”，还不是完全聚合后的稳定规则库。由于没有 exact duplicate，单纯本地确定性规则很难继续压缩；如果后续要进一步提高规则质量，需要做语义聚合，这一步会调用模型 API。

### 下一步

当前 1-6 步主流程已跑通，且第一轮质量评估达到“可用但有已知缺口”状态。

当前 embedding 输入为：

- `results/unified_rules_3000/rule_embedding_input.json`

推荐重聚类阈值：

- `similarity_threshold=0.74`

执行前先运行：

```powershell
D:\conda_envs\physicsverifier\python.exe scripts\unified_rules_pipeline.py embedding-command --dataset 3000
```

下一步建议：先在服务器跑 30 条 `top_down_verifier` 端到端检索质量评估；根据空规则选择率和错误 topic 分布，再决定优先补哪些无 cluster topic。

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
