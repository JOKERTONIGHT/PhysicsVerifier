# SFT 效果不佳：原因、案例与改进

两次微调都把 Qwen3-8B **「写完就停」** 打坏了。模型不是忘了 `\boxed{}`，也不是通用能力训崩：短问答、指令、代码、翻译都还在。失败形态是进入重复循环，顶满 8192 token 被切断，答案框因此出不来。把题解写长、换标注模型，都救不了。GRPO 仍应从 **base** 起训。

---

## 1. 指标怎么算

实现见 `evaluation/benchmarks/hipho/score_hipho_predictions.py`。正式对照用 **65 题可信集**、每题 **k=4**、温度 0.6、最长 **8192** token（`eval_heldout_fast.sh`）。

| 指标 | 含义 | 计算 |
|---|---|---|
| `part_avg@4` | 小题正确率，主门禁 | 每题 4 次采样，每次看「命中了几个金标小问 / 小问总数」，先对 4 次取平均，再对 65 题取平均。小问对齐用全部 `\boxed{}`，不是只看最后一个。 |
| `item_acc` | 整题全对 | 一次生成里所有金标小问都命中才算 1。旧 SFT 评测用的是这一栏、且 k=1、88 题。 |
| `no_boxed` | 没写出答案框 | 全文抽不到 `\boxed` / `\fbox`。多数其实是写满上限被砍断，不是故意不框。 |
| `repetition` | 尾巴在循环 | 最后 200 个字符在全文中至少出现 3 次（`looks_repetitive`）。 |
| `truncated` | 看起来没写完 | `\boxed{` 括号未闭合；或全文没有 boxed 且结尾不是句读。 |
| `degrade` | 终止变差的合计 | `no_boxed + repetition`（同一条可以两项都中，故可大于「出问题的条数」）。 |

门禁（相对同一套 base 分数）：`part_avg ≥ 0.252`，且 `degrade ≤ 0.05`，且 `no_boxed ≤ 0.05`。

---

## 2. 数字

| 模型 | 口径 | part_avg@4 | no_boxed | repetition | 读法 |
|---|---|---|---|---|---|
| base Qwen3-8B | 65 题 ×4 | **0.252** | **1.5%** | **0.4%** | 几乎总会收束 |
| 全参 SFT ckpt-54 | 88 题 ×1（旧） | 整题正确率 1.1%（base 同口径约 5.7%） | 38.6% | 22.7% | 停止已坏；与下行不可直接比分数，只能比失败形态 |
| RFT LoRA step 20 | 65 题 ×4 | 0.145 | 17.3% | 9.6% | 第 20 步就过不了门禁 |
| RFT step 40–138 | 同上 | 0.14–0.18 | 15%–22% | 12%–17% | 继续训也不恢复 |

长度（字符）变成双峰：短的比 base 更短，长的到一两万字。「写完推导 → boxed → 停」的中间态变少。

| | p50 | p90 | max | 无 boxed 且截断 |
|---|---|---|---|---|
| base | 4404 | 8086 | 28076 | 4/260（1.5%），且这 4 条也都是超长 |
| SFT ckpt-54 | 2040 | 16459 | 25562 | 34/88 无框，32 条判截断 |
| RFT step 20 | 2679（比 base 短） | 19242 | 32554 | 45 条无框里 43 条截断、44 条正文 >12000 字 |

---

## 3. 失败长什么样

1. 开头还在建模；
2. 进入 `### Step N` 或清单；
3. 某一行公式开始原样复制；
4. 顶到上限，没有 `\boxed{}`。

**SFT，题 `110_651`。** 前面写 \(\mu=GM\)，尾巴同一行复制约 8300 字，0 个 boxed：

```text
\frac12 m\left(\frac{v}{2}\right)^2
=\frac12 m\left(\frac{v}{2}\right)^2
...
```

**RFT step 20，题 `152_126`。** 开头已是 Qwen 口吻（`We are given...` / `### Step 1`），约 18000 字后变成 `\text \text} \text{...`。同一题多个样本同一种死法。

Base 也会偶发（约 1.5%）。微调把它放大到 17%–38%。无 boxed 的样本绝大部分同时 `truncated` 且超长——看起来像「没写框」，其实是 **循环写满了额度**。

---

## 4. 为什么 SFT 会让模型写完不停

SFT 优化的是 **下一个 token 的交叉熵**，不是「这一段该不该结束」。结束符 EOS 只在每条样本的最后出现一次。训练时教师强迫（teacher forcing）总把正确答案喂给模型；推理时温度 0.6、最长 8192，前缀稍微走偏，停止概率就会输给「再写一个 token」。

下面三条机制叠在一起。

### 4.1 停止只在一个长度上被监督，推理却在别的长度上采样

Base 自己解题时，中位大约 4400 字再收束——这是预训练 + 指令对齐里学到的「这段话写完了」。

原始 SFT 却把目标改成中位 **1103 字** 的教材体。模型只在「短证明刚写完」这种前缀上见过 EOS。推理时 Qwen 仍倾向写长：过了 1100 字之后的前缀，**SFT 从未教过此处该不该停**。短停法用不上，原生长停法又被全参 3 epoch 冲掉，于是一部分草草收束（答错），一部分滑进循环。

RFT 把长度对齐到 4410 字，EOS 仍然只出现在序列末尾一次。几千个 token 的梯度几乎全在「继续写步骤」，停止信号相对极弱。所以 **补长度不等于补停止**。

### 4.2 训练目标在教「接着写」，有的还教「boxed 不是句号」

交叉熵对「再写一个 `### Step`、再写一行公式」给正梯度，对「这里结束」只在最后一步给一次。

- **短 SFT（离策略）：** 489/554 条是更强模型看过金标后整理的 cascade，292 条带 `hint_gold`。这种文本没有「搜完再提交」的过程，boxed 像填空而不是终点。例 `164_812` 直接写出 \(p(r)=p_o+4S/r\) 再整理公式。推理时没有提示，模型既不会倒推，也不知道该在何处停。
- **长 RFT（在策略）：** 274 条全是 base 答对的自采样，但 **274 条都有 markdown 标题，231 条 emoji，147 条多个 boxed**。`### Step k` 后面最像的还是 `### Step k+1`；中间已经出现过 boxed，等于告诉模型 **框不是终止符**。例 `208_387` 的开头（`### Step 1: Understand the situation`）和崩掉后 heldout 的开头是同一种文风。

自相似格式（步骤标题、方程链、`\text{`）的局部续写概率处处都高。一旦进去，EOS 很难赢过「再来一步」。这是案例 A/B 尾巴死循环的直接原因。

### 4.3 训练看不到「写歪之后」该怎么停

训练时每个位置的下一个词都是标准答案，模型从未在「公式写破、括号没闭上」的前缀上学习。推理时只要一个 token 离开训练流形（坏掉的 LaTeX、多余的 `###`），最高似然续写往往是 **再重复刚才那一段**。SFT 不惩罚这种前缀，因为数据里没有它们。

Base 里循环大约 1.5%，本来被「写完 EOS」压着。SFT/RFT 用几百条尖分布去改续写核：短解冲掉原停止，长 markdown 又放大「继续写 Step」的盆地。稀有失败变成主失败。RFT 从 step 20 崩到 138 都不恢复，说明 **不是欠拟合，是停止统计被改写了**。

因此：换更强标注模型、把 target 改成 3500–5000 字，都还在同一套交叉熵里加「继续写」的样本，不会凭空出现「写完就停」的梯度。

---

## 5. 两条数据路线（同一坑）

| | 原始 SFT | RFT 诊断 |
|---|---|---|
| 文件 | `data/rl/sft_solutions.jsonl`（554） | `data/rl/rft_solutions_dedup.jsonl`（274） |
| 来源 | cascade 489 + API 41 + local 12 + 手写 12 | base 答对的 rollout，每题 1 条 |
| 中位长度 | 1103 字 | 4410 字 |
| 风格 | 无标题、无 emoji、单 boxed | 全是标题；多数 emoji、多 boxed |
| 配方 | 全参、3 epoch、lr \(1\times10^{-5}\)、`max_length` 4096 | LoRA rank 16、lr \(2\times10^{-5}\)、`max_length` 8192 |
| 结果 | 88 题正确率掉到 1.1%，38.6% 无框 | 65 题 part_avg 0.25→0.15，停止指标全面恶化 |

候选池约 1/3 `DISAGREE`（更强模型按题干推不出金标）。门禁 553/554 绿灯只说明格式能过，不说明无提示可解。

长链 503 条里 267 条是同一批带标题的 base 锚点；清洗后只留 205 条。

---

## 6. 若仍要 SFT

默认：**先不 SFT**，outcome-only GRPO 从 base 起（截断无框得 0，这才是终止信号）。若必须热身，目标改成「保住停止」，不要模仿更长的 markdown。

**数据（三条都要满足）：**

1. 无提示做对：去掉 `hint_gold`、cascade、DISAGREE；金标用 `answers_equivalent` 复核。
2. 干净收束：无 `#` 标题、无 emoji、恰好一个 `\boxed{}` 且在文末；boxed 后截断再接 EOS。用 `screen_training_data.py` 的 **sft 模式**（不要 `mode=rft`）和 `clean_longchain_anchors.py`。
3. 长度跟「base 答对且未截断」的中位靠拢（约 1500–4500 字），不要专挑超长 `### Step`。

优先：过 sft 门的 base 正确 rollout（条数变少是预期）→ 无 hint 的 API/人工解。不要把「答对的啰嗦 markdown」当主力。

**配方：** LoRA rank 8–16，1 epoch，lr 从 \(5\times10^{-6}\) 试起，`max_length` 约 4k。每 10 步用 65 题看 `no_boxed` / `repetition` / `part_avg`；相对 base 一旦 `no_boxed > 5%` 或重复率上升就停（loss 仍在降也停）。先 100–200 条试跑，过门禁再加量。

**不要：** hint/cascade 当主体；`mode=rft` 放行标题；为补长度挑超长 rollout；全参多 epoch 只盯 train loss；从已崩 ckpt 接着训；指望换标注模型修好停止。

---

## 7. 通用域探针：不是训崩

「训崩」这里指灾难性遗忘：常识、指令跟随、中英、算术、代码一起没了。用 10 道短题、greedy、最多 80 个新 token、关闭 thinking，对 **base / 全参 SFT ckpt-54 / RFT LoRA step 20** 各跑一遍（脚本 `training/swift/probe_general_qa.py`，原始回答 `logs/general_qa_{base,sft54,rft20}.json`）。

| 题 | 问 | base | SFT-54 | RFT-20 |
|---|---|---|---|---|
| 常识英 | 法国首都，一句话 | The capital of France is Paris. | 同左 | 同左 |
| 常识中 | 中国首都，一句话 | 中国的首都是北京。 | 同左 | 同左 |
| 算术 | 只用数字：17×24 | 408 | 同左 | 同左 |
| 指令 | 只用一个词：晴空颜色 | 蓝色 | 同左 | 同左 |
| 逻辑 | 只答是/否 | 是 | 同左 | 同左 |
| 代码 | 只写 `reverse_list` | `return xs[::-1]` | 同左 | 同左 |
| 翻译 | 光合作用一句英译 | Photosynthesis converts light energy into chemical energy. | 同左 | 同左 |
| 文学 | Hamlet 作者，只写名字 | William Shakespeare | William Shakespeare. | 同 base |
| 格式 | 三个奇数，逗号分隔 | 1, 3, 5 | 1,3,5 | 同 base |
| 中文简述 | ≤30 字解释光合作用 | 植物利用阳光将二氧化碳和水转化为葡萄糖和氧气。 | 同左 | 同左 |

RFT 与 base **10/10 字面相同**。SFT 有两处标点/空格差，内容全对。十条都没有循环、没有 markdown 步骤标题、都在 80 token 内正常结束。

物理 heldout 也不是胡言乱语：SFT 88 条全部用通顺英文开场，44 条在 80–2000 字内写完并 boxed；RFT-20 的 260 条同样全是通顺英文/LaTeX，短样本仍会聊天式说明「题干信息不够」。坏的是 **长解码的停止**，不是权重变成噪声。

因此：SFT/RFT 后 HiPhO 变差，**不能**解释成通用能力丧失。门禁用 `no_boxed` / `repetition` / `part_avg` 盯的是终止，不是 MMLU。

---

## 8. 相关文件

| 文件 | 用途 |
|---|---|
| `score_hipho_predictions.py` | 上表全部指标 |
| `probe_general_qa.py` | 通用短问答探针 |
| `sft_solutions.jsonl` | 原始短 SFT，勿直接再训 |
| `rft_solutions_dedup.jsonl` | 崩掉的 RFT，风格反面教材 |
| `sft_solutions_longchain_clean.jsonl` | 已去标题/emoji，仍需再筛 hint |
| `screen_training_data.py` | 风格门，必须 sft 模式 |
| `docs/outcome_rl_runbook.md` | 从 base 做 outcome-only GRPO |
