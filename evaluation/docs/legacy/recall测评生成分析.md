# Recall测评生成分析（扩充版）

## 1. 背景与当前结果

当前 recall 测评结果（`results/eval_pipeline_v0412_qwen30b_a3b_instruct/metrics_qwen30b_a3b_instruct.json`）显示：

- recall 数据集 20 条，GT 错误总数 47 条。
- 命中的 GT 错误仅 3 条，错误级召回率为 0.0638。
- 样本级触发率（`pred_has_diagnostic=true`）为 0.8，说明检查器多数样本有输出，但和 GT 对齐很差。

这说明核心矛盾不是“完全没报错”，而是“报错内容和 GT 错误语义不对齐”。

## 2. 现有链路梳理与风险点

### 2.1 GT 生成链路（`scripts/build_physics_eval_sets.py`）

1. 从 recall 样本抽题。
2. 调用强模型生成 `physics_error_examples`。
3. 只要“提取到至少 1 条 error 文本”就记为成功，不做物理相关性校验。
4. 对不规范输出会做 regex/自由文本修复提取，可能保留了格式对但语义偏的条目。

关键风险：

- 生成目标偏“模板完整”而非“证据绑定”。
- 没有要求每条 error 必须绑定题干/学生答案的证据片段。
- 没有做“学科域一致性过滤”和“规则库可映射性过滤”。

### 2.2 Recall 评测链路（`scripts/evaluate_physics_eval_sets.py`）

1. `findings` 来源于：
   - top-down 的 `diagnostics`
   - symbolic audit 的 `experience_code_checks` 中 fail 项
2. GT 与 findings 匹配方式：
   - 优先 LLM 语义覆盖判断（若可用）
   - 否则退化到关键词重叠匹配

关键风险：

- `findings` 里混有“跨题型/跨领域规则提示”，噪声较大。
- 匹配只输出覆盖布尔值，不落地“哪条 finding 对应哪条 GT”的可审计证据。

## 3. 使用的 GT 生成 Prompt（现状）

```python
system_prompt = (
    "You are a strict physics evaluator for building a recall benchmark for a rule checker. "
    "Extract concise, physics-rule-grounded errors from the student answer. "
    "Each error must represent a GENERALIZABLE experience rule, not a one-off case detail. "
    "Each error should be directly mappable to a checkable rule condition. "
    "Use English only. No rubric references."
)
user_prompt = (
    f"Question:\n{question}\n\n"
    f"Student answer:\n{prediction}\n\n"
    f"Reference answer:\n{answer}\n\n"
    "Return JSON only with this schema:\n"
    "{\n"
    "  \"errors\": [\n"
    "    {\"error\": \"In <condition>, <expression/result> should satisfy <rule>, but the answer <violation>.\"}\n"
    "  ]\n"
    "}\n\n"
    "Style requirements for each item:\n"
    "1) Must use a 3-part rule form: CONDITION -> SHOULD RULE -> VIOLATION.\n"
    "2) CONDITION should be a reusable scenario (e.g., uniform acceleration, conservation law, boundary condition, symmetry), not a sample-specific sentence.\n"
    "3) SHOULD RULE should be a general physics relation/constraint (formula family, sign/monotonicity, conservation, dimensional consistency).\n"
    "4) VIOLATION should describe how the answer conflicts with that rule.\n"
    "5) Keep one rule violation per item.\n"
    "6) Avoid over-specific details: do NOT depend on sample id, exact numeric substitution, or one-time constants unless strictly necessary. Prefer variable-based wording.\n"
    "7) Prefer naming the rule family when possible (e.g., Newton's second law, energy conservation, continuity, boundary matching, unit consistency).\n"
    "8) Each item should be reusable as an experience rule template for similar problems.\n\n"
    "Examples (do not copy, adapt to this sample):\n"
    "- In uniform acceleration, displacement should satisfy s=v0 t+1/2 a t^2, but the answer uses a constant-velocity form s=vt.\n"
    "- For ideal gas at fixed n and T, pressure-volume should satisfy PV=const, but the answer treats P as independent of V.\n"
    "- Under valid boundary conditions, the final expression should satisfy continuity at the interface, but the answer violates that continuity condition.\n\n"
    f"Output up to {max_errors} errors."
)
```

问题不在格式要求本身，而在于缺少“先找错，再抽象”的强约束与可验证约束。

## 4. 逐案例分析（扩充版）

本节对每个案例都给出：

1. 题目大意
2. 生成错误列表完整英文原文
3. 对应中文翻译
4. 逐条核查（该错误是否真实存在、表述是否准确）
5. 与检查器输出及 recall 的对照

### 4.1 Case A：189_957（高质量正例，完全命中）

#### 题目大意

各向异性摩擦起动问题：在 $F_x=-\frac{|N|}{|v|}\mu_x v_x,\;F_y=-\frac{|N|}{|v|}\mu_y v_y$ 的模型下，求受力随时间线性增长时的起动时刻。

#### 生成错误列表（完整英文原文）

1. In anisotropic friction models where force components are coupled by velocity magnitude (e.g., $F_i \propto v_i/|v|$), the threshold for motion should satisfy an elliptical boundary condition $(\frac{F_x}{\mu_x})^2 + (\frac{F_y}{\mu_y})^2 = N^2$, but the answer treats the $x$ and $y$ thresholds as independent scalar limits.
2. For a body at rest subjected to an external force vector, the onset of motion should occur when the applied force vector reaches the boundary of the static friction region, but the answer uses a logic based on the minimum time for individual components to reach independent maximums.

#### 对应中文翻译

1. 在各向异性摩擦中，摩擦分量通过速度模长耦合，起动阈值应满足椭圆边界；但答案把 x、y 方向阈值当成相互独立的标量比较。
2. 对静止物体受外力起动，应由“外力向量达到静摩擦域边界”判定；但答案使用了“分量分别先到阈值”的最小时间逻辑。

#### 逐条核查

- 错误 1：确实存在，表述准确性高。学生解答明确使用了 $F_x=\mu_xN$ 或 $F_y=\mu_yN$ 的独立触发条件，和各向异性耦合边界不一致。
- 错误 2：确实存在，表述准确性高。学生将向量边界问题降为分量最小值判据，物理判据层面有偏差。

#### 检查器对照与 recall

- metrics: `gt_error_count=2`, `pred_finding_count=5`, `matched_error_count=2`, `sample_error_recall=1.0`。
- top-down 输出核心诊断与 GT 高一致：均指向“独立分量阈值判据不成立”。

结论：这是“GT 质量高 + 检查器触发有效 + 匹配语义一致”的标准正例。

---

### 4.2 Case B：157_809（高噪声反例，零触发）

#### 题目大意

杆上质点在 $\omega=0$ 与 $\omega>0$ 条件下的静止判据，本质是受力分解和摩擦平衡。

#### 生成错误列表（完整英文原文）

1. In this case, the solution should satisfy: INCLINED MEMBER FORCE RESOLUTION -> THE COMPONENT OF A HORIZONTAL FORCE PARALLEL TO THE MEMBER SHOULD BE CALCULATED AS F * COS(THETA) -> THE COMPONENT WAS INCORRECTLY CALCULATED USING THE WRONG TRIGONOMETRIC FUNCTION OR ANGLE REFERENCE., but the answer does not satisfy this requirement.
2. In this case, the solution should satisfy: STATIC EQUILIBRIUM OF A RIGID BODY -> THE ALGEBRAIC SUM OF ALL EXTERNAL MOMENTS ABOUT ANY ARBITRARY POINT MUST EQUAL ZERO -> THE MOMENT SUMMATION WAS INCOMPLETE OR NON-ZERO., but the answer does not satisfy this requirement.
3. In this case, the solution should satisfy: CONSERVATION OF ENERGY IN A CLOSED SYSTEM -> TOTAL MECHANICAL ENERGY MUST REMAIN CONSTANT IN THE ABSENCE OF NON-CONSERVATIVE FORCES -> ENERGY WAS NOT CONSERVED WITHOUT AN IDENTIFIED EXTERNAL WORK SOURCE., but the answer does not satisfy this requirement.
4. In this case, the solution should satisfy: FRICTION ON A SLIDING BLOCK -> THE KINETIC FRICTION FORCE SHOULD BE CALCULATED AS THE PRODUCT OF THE KINETIC COEFFICIENT AND THE NORMAL FORCE -> THE FRICTION FORCE WAS CALCULATED USING TOTAL WEIGHT INSTEAD OF THE NORMAL COMPONENT., but the answer does not satisfy this requirement.

#### 对应中文翻译

1. 斜杆受力分解应按正确角度计算平行分量；但答案使用了错误三角关系或角度参照。
2. 刚体静平衡应满足任意点力矩和为零；但答案力矩求和不完整或不为零。
3. 无非保守力时总机械能应守恒；但答案出现了无外功来源的能量不守恒。
4. 滑块动摩擦应为 $\mu N$；但答案把总重力直接当摩擦来源而非法向分量。

#### 逐条核查

- 错误 1：可能部分相关，但证据不足。学生在旋转部分受力分解确有可疑项，但 GT 过于武断。
- 错误 2：基本不准确。该题主要是质点沿杆平衡，不是典型“刚体任意点力矩平衡”主线。
- 错误 3：不准确。学生解法未以能量法为主，直接给出“能量不守恒”缺乏题内证据。
- 错误 4：不准确。学生文本并未明确“用总重力替代法向分量算摩擦”的直接证据。

#### 检查器对照与 recall

- metrics: `gt_error_count=4`, `pred_finding_count=0`, `matched_error_count=0`。
- top-down: 主题被路由到 `Error Analysis (Absolute, Relative, Percentage Errors)`，`diagnostics=[]`。

结论：该例的问题不仅是“检查器没触发”，更是“GT 语义噪声高、可验证性弱”。

---

### 4.3 Case C：194_10（有触发但严重错配）

#### 题目大意

双体引力波四极矩辐射功率系数 $\xi$ 的推导。

#### 生成错误列表（完整英文原文）

1. In this case, the solution should satisfy: Calculation of a binary system quadrupole moment tensor -> The components must satisfy the trace-free condition for gravitational radiation modeling -> The resulting tensor yields a non-zero trace., but the answer does not satisfy this requirement.
2. In this case, the solution should satisfy: Analysis of isolated system thermodynamic processes -> The total entropy must strictly remain constant or increase over time -> The calculation results in a net decrease in total entropy., but the answer does not satisfy this requirement.
3. In this case, the solution should satisfy: Modeling relativistic particle kinematics -> The magnitude of the velocity vector must not exceed the speed of light -> The transformed coordinates result in a superluminal velocity or imaginary Lorentz factor., but the answer does not satisfy this requirement.
4. In this case, the solution should satisfy: Application of Kirchhoff’s Loop Rule to a closed circuit -> The algebraic sum of all potential differences around any closed loop must equal zero -> The sum of voltage drops and rises fails to return to the starting potential., but the answer does not satisfy this requirement.

#### 对应中文翻译

1. 双星四极矩张量应满足引力辐射建模一致性（如无迹/规范一致）；但答案给出不一致张量结果。
2. 孤立系统总熵应不减；但答案导致总熵下降。
3. 相对论运动学要求速度不超光速；但答案出现超光速或虚数洛伦兹因子。
4. 基尔霍夫回路定律要求回路电势和为零；但答案未闭合。

#### 逐条核查

- 错误 1：有一定相关性，但表述不够精确。该题确有四极矩系数/推导错误，但“trace-free”并非唯一核心失误点。
- 错误 2：不成立。题目与热力学熵无关。
- 错误 3：基本不成立。学生并未进行超光速推导。
- 错误 4：完全不成立。题目与电路无关。

#### 检查器对照与 recall

- metrics: `gt_error_count=4`, `pred_finding_count=6`, `matched_error_count=0`。
- top-down/symbolic audit 均出现跨域噪声（如 Cherenkov、其他非本题规则触发），导致和 GT 交集极低。

结论：该例属于“GT 过拟合模板 + 检查器跨域噪声”双向错位。

---

### 4.4 Case D：97_588（部分命中，值得保留）

#### 题目大意

单摆中竖直速度分量最大时的 $\cos\theta$。

#### 生成错误列表（完整英文原文）

1. In solving quadratic equations for physical parameters, the selected root should satisfy the physical domain constraints of the system (e.g., cos θ ≥ cos θ₀), but the answer selects the negative root which falls outside the valid range of motion.
2. In optimization problems for continuous functions on a closed interval, the identified maximum should correspond to the highest value among critical points and endpoints, but the answer concludes the maximum occurs at the lowest point (θ=0) where the vertical velocity is zero.
3. In maximizing a product of two dependent variables (v and sin θ), the extremum should be found by differentiating the entire product expression, but the answer incorrectly assumes the maximum occurs solely where one factor (sin θ) is maximized.

#### 对应中文翻译

1. 解物理二次方程时应选满足物理取值域的根（如 $\cos\theta \ge \cos\theta_0$）；但答案选了超出有效运动区间的负根。
2. 闭区间极值应比较临界点与端点；但答案断言最大值在 $\theta=0$，此时竖直速度分量为 0。
3. 对 $v\sin\theta$ 这类耦合乘积求最大值，应对整体求极值；但答案把最大化简化为只看单一因子。

#### 逐条核查

- 错误 1：基本成立。学生确有根选择与物理区间矛盾问题。
- 错误 2：成立。其文中反复给出“$\cos\theta=1$”与“竖直速度最大”并存的自相矛盾结论。
- 错误 3：成立。学生确实在中段采用过“只看 sinθ 最大”的错误直觉。

#### 检查器对照与 recall

- metrics: `gt_error_count=3`, `pred_finding_count=6`, `matched_error_count=1`, `sample_error_recall=0.3333`。
- top-down 中至少有 1 条直接命中“把竖直速度最大点误判为 θ=0”的核心错误；其余诊断含一定规则体系噪声。

结论：该例是“GT 质量较高但 checker 对齐不完整”的中间样本，可作为提升 recall 的重点对象。

---

### 4.5 Case E：198_283（GT 截断异常）

#### 题目大意

厕纸卷在杆上静止条件，核心是力矩平衡与摩擦临界条件。

#### 生成错误列表（完整英文原文）

1. "error": "In torque balance for a roll with hanging material, the lever arm for the hanging weight should be the radius of the roll ($R_o$), but the answer

#### 对应中文翻译

1. （文本被截断）在有悬挂纸带的卷筒力矩平衡中，悬挂重力的力臂应与卷筒半径 $R_o$ 相关，但答案……

#### 逐条核查

- 形式层面：该 GT 条目是“截断字符串”，语义不闭合，按数据质量标准应判定为无效标注。
- 实质层面：学生解答确有“$\tau_{friction}=\mu N\times r$（用杆半径当力臂）”的问题，潜在错误方向是对的，但 GT 无法作为可匹配监督信号。

#### 检查器对照与 recall

- metrics: `gt_error_count=1`, `pred_finding_count=5`, `matched_error_count=0`。
- top-down 中存在与力臂问题高度相关的诊断（指出应使用 $R_o$ 而非 $r$），但因 GT 截断，匹配难以成立。

结论：这是“标注质量直接破坏 recall 评测有效性”的典型案例。

---

### 4.6 Case F：264_81（GT准确但 checker 噪声高）

#### 题目大意

双星系统质量估计，应基于题目给定观测量推导，不应引入拍脑袋参数。

#### 生成错误列表（完整英文原文）

1. In quantitative physics problems, numerical results should be derived from the specific parameters provided in the problem context, but the answer substitutes arbitrary values for the orbital radius and period.

#### 对应中文翻译

1. 在定量物理题中，数值结果应由题目给定参数推导；但答案自行代入了任意轨道半径和周期。

#### 逐条核查

- 错误 1：成立且准确。学生明确写出“assume typical values: d=1e12 m, T=1e7 s”，属于无依据代参。

#### 检查器对照与 recall

- metrics: `gt_error_count=1`, `pred_finding_count=6`, `matched_error_count=0`。
- top-down 有 1 条相关诊断（Kepler 公式应用问题），但也混入多条明显跨题型规则，导致语义匹配信号被稀释。

结论：这是“GT 本身质量尚可，但 checker 输出噪声/偏航导致不命中”的代表。

---

### 4.7 Case G：83_626（生成失败导致 GT 为空）

#### 题目大意

电子束在长螺线管中聚焦，求电流。学生答案约 6.84A，参考约 6.75A，数值接近。

#### 生成错误列表

- 空列表：`physics_error_examples=[]`（`strong_model_failed`，`no_parseable_errors_from_model_output`）。

#### 说明

- 这是 recall 集中的“无 GT 错误样本”，会削弱 recall 评测信号纯度。
- metrics 显示 `gt_error_count=0`, `pred_finding_count=0`，该样本对 recall 无贡献但占用预算。

结论：应在构建 recall 集时加入“GT 非空且可解析”的硬校验。

## 5. 归纳：当前 recall GT 生成的潜在问题

### 5.1 生成目标偏模板，缺少证据锚定

当前 prompt 强调 CONDITION->RULE->VIOLATION 结构，但未强制“每条错误必须引用学生答案中的具体证据片段”。

后果：

- 文本看起来像规则，但可能是“合理猜测”而非“该题确实犯错”。

### 5.2 过度生成与泛化噪声

`max_errors` 默认 4，模型容易补齐到多条，带来跨域错误（热力学、电路、相对论等无关条目）。

### 5.3 缺少“规则库可映射”约束

GT 文本虽自然语言可读，但没有硬约束要求可映射到已有 rule family/rule id，导致与 checker 的可检测空间不一致。

### 5.4 评测匹配可解释性不足

当前指标只给 `matched_error_count`，不输出“GT_i 匹配到 finding_j 的证据”，导致难以定位到底是 GT 问题、checker 问题，还是 matcher 问题。

### 5.5 GT 文本截断/损坏未被拦截

在 198_283 等样本中，`physics_error_examples.error` 出现未闭合字符串，说明生成后缺少结构完整性与语义完整性校验。

后果：

- 即使“错误方向”可能合理，评测也无法稳定匹配，直接拉低 recall 可解释性。

### 5.6 Recall 集混入 GT 为空样本

在 83_626 中，`expected_has_physics_error=true` 但 `physics_error_examples=[]`，属于“应有错误但标注缺失”。

后果：

- 样本占用预算却不提供有效监督信号，造成 recall 评测稀释。

## 6. 解决方案（按落地优先级）

### 6.1 先做“可验证 GT”最小改造（高优先）

将 GT schema 从单字段 `error` 扩为：

```json
{
  "error": "...",
  "rule_family": "...",
  "evidence_quote": "来自学生答案的原文片段",
  "why_wrong": "该片段为何违反规则",
  "confidence": 0.0
}
```

硬性校验规则：

1. `evidence_quote` 必须是学生答案子串。
2. `rule_family` 必须能检索到 catalog 中候选规则（阈值可设）。
3. 若不满足，丢弃该条并重生成。

### 6.2 生成流程改成“两阶段”

阶段 A（事实抽取）：仅抽“学生答案中的明确错误断言”，不做泛化。

阶段 B（规则抽象）：把阶段 A 每条错误映射成可复用规则模板，且保留证据绑定。

这样可避免“先想规则再硬套错误”的倒置过程。

### 6.3 增加跨域噪声过滤

在 GT 生成后做轻量分类器或词表过滤，若题目主题为引力波/力学，出现电路/Kirchhoff/Cherenkov 等低相关词则降权或重生成。

### 6.4 降低过度生成

把固定 `max_errors=4` 改为按证据数量自适应（如 1-2 条为主）。

建议策略：

- 默认最多 2 条。
- 仅当存在 3 个以上互不重叠证据片段时才允许 >=3 条。

### 6.5 评测阶段增加可解释日志

在 `evaluate_physics_eval_sets.py` 输出中新增：

- `gt_to_finding_alignment`: 每条 GT 对应的最相近 finding 和匹配理由。
- `match_method`: `llm_semantic` 或 `keyword`。
- `match_confidence`。

这样可以快速定位失败来源（GT 偏、checker 偏、matcher 偏）。

### 6.6 增加 GT 质量闸门（上线前强校验）

在写入 recall 数据前新增 hard checks：

1. `physics_error_examples` 非空。
2. 每条 `error` 必须是完整句（最小长度、末尾标点、无未闭合引号）。
3. JSON 严格可解析且字段齐全。
4. 不通过则重试生成；重试后仍失败则剔除样本并补抽。

## 7. 建议的短周期实验

### 7.1 A/B 对照（同一批 20 条 recall 样本）

- A 组：现有生成策略。
- B 组：两阶段 + 证据锚定 + 规则库映射过滤。

### 7.2 重点观察指标

1. `valid_gt_rate`：GT 中可映射规则占比。
2. `noise_gt_rate`：跨域无关错误占比。
3. `recall_error_level`：错误级召回。
4. `gt_checker_overlap_rate`：GT 与 checker 同 rule family 比例。

预期：即使 checker 本身不改，B 组也应显著提升“可匹配性”和可解释性。

## 8. 一句话总结

当前 recall 低的主因并非“检查器完全没能力报错”，而是“GT 生成缺乏证据约束与规则映射约束，导致语义漂移”；先把 GT 做成可验证、可映射、可审计的形式，才能让 recall 指标真正反映检查器能力。
