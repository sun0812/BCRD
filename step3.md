# 步骤 3：摘要级筛选

筛选时间：2026-07-15 至 2026-07-16（Asia/Shanghai）

## 筛选轴

每条记录按四项判断：①是否为长时序决策；②是否给替代动作产生下游 cost/value；③是否在有限查询预算下选择动作；④是否利用 EOS 冲突/资源结构。摘要或题名无法支持的轴记为“未知”，不据此补造结论。

## 保留候选

| 工作 | 时序决策 | 替代动作下游值 | 查询预算/采样 | EOS 结构 | 摘要级判断 |
|---|---:|---:|---:|---:|---|
| SeaRNN（2018） | 是 | 是 | 是 | 否 | 机制最接近，必须全文核验 |
| LOLS（2015） | 是 | 是 | 部分 | 否 | 局部 rollout cost 与 reference/learner 混合先例 |
| AggreVaTe（2014） | 是 | 是 | 部分信息探索 | 否 | cost-to-go imitation 的基础先例 |
| Herrmann–Schaub（2022） | 是 | 是 | MCTS 模拟预算 | 是 | EOS 中 MCTS-Q 到快速网络的同域先例 |
| Jacquet 等（2025） | 是 | 是/搜索值 | MCTS 预算 | 是 | EOS GNN 策略和在线搜索近邻 |
| RILO（2019） | 是 | 否 | 否 | 否 | 只检验异构动作空间模仿，不是直接威胁 |
| Mercado-Martínez 等（2025） | 是 | DQN-Q | 探索而非离线查询预算 | 是 | EOS 图动作评分近邻，机制不同 |

## 次级相关记录

- EOS 直接相关：Hadj-Salah 等 DRL 调度、Long 等变规模资源感知 DRL、multi-type/multi-satellite scheduling、DQL+ensemble heuristics、DRL-guided ACO、edge-intelligence acquisition/processing、wildfire detection-to-scheduling、safe RL、super-agile constraint programming、EOS-Bench。它们说明同域学习和混合求解已拥挤，但题名/摘要未显示固定预算的 counterfactual footprint selector。
- 主动约束获取边界：Belaid，*Optimizing Earth Observation Schedules under Unknown Operational Constraints: Active Constraint Acquisition*（arXiv:2604.13283，2026）在 EO 场景主动查询 feasibility oracle 以识别未知操作约束。它与 CCOD 都关心昂贵查询，但查询对象不同：前者学习约束可行性，后者计划查询备选动作的 $H$ 步 value；摘要级证据未显示固定 $B$ 的 EOS action footprint、balanced tail coverage 或等预算 closed-loop 比较，因此保留为边界工作而非直接覆盖。
- 模仿学习近邻：Offline Imitation Learning with Variational Counterfactual Reasoning、Greedy-DAgger、targeted exploration、若干机器人/车辆/视觉模仿工作。它们分别覆盖“反事实推断”“减少 rollout”“定向探索”或一般行为克隆，但没有形成当前 CCOD 的 EOS 查询分配协议。
- 查询 3 返回的大部分 conflict graph、resource allocation、diverse selection 记录来自通信、聚类、主动学习或一般组合优化；没有一条能从摘要建立“EOS 候选 footprint 驱动 counterfactual query allocation”的直接重合。

## 排除原则

- “satellite/Earth”只指遥感成像、材料或通信而非排程的记录，排除。
- “schedule”只指学习率或训练日程的记录，排除。
- imperfect-information games 中的 Counterfactual Regret Minimization 与本文 opportunity cost 不同，排除为术语噪声。
- 只有题名且没有摘要/全文的 EOS 算法保留为 related-work 候选，不将其当作机制证据。

摘要级筛选后，七篇高风险工作进入全文深读；EOS-Bench 作为实验协议和领域锚点，不计为方法威胁。

面向 `budgeted counterfactual`、`action-value query selection` 与上述 EOS 组合条件的定向检索没有发现一篇同时包含 selected-only 状态重建、约束 footprint 固定 $B$ 分配、balanced tail coverage 和等预算闭环收益的论文。该结论受 OpenReview/OpenAlex/Semantic Scholar/DBLP 失败与关键词召回限制，只能解释为“当前证据集中未发现”，不能解释为对全部文献空间的穷尽性排除。
