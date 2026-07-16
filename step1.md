# 步骤 1：拆解当前 CCOD 新颖性主张

检索时间：2026-07-15（Asia/Shanghai）
复核时间：2026-07-16（Asia/Shanghai）

## 研究对象

本文当前方案是 **CCOD（Conflict-Covered Opportunity-Cost Distillation，冲突覆盖机会代价蒸馏）**，不是此前的 BCRD 双头 regret 方案。研究对象是 EOS-Bench 中由异构求解器最终解规范重放得到的 `solution-induced constructive traces`。这些轨迹只暴露已执行动作，不能直接说明同一状态下其他可行动作的下游机会代价。

截至本轮检索与全文复核结束，正式冻结包仍为**零标签**：100 个状态、1,570 条查询只完成了身份、候选前缀和执行计划冻结，尚未运行任何 $Q_H$，因此下述判断只评价论文主张与既有工作的关系，不包含效果结论。

## 四条新颖性轴

1. **问题轴**：在动态、强约束、候选数变化的 task–satellite–window 动作集中，普通 one-hot 行为克隆把所有未选可行动作视为同类负例，忽略它们对未来窗口、姿态转移和资源可行域的不同影响。
2. **通用机制轴**：强制替代动作并执行有限视域 continuation，得到局部 action cost/value，再把相对次序蒸馏到部署时无搜索的策略。此轴已有 SeaRNN、LOLS、AggreVaTe/AggreVaTeD 等充分先例，不能作为“首次”贡献。
3. **可守机制轴**：在固定查询上限 $B$ 下，不按均匀、当前策略概率或 top-k 取动作，而以 EOS 约束一致的 conflict/resource footprint 分配查询；footprint 描述未来窗口删除、任务优先级质量损失、最后机会、姿态转移余量和资源压力。
4. **证据轴**：查询分配首先必须在穷举 dev 状态上提高容量平衡双尾覆盖 $\operatorname{BTailCov}_{16}$，随后才允许检验 train→dev 排序与完整排程收益；离线覆盖与闭环收益必须同时成立。

## 当前最小主张

> 在相同状态、候选集、continuation、查询上限、模型、损失、种子和训练步数下，EOS conflict/resource footprint 查询分配能否比 Uniform、Policy-sampling、Policy-top-K 与通用特征 farthest-first 更有效地覆盖真实机会代价的两端，并将该额外信息转化为 held-out EOS-Bench instances 上的可行完整排程收益。

## 明确不声称

- 不声称首次使用替代动作 rollout、cost-to-go、局部 regret/cost vector、pairwise ranking 或无搜索蒸馏。
- 不把单头评分器、beam-8 核验、确定性 continuation 或内容寻址缓存单独包装成算法创新。
- 不把“卫星调度使用图网络”作为创新；Herrmann–Schaub、Jacquet、Mercado-Martínez 等已覆盖同域价值学习、GNN 或搜索结合。
- 不把 100-state 与 500-state 工程门槛解释为统计显著性；最终推断只能在后续冻结的 30–60 个独立 instance paper set 上进行。

## 预期威胁

最接近的单篇机制威胁是 **SeaRNN**：它已经具有 forced alternatives、完成式 rollout、局部 cost vector、uniform/policy/top-k 子采样以及测试时无 rollout。CCOD 只有在 EOS footprint 查询分配相对这些公平基线形成可重复的查询效率增益时，才保留方法性 delta。
