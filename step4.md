# 步骤 4：确定七篇全文高风险候选

确定时间：2026-07-16（Asia/Shanghai）

## 候选与入选理由

1. **Leblond 等，SeaRNN（ICLR 2018）**：forced alternative、rollout cost vector、uniform/policy/biased/top-k 子采样和无 rollout 部署几乎复现 CCOD 的通用骨架，是最近单篇威胁。
2. **Chang 等，LOLS（ICML 2015）**：对每个合法动作执行 rollout，并研究 learned/reference/mixed roll-in/roll-out；直接约束“局部机会代价”不能被声称为新概念。
3. **Ross 与 Bagnell，AggreVaTe（2014）**：从 learner-visited state 探索动作并学习专家 cost-to-go，还明确讨论部分信息下均匀动作探索低效，是理论谱系根节点。
4. **Herrmann 与 Schaub（JAIS 2022）**：在 EOS 调度中用 MCTS 产生 state–action value，再回归为快速网络，直接否定“首次在 EOS 将搜索价值蒸馏到无搜索策略”。
5. **Jacquet 等（IWPSS 2025 / arXiv 2024）**：GNN 直接给动态候选打分，PPO 学习并可叠加 PUCT MCTS；是表示、闭环质量与在线搜索的强同域对照。
6. **Qi 等，RILO（2019）**：处理专家与学习者动作空间不同、仅有状态观察和稀疏奖励；用于检验“异构动作”是否已覆盖 CCOD，但其机制最终不是反事实查询。
7. **Mercado-Martínez、Soret 与 Jurado-Navas（ICMLCN 2025）**：单星 AEOS 冲突图、两层 GAT、逐动作 Q 值与 DQN；用于检验“EOS 图动作评分/资源感知”边界。

## 未进入七篇的工作

- DAgger、Deeply AggreVaTeD、Policy Distillation 与 targeted exploration 是重要引用，但其关键作用已由 AggreVaTe/SeaRNN 的全文机制覆盖；在轴向表中作为谱系证据引用。
- EOS-Bench 是数据、约束与评测锚点，不是对 CCOD 机制的新颖性威胁。
- 搜索返回的其他 EOS DRL/启发式工作多数只有题名或摘要，适合 related work 扩展，不足以替代七篇全文证据。

## 证据要求

每篇必须至少记录三类独立证据：问题定义、方法细节、实验结果或作者明确局限。只有摘要相似而没有方法证据的工作不能支撑 Level 2 判断。逐篇证据见 [`step5.md`](./step5.md)。
