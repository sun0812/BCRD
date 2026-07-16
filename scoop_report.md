# CCOD 新颖性核验报告

报告日期：2026-07-16（Asia/Shanghai）

对象：`方案_v2.md` 中的 CCOD（Conflict-Covered Opportunity-Cost Distillation）
证据状态：零正式标签；100-state / 1,570-query 计划已冻结但尚未执行 $Q_H$

## 执行结论

**Scoop 等级：Level 2 — High Overlap。**

CCOD 的通用学习骨架并不新：AggreVaTe 已用 action cost-to-go 训练 imitation policy；LOLS 已在合法替代动作上执行 rollout 并学习局部代价；SeaRNN 已把 forced alternatives、完成式 rollout、局部 cost vector、uniform/policy/top-k 子采样和测试时无 rollout 组合起来。Herrmann–Schaub 又已在 EOS 调度中把 MCTS state–action value 回归到快速无搜索网络。最近的单篇机制威胁是 **SeaRNN**。

本轮没有发现一篇论文同时覆盖以下组合：

1. 从异构 EOS solver 最终解构造可哈希、可精确重放的 solution-induced traces；
2. 在动态 task–satellite–window 候选集上按真实约束的 conflict/resource footprint 分配固定上限 $B$ 的 continuation queries；
3. 以 exhaustive dev 的容量平衡双尾覆盖和 train→dev 完整 schedule 收益共同检验查询效率。

因此方案不是“完全被 scoop”，但其可守部分只剩 **EOS-specific footprint query allocation 及其严格证伪协议**。该部分仍是待验证假设，不能在零标签阶段写成已实现效果。

## 1. 当前主张的冻结版本

### 1.1 问题

EOS-Bench 的可行动作由任务、卫星和窗口共同确定，动作会通过时间冲突、姿态转移、窗口稀缺与星上资源改变未来可行域。异构求解器只提供不同质量的最终可行 assignment sets；规范重放得到的构造轨迹只标出实际动作，普通 one-hot BC 无法表达未选可行动作之间的机会代价差异。

### 1.2 方法

CCOD 对冻结状态执行确定性、有界视域 continuation，得到 $Q_H(s,a)$；每个状态最多查询 $B$ 个动作。它以未来窗口删除、受影响任务优先级质量、最后可行机会、转移 slack 和资源压力组成 EOS footprint，并在高风险与多样性之间分配查询。训练使用已查询动作之间的相对 order，部署时单头 scorer 一次性给 capped feasible set 打分，不运行 rollout/MCTS/solver。

### 1.3 唯一可守因果链

$$
\text{约束一致 footprint}
\longrightarrow
\text{相同预算下更好的双尾覆盖}
\longrightarrow
\text{更好的 held-out 排序}
\longrightarrow
\text{更好的可行完整排程}.
$$

任一箭头失败，都必须收缩或撤回方法主张。

## 2. 检索协议与错误

2026-07-15 使用 paper-search 运行三组公开查询，年份 2000–2026：

1. `Earth observation satellite scheduling learning action ranking solver schedules`；
2. `budgeted counterfactual cost-to-go imitation learning large action space`；
3. `conflict graph resource footprint diverse query selection combinatorial scheduling`。

arXiv 与 Crossref 每组各返回 10 条。查询 1 召回 Jacquet、Mercado-Martínez、Hadj-Salah、EOS-Bench、edge intelligence、wildfire scheduling、safe RL、DQL/heuristic 等同域工作；查询 2 召回 RILO、state-space imitation、Offline Imitation Learning with Variational Counterfactual Reasoning、Budgeted Batch Mode Active Learning 及大量机器人/因果/CFR 噪声；查询 3 主要返回跨域 conflict graph、资源分配、多样性选择、active learning 与组合调度，没有直接 EOS footprint selector。

定向补充检索发现 Belaid 2026 的 *Optimizing Earth Observation Schedules under Unknown Operational Constraints: Active Constraint Acquisition*（arXiv:2604.13283）。摘要显示它在 EO 调度中主动查询 feasibility oracle，以识别未知操作约束。它证明 EO 领域已有“主动选择昂贵查询”的相邻方向，但查询目标是约束可行性，而不是 CCOD 在已重建状态上对备选动作执行 $H$ 步 value query；摘要也未显示固定 $B$ 的 action footprint allocation、balanced tail coverage 或同预算闭环蒸馏收益，因此保守归为边界工作而非直接覆盖。

来源错误完整保留：

- OpenReview：三组均 HTTP 400，`Invalid username or password`；
- OpenAlex：三组均 HTTP 401，`Unauthorized`；
- Semantic Scholar：三组均 HTTP 403，`Forbidden`；
- DBLP：查询 1 为 0 条，查询 2、3 为 HTTP 503；
- 仓库 `.venv` 首次依赖不全，后改用 `/opt/homebrew/bin/python3` 完成运行。

完整逐查询高层结果见 [`allinone.md`](./allinone.md)。鉴于三类主 API 失败，本报告不能声称“穷尽了所有 2000–2026 论文”；结论依靠 arXiv/Crossref 召回、经典谱系补召回及七篇全文交叉验证。

本轮针对 budgeted counterfactual/action-value query selection 的定向检索没有发现一篇同时具有 EOS selected-only 状态重建、约束 footprint 固定 $B$ 分配、balanced tail coverage 和等预算 closed-loop 收益的论文。这只表示当前可访问证据中未发现，不构成穷尽性新颖性证明。

## 3. 七篇全文证据

### 3.1 SeaRNN（Leblond 等，ICLR 2018）

- **问题**：MLE 只推高 ground-truth token，不能利用结构化终局 loss 区分不同错误，也存在 train/test exposure mismatch。
- **方法**：roll-in 后在指定位置强制替代 token，以 greedy/beam 完成序列并得到 action cost vector；用最小 cost 的 log-loss 或 cost-derived soft target 的 KL 训练。
- **预算机制**：为降低 $|A|T$ 次 rollout，明确比较 uniform、current-policy sampling、biased policy sampling 和 top-k，并总是纳入 ground truth。
- **实验/局限**：OCR、拼写纠错和 IWSLT14 翻译；子采样可行但最佳 selector 随 loss/任务变化，大量前向 rollout 仍是瓶颈。
- **威胁**：覆盖 CCOD 几乎全部通用骨架；没有 EOS footprint 和双尾/闭环联合证据。

### 3.2 LOLS（Chang 等，ICML 2015）

- **问题**：reference policy 次优时，学习到“接近 teacher”不等于得到好策略。
- **方法**：在访问状态对合法动作做一步偏离，用 learned/reference/mixed rollout 到终局，形成局部动作 cost；保证兼顾 reference regret 与自身一步偏离 regret。
- **实验/局限**：cost-sensitive classification、POS 与 dependency parsing；弱 reference 下 learned roll-in + mixture rollout 更好，但只保证局部最优，rollout choice 会改变 target。
- **威胁**：次优 solver、替代动作相对代价和 mixed continuation 的一般思想均已有先例。

### 3.3 AggreVaTe（Ross 与 Bagnell，2014）

- **问题**：0/1 imitation loss 不表达错误后果，learner 又会访问不同状态分布。
- **方法**：在 learner trajectory 的随机时刻探索动作，由专家完成余下轨迹并提供 cost-to-go；聚合 $(s,t,a,Q)$ 后做 cost-sensitive learning/Q regression。
- **理论/局限**：给出 no-regret reduction；部分信息下明确称 uniform action exploration 低效并建议 contextual bandit。专家可能对受限策略类过度乐观，每个 Q 样本可能需要整轨 rollout。
- **威胁**：有限 cost-to-go 查询和“更聪明地选查询”都不能作为新概念。

### 3.4 Herrmann–Schaub（JAIS 2022）

- **问题**：Earth-observing satellite 的 charging/downlink/imaging/desaturation mode scheduling 与星上快速重规划。
- **方法**：heuristic/random-rollout MCTS 产生 state–action values，神经网络回归后直接选动作。
- **实验/局限**：MCTS 接近 GA，value network 与 MCTS 指标相当并报告约六个数量级加速；但任务是 target-agnostic mode control、动作小，不是 EOS-Bench target-window 排程，GA 对比只检查部分初始条件。
- **威胁**：同域“搜索价值 → 无搜索网络”已被直接覆盖。

### 3.5 Jacquet 等（IWPSS 2025）

- **问题**：单颗敏捷 EOS 在窗口/机动冲突下选择 acquisitions 最大化 utility。
- **方法**：动态图 GNN 产生候选 logits/value，masked PPO 学习，部署可贪心或叠加 PUCT MCTS。
- **实验/局限**：639 个约百规模训练、27 个同规模未见测试，并报告更大规模外推；最大条目可低于 greedy。MCTS 能提高网络结果，但在高 branching factor 下深度浅、耗时显著。当前 chronological insertion 和有限层消息传递限制 foresight。
- **威胁**：EOS 图策略、候选打分与 online search 已有强对照；CCOD 必须靠离线 query efficiency 与低在线时延区分。

### 3.6 RILO（Qi 等，2019）

- **问题**：专家/学习者动作空间不同、只有 state-only expert observations、奖励稀疏。
- **方法**：state-pair GAIL discriminator 与 success-adaptive self-exploration，自动减弱 imitation reward。
- **实验/局限**：全/部分观测 grid worlds、不同动作组合和 ViZDoom；依赖环境交互与对抗 reward，没有显式 forced alternative、局部 $Q_H$、查询预算或 EOS 约束。
- **威胁**：覆盖“异构动作使模仿困难”的 framing，不覆盖 CCOD 机制。

### 3.7 Mercado-Martínez 等（ICMLCN 2025）

- **问题**：单颗 AEOS 联合选择目标与拍摄时刻，并考虑云、湍流、分辨率、能量和存储。
- **方法**：target–time 动作构成有向冲突图，两层 GAT + FC 输出候选 Q，使用 $\epsilon$-greedy DQN。
- **实验/局限**：40/60/80/100 targets、每目标三个时刻、5,000 episodes、每设置 200 tests；报告废片与姿态能耗浪费明显降低，但使用合成单星环境和 A100，多星通信留作未来工作。
- **威胁**：EOS conflict graph、GAT 与候选长期 Q 已有先例；不覆盖 solver traces 或固定离线查询 selector。

逐篇更完整的证据链见 [`step5.md`](./step5.md)。

## 4. 机制重合矩阵

| 机制 | AggreVaTe | LOLS | SeaRNN | Herrmann | Jacquet | Mercado | CCOD 计划 |
|---|---:|---:|---:|---:|---:|---:|---:|
| cost-to-go / action value | 是 | 是 | 是 | 是 | 是 | 是 | 是 |
| forced alternative + completion | 部分 | 是 | 是 | MCTS | MCTS | 环境探索 | 是 |
| 固定部分动作查询 | 是/部分信息 | 通常全动作 | 是 | 模拟预算 | 搜索预算 | 否 | 是 |
| uniform/policy/top-k 对照 | 非完全 | 否 | 是 | 否 | PUCT | $\epsilon$-greedy | 必须 |
| EOS 冲突/资源状态 | 否 | 否 | 否 | 是 | 是 | 是 | 是 |
| EOS footprint 驱动查询分配 | 否 | 否 | 否 | 否 | 否 | 否 | 待验证 |
| 无搜索部署 | 是 | 是 | 是 | 是 | 可选 | 是 | 计划是 |
| 双尾覆盖 + train→dev 闭环 | 否 | 否 | 否 | 否 | 否 | 否 | 待验证 |

## 5. 最近威胁与精确 delta

### 最近威胁

SeaRNN 是最近单篇威胁，因为它不是只共享“rollout”一个词，而是共享完整的工程结构：替代动作、完成式 cost、局部 cost-sensitive target、动作子采样、uniform/policy/top-k 与测试时无 rollout。

### 条件性 delta

> 不同于 SeaRNN 在 token 空间以均匀或模型概率抽取替代动作，CCOD 在精确重放的 EOS schedule states 上，用未来窗口冲突和星上资源 footprint 分配固定上限的 continuation queries；只有这种分配在公平设置中同时提高 $\operatorname{BTailCov}_{16}$ 和 held-out 完整排程收益时，才构成可辩护的领域方法贡献。

“solution-induced traces”解决的是数据语义与可复现性；“footprint”解决的是查询分配；“BTailCov + closed-loop”解决的是证据。三者组合可以形成论文，但单独任何一项都不足以支撑通用 ML 新颖性。

## 6. 证伪门槛

### 6.1 100-state 完整性与三态语义

- 100 states / 1,570 queries 全成功才为 `complete`；只有此状态才计算 signal gate。
- timeout、RSS、worker/process、attempt exhaustion、interrupt 为 `incomplete + not_evaluated + method_decision=null`。
- identity/hash/cache/frozen/runner drift 为粘性 `invalid + not_evaluated + method_decision=null`，修复后换新 run directory。
- `incomplete`/`invalid` 是执行证据，不是方法 No-Go；只有 `complete` 才能产生 `pass/go` 或 `fail/no_go`。

### 6.2 信号与 continuation

- signal-eligible 至少 80/100；Type-7 spread $\ge0.01$ 总通过率至少 60%，两个 dev instance 各至少 50%。
- 10 个固定状态的 greedy–beam-8 median Spearman 至少 0.70；undefined 即失败。
- 至少 20 个 $17\le C_s\le128$ 状态全动作穷举，为 selector 提供 ground truth。

### 6.3 selector

令 $q_s=\max(1,\lceil0.1C_s\rceil)$，$m_s=\min(q_s,\lfloor |U_B(s)|/2\rfloor)$，容量平衡双尾覆盖为

$$
\operatorname{BTailCov}_{B}(s)=\frac12\left[
\frac{\min\{|U_B\cap\operatorname{Best}_{q_s}|,m_s\}}{m_s}+
\frac{\min\{|U_B\cap\operatorname{Worst}_{q_s}|,m_s\}}{m_s}
\right].
$$

只在 exhaustive dev states 使用。CCOD 的 mean $\operatorname{BTailCov}_{16}$ 必须比最佳 Uniform/Policy-sampling/Policy-top-K/generic-farthest-first 至少 +0.10，且两个 dev instance 的均值差都严格为正。两个 clusters 阶段禁止 CI 和显著性语言。

### 6.4 train→dev 闭环

- 500 个训练状态只来自 train instances；dev 只用于 selector、model selection 和闭环验证。
- pairwise AUC $\ge0.55$，Spearman $\ge0.10$，undefined 即失败。
- 两个 dev instance 的 $F_\omega$ delta 均非负，均值严格大于 0，feasibility 100%。
- footprint permutation 应使覆盖与收益回落到 uniform；否则撤回 footprint 因果解释。

### 6.5 最终证据

500-state 通过后才冻结 3 个规模档、30–60 独立 instances 的 paper set。最终 5 seeds 的 CCOD 相对最佳 sampled cost-vector baseline 的 paired $F_\omega$ difference 95% CI 必须大于 0，feasibility 不降，online RT 不超过 BC 的 1.2 倍。

## 7. 写作建议

### 可以写成主要贡献

1. 异构 EOS 最终解到可验证 solution-induced traces 的规范协议；
2. EOS constraint-consistent conflict/resource footprint query allocation；
3. 在相同查询成本下联合报告双尾覆盖、排名、完整 schedule、可行性、在线时延和离线标签成本。

### 不能写成主要贡献

- forced alternative rollout；
- cost-to-go / regret / Q distillation；
- uniform、policy 或 top-k sampling；
- GNN/集合编码器、单头 scorer、pairwise loss；
- 测试时不运行搜索。

### 阴性结果退路

若 signal 存在但 footprint selector 不占优，转为 EOS counterfactual labeling benchmark / diagnostic study；若 100-state 机会代价 spread 本身不足，立即停止该路线；若只有离线覆盖提高而完整 schedule 不改善，不写 CCOD 方法论文。

## 8. 最终建议

- 航天调度、智能优化、应用 AI venue：**Conditional Go**，前提是预注册 selector 与闭环门槛通过。
- 通用 ML venue：当前证据为 **No-Go**；只有 footprint 产生强、可迁移、可解释且经负对照确认的 query-efficiency 增益，才重新评估。
- 计算资源：继续用 Mac 完成 100-state 与 500-state gates；在本地证据通过前不租 3090。

## 参考文献

1. S. Ross and J. A. Bagnell, “Reinforcement and Imitation Learning via Interactive No-Regret Learning,” 2014.
2. K.-W. Chang et al., “Learning to Search Better than Your Teacher,” ICML, 2015.
3. W. Sun et al., “Deeply AggreVaTeD,” ICML, 2017.
4. R. Leblond et al., “SeaRNN: Training RNNs with Global-Local Losses,” ICLR, 2018.
5. A. P. Herrmann and H. Schaub, “Monte Carlo Tree Search Methods for the Earth-Observing Satellite Scheduling Problem,” JAIS, 2022.
6. A. Jacquet et al., “Earth Observation Satellite Scheduling with Graph Neural Networks and Monte Carlo Tree Search,” IWPSS, 2025.
7. H. Qi et al., “Reinforced Imitation in Heterogeneous Action Space,” 2019.
8. A. M. Mercado-Martínez et al., “An Energy-Efficient Learning Solution for the Agile Earth Observation Satellite Scheduling Problem,” ICMLCN, 2025.
9. Q. Yin et al., “EOS-Bench: A Comprehensive Benchmark for Earth Observation Satellite Scheduling,” 2026.
10. Belaid, “Optimizing Earth Observation Schedules under Unknown Operational Constraints: Active Constraint Acquisition,” arXiv:2604.13283, 2026.
