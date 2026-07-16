# 步骤 5：七篇候选全文深读

全文复核时间：2026-07-16（Asia/Shanghai）

本步骤只使用已下载全文，不把题名推断当作方法事实。每篇均记录“问题—方法—实验/局限”三类证据，并说明它对当前 CCOD 主张的约束。

## 1. SeaRNN：最近的单篇机制威胁

**问题证据。** Leblond 等指出，MLE/teacher forcing 只提高 ground-truth token 概率，既不利用结构化终局损失，也不能区分“接近”和“远离”参考输出的错误；这与 CCOD 对 one-hot BC 的诊断在抽象层高度同构。

**方法证据。** SeaRNN 先 roll-in 到某个位置，再逐一强制 token，随后用 greedy 或 beam 完成序列，按最终结构化误差得到该位置的 cost vector。它用最小成本动作的 log-loss，或把完整 cost vector 转成软分布的 KL loss。为控制 $|A|T$ 次 rollout 的成本，论文进一步对位置和 token 子采样；token 策略明确包括 uniform、当前策略采样、偏向低概率动作的采样和 policy top-k，并始终包含 ground-truth 动作。测试阶段只运行训练后的 RNN，不需要 rollout。

**实验与局限证据。** 论文在 OCR、拼写纠错和 IWSLT 2014 德英翻译上测试；OCR/拼写实验使用 greedy 计算 cost 和解码，子采样实验显示不同损失偏好的 selector 不同，top-k 并非对所有设置都最好。作者明确把大量前向 rollout 视为主要计算瓶颈，并指出子采样只是使其在大词表上可行的近似。

**对 CCOD 的约束。** forced alternatives、rollout cost vector、动作子采样、公平的 uniform/policy/top-k 基线和无搜索部署都不能再声称为新。可能保留的差异只有：SeaRNN 的采样由 token 概率/均匀性驱动，而 CCOD 试图用 EOS 冲突与资源 footprint 在固定预算下覆盖机会代价两端；该差异必须由 $\operatorname{BTailCov}_{16}$ 和闭环收益证实。

## 2. LOLS：局部 rollout cost 与弱参考策略

**问题证据。** Chang 等研究参考策略不最优时的 learning-to-search；仅保证接近 reference 可能把学习器锁死在较差策略上，因此要求相对学习器自身的一步偏离也达到局部最优。

**方法证据。** LOLS 在 roll-in 访问的状态对合法动作执行一次偏离，然后用 learned、reference 或二者 mixture rollout 到终局，以终局 loss 构造动作 cost。其分析同时包含相对 reference 的 regret 和相对 learned-policy one-step deviation 的 regret；局部动作代价本质上由各动作 rollout 终局 loss 的相对差决定。

**实验与局限证据。** 论文在 KDDCup 99 派生的 cost-sensitive classification、POS tagging 和 Penn Treebank dependency parsing 上比较不同 roll-in/roll-out 组合。结果显示 reference 差时，reference roll-out 也会明显变差，而 learned roll-in + mixture rollout 更稳；作者还指出 mixture rollout 可能产生比纯 reference 更大的渐近 regret，算法只保证局部而非全局最优。

**对 CCOD 的约束。** “异构求解器可能是次优 teacher”“用相对 rollout 结果修正 selected-only label”均已有先例。CCOD 的 solution-induced traces 和固定 deterministic continuation 是域适配与协议选择，不自动构成理论创新。

## 3. AggreVaTe：cost-to-go imitation 的基础边界

**问题证据。** Ross 与 Bagnell 认为仅模仿专家即时动作忽略错误后果，而且 learner 会改变自己访问的状态分布；应在 learner-induced states 上比较动作的未来 cost-to-go。

**方法证据。** AggreVaTe 每轮执行当前策略，在随机时刻探索动作，再让专家完成余下轨迹并得到 $Q^*$；样本形如 $(s,t,a,\widehat Q)$，跨轮聚合后训练 cost-sensitive classifier、ranking 模型或 Q regressor。论文还直接讨论部分信息场景：均匀探索动作简单但低效，可把“选哪个动作查询”视为 contextual bandit。

**理论/局限证据。** 该文核心证据是 no-regret reduction 与有限样本分析，而不是 EOS 实验。作者明确列出两类限制：若专家远强于策略类，专家 cost-to-go 会过度乐观；每个 state-action cost-to-go 可能要执行一整条轨迹，代价高于每轨迹获得多个标签的 DAgger。

**对 CCOD 的约束。** cost-to-go imitation、部分动作反馈和“查询选择可提高样本效率”的一般动机都已存在。CCOD 只能把贡献放在可计算的 EOS footprint、冻结预算和同预算证伪协议上，不能把“有限查询 cost-to-go”写成首次。

## 4. Herrmann–Schaub：同域 MCTS-Q 无搜索蒸馏

**问题证据。** 论文研究资源受限的 Earth-observing satellite 闭环 mode scheduling：卫星在 charging、downlink、imaging、desaturation 等动作间决策，目标是安全管理电池、数据缓存和反作用轮并最大化下传科学数据。作者明确指出计算密集规划不利于星上快速重规划。

**方法证据。** 论文比较 random/heuristic rollout 的 MCTS，并用 MCTS 生成 state–action value 训练数据；神经网络回归这些值，部署时直接用 value network 选动作。网络状态包含轨道位置/速度、姿态误差、角速度、轮速、电量、缓存、地面站可见性和视域进度。

**实验与局限证据。** 论文对 MCTS 超参数和网络结构作搜索，以遗传算法估计最优性差距；heuristic-rollout MCTS 接近 GA，训练后的 state–action value network 达到与 MCTS 相当或更好的任务指标，并报告约六个数量级的执行加速。边界也很明确：该任务是 target-agnostic nadir science-mode scheduling，动作空间相对小，不是 EOS-Bench 的动态 task–satellite–window 集；GA 最优性对比只用 100 个初始条件中的前 10 个。

**对 CCOD 的约束。** “在 EOS 中离线搜索动作价值并蒸馏为快速、无搜索网络”已被直接覆盖。CCOD 的差异必须落到 target-window 候选、solution-induced traces 和 footprint query allocation，而非价值蒸馏本身。

## 5. Jacquet 等：EOS GNN 策略与在线 MCTS

**问题证据。** 论文处理单颗敏捷卫星在可见时间窗与姿态机动约束下选择 acquisitions、最大化累计 utility 的 EOS 调度问题；候选集和冲突关系随选中动作动态变化。

**方法证据。** 方法用图表示 acquisitions 与冲突，消息传递 GNN 为可选节点产生 logits/value，以 masked PPO 学习顺序策略；推理可直接贪心，也可将 learned network 作为 PUCT MCTS 的先验/新节点估值以继续搜索。

**实验与局限证据。** 论文在 639 个约 100-acquisition 问题上训练、27 个未见同规模问题上测试，并报告从 100 到 1,591 acquisitions 的外推；约 100 acquisitions 时接近 RAMP 且优于 greedy，但最大 1,591-acquisition 条目低于 greedy。MCTS(100/1000)能改善 vanilla network，却分别带来约 138/1,248 分钟的总测试耗时。作者承认当前按时间顺序插入，GNN foresight 受层数限制，MCTS 在大 branching factor 下只能探索很浅且耗时不可忽略。

**对 CCOD 的约束。** 图表示动态冲突、GNN 动作打分和“网络 + 在线 MCTS”已有强同域对照。CCOD 若成立，卖点应是离线查询效率和接近 BC 的无搜索时延，而不是 GNN 本身；实验至少要讨论 Jacquet 的 GNN/PPO/MCTS 质量—时延边界。

## 6. RILO：异构动作空间模仿，但不是反事实查询

**问题证据。** Qi 等考虑专家和学习者可能拥有不同动作空间、只提供 expert state-only trajectories、环境奖励又稀疏的模仿学习。纯粹匹配专家行为可能限制拥有更强动作集的 learner。

**方法证据。** RILO 扩展 GAIL，用 state-pair discriminator 代替 state-action discriminator；比较 consecutive、single-state、random-time-gap 和 average-per-time 判别奖励，并用随成功率变化的 Bernoulli self-exploration 开关，逐步从 imitation reward 转向 sparse environment reward。

**实验与局限证据。** 论文在全观测/部分观测 grid worlds 系统改变 4-way、king、knight、16-way 专家—学习者动作组合，并在 ViZDoom 检验像素输入；实验问题集中于 self-exploration、动作空间差异和 learner 是否超过 expert。它依赖环境交互、稀疏成功奖励和对抗训练，没有 forced alternatives、候选查询预算、局部 $Q_H$ 或 EOS 约束。

**对 CCOD 的约束。** “异构动作空间使普通模仿不足”不是新颖叙述，但 RILO 并不覆盖从异构 solver 最终解重放轨迹、查询同一 EOS 状态替代动作或分配 footprint budget，故只属于 framing 邻居而非直接 scoop。

## 7. Mercado-Martínez 等：EOS 冲突图上的 GAT-DQN

**问题证据。** 论文研究单颗敏捷 EOS 的目标序列和拍摄时刻联合选择，同时考虑云、湍流、分辨率、能量和存储；每个 target–capture-time 对是一个动作，冲突决定动作后续是否仍可行。

**方法证据。** 作者构造有向图，两层单层 GAT 编码节点/边，再由全连接层输出每个动作的一维 quality/Q；使用经验回放与 $\epsilon$-greedy DQN 学习，选中动作后更新图，直到可行动作耗尽。

**实验与局限证据。** 训练集包含 $N\in\{40,60,80,100\}$ 的合成单星实例，每个目标只离散为三个可选拍摄时刻，训练 5,000 episodes，并在每种设置 200 个实例上比较 MaxResolution 与 MaxTargets。论文报告丢弃图像减少超过 60%、姿态机动能耗浪费最多减少 78%，但普通条件下部分规模的 observation profit 仍低于某些基线；计算使用 A100。作者把多星网络、处理与通信联合优化留作未来工作。

**对 CCOD 的约束。** EOS conflict graph、候选级图注意力和 long-term Q 动作评分均非新。该工作不使用异构 solver traces，不离线强制替代动作，不在固定 counterfactual budget 下比较 selectors，也不报告双尾覆盖，因此没有覆盖 CCOD 最窄的 footprint query-allocation 主张。

## 深读汇总

- **已有通用机制**：forced alternative、rollout/cost-to-go、局部 cost vector、部分动作采样、策略/Q 蒸馏、图动作评分。
- **同域已有机制**：EOS MCTS-Q 网络、EOS GNN/PPO/MCTS、EOS GAT-DQN 与资源状态编码。
- **尚可检验的组合**：从 EOS-Bench solution-induced traces 出发，按约束一致 footprint 在固定上限 $B$ 内分配查询，并以容量平衡双尾覆盖及 train→dev 完整 schedule 收益共同证伪。
