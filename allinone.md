# CCOD 新颖性检索总账

首次检索：2026-07-15（Asia/Shanghai）

全文复核与归档：2026-07-16（Asia/Shanghai）

检索窗口：2000–2026
检索对象：当前 CCOD（Conflict-Covered Opportunity-Cost Distillation），不是旧 BCRD 双头方案

## 0. 证据状态与隐私边界

本轮检索发生在任何正式 counterfactual label 产生之前。正式包 `output/ccod_diagnostic_v1_py310_final_9d74e46` 仍只有 100 个状态和 1,570 条冻结查询计划，没有 $Q_H$、label 或方法判断。因此本文件只回答“主张是否与既有工作重合”，不回答“方法是否有效”。

外部接口只收到以下三个公开、泛化后的查询，没有收到完整未发表方法文字、冻结 split、footprint 公式或预注册阈值。

1. `Earth observation satellite scheduling learning action ranking solver schedules`
2. `budgeted counterfactual cost-to-go imitation learning large action space`
3. `conflict graph resource footprint diverse query selection combinatorial scheduling`

## 1. 来源状态总表

| 来源 | 查询 1 | 查询 2 | 查询 3 |
|---|---|---|---|
| arXiv | 10 条 | 10 条 | 10 条 |
| Crossref | 10 条 | 10 条 | 10 条 |
| OpenReview | HTTP 400：`Invalid username or password` | HTTP 400：同左 | HTTP 400：同左 |
| OpenAlex | HTTP 401：`Unauthorized` | HTTP 401：同左 | HTTP 401：同左 |
| Semantic Scholar | HTTP 403：`Forbidden` | HTTP 403：同左 | HTTP 403：同左 |
| DBLP | 0 条，无匹配 | HTTP 503 | HTTP 503 |

首次用仓库 `.venv` 执行时还遇到依赖缺失，随后改用 `/opt/homebrew/bin/python3` 完成上述运行。错误来源均保留，不把失败接口写成“零相关论文”。API 返回的 citation metadata 不完整，本次不据此做引用量排序。

## 2. 查询 1：学习型 EOS 调度

查询：`Earth observation satellite scheduling learning action ranking solver schedules`

### 2.1 arXiv 的 10 条高层结果

| # | 返回工作或主题 | 处理 |
|---:|---|---|
| 1 | Jacquet 等：*Earth Observation Satellite Scheduling with Graph Neural Networks and Monte Carlo Tree Search*（arXiv:2408.15041） | 高相关；进入全文深读 |
| 2 | Soret 等：*Edge Intelligence for Satellite-based Earth Observation: Scheduling Image Acquisition and Processing*（2026） | 同域新近工作；侧重采集—处理联合资源调度，不是反事实 selector |
| 3 | *Optimal Linear Decay Learning Rate Schedules and Further Refinements* | “schedules”词面误召回；排除 |
| 4 | Mercado-Martínez 等：*An Energy-Efficient Learning Solution for the Agile Earth Observation Satellite Scheduling Problem*（arXiv:2503.04803） | 高相关；进入全文深读 |
| 5 | 异构卫星集群学习/资源决策工作（HADT 类） | 同域 framing；没有 solver-trace counterfactual 查询证据 |
| 6 | *Automating the Wildfire Detection and Scheduling Pipeline with Maneuverable Earth Observation Satellites*（2026） | 检测—调度闭环应用；不覆盖当前机制 |
| 7 | Hadj-Salah 等：*Schedule Earth Observation Satellites with Deep Reinforcement Learning*（2019） | EOS DRL 基线/related work |
| 8 | Yin 等：*EOS-Bench: A Comprehensive Benchmark for Earth Observation Satellite Scheduling*（2026） | 数据与协议锚点，不是方法威胁 |
| 9 | Wang、Han、Leus 等关于多敏捷 EOS 调度与时变转移的工作（2018） | 经典同域优化；不含当前学习信号 |
| 10 | Picard 等关于 auction/distributed satellite scheduling 的工作（2021） | 分布式求解近邻；不含 counterfactual query allocation |

### 2.2 Crossref 的 10 条高层结果

| # | 返回工作或主题 | 处理 |
|---:|---|---|
| 1 | *Multi-Type, Multi-Satellite Scheduling for Heterogeneous Earth Observation Satellite Constellations*（2025） | 同域调度，题名级保留 |
| 2 | *Safe Reinforcement Learning for Autonomous Constraint-Aware Earth Observation Satellite Scheduling*（2026） | 安全/约束 RL 近邻；未见离线 footprint selector |
| 3 | time-dependent multi-agile EOS 的 DQL + ensemble heuristics（SSRN 版本） | 同域混合学习，保留为 related work |
| 4 | super-agile EOS 的 constraint-programming model（2026） | 精确/约束基线，不是学习机制威胁 |
| 5 | *Satellite Task Scheduling System* 书章节（2023） | 背景资料 |
| 6 | improved deep-Q-learning EOS scheduling（2025） | 同域 DRL 近邻 |
| 7 | DRL + rule 的 EOS scheduling（2024） | 同域混合基线 |
| 8 | RL controller 引导的 ALNS（SSRN） | learned search control 近邻，不是查询标签分配 |
| 9 | integrated agile-EOS scheduling（SSRN） | 同域问题近邻，方法信息不足 |
| 10 | Mercado-Martínez 等 energy-efficient EOS learning（会议记录） | 与 arXiv #4 去重 |

### 2.3 查询 1 结论

EOS 学习调度已经覆盖 DRL、GNN、DQN、MCTS、规则/元启发式混合与资源感知。查询没有发现以 solver final schedules 为起点、用 EOS footprint 分配固定 counterfactual query budget 的直接论文，但“图动作评分”“长期 Q”“在线搜索”和“无搜索网络”都已有先例。

### 2.4 2026 主动约束获取边界工作

Belaid 的 *Optimizing Earth Observation Schedules under Unknown Operational Constraints: Active Constraint Acquisition*（arXiv:2604.13283，2026）按摘要级证据应单独列为边界工作。它研究未知操作约束下的 EO schedule optimization，通过主动查询 feasibility oracle 获取约束信息，因此说明“在 EO 调度中主动选择昂贵 oracle queries”并非空白。

它与 CCOD 的问题设定和查询对象不同：

- Belaid 查询某个 schedule/decision 在未知操作约束下是否可行，目标是获取 constraint information；
- CCOD 计划在已知并精确执行 EOS-Bench 约束的状态上，查询备选动作经固定 $H$ continuation 后的 $Q_H$，目标是产生局部排序监督；
- 摘要级证据未显示 Belaid 使用 selected-only solver state reconstruction、按 action conflict/resource footprint 分配固定 $B$、计算 balanced tail coverage，或比较同查询预算下的蒸馏策略完整 schedule 收益。

因此它必须在 related work 中引用，但当前属于邻近边界而不是对 CCOD 的直接覆盖。该判断只基于摘要级证据，若后续取得全文应重新核验 query acquisition function、oracle feedback 和下游评测。

## 3. 查询 2：预算化 counterfactual cost-to-go imitation

查询：`budgeted counterfactual cost-to-go imitation learning large action space`

### 3.1 arXiv 的 10 条高层结果

| # | 返回工作或主题 | 处理 |
|---:|---|---|
| 1 | Qi 等：*Reinforced Imitation in Heterogeneous Action Space*（RILO，2019） | 进入全文；只重合异构动作 framing |
| 2 | *Interactive Imitation Learning in State-Space*（2020） | 一般交互模仿近邻；未显示替代动作预算 |
| 3 | 车辆控制/驾驶 imitation | 应用近邻；排除为直接威胁 |
| 4 | neuro-symbolic long-horizon imitation/planning | 长视域近邻；无 EOS 查询分配证据 |
| 5 | one-shot visual imitation via meta-learning | 演示稀缺，不是 cost-to-go query |
| 6 | coarse-to-fine robot imitation from one demonstration | 分层模仿，不是当前机制 |
| 7 | JUICER 相关工作 | 交互/纠错学习近邻；没有 EOS footprint |
| 8 | RuleML/规则增强学习条目 | 词面噪声或机制弱相关 |
| 9 | 电力系统 counterfactual/决策条目 | “counterfactual”跨域误召回 |
| 10 | coordinated multi-agent imitation | 多智能体 framing；无固定替代动作查询证据 |

### 3.2 Crossref 的 10 条高层结果

| # | 返回工作或主题 | 处理 |
|---:|---|---|
| 1 | *Offline Imitation Learning with Variational Counterfactual Reasoning*（NeurIPS 2023） | “counterfactual”重要近邻；侧重潜变量推断，不是显式 forced-action continuation selector |
| 2 | *Budgeted Batch Mode Active Learning*（ICPR） | 预算查询的一般先例；不含顺序 cost-to-go |
| 3 | 目标条件模仿中的 counterfactual reward estimation | 反事实奖励近邻；机制不同 |
| 4 | rollout-efficient / student-efficient imitation | 查询成本近邻；保留为讨论 |
| 5 | contextual-bandit / active-query imitation | 一般动作选择先例；不具 EOS 结构 |
| 6 | 机器人运动或驾驶 imitation 条目 | 跨域应用；低相关 |
| 7 | 一般大动作空间 RL 条目 | 问题规模近邻；无 solver traces |
| 8 | 因果/反事实表示学习条目 | 术语近邻；无强制动作 rollout |
| 9 | imperfect-information counterfactual regret 条目 | CFR 术语噪声；排除 |
| 10 | 与学习预算或 cost-sensitive classification 相关的跨域条目 | 只具词面重合 |

### 3.3 查询 2 结论

API 召回对 “counterfactual” 极其不精确，但通过全文谱系补召回后，AggreVaTe、LOLS 与 SeaRNN 明确构成强先例：cost-to-go imitation、forced alternative、cost vector 和子采样都不是新机制。RILO 只说明“专家/学习者动作不同”的 framing 已存在，不覆盖当前 EOS 查询协议。

## 4. 查询 3：conflict/resource footprint 与多样化查询

查询：`conflict graph resource footprint diverse query selection combinatorial scheduling`

### 4.1 arXiv 的 10 条高层结果

这组词将多个成熟领域混在一起，返回 10 条均未形成直接威胁。按运行输出的高层主题依次为：

| # | 高层主题 | 判断 |
|---:|---|---|
| 1 | conflict-graph coloring / allocation | 有冲突图，无 counterfactual learning |
| 2 | 通信或频谱资源调度 | 有资源约束，不是 EOS 构造排程 |
| 3 | diverse subset / representative selection | 有多样性选择，无顺序 cost-to-go |
| 4 | 主动学习 query selection | 有预算查询，无 EOS 约束 footprint |
| 5 | job-shop / machine scheduling | 有组合调度，无 solver-trace distillation |
| 6 | graph-based combinatorial optimization | 有图决策，无当前标签协议 |
| 7 | 多智能体资源分配 | 有资源状态，无替代动作 continuation |
| 8 | environmental/resource footprint 分析 | “footprint”语义误召回 |
| 9 | network conflict resolution | “conflict”语义误召回 |
| 10 | 一般 diversity-aware optimization | 只能支持 generic farthest-first 基线 |

### 4.2 Crossref 的 10 条高层结果

Crossref 同样返回 10 条跨域结果；按运行输出逐条归纳的高层主题如下。这里保留的是检索审计所需的主题与筛选判断，不反向补造未保留的噪声题名。

| # | 高层主题 | 判断 |
|---:|---|---|
| 1 | conflict-graph scheduling | 有图冲突，无顺序 action-value query |
| 2 | 无线/频谱 resource allocation | 资源预算语义不同 |
| 3 | 云/边缘计算 resource scheduling | 非 EOS，查询对象不同 |
| 4 | job-shop / machine scheduling | 有组合调度，无 offline counterfactual labels |
| 5 | budgeted batch active learning | 有查询预算，是通用选择基线的先例 |
| 6 | diversity-aware sample selection | 有多样性，应由 generic farthest-first 控制 |
| 7 | environmental/resource footprint accounting | “footprint”语义误召回 |
| 8 | graph optimization / subset selection | 有图与子集，无 continuation value |
| 9 | multi-objective combinatorial scheduling | 问题结构近邻，学习信号不同 |
| 10 | 一般 conflict/resource scheduling 综述或应用 | 背景材料，不是直接机制证据 |

逐条检查题名/摘要后：

1. 没有工作同时出现 sequential forced action、continuation cost 和 EOS scheduling；
2. 没有工作以未来可行窗口删除或姿态/星上资源影响定义查询 footprint；
3. diversity/active-learning 条目足以要求 CCOD 设置 generic feature farthest-first 与 footprint permutation 负对照；
4. 不能因该组零直接命中就宣称“首次结构化查询”，因为 active selection 本身是成熟通用思想。

### 4.3 查询 3 结论

检索没有发现 EOS conflict/resource footprint query allocator 的直接同构工作，但这条查询精度最低。可守主张必须限定到“EOS 约束一致 footprint 在公平固定预算下的可测增益”，不能扩大为“首次用结构/多样性选查询”。

## 5. 模型知识补召回与去重后的全文集合

| 工作 | 年份 | 全文证据文件 | 主要威胁轴 |
|---|---:|---|---|
| SeaRNN: Training RNNs with Global-Local Losses | 2018 | `papers/searnn_2018.txt` | forced alternatives、cost vector、uniform/policy/top-k、无搜索部署 |
| Learning to Search Better than Your Teacher（LOLS） | 2015 | `papers/lols_2015.txt` | 局部 rollout cost、弱 reference、mixed rollout |
| Reinforcement and Imitation Learning via Interactive No-Regret Learning（AggreVaTe） | 2014 | `papers/aggrevate_2014.txt` | learner-state cost-to-go、部分信息动作探索 |
| Monte Carlo Tree Search Methods for the Earth-Observing Satellite Scheduling Problem | 2022 | `papers/herrmann_schaub_2022.txt` | EOS MCTS-Q 到快速 value network |
| Earth Observation Satellite Scheduling with Graph Neural Networks and Monte Carlo Tree Search | 2025 | `papers/jacquet_gnn_mcts_2025.txt` | EOS GNN/PPO、在线 PUCT MCTS |
| Reinforced Imitation in Heterogeneous Action Space | 2019 | `papers/reinforced_imitation_2019.txt` | 异构动作空间与 state-only imitation |
| An Energy-Efficient Learning Solution for the Agile Earth Observation Satellite Scheduling Problem | 2025 | `papers/energy_efficient_eos_2025.txt` | EOS conflict graph、GAT-DQN 候选 Q |

EOS-Bench 全文作为数据/约束锚点保留，但不计入七篇方法威胁。

## 6. 逐篇全文证据摘要

### SeaRNN

- 问题：MLE 不能用结构化终局 loss 区分不同错误。
- 方法：roll-in 后强制每个替代 token，greedy/beam 完成并构造 cost vector；大词表使用 uniform、policy、biased、top-k 子采样。
- 实验/局限：OCR、拼写纠错和机器翻译；rollout 成本是主要瓶颈，子采样有效但不同 loss 的最佳 selector 不一致。

### LOLS

- 问题：参考策略次优时，单纯模仿不能超过 teacher。
- 方法：一步偏离后以 learned/reference/mixed policy rollout，学习相对动作终局 cost。
- 实验/局限：cost-sensitive classification、POS、dependency parsing；弱 reference 下 mixture rollout 更好，但只给局部最优保证。

### AggreVaTe

- 问题：即时 imitation error 忽略后续后果和 learner-induced distribution。
- 方法：在 learner state 随机时间探索动作，由专家完成并给 cost-to-go，聚合训练 cost-sensitive policy/Q regressor。
- 理论/局限：no-regret reduction；专家可能对受限策略类过度乐观，每个 Q 样本可能需要整条 rollout。

### Herrmann–Schaub

- 问题：EOS 资源管理与星上快速重规划。
- 方法：MCTS 产生 state-action values，网络回归后直接选 mode action。
- 实验/局限：近 GA 的 MCTS、网络约六个数量级加速；任务是 target-agnostic mode scheduling、动作小，非 EOS-Bench target-window 候选。

### Jacquet 等

- 问题：单星 acquisition utility scheduling。
- 方法：动态图 GNN + masked PPO，推理可贪心或加 PUCT MCTS。
- 实验/局限：639 个约百规模训练、27 个同规模测试，并外推更大实例；最大规模可能输给 greedy，MCTS 时间不可忽略且搜索深度浅。

### RILO

- 问题：专家/学习者动作空间不同且只有 state-only demonstrations、环境奖励稀疏。
- 方法：state-pair GAIL reward + success-adaptive self-exploration。
- 实验/局限：grid worlds 与 ViZDoom；依赖环境交互，没有 forced alternatives、查询预算或 EOS 约束。

### Mercado-Martínez 等

- 问题：单星 target-time 选择，兼顾气象、图像质量、能量和存储。
- 方法：冲突有向图、两层 GAT + 一维动作 Q、DQN。
- 实验/局限：40–100 targets、每目标三时刻、5,000 episodes、每设置 200 tests；合成单星且用 A100，不含异构 solver traces 或离线 selector。

完整三类证据与边界见 [`step5.md`](./step5.md)。

## 7. 轴向结论

| 主张 | 证据判断 |
|---|---|
| 替代动作 rollout / cost-to-go 是新机制 | 否；AggreVaTe、LOLS、SeaRNN 已覆盖 |
| 大动作空间中只查部分动作是新机制 | 否；SeaRNN 已比较 uniform/policy/top-k，AggreVaTe 讨论部分信息探索 |
| 将搜索/Q 蒸馏到无搜索策略是 EOS 首次 | 否；Herrmann–Schaub 直接覆盖 |
| EOS 图候选长期价值评分是首次 | 否；Jacquet 与 Mercado-Martínez 已覆盖 GNN/PPO/MCTS 或 GAT-DQN |
| EOS constraint-consistent footprint 用于固定 counterfactual query allocation | 本轮未发现直接同构；仍需实验证明相对通用 selector 的独立价值 |
| $\operatorname{BTailCov}_{16}$ + train→dev closed-loop 的联合证伪协议 | 当前方案特定的评测组合；是证据设计，不宜夸成一般算法理论 |

## 8. Scoop 结论

**Level 2 — High Overlap。** 最近单篇威胁是 SeaRNN。当前可辩护 delta 是条件性的：

> CCOD 在精确 EOS schedule states 上，用冲突/资源 footprint 分配固定上限的 continuation queries；只有它相对 Uniform、Policy-sampling、Policy-top-K 与 generic farthest-first 同时提高 exhaustive dev 的容量平衡双尾覆盖，并提高 train→dev 的可行完整 schedule，方法贡献才成立。

若只证明 cost-to-go 比 one-hot BC 信息更多，结论已被既有谱系充分预期；若 footprint 不优于通用 selectors，应将工作降级为诊断/benchmark，而不是继续声称新算法。

针对 budgeted counterfactual/action-value query selection 的定向检索，没有发现一篇同时具备 EOS selected-only 状态重建、约束 footprint 固定 $B$ 分配、balanced tail coverage 和等预算闭环收益的工作。这里的措辞必须保持为“本轮、当前可访问证据中未发现”：三类主接口失败且关键词检索天然低召回，它不是穷尽性新颖性证明。

## 9. 参考入口

1. Ross 与 Bagnell，*Reinforcement and Imitation Learning via Interactive No-Regret Learning*，2014，arXiv:1406.5979。
2. Chang 等，*Learning to Search Better than Your Teacher*，ICML 2015。
3. Sun 等，*Deeply AggreVaTeD*，ICML 2017。
4. Leblond 等，*SeaRNN*，ICLR 2018。
5. Herrmann 与 Schaub，*Monte Carlo Tree Search Methods for the Earth-Observing Satellite Scheduling Problem*，JAIS 2022，doi:10.2514/1.I010992。
6. Jacquet 等，*Earth Observation Satellite Scheduling with Graph Neural Networks and Monte Carlo Tree Search*，IWPSS 2025，arXiv:2408.15041。
7. Qi 等，*Reinforced Imitation in Heterogeneous Action Space*，2019，arXiv:1904.03438。
8. Mercado-Martínez、Soret 与 Jurado-Navas，*An Energy-Efficient Learning Solution for the Agile Earth Observation Satellite Scheduling Problem*，ICMLCN 2025，arXiv:2503.04803，doi:10.1109/ICMLCN64995.2025.11140302。
9. Yin 等，*EOS-Bench: A Comprehensive Benchmark for Earth Observation Satellite Scheduling*，2026，arXiv:2604.25782。
10. Belaid，*Optimizing Earth Observation Schedules under Unknown Operational Constraints: Active Constraint Acquisition*，2026，arXiv:2604.13283。
