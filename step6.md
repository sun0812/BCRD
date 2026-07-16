# 步骤 6：逐轴比较与 Scoop 等级

比较时间：2026-07-16（Asia/Shanghai）

## 机制矩阵

“CCOD”一行表示当前预注册方案，不表示实验已经完成。

| 工作 | 替代动作下游值 | 有限动作查询/采样 | 结构驱动查询分配 | EOS 动态候选/资源 | 无搜索部署 | 双尾覆盖 + 完整排程证据 |
|---|---|---|---|---|---|---|
| SeaRNN | 是：forced token + 完成式 rollout | 是：uniform/policy/biased/top-k | 否：依赖概率或均匀性 | 否 | 是 | 否 |
| LOLS | 是：一步偏离 + mixed/reference/learned rollout | 主要为合法动作 cost vector | 否 | 否 | 是 | 否 |
| AggreVaTe | 是：专家 cost-to-go | 是：部分信息动作探索 | 只提出 contextual-bandit 方向 | 否 | 是 | 否 |
| Herrmann–Schaub | 是：MCTS state–action value | 是：MCTS 模拟预算 | 否：heuristic/random rollout | 是，但为少量 mode actions | 是：Q 网络替代 MCTS | 否 |
| Jacquet 等 | 是：PPO/value 与可选 MCTS | 是：在线 MCTS budget | 否：PUCT 搜索，不是离线标签 selector | 是：单星 acquisition graph | 直接策略时是；MCTS 版否 | 否 |
| RILO | 否：state-pair imitation reward | 否 | 否 | 否 | 是 | 否 |
| Mercado-Martínez 等 | 是：DQN-Q | $\epsilon$-greedy 环境探索，不是离线查询预算 | 否 | 是：单星 target-time conflict graph | 是 | 否 |
| Belaid（2026，摘要级） | 否：查询 feasibility，而非动作 $Q_H$ | 是：主动选择 oracle query | 面向未知约束获取，不是 action footprint | 是：EO schedule | 不适用 | 否 |
| CCOD（计划） | 是：固定 $H$ 的 $Q_H$ 和 pairwise order | 是：每状态上限 $B$ | **计划是**：EOS conflict/resource footprint | **是**：EOS-Bench task–satellite–window | **计划是** | **计划是；尚无标签、尚无结果** |

## 逐轴判断

### 轴 A：问题表述

“one-hot 监督无法表达不同局部错误的结构化终局代价”已由 SeaRNN/L2S 明确提出；“参考策略可能次优”已由 LOLS 处理；“异构动作空间使模仿变难”已由 RILO 处理。因此 CCOD 的 framing 有现实意义，但不是全新问题类型。

### 轴 B：训练信号

forced alternatives、rollout cost vector、expert/learner cost-to-go、Q regression 和 pairwise/soft cost learning都在既有谱系内。把 $Q_H(s,a)-Q_H(s,a^{\mathrm{obs}})$ 称为 opportunity cost 可以改善语义，但不能改变其高重合事实。

### 轴 C：查询效率

SeaRNN 已经在大动作空间中比较 uniform、policy 与 top-k 子采样；AggreVaTe 也明确指出均匀动作探索低效。因此“预算化采样”仍是通用机制。尚可守的是：footprint 必须由真实 EOS 约束的未来可行域变化定义，而不是一般 uncertainty/diversity 的换名。

Belaid 2026 又给出 EO 未知操作约束下的 active constraint acquisition：它主动查询 feasibility oracle，进一步说明“EO + 主动昂贵查询”的组合本身也不是空白；但它不查询备选动作的 bounded-horizon value，不覆盖 CCOD 的 action-footprint 双尾目标。该判断目前只基于摘要级证据。

### 轴 D：领域结合

Herrmann–Schaub、Jacquet 和 Mercado-Martínez 已覆盖 EOS 中的 state-action value、图候选评分、资源状态与搜索/无搜索策略。单纯把 SeaRNN 移植到卫星调度只属于应用组合。可能形成方法贡献的是：constraint-consistent footprint 是否真的提高每次昂贵 continuation query 的信息量。

### 轴 E：证据协议

容量平衡双尾覆盖是当前方案最重要的可证伪接口。对候选数 $C_s$、预算内查询集合 $U_B(s)$，令

$$
q_s=\max\!\left(1,\left\lceil0.1C_s\right\rceil\right),\qquad
m_s=\min\!\left(q_s,\left\lfloor\frac{|U_B(s)|}{2}\right\rfloor\right),
$$

则

$$
\operatorname{BTailCov}_{B}(s)=\frac12\left[
\frac{\min\{|U_B(s)\cap\operatorname{Best}_{q_s}(s)|,m_s\}}{m_s}+
\frac{\min\{|U_B(s)\cap\operatorname{Worst}_{q_s}(s)|,m_s\}}{m_s}
\right].
$$

该指标需要穷举 $Q_H$，只能用于预注册的 exhaustive dev states，不能在非穷举 500-state 集上伪造。它本身也不是理论新颖性；其价值在于让 footprint selector 的主张可以被公平否定。

## 最终等级

**Level 2 — High Overlap（高重合）。**

理由：最近威胁 SeaRNN 已经覆盖当前方法的通用机制骨架，LOLS/AggreVaTe 覆盖局部 cost-to-go 谱系，Herrmann–Schaub 覆盖同域价值蒸馏。尚未发现完全相同的 EOS footprint query-allocation 论文，因此不是“完全被 scoop”；但也绝不能按 Level 1 将整个 CCOD 宣称为全新。

## 条件性 delta

> 不同于 SeaRNN 以 uniform、policy probability 或 top-k 在 token 空间抽取替代动作，CCOD 在精确重放的 EOS-Bench schedule states 上，以约束一致的 conflict/resource footprint 在固定上限 $B$ 内分配有限 continuation queries；其贡献只有在相同 continuation 和模型下，同时提高容量平衡双尾覆盖与 train→dev 完整可行排程收益时成立。

这是**待证假设**，不是已取得结论。正式包仍为零标签，不能使用“有效”“提升”或“优于”等完成时表述。
