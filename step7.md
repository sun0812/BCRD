# 步骤 7：可执行 delta 与证伪门槛

冻结复核时间：2026-07-16（Asia/Shanghai）

## 建议保留的一句话贡献

> CCOD 检验一种 EOS 特定的反事实查询分配假设：由真实调度约束计算的 conflict/resource footprint，能否在固定查询上限内，比 SeaRNN-style 与通用多样性选择器更充分地覆盖机会代价两端，并把额外覆盖转化为 held-out instances 的完整排程收益。

“检验”比“提出一种全新的 cost-to-go 学习范式”更准确，也让阴性结果仍能形成可复现诊断。

## 必须执行的分层证伪

### 1. 100-state 信号门槛

- 正式冻结包必须 100 states / 1,570 queries 全部成功，才是 `complete`，才能计算 signal gate。
- `query_timeout`、`state_timeout`、`rss_exceeded`、`worker_exit`、`worker_error`、`attempt_exhausted`、`interrupted` 一律为 `incomplete + not_evaluated + method_decision=null`。
- `identity_mismatch`、`cache_corrupt`、`hash_mismatch`、`frozen_drift`、`runner_drift` 一律为 `invalid + not_evaluated + method_decision=null`；`invalid` 是粘性终态，修复后换新 run directory。
- 只有 `complete` 才允许按冻结阈值得到 `pass/go` 或 `fail/no_go`。工程中断绝不能伪装成方法 No-Go。
- signal-eligible 至少 80/100；Type-7 spread 不低于 0.01 的总通过率至少 60%，两个 dev instance 各至少 50%。

### 2. continuation 可信度门槛

- 至少 20 个满足 $17\le C_s\le128$ 的 dev states 做全动作穷举。
- 固定 10 个状态做 greedy 与 beam-8 continuation 排名核验，median Spearman 至少 0.70；相关未定义即失败。
- 排名不稳时先修 continuation，不能用更复杂 scorer 掩盖 teacher 噪声。

### 3. selector 主门槛

- 比较 Uniform-K、Policy-sampling-K、Policy-top-K、generic feature farthest-first 和 CCOD；共享状态、候选集、continuation、预算、模型、loss、随机种子和训练步数。
- 在 20 个 exhaustive dev states 上，CCOD 的 mean $\operatorname{BTailCov}_{16}$ 必须比最佳通用 selector 至少高 0.10。
- `cities_08` 与 `cities_04` 两个 instance-level mean difference 都必须严格大于 0。
- 当前只有两个 dev clusters，不报告 CI、$p$ 值或“显著”。

### 4. 500-state train→dev 门槛

- 500 个拟合状态只能来自六个 train instances，dev 不得参与拟合或标签选择。
- validation pairwise AUC 至少 0.55，Spearman 至少 0.10；任一未定义即失败。
- 两个 dev instance 的完整 schedule $F_\omega$ delta 均非负、均值严格大于 0，feasibility 必须为 100%。
- 只提高离线覆盖/排序而不改善完整 schedule 趋势，判为方法性 No-Go，不建立 paper set。

### 5. footprint 因果负对照

- 在保持所有其他量不变时置乱 footprint；$\operatorname{BTailCov}_{16}$ 与下游增益应回落到 uniform 区间。
- 若置乱不回落，说明收益来自预算、模型或实现细节，撤回 footprint 因果解释。

### 6. 最终 paper-set 门槛

- 500-state 通过后，才在任何新 $Q_H$ 产生前冻结 3 个规模档、30–60 个独立 instances 的 identity、train/dev/sealed-test、objective、features、continuation、pairwise epsilon 和 seeds。
- 最终 5 seeds 的 CCOD 相对最佳 sampled cost-vector baseline 的 $F_\omega$ paired difference 95% CI 必须大于 0；feasibility 不下降，online RT 不超过 BC 的 1.2 倍。

## 论文写作边界

- 主要创新写成“EOS footprint query allocation + 可审计标签协议 + 查询成本/覆盖/闭环联合证据”。
- SeaRNN、LOLS、AggreVaTe/AggreVaTeD、Herrmann–Schaub 必须在引言末尾和 related work 中前置承认。
- 单头 scorer、pairwise loss、确定性 $H=5$ continuation 是实现构件，不单列为新理论。
- 阴性结果的退路是“EOS counterfactual labeling benchmark / diagnostic study”，而不是继续堆模型或扩大 GPU 预算。

## 当前行动结论

继续在 Mac 上完成零标签计划的只读深验与单 state 试运行；100-state `complete` 且通过信号/continuation 门槛前不租 3090。该资源决策与 Level 2 判断一致：先证明 EOS-specific delta 存在，再为训练付费。
