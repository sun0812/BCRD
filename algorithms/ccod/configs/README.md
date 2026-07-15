# CCOD 诊断预注册

`diagnostic_v1.json` 在查看 100-state 反事实标签前冻结实例划分、来源、状态预算、查询预算和工程 Go/No-Go 门槛。

正式生成与标签执行固定使用 CPython 3.10.20（仓库命令为 `.venv/bin/python`）；脚本会在读取任何来源或恢复状态前校验解释器身份，其他 Python 版本直接失败，不能沿用或改写 query identity。

## 数据边界

10 个 Sats1/M100 实例按规范化场景语义内容生成身份，并显式冻结为：

- train：`cities_03, cities_05, cities_01, cities_09, cities_06, cities_07`；
- dev：`cities_08, cities_04`；
- sealed test：`cities_02, cities_10`。

同一实例的全部 solver、objective、trace、state 与 label 必须跟随实例进入同一 fold。100-state diagnostic 只能读取 dev；test 在方法、阈值、selector 和训练超参数冻结前不得生成标签或进行完整调度比较。路径只作为 `root_id + relative_path + sha256` 的可搬迁引用，不进入科学身份。

## 来源与选样边界

首轮诊断只使用 balanced objective 下 SA、GA、ACO 三类 solution-induced traces。先按完整 `state_hash` 合并重复状态，再用稳定哈希选定 canonical source；禁止用多个来源中的 observed action 人为调整 SKIP 比例。

最终选 100 个 actionable 状态，两个 dev 实例各 50 个。选样只允许使用来源、step、candidate count、canonical observed 类型和哈希身份；禁止读取 $Q_H$、优势、模型分数或任何下游标签。预标签审计发现全部 124 个 canonical observed-SKIP 状态的候选数都为 1，因而与 actionable 条件互斥；首轮诊断将 observed-SKIP 选样目标固定为 0，但每个状态仍强制查询 SKIP。时间、candidate count 与 solver 仅做边际平衡，不尝试覆盖它们的笛卡尔积。

为避免看过标签后挑选验证状态，100-state selection 还硬性保证两个 dev 实例各至少 10 个状态满足 `17 <= candidate_count <= 128`。随后以稳定哈希按实例各选 10 个 exhaustive 状态，再从这 20 个状态中按实例各选 5 个做 beam-8 近似标签；二者必须在同一状态上比较。

## 查询与门槛

每状态查询 `min(16, candidate_count)` 个唯一动作：先放 canonical observed 与 SKIP，去重后按 `eosbench-ccod-uniform-rank-v1` 的 SHA-256 排序无放回补齐。输入候选顺序、Python hash seed 和 source alias 顺序不得改变计划。

信号分别在 all、actionable（`candidate_count>=2`）和 signal-eligible（`candidate_count>=10`）状态上报告。只有恰好 100 个唯一状态、全部预注册 query 成功、至少 80 个 signal-eligible 状态时才允许判定：至少 60% signal-eligible 状态满足 Type-7 $P_{90}(Q_H)-P_{10}(Q_H)\ge0.01$，且两个 dev 实例各自通过率不低于 50%。基础设施缺失或缓存损坏记为 `incomplete/invalid`，不得误记为方法 No-Go。

这里只有两个 dev instance，因此该门槛只是租卡前工程筛查，不报告 state-level 置信区间，也不声称统计显著。信号通过后仍需至少 20 个小状态的 exhaustive 排序和 10 个状态的 greedy-vs-beam-8 强标签比较；所有 Mac 本地门槛通过前不租 RTX 3090。
