# Cursor / Agent 项目交接说明：BCRD for EOS-Bench

> 项目名称：**BCRD: Budgeted Counterfactual Regret Distillation for Real-Time Multi-Satellite Earth Observation Scheduling**  
> 中文：**基于 EOS-Bench 的预算化反事实遗憾蒸馏多卫星调度方法**  
> 当前目标：先跑通 EOS-Bench 数据处理与小规模调度闭环，再训练轻量神经调度器。  
> 使用环境：前期使用 Mac + Cursor 做代码阅读、数据检查、轨迹导出、小规模 debug；正式训练阶段再租单卡 GPU。  
> 重要约束：不要做大模型、不要做 diffusion、不要做重型 MARL、不要复现 MAPPO/MAT/REDA，先做一个轻量、可跑通、可消融的核心方法。

---

## 1. 我现在在做什么？

我正在基于 **EOS-Bench** 做一个实时多卫星对地观测任务调度方法。

EOS-Bench 是一个 Earth Observation Satellite Scheduling benchmark，包含：

- 多卫星对地观测调度场景；
- scenario JSON；
- 官方求解器；
- MIP / heuristic / SA / GA / ACO / PPO 等 baseline；
- TP / TCR / TM / BD / RT 等评估指标；
- 运行 solver 后可以输出 schedule JSON。

我的目标不是重新写一个卫星调度模拟器，也不是从零写传统优化器，而是：

> 利用 EOS-Bench 的场景、约束、solver 输出和评估器，训练一个轻量神经调度器，让它接近强求解器的调度质量，但推理速度更快，并且比普通行为克隆更少出现长期调度劣化。

---

## 2. 这个问题是单卫星还是多卫星？

这是 **多卫星任务调度**，不是单卫星规划。

单步动作可以理解为：

```text
a_t = (satellite_i, task_j, observation_window_k)
```

也就是从当前所有合法候选动作中选择：

```text
哪颗卫星，在什么窗口，执行哪个任务
```

所以它是一个多卫星星座下的全局任务分配与序列调度问题。

---

## 3. 方法定位

这个项目不是：

```text
不是 diffusion
不是在线强化学习 PPO/MAPPO
不是多智能体大 Transformer
不是纯运筹优化器
```

更准确的定位是：

```text
solver distillation
imitation learning
learning-to-optimize
offline sequential decision learning
learning-augmented scheduling
```

也就是说：

> 我们从 EOS-Bench 官方 solver 生成的调度结果中学习快速调度策略。

---

## 4. 核心方法：BCRD

方法名：

```text
BCRD = Budgeted Counterfactual Regret Distillation
```

中文：

```text
预算化反事实遗憾蒸馏
```

### 4.1 普通 BC 的问题

普通行为克隆只学习：

```text
专家在当前状态下选了哪个动作
```

但是 EOS 多卫星调度中有很多动作：

```text
当前合法，但未来有害
```

例如：

- 当前拍摄一个中等收益任务，导致后面高价值窗口错过；
- 当前安排某颗卫星过载，导致后续负载不均；
- 当前姿态转移合法，但破坏后续任务链；
- 当前消耗存储 / 电量，导致未来可行动作减少。

所以普通 BC 容易产生 **long-term imitation regret**，即长期模仿遗憾。

---

### 4.2 BCRD 的核心思想

不要只学专家动作，还要学习：

```text
如果当前选择另一个合法动作，它会造成多少未来损失？
```

我们对每个状态采样少量反事实动作，而不是对所有候选动作都 rollout。

```text
每个状态只采样 K 个反事实动作
K = 1 / 3 / 5
```

这就是 **budgeted**，预算化。

然后计算每个反事实动作的 future regret：

```math
\Delta(s_t,a)
=
\max
\left(
0,
\frac{
R^{ref}_{t:t+H}
-
R^{rollout(a)}_{t:t+H}
}{
|R^{ref}_{t:t+H}|+\epsilon
}
\right)
```

其中：

- \(R^{ref}_{t:t+H}\)：专家轨迹未来 H 步收益；
- \(R^{rollout(a)}_{t:t+H}\)：当前先执行反事实动作 a，再用 cheap rollout policy 补未来 H 步得到的收益；
- \(\Delta(s_t,a)\)：动作 a 的未来遗憾。

---

### 4.3 推理时如何用 regret？

普通 BC：

```math
a_t =
\arg\max_{a \in A_{valid}(s_t)}
f_\theta(s_t,a)
```

BCRD：

```math
a_t =
\arg\max_{a \in A_{valid}(s_t)}
[
f_\theta(s_t,a,z_I)
-
\lambda \Delta_\phi(s_t,a,z_I)
]
```

其中：

- \(f_\theta\)：policy scorer，学习专家偏好；
- \(\Delta_\phi\)：regret estimator，预测动作未来遗憾；
- \(z_I\)：scenario feature；
- \(A_{valid}\)：合法动作集合；
- \(\lambda\)：遗憾惩罚系数。

这就是 **regret-regularized feasible decoding**。

---

## 5. 当前真正的创新点

请不要把这个项目理解成“BC + Risk + Scenario 拼模块”。

真正的创新点是：

### 创新 1：Budgeted Counterfactual Regret Estimation

用少量反事实动作 rollout 估计长期遗憾，避免全候选 rollout 的高成本。

### 创新 2：Informative Counterfactual Action Mining

反事实动作不是纯随机采样，而是优先选择信息量高、容易误导 BC 的动作，例如：

- 高即时收益动作；
- 高冲突动作；
- 高资源消耗动作；
- BC 可能高分但非专家动作；
- 随机合法动作。

### 创新 3：Regret-Regularized Feasible Decoding

推理阶段在合法动作集合内结合：

```text
专家偏好分数 - 遗憾惩罚
```

从而减少当前合法但长期有害的动作。

---

## 6. 目前不做什么？

请所有 agent 注意，当前阶段不要擅自扩展方向。

不要做：

```text
不要做 Diffusion Scheduler
不要做 MAT / MADT / MAPPO / REDA 复现
不要做大 Transformer
不要做 GNN，除非后续明确要求
不要做全候选动作反事实 rollout
不要做 s1000 全量训练
不要重新实现 EOS-Bench 已有 solver
不要绕过 EOS-Bench 官方 evaluation_metrics.py
不要把 multi-solver teacher 用在测试推理阶段
```

尤其注意：

> 多求解器只用于训练专家池构造，测试推理时不能先跑所有 solver 再选最优，否则会产生 oracle / 数据泄漏问题。

---

## 7. EOS-Bench 相关仓库和数据

### 7.1 GitHub

```text
https://github.com/Ethan19YQ/EOS-Bench
```

### 7.2 arXiv

```text
https://arxiv.org/abs/2604.25782
```

### 7.3 Hugging Face Dataset

```text
https://huggingface.co/datasets/Ethan19YQ/EOS-Bench
```

---

## 8. 当前需要先确认的问题

官方 Hugging Face 数据包里可能有：

```text
output_s10.7z
output_s20.7z
output_s50_cc.7z
...
```

但目前不能默认其中已经包含所有 solver 的 expert schedule JSON。

因此第一步必须检查：

```bash
7z l output_s10.7z | grep -E "schedules|scheduler_|profit_first|sa|aco|mip|ppo" | head -100
7z l output_s10.7z | grep -E "Scenario_.*\.json" | head -20
```

如果压缩包里有：

```text
output/schedules/scheduler_*.json
```

则可以直接解析官方 schedule。

如果没有，则需要用 EOS-Bench 官方 solver 小规模复跑生成 schedule：

```bash
python main_scheduler.py --algos profit_first sa aco --workers_other 4
```

---

## 9. EOS-Bench 代码中已确认的信息

EOS-Bench 中 schedule JSON 由 `schedulers/schedule_output.py` 生成。

典型字段包括：

```text
scenario_id
start_time
end_time
assignments
unassigned_tasks
metrics
```

每个 assignment 包含：

```text
task_id
satellite_id
sat_start_time
sat_end_time
sensor_id
orbit_number
data_volume_GB
power_cost_W
sat_angles
priority
ground_station_id, if applicable
gs_start_time, if applicable
gs_end_time, if applicable
```

这些字段可以用于构造专家 schedule。

但是训练行为克隆还需要进一步重建：

```text
state_t
candidate_actions_t
valid_mask_t
expert_action_t
future_return_H
```

所以还需要写 `schedule-to-trajectory` 转换器。

---

## 10. 当前工程阶段

当前先做 **数据处理与接口打通**，不是正式训练。

### 阶段 0：数据检查

目标：

```text
确认数据包是否含 schedule JSON；
确认 scenario JSON 字段；
确认 schedule JSON 字段；
确认官方 solver 是否能跑通小规模。
```

产出：

```text
docs/data_inventory.md
```

---

### 阶段 1：EOS-Bench 接口报告

需要阅读：

```text
main_scheduler.py
schedulers/schedule_output.py
schedulers/constraint_model.py
schedulers/engine.py
algorithms/candidate_pool.py
algorithms/objectives.py
schedulers/evaluation_metrics.py
```

输出：

```text
docs/eosbench_interface_report.md
```

报告需要回答：

1. solver 如何被调用；
2. schedule JSON 如何保存；
3. assignments 字段结构；
4. 候选动作在哪里生成；
5. 合法动作在哪里判断；
6. evaluation metrics 如何计算；
7. learned scheduler 应该接入哪个接口。

---

### 阶段 2：Schedule-to-Trajectory Exporter

新增脚本：

```text
scripts/export_trajectories.py
```

输入：

```text
scenario JSON
schedule JSON
```

输出：

```text
data/trajectories_s10.parquet
data/trajectories_s20.parquet
```

每条样本包含：

| 字段 | 说明 |
|---|---|
| instance_id | 实例编号 |
| timestep | 决策步 |
| scenario_features | 场景复杂度特征 |
| state_features | 当前状态特征 |
| candidate_features | 候选动作特征 |
| valid_mask | 合法动作 mask |
| expert_action_index | 专家动作下标 |
| reward | 当前动作收益 |
| future_return_H | 专家未来 H 步收益 |

注意：

> 尽量复用 EOS-Bench 现有 candidate_pool / constraint_model，不要重复造约束逻辑。

---

### 阶段 3：普通 BC baseline

新增：

```text
models/bc_policy.py
train_bc.py
```

实现：

```text
BC only
BC + Mask
BC + Scenario
```

训练目标：

```math
L_{BC}
=
-\log
\frac{
\exp f_\theta(s_t,a_t^*)
}{
\sum_{a \in A_{valid}(s_t)}
\exp f_\theta(s_t,a)
}
```

---

### 阶段 4：Counterfactual Regret Label

新增：

```text
scripts/build_counterfactual_regret.py
```

功能：

1. 读取 trajectory；
2. 对每个状态采样 K 个反事实合法动作；
3. 先执行反事实动作；
4. 用 cheap rollout policy 补未来 H 步；
5. 计算 normalized regret label；
6. 保存 regret training data。

建议 debug 设置：

```text
K = 1
H = 5
```

主实验设置：

```text
K = 3
H = 10
```

---

### 阶段 5：BCRD 模型

新增：

```text
models/bcrd_model.py
train_bcrd.py
```

模型输出：

```text
policy_score
regret_score
```

损失：

```math
L = L_{BC} + \beta L_{regret}
```

其中：

```math
L_{regret}
=
Huber(
\Delta_\phi(s_t,a),
\Delta(s_t,a)
)
```

可选排序损失：

```math
L_{rank}
=
\max(
0,
m -
[
\Delta_\phi(s_t,a_{bad})
-
\Delta_\phi(s_t,a_t^*)
]
)
```

---

### 阶段 6：Learned Scheduler 接入 EOS-Bench

新增：

```text
schedulers/learned_bcrd_scheduler.py
```

推理逻辑：

```python
policy_score = model.policy(candidate_features)
regret_score = model.regret(candidate_features)

final_score = policy_score - lambda_regret * regret_score
final_score[~valid_mask] = -inf

action = argmax(final_score)
```

要求：

```text
能被 main_scheduler.py 调用；
能输出 schedule JSON；
能使用 evaluation_metrics.py 计算 TP / TCR / TM / BD / RT。
```

---

## 11. 实验设计

### 11.1 官方 baseline

| 方法 | 是否必须 |
|---|---|
| Profit-first / Greedy | 必须 |
| SA | 必须 |
| ACO | 必须 |
| MIP | 小规模必须 |
| PPO | 可选 |

### 11.2 学习型 baseline

| 方法 | 作用 |
|---|---|
| BC only | 证明不是普通模仿即可 |
| BC + Mask | 验证合法动作约束 |
| BC + Scenario | 验证 scenario context |
| BC + Value | 证明 regret 不是普通 value head |
| BC + Risk-BCE | 对比二分类 risk |
| BC + Regret | 核心模块 |
| Ours / BCRD | Mask + Scenario + Regret Decoding |

### 11.3 消融

| 消融 | 设置 |
|---|---|
| K | 1 / 3 / 5 |
| H | 5 / 10 / 20 |
| lambda | 0 / 0.1 / 0.3 / 0.5 / 1.0 |
| action mining | random vs informative |
| teacher | Greedy only / SA only / ACO only / multi-solver pool |

### 11.4 难场景分组

必须单独评估：

```text
low conflict vs high conflict
low load vs high load
low opportunity vs high opportunity
low timeline overload vs high timeline overload
```

核心 claim：

> BCRD 应该在高冲突、高负载、高时间线过载场景下比普通 BC 更稳。

---

## 12. 评价指标

### EOS-Bench 官方指标

```text
TP  = Task Profit
TCR = Task Completion Rate
TM  = Timeliness Metric
BD  = Balance Degree
RT  = Runtime
```

### 本文新增指标

```text
Regret Prediction MAE
Regret Ranking Accuracy
High-Regret Action Rate
Missed High-Value Task Rate
Generalization Gap
Per-decision latency
```

---

## 13. 当前资源策略

### 13.1 Mac + Cursor 适合做

```text
clone 仓库
检查数据包结构
解压 s10 / s20
读 EOS-Bench 接口
跑小规模 profit_first / SA / ACO
写 export_trajectories.py
写 feature extractor
写 BC baseline
小规模 K=1/H=5 regret label debug
单元测试
```

### 13.2 后续租卡做

```text
s50 / s100 正式训练
多组 ablation
K=3 / K=5 regret label 批量生成
正式评估和日志复现
```

### 13.3 不要现在租卡做

```text
装环境
查字段
修 JSON
写 exporter
跑小规模 debug
```

---

## 14. 今天白天的执行顺序

### Step 1：检查数据包

```bash
7z l output_s10.7z | grep -E "schedules|scheduler_|profit_first|sa|aco|mip|ppo" | head -100
7z l output_s10.7z | grep -E "Scenario_.*\.json" | head -20
```

产出：

```text
docs/data_inventory.md
```

---

### Step 2：跑一个最小 solver case

如果没有 schedule JSON：

```bash
python main_scheduler.py --algos profit_first sa aco --workers_other 4
```

产出：

```text
output/schedules/*.json
```

---

### Step 3：输出接口报告

让 Cursor 读代码，不要先改代码。

产出：

```text
docs/eosbench_interface_report.md
```

---

### Step 4：写 trajectory exporter

产出：

```text
scripts/export_trajectories.py
data/trajectories_s10.parquet
```

---

### Step 5：BC debug

产出：

```text
train_bc.py
outputs/debug_bc_metrics.json
```

---

## 15. 给 Cursor 的第一条 Prompt

```text
你现在是这个项目的代码 agent。请先不要修改代码，只做代码和数据结构检查。

项目目标：
我们要基于 EOS-Bench 做 BCRD: Budgeted Counterfactual Regret Distillation，用官方 solver schedule 构造多卫星调度的 imitation learning 数据，再训练轻量神经调度器。当前不是做大模型、不是 diffusion、不是在线 MARL。

请完成以下任务：

1. 检查当前项目目录和数据目录：
   - 是否有 output/schedules/
   - 是否有 scheduler_*.json
   - 是否有 sa / aco / mip / profit_first / ppo 相关 schedule JSON
   - 如果数据仍在 .7z 中，请使用 7z l 查看压缩包内部是否包含 schedules

2. 随机打开一个 Scenario_*.json，总结字段结构。

3. 如果存在 schedule JSON，随机打开一个，总结：
   - scenario_id
   - assignments 字段
   - 每个 assignment 包含哪些字段
   - metrics 字段

4. 阅读以下文件：
   - main_scheduler.py
   - schedulers/schedule_output.py
   - schedulers/constraint_model.py
   - schedulers/engine.py
   - algorithms/candidate_pool.py
   - schedulers/evaluation_metrics.py

5. 输出 docs/data_inventory.md 和 docs/eosbench_interface_report.md，回答：
   - solver 如何运行
   - schedule JSON 如何生成
   - 候选动作在哪里生成
   - 合法性在哪里判断
   - 如何新增 learned scheduler
   - 是否可以直接从已有 schedule JSON 构造训练数据

不要修改核心代码。不要新增模型。不要训练。只输出检查报告。
```

---

## 16. 给 Cursor 的第二条 Prompt：实现 exporter

```text
请基于前一步接口报告，实现 scripts/export_trajectories.py。

目标：
将 EOS-Bench scenario JSON + schedule JSON 转换为 imitation learning 训练数据。

要求：
1. 输入参数：
   --scenario <path>
   --schedule <path>
   --out <path>
   --horizon H

2. 读取 schedule 中的 assignments，按 sat_start_time 排序。

3. 每一步重建当前已执行 assignment 集合。

4. 调用 EOS-Bench 现有 candidate_pool / constraint_model 逻辑，生成当前候选动作集合和 valid mask。不要重新实现约束判断。

5. 找到当前 expert assignment 对应的 candidate action index。

6. 为每个样本导出：
   - instance_id
   - timestep
   - state_features
   - candidate_features
   - valid_mask
   - expert_action_index
   - reward
   - future_return_H
   - scenario_features

7. 输出 parquet；如果缺依赖，允许先输出 jsonl。

8. 加入详细日志：
   - 总 timestep 数
   - 成功匹配 expert action 数
   - 匹配失败数和原因
   - 平均候选动作数量

9. 不要改动 EOS-Bench 核心调度逻辑。
```

---

## 17. 最终论文表达

不要写：

```text
我们提出多智能体大模型
我们提出 diffusion
我们全面超过优化器
```

要写：

```text
We identify long-term imitation regret as a key failure mode of solver-distilled neural schedulers for multi-satellite EO scheduling. We propose Budgeted Counterfactual Regret Distillation, which uses a small number of informative counterfactual probes to estimate downstream degradation and distills this signal into a regret-regularized feasible decoder.
```

中文：

```text
本文指出长期模仿遗憾是求解器蒸馏神经调度器在多卫星 EO 调度中的关键失败模式。为此，本文提出预算化反事实遗憾蒸馏方法，通过少量信息量高的反事实动作探测估计后续调度劣化，并将该信号蒸馏到遗憾正则化可行解码器中。
```

---

## 18. 当前阶段验收标准

今天阶段只要做到这些就算成功：

```text
1. 明确 output_s10.7z 里有没有 schedules；
2. 能打开并理解 Scenario_*.json；
3. 能打开并理解 schedule JSON，或者能复跑 solver 生成 schedule JSON；
4. 有 data_inventory.md；
5. 有 eosbench_interface_report.md；
6. export_trajectories.py 有初版；
7. 至少成功导出一个小 scenario 的 trajectory 样本。
```

训练不是今天的重点。  
今天的重点是把数据和接口跑通。
