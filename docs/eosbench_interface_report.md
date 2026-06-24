# EOS-Bench 接口报告（BCRD 视角）

> 不修改核心代码、不新增模型、不训练。仅做接口梳理。
>
> 目标：弄清楚 solver 怎么跑、schedule JSON 怎么生成、候选动作和合法性在哪里、怎么把已有 schedule 复用成 imitation learning 数据、以及如何注册一个 learned scheduler。

---

## 1. solver 如何运行

### 调用栈（自上而下）

```
main_scheduler.py
└── run_benchmark(...)                                       批量驱动
    └── _worker_run_single(job)                              子进程
        └── run_single_scheduling(problem_json, class_id, algo_name, ...)
            ├── load_scheduling_problem_from_json(scenario_json) ──► SchedulingProblem
            ├── ConstraintModel(problem, placement_mode, ...)
            ├── create_algorithm(algo_name, weights, cfg_overrides)
            ├── SchedulingEngine(problem, cm, algorithm)
            │     └── engine.run()
            │           ├── cm.build_initial_schedule()       priority-desc greedy
            │           └── algorithm.search(problem, cm, initial_schedule) ──► Schedule
            ├── compute_evaluation_metrics(problem, schedule, runtime, ...)
            ├── save_schedule_to_json(schedule, problem, path, metrics)
            └── plot_schedule_gantt(...)
```

### 关键文件/职责

| 文件 | 职责 |
|---|---|
| `main_scheduler.py` | 入口；解析 CLI、做批量并发、组装 `class_id × algo × objective_weights × scenario` 任务矩阵；处理 PPO 训练/测试 |
| `schedulers/scenario_loader.py` | JSON → `SchedulingProblem`（含 satellites / tasks / windows / comm_windows） |
| `schedulers/constraint_model.py` | 约束 + 单任务可行 assignment 构造 + 初始解 greedy + 简单目标函数 |
| `schedulers/engine.py` | `SchedulingEngine` 协调；定义 `BaseSchedulerAlgorithm` Protocol |
| `algorithms/factory.py` | algo_name → 算法实例（dispatch） |
| `algorithms/objectives.py` | `ObjectiveWeights`（frozen dataclass, 4 维, normalized）+ `ObjectiveModel`（profit/completion/timeliness/balance 0~1 加权打分） |
| `algorithms/candidate_pool.py` | 单任务候选 Assignment 枚举（多种 placement） |
| `algorithms/mip.py / heuristics.py / meta_sa.py / meta_ga.py / meta_aco.py / ppo/` | 具体算法 |
| `schedulers/schedule_output.py` | `save_schedule_to_json` + `plot_schedule_gantt` |
| `schedulers/evaluation_metrics.py` | `compute_evaluation_metrics`（TP/TCR/BD/TM/RT/RV/mip_gap） |

`ObjectiveWeights / ObjectiveModel` 的注意点：

- `ObjectiveWeights(w_profit, w_completion, w_timeliness, w_balance)` 是 frozen dataclass，调用 `.normalized()` 后 4 个分量和为 1（全 0 时回退到 `(1,0,0,0)`）。
- `ObjectiveModel.score(schedule)`：返回 `Σ wᵢ · scoreᵢ`，其中各 `scoreᵢ` 已 clip 到 `[0,1]`（profit 按总优先级归一，completion 按总任务数归一，timeliness_score = 1 − TM，balance_score = BD）。
- 用于 MIP/SA/GA/ACO/PPO 的"在搜索内部"评估；和 `evaluation_metrics.py` 的 TP/TCR/TM/BD 数值口径一致，但是后者输出绝对值，前者输出 0~1 归一值。BCRD 训练 reward / regret 时建议 **复用 `ObjectiveModel.score`**，避免重新定义目标。

### 默认 IO 约定

- 输入扫描目录：`BASE_DIR/output/`（递归 `Scenario_*.json`，跳过 `output/schedules/`）
- 输出目录：`BASE_DIR/output/schedules/`（按 scenario_json 相对路径分目录）
- PPO 模型目录：`BASE_DIR/output/models/`

### 算法 `search` 协议（核心抽象）

```python
class BaseSchedulerAlgorithm(Protocol):
    def search(
        self,
        problem: SchedulingProblem,
        constraint_model: ConstraintModel,
        initial_schedule: Schedule,
    ) -> Schedule: ...
```

每个 algorithm 都实现这一个方法返回 `Schedule`（一组 `Assignment`），其它一切（IO、指标、可视化）都在外层。

---

## 2. schedule JSON 如何生成

由 `schedulers/schedule_output.py::save_schedule_to_json` 写出：

```
result = {
    "scenario_id": problem.scenario_id,
    "start_time":  problem.start_time.isoformat(),
    "end_time":    problem.end_time.isoformat(),
    "assignments": [...],          # 由 schedule.assignments 直接遍历
    "unassigned_tasks": sorted(all_task_ids - assigned_task_ids),
    "metrics": { TP, TCR, BD, TM, RT, RV }     # 当 metrics 非 None
}
```

逐 assignment 写出字段（no-GS 分支）：

```
task_id, satellite_id, sat_start_time, sat_end_time,
sensor_id, orbit_number, data_volume_GB, power_cost_W,
sat_angles, priority
```

有 GS 时多写 `ground_station_id / gs_start_time / gs_end_time`。

**注意写入逻辑里 *丢失* 了内存中的 `sat_window_id` 与 `gs_window_id`**。要把 JSON 里的 assignment 还原回 Scenario 的 `observation_windows[i].time_windows[j]`，需要用 `(task_id, satellite_id, sensor_id, orbit_number, sat_start_time ∈ [w.start, w.end])` 做 lookup。

---

## 3. 候选动作在哪里生成

两个并行实现，行为一致但服务对象不同：

### (a) `ConstraintModel.build_feasible_assignment_for_task`

`schedulers/constraint_model.py:372` — 给一个 task 找 **一个** 可行 Assignment（用于 greedy 初始解 + SA/GA/ACO 修补）。

placement 模式由 `placement_mode ∈ {earliest, center, latest}` 决定，agile 卫星按 `time_step` 离散选 `k`，并把对应的 `agile_data` 切片塞入 `Assignment.sat_angles`。non_agile 卫星只允许窗口中心。

### (b) `algorithms.candidate_pool.enumerate_task_candidates`

`algorithms/candidate_pool.py:82` — 给一个 task 返回 **一批** 候选 Assignment（用于 MIP / heuristics）。

特点：

- 对每个 observation_window，枚举所有可能的子窗口起点 `k` （0 ~ `max_k = floor((window_dur - task_dur)/time_step)`），优先放入 `[earliest, center, latest]` 三个 must；其余 shuffle。
- 有 GS 时再为每个观测起点 × 每个 comm_window 产生若干 downlink placement（earliest / center / latest / 若干随机）。
- 配置参数：`max_candidates`（截断）、`random_samples_per_window`、`prefer_must_first`、`seed`。

**对 BCRD 的意义**：这是 离线枚举可行动作集 / 计算反事实 baseline 的现成入口。不依赖任何 solver。

---

## 4. 合法性在哪里判断

唯一权威：`schedulers/constraint_model.py::ConstraintModel.is_feasible_assignment(assignment, schedule)`

逐条检查（约 6 大类）：

1. **一对一**：task 已有 assignment ⇒ 拒绝。
2. **时长**：`sat_end - sat_start >= task.required_duration`。
3. **下行段（仅当 has_ground_stations）**：必须有 `gs_*` 字段；`gs_start > sat_end`。
4. **同卫星无重叠 + 姿态切换时间**：
   - 对同卫星上每个已有 assignment `a`，按时间顺序计算 `transition_s` = `_transition_time_s(satellite_id, prev, next)`：
     - non_agile：固定 `non_agile_transition_s`（默认 10 s，可被 main 覆盖）
     - agile：调用 `transition_utils.compute_transition_time_agile(Δg, agility_profile)`，`Δg = |Δroll|+|Δpitch|+|Δyaw|`（取相邻 assignment 的端点姿态）
   - 观测段 vs 观测段、观测段 vs 通信段、通信段 vs 通信段都不能 overlap。
5. **GS 排他**：同一 ground_station 同一时段只能服务一个任务。
6. **每轨资源**：在 `(satellite_id, orbit_number)` 上 ∑`data_volume_GB ≤ max_data_storage_GB`，∑`power_cost_W ≤ max_power_W`（上限为 0 时视作不限）。

**对 BCRD 的意义**：counterfactual rollout / regret 计算时，把候选动作 + 当前部分 schedule 投到这函数即可，不需要重跑任何 solver。

---

## 5. 如何新增一个 learned scheduler

最小侵入做法是把它做成实现 `BaseSchedulerAlgorithm` 协议的类。

### 步骤（仅描述，不修改代码）

1. **新建文件**：`algorithms/learned_bcrd.py`
2. **实现接口**

   ```python
   from schedulers.scenario_loader import SchedulingProblem
   from schedulers.constraint_model import ConstraintModel, Schedule, Assignment
   from algorithms.candidate_pool import enumerate_task_candidates

   class BCRDLearnedScheduler:
       def __init__(self, policy, max_candidates=256, placement_mode="earliest"):
           self.policy = policy                   # 你的轻量 NN
           self.max_candidates = max_candidates
           self.placement_mode = placement_mode

       def search(self, problem, constraint_model, initial_schedule) -> Schedule:
           schedule = Schedule()                  # 或 copy(initial_schedule)
           for task in self._order_tasks(problem):
               cands = enumerate_task_candidates(
                   problem, task,
                   placement_mode=self.placement_mode,
                   downlink_duration_ratio=constraint_model.downlink_duration_ratio,
                   max_candidates=self.max_candidates,
               )
               feasible = [c for c in cands if constraint_model.is_feasible_assignment(c, schedule)]
               if not feasible:
                   continue
               idx = self.policy.select(problem, task, feasible, schedule)   # argmax / sample
               schedule.assignments.append(feasible[idx])
           return schedule
   ```

3. **在 `algorithms/factory.create_algorithm` 中注册一个分支**（例如 `if name in ("bcrd", "learned"):`），把 `cfg_overrides` 里的 `policy_ckpt` 路径或 policy 对象传进去。
4. **可选**：`main_scheduler.run_benchmark.all_algo_specs` 里加一条 `{"class_id": 5, "algo_name": "bcrd"}` 让批跑可见。
5. **PPO 已有模板可参考**：`algorithms/ppo/learning.py` 里的 `PPOLearningScheduler.search(problem, constraint_model, initial_schedule, base_dir=...)` 就是同一接口的具体实现。

> 用 PPO 当模板的好处：它已经把 NN inference + 候选枚举 + 合法性兜底全部跑通；BCRD 只是把 actor-critic 换成蒸馏好的 policy。

---

## 6. 能否直接从已有 schedule JSON 构造训练数据

**结论：可以**，并且 EOS-Bench 已经把所有必要 hook 暴露好了。

### 数据可还原性

每个 schedule JSON 里 `assignments[i]` 提供：
`(task_id, satellite_id, sensor_id, orbit_number, sat_start_time, sat_end_time, sat_angles)`。

每个 Scenario JSON 里 `observation_windows` 提供：所有 `(satellite_id, sensor_id, mission_id, time_windows[])`。

两者按 `(task_id, satellite_id, sensor_id, orbit_number, sat_start_time ∈ window)` 唯一匹配 → 复原 **expert 选了哪个 window** 以及 **窗口内的子偏移 k**（由 `(sat_start - window.start)/time_step` 反推）。

### 推荐的 IL 样本构造方式（伪流程）

1. 加载 scenario → `SchedulingProblem`，初始化空 `Schedule`。
2. 解析 schedule JSON 的 `assignments`，**按 `sat_start_time` 升序**遍历（这是最自然的 step-by-step replay 顺序；如果想匹配 build_initial_schedule 的 greedy 行为，也可以按 `priority` 降序，但与 MIP/SA 结果未必一致）。
3. 在每一步：
   - 取当前 `task`；
   - 用 `enumerate_task_candidates(problem, task, ..., max_candidates=K)` 拿候选集；
   - 用 `ConstraintModel.is_feasible_assignment(c, schedule)` 过滤 → 得到可行候选 `feasible`；
   - 在 `feasible` 中按 expert 字段唯一定位 expert action 的索引 `i*`；这就是监督标签；
   - 把 expert assignment 追加到 `schedule.assignments`，进入下一步。
4. 对 `unassigned_tasks`：监督信号是 "no-op / skip"，相当于 expert 选择了 "弃选" 这个动作（在动作空间里加一个 NULL action 或 mask 掉即可）。
5. 把 (state, candidate_set, label) 序列化成一份训练数据；BCRD 的 budgeted counterfactual regret 在这之上叠加：每步对 non-expert 候选做 1-step lookahead（用 `is_feasible_assignment` + `evaluate` 估 regret），按预算选 top-B 个反事实样本蒸馏。

### 注意点

- **缺 `sat_window_id`**：JSON 没存，需要靠 `(satellite_id, sensor_id, orbit_number, sat_start_time ∈ window)` 三/四元组在 scenario 的 `observation_windows` 里反查。鲁棒做法：构建一个 `(sat, sensor, orbit) → [windows]` 索引后做时间区间命中。
- **顺序不唯一**：MIP 是整体求解，输出 assignments 顺序不一定是任何启发式的执行顺序；这意味着 IL replay 出的 "expert 中间状态" 是合成的，而不是 MIP 真实的内部状态。对 imitation 来说没问题，对严格 inverse RL 需要谨慎。
- **expert 间不一致**：同一 scenario 的 15 份 schedule 给出不同 (task → window) 标签。BCRD 自然可以把它们看成一个多模态 expert 分布（or 按权重组做条件蒸馏）。
- **placement_mode / agility_profile / downlink_ratio** 必须和当年跑 solver 时一致，否则 `is_feasible_assignment` 的姿态切换时间会算错。从 `runlog_batch_20260427_191631.txt` 看是 `placement=earliest, downlink=1.0, agility=Standard-Agility, transition_s=10.0, unassigned_penalty=1000.0`。
- **数据量**：单场景给出 15 份 expert × 500 task ≈ 7500 (state, action) pair。要训轻量神经调度器肯定不够，得跑 `main_generate.py` + `main_scheduler.py` 扩到 `Scenario_Level_Results.xlsx` 中那一整套场景。

---

## 7. 各 solver 内部行为（深度核对）

读完 `mip / heuristics / meta_sa / meta_ga / meta_aco / ppo` 后，按"对 BCRD imitation learning exporter 的影响"维度逐个梳理：

### 7.1 公共组件

| 组件 | 文件 | 行为 |
|---|---|---|
| 任务排序（缺省 greedy） | `ConstraintModel.build_initial_schedule` | **priority desc** |
| 任务排序（RL env） | `schedulers/rl_env.py::RLSchedulingEnv.reset` | **priority desc**（与 greedy 一致）|
| 候选枚举 | `algorithms/candidate_pool.py::enumerate_task_candidates` | 每 window 枚举所有可行子偏移 k，agile 模式下 `[0, center, max_k]` 先入 + 其余 shuffle |
| 单候选合法性 | `ConstraintModel.is_feasible_assignment(a, schedule)` | 6 条约束（任务唯一性 / 时长 / 下行 / 同卫星无重叠+姿态切换 / GS 排他 / 每轨资源） |
| 姿态切换 | `schedulers/transition_utils.py::compute_transition_time_agile` | agile：分段函数（Δg≤10s 固定 11.66；之后随 profile 速度线性）；non_agile：固定 `non_agile_transition_s`（缺省 10s） |
| 评分（搜索内部） | `algorithms/objectives.py::ObjectiveModel.score` | 4 项 [0,1] 加权和（profit + completion + (1−TM) + BD） |
| 评分（最终输出） | `schedulers/evaluation_metrics.py::compute_evaluation_metrics` | 绝对值 TP / TCR / BD / TM / RT |

### 7.2 各算法 `search()` 行为

#### MIP（`algorithms/mip.py`）

1. 调 `enumerate_task_candidates`，每任务取 ≤ `max_candidates_per_task=128`（+ `random_samples_per_window=1`）。
2. 对所有 (task, candidate_k) 两两预计算 `assignments_conflict ∨ _satellite_conflict_with_transition`。
3. 建 PuLP MILP：
   - 二元变量 `x[(tid,k)]`、连续 `z[tid] ∈ [0,1]`、`y[sid]` 每卫星时长、`mu` 平均、`d[sid]` 偏差、`W` 总时长、`T` 总任务数。
   - 约束：`∑_k x[tid,k] ≤ 1`、`z[tid] = ∑_k x[tid,k]`、冲突对 `≤ 1`、每轨 storage / power ≤ 上限、`T ≥ enforce_min_scheduled_tasks=1`。
   - 目标：`primary_boost * (wp·profit_norm + wc·completion_norm + wt·timeliness_score) − wb·imbalance_norm`（lexicographic 风格，先满足主目标再追平衡）。
4. CBC/GLPK 求解；解析 log 得 `mip_gap`。
5. **关键**：返回前把所有 `x*≈1` 的 assignment 按 `sat_start_time` 升序逐个 `is_feasible_assignment` 过滤再 append（"repair"）。所以 `schedule.assignments` 顺序 = MIP 选中按时间排序。
6. 偶尔会有"chosen != feasible_kept" 不一致（log 里会打印）。
7. **限制**：`main_scheduler` 把 MIP 限定在 `Sats ≤ 20`，否则跳过。

#### Profit-first（`algorithms/heuristics.py::HeuristicProfitFirstScheduler`）

- 一次性枚举所有 task 的候选 → priority desc 排序 → 同优先级块 shuffle → 逐任务 `_pick_best_feasible_for_task`。
- `key_fn = (−priority, finish_ts, busy)`，且 `early_accept=True`（缺省）：**只要第一个合法候选就接受**，不再扫描。
- 候选内部顺序 shuffle，因此第一个合法 ≠ earliest，**有随机性**。

#### Completion-first（同上 `Completion`）

- 每轮抽样 `completion_scan_tasks_per_iter=256` 个剩余任务，选"可行候选数最少"的（MRV：minimum remaining value）。
- `early_accept=True`，找到第一个可行就停。
- fallback 时按 `(finish_ts, busy, −priority)` 取 key 最小。
- **不是按 priority 顺序**，纯按"剩余可行性最紧迫"的优先级。

#### Timeliness-first / Balance-first

- 同样在 candidate_pool 上 `_pick_best_feasible_for_task`，区别只在 `key_fn`：
  - Timeliness: `(sat_start_time, −priority, busy, finish_ts)`
  - Balance: `(workload_after_add, finish_ts, busy, −priority)`

#### SA（`algorithms/meta_sa.py`）

- 从 `ConstraintModel.build_initial_schedule()` 出发；每步随机选一个 task → 拆掉它的当前 assignment → 调 `cm.build_feasible_assignment_for_task(randomized=True)` 找新位置 → Metropolis 接受。
- 不在 candidate_pool 上工作，而是直接 mutate Schedule。
- 用 `ObjectiveModel.score` 当目标。

#### GA（`algorithms/meta_ga.py`）

- 染色体 = 每个 task 在 candidate_map 里的 k 索引（−1 表示不选）。
- 解码 `_decode`：按"候选少的任务优先"排序解码；逐基因尝试，命中 `is_feasible_assignment` 才入 schedule。
- 锦标赛选择 + 交叉变异 + 精英保留。

#### ACO（`algorithms/meta_aco.py`）

- 每蚁：把 task 顺序 shuffle，对每个 task 按 `(τ^α · η^β · bal_factor)` 加权采样候选；用自维护 `_AntState`（快速可行性 + 资源簿记）筛 → 末尾再过一次 `ConstraintModel.is_feasible_assignment` 兜底。
- 静态 heuristic：`η = wp·profit_norm + wc·1 + wt·(1−delay/horizon)`。

#### PPO（`algorithms/ppo/learning.py + schedulers/rl_env.py`）

- **测试推理 `.search()`**：
  1. 用 `RLSchedulingEnv(max_actions=256, …)`；
  2. task 按 **priority desc** 排序；
  3. 每步：`enumerate_task_candidates(max_candidates=max_actions−1, random_samples_per_window=0)` → `is_feasible_assignment` 过滤 → 加上 index 0 = skip → mask 拼到固定长度；
  4. policy `greedy(state, mask)` 选 index；越界/非法 → 视为 skip 并罚 `unassigned_penalty`；
  5. reward = `reward_scale × Δ ObjectiveModel.score`。
- state 维度 = 5（无 GS）或 7（有 GS）：`[pr/10, dur/T, remaining_ratio, sat_load_mean_normed, sat_load_std_normed, (gs_load_mean_normed, gs_load_std_normed)]`。
- **PPO 已经把 BCRD 想要的"动作空间 = skip ∪ feasible candidates"实现好了。这就是 exporter 应该复用的决策过程。**

### 7.3 多 expert 的"决策协议"对照表

| solver | 任务遍历顺序 | 决策本质 | 是否可逐步 replay |
|---|---|---|---|
| MIP | 全局 → 输出按 `sat_start_time` 排序 + repair | 整解组合优化 | 可，但不忠实于 MIP 内部 |
| profit_first | priority desc | 贪心 first-feasible | 可，忠实 |
| completion_first | MRV (per-iteration MRV ranking) | 贪心 first-feasible | 可，但 task 顺序非确定 |
| timeliness_first | earliest-start | 贪心 best-feasible | 可，忠实 |
| balance_first | priority desc → key=workload_after | 贪心 best-feasible | 可，忠实 |
| SA | initial greedy + random task swap | 局部搜索 | 不可（轨迹是反复 swap，不是一次性 step） |
| GA | 不按 task 顺序，染色体优化 | 群体进化 | 不可 |
| ACO | 随机 task 顺序 + 加权采样 | 群体随机构造 | 不可 |
| PPO | priority desc | NN action selection | 可，忠实（就是 RLSchedulingEnv） |

**对 BCRD exporter 的关键含义**：

- **MIP / profit_first / balance_first / PPO** 的 schedule 可以直接当 expert 在 "priority-desc + step-by-step decision" 协议下 replay 得到 (state, candidate_set, expert_action_index) 数据。
- **completion_first** 也能 replay，但其 task 顺序不是 priority-desc，replay 出的中间状态会和它真实运行时不同 — 对 BC label 不影响，但要意识到"专家中间状态是合成的"。
- **SA / GA / ACO** 没有清晰的逐步决策语义，**不建议**逐 step 蒸馏；它们的产出更适合作为"最终解监督信号"（i.e., 只用最终 assignments 作为多 expert 监督，不试图复原中间状态）。

---

## 8. exporter 设计前置约束（汇总）

> 这是写 `scripts/export_trajectories.py` 之前必须钉住的几个点。

### 8.1 决策协议选型

Brief §16 step 2 写了"按 `sat_start_time` 排序"。但从 7.3 表看：

- 如果按 `sat_start_time` 排序：忠实于 MIP 的输出顺序，但和 PPO/heuristics 内部决策顺序（priority desc）**不一致**。
- 如果按 `priority desc` 排序：忠实于 RL env / 大多数启发式 / `build_initial_schedule`，但 MIP 的中间状态变成"合成的"。

→ **推荐**：用 **priority desc** 协议（与 `RLSchedulingEnv` / `build_initial_schedule` 对齐），因为：

1. 这是 EOS-Bench 原生 RL 框架的协议，未来 BCRD 推理也要走这条；
2. PPO/heuristics expert 在此协议下是 *忠实* 的；
3. MIP 在此协议下仍可被解释为"理想专家给出 task→window 映射"，BCRD 只需学这个映射，不在乎 MIP 内部 LP 决策顺序；
4. 同协议下复用 `RLSchedulingEnv` 的 state/mask 抽取代码，零额外约束实现。

（如果坚持按 `sat_start_time` 排序，记得文档化"中间状态是合成的"这一警告。）

### 8.2 expert action 匹配键

JSON 缺 `sat_window_id`，所以匹配键应该是：

```
(task_id, satellite_id, sensor_id, orbit_number, sat_start_time)
```

实操：先按 `(satellite_id, sensor_id, orbit_number)` 在 Scenario 的 `observation_windows` 索引到 unique 候选 window，再校验 `sat_start_time ∈ [w.start, w.end]`。子偏移 `k = round((sat_start − w.start)/time_step)`。

### 8.3 "expert action 在 replay 时不可行" 情况

priority desc 协议下，可能出现：

- 高优先级任务先消耗了某卫星的时段；
- expert 在该卫星上放了低优先级任务，replay 到该 task 时 `is_feasible_assignment` 已经拒绝它。

这是 **MIP 之类整体解 vs greedy replay 的固有 mismatch**。处理方案：

| 方案 | 说明 |
|---|---|
| A) 视为 SKIP 监督 | label = action 0（skip）；保留为合法监督信号；统计这种 case 的比例 |
| B) 跳过该 step | 不出样本，仅记录 log；不影响其它步 |
| C) 重新 priority-desc 顺序保留 expert 选择 | 在 replay 之前先把 expert assignment 直接 inject 到 schedule（"oracle replay"），单步只看 state/candidate，但跳过该步的 expert label 反查 |

→ **推荐方案 A**（行为克隆里最自然，且能让模型学到"看起来漂亮但实际不可行所以应该 skip"的判断）。

### 8.4 候选集合稳定性

`enumerate_task_candidates(seed=...)` 用 rng shuffle，**不固定 seed 时候选顺序每次不同**。BCRD 训练时：

- exporter 必须传入固定 seed（建议 `seed=0` 或 `_stable_seed(scenario_id|task_id)`），否则 expert_action_index 不可重现。
- 推理时也要传同样 seed；或干脆**不用 index 当 label**，而用候选本身的特征 hash / 字段做匹配。

### 8.5 placement / agility / downlink 参数复盘

`docs/runlog_batch_20260427_191631.txt` 显示官方跑 schedule 时的参数：

```
placement_mode    = earliest
downlink_ratio    = 1.0
unassigned_penalty= 1000.0
agility_profile   = Standard-Agility
non_agile_transition_s = 10.0
```

exporter 必须用同一组参数构造 `ConstraintModel`，否则 `is_feasible_assignment` 与 expert 决策不自洽（姿态切换时间会算错）。

### 8.6 future_return_H 怎么算

Brief 要求 `future_return_H = 专家未来 H 步收益`。在 priority-desc 协议下，对样本 i：

```
future_return_H(i) = ObjectiveModel.score(schedule_after_step_min(i+H, N))
                   − ObjectiveModel.score(schedule_after_step_i)
```

`ObjectiveModel.score` ∈ [0,1]，差分量级 ≤ 0.x，可乘 `reward_scale=10.0` 与 PPO env 对齐。

### 8.7 unassigned 任务的处理

`unassigned_tasks` 在 JSON 里就是 `task_id` 列表。在 priority-desc replay 中：

- 任务被遍历到时；
- 在 candidate set 中没有任何 candidate 与 expert "选窗口"匹配（因为 expert 根本没选）；
- → label = action 0（skip）。

这天然把 unassigned 编码进 IL 信号，无需额外处理。

---

## 9. 报告小结

- solver 入口在 `main_scheduler.py::run_single_scheduling`，所有算法走统一 `BaseSchedulerAlgorithm.search(problem, cm, initial_schedule) -> Schedule`。
- schedule JSON 由 `save_schedule_to_json` 写出，包含 expert 选窗 + 时间 + 姿态切片 + 指标。
- 候选动作枚举在 `algorithms/candidate_pool.enumerate_task_candidates`，合法性判断在 `ConstraintModel.is_feasible_assignment`。两者足以离线构造 BCRD 所需 (候选集, 可行性, regret) 三元组。
- 新增 learned scheduler 只需：实现 `.search(...)` 接口 + 在 `factory.create_algorithm` 注册一行 → 即可纳入 `main_scheduler.py` 的批跑评测。
- 已有 schedule JSON **足以** 启动 imitation learning 的最小验证；但要做到论文体量，要先用现有 `main_generate.py + main_scheduler.py` 扩数据。

