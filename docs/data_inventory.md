# EOS-Bench 数据资产盘点（BCRD 视角）

> 仅做检查报告，不修改代码、不新增模型、不训练。
>
> 项目目标：基于官方 solver 产生的 schedule 构造多卫星调度的 imitation learning 数据，进而训练轻量神经调度器（BCRD: Budgeted Counterfactual Regret Distillation）。

---

## 1. 顶层目录

`EOS-Bench/`

```
.git/                     仓库
CITATION.cff
README.md
Scenario_Level_Results.xlsx  汇总指标表（全场景）
algorithms/                  调度算法实现（mip / 启发式 / 元启发 / PPO + 公共候选池）
core/                        scenario 数据模型（Pydantic 风格）
docs/                        本次发布附带的样例数据 + 样例输出 + 可视化
draw/                        画图模块
input/                       scenario / cities_data 输入字典
schedulers/                  scheduling 引擎 + 约束模型 + IO + 评测
utils/                       Orekit visibility 工具
main_generate.py             从 input/ 生成 Scenario_*.json
main_scheduler.py            批量跑算法生成 scheduler_*.json
main_draw.py                 画图入口
```

无以下目录（按 `main_scheduler.py` 默认值，本地未生成）：

- `output/`
- `output/schedules/`
- `output/models/`

也未发现任何 `.7z` 压缩包。

---

## 2. 调度结果（schedule JSON）盘点

仓库自带的 schedule 数据全部在 `docs/`（这是仓库展示用样例，并不是 main_scheduler 默认的 `output/schedules/` 输出位置）。

**单一场景**：`Scenario_S1_Sats20_M500_T0.5d_dist0`
（20 颗卫星，500 个任务，时长 0.5 天，dist=0）

| 类别 | 算法 | 文件数 | 备注 |
|---|---|---|---|
| c1 | `mip` | 3 | 权重 `p1_c0_t0_b0` / `p0_c1_t0_b0` / `p0.25_c0.25_t0.25_b0.25` |
| c2 | `profit_first` | 1 | implicit 权重 |
| c2 | `completion_first` | 1 | implicit 权重 |
| c3 | `sa` | 3 | 三组权重 |
| c3 | `ga` | 3 | 三组权重 |
| c3 | `aco` | 3 | 三组权重 |
| c4 | `ppo` | 1 | 模型名 `ppo_model_p1_c0_t0_b0` |

合计 **15 个** schedule JSON。

文件命名规则（来自 `main_scheduler.run_single_scheduling`）：

```
scheduler_<scenario_stem>_c<class_id>_<algo>_p<wp>_c<wc>_t<wt>_b<wb>.json     # c1/c3
scheduler_<scenario_stem>_c2_<algo>_implicit.json                              # c2
scheduler_<scenario_stem>_c4_ppo_<model_stem>.json                             # c4
```

附带的非 JSON 物料：

- `Scenario_S1_Sats20_M500_T0.5d_dist0.json`：唯一一个场景文件（≈60 MB，含 3052 个 observation_windows）
- `orbit.czml`、`cesium_viewer.html`、`gantt_viewer.html`、`metrics_viewer.html`、`visualisation.gif`：可视化
- `runlog_batch_20260427_191631.txt`：原始批跑日志（含 MIP/SA/GA/ACO/PPO 的 TP/TCR/BD/TM/RT/Gap）

---

## 3. Scenario JSON 字段结构

样例：`docs/Scenario_S1_Sats20_M500_T0.5d_dist0.json`

顶层字段：

```
scenario_id          : "Scenario_S1"
scenario_type        : "multi_sat_multi_gs"   # 不过本场景的 ground_stations 实际为空
metadata             : dict   { name, creation_time, duration, time_step, description, extra }
satellites           : list[20]
missions             : list[500]
observation_windows  : list[3052]
```

未出现 `ground_stations` / `communication_windows`，所以本场景属于 “no-GS” 模式（`save_schedule_to_json` 中 `has_gs == False` 分支）。

### satellite 字段

```
id                          str
orbital_type                "LEO"
orbital_params              dict (Kepler 六根 + epoch)
satellite_specs             { max_data_storage_GB, max_operation_time_s, max_power_W,
                              max_battery_capacity_Wh, max_fuel_kg }
observation_capability      { sensors: [ { sensor_id, resolution, sensor_mode,
                                            field_of_view_deg, observation_swath_width_km,
                                            data_rate_Mbps, power_consumption_W,
                                            min_elevation_angle_deg } ] }
maneuverability_capability  { maneuverability_type ∈ {"agile","non_agile"},
                              slew_rate_deg_per_s, max_pitch_angle_deg, max_yaw_angle_deg,
                              max_roll_angle_deg, stabilization_time_s }
```

### mission 字段

```
id                       "M001" .. "M500"
target_location          { latitude, longitude, altitude_km }
priority                 float
observation_requirement  { duration_s, required_resolution, required_mode,
                           min_elevation_angle_deg }
```

### observation_window 字段（核心，用来构造候选动作）

```
satellite_id  str
sensor_id     str
mission_id    str
time_windows  list of {
    start_time     ISO8601 UTC
    end_time       ISO8601 UTC
    orbit_number   int
    agile_data     { pitch_angles, yaw_angles, roll_angles }  # 与窗口 time_step 等长的逐采样姿态序列
    non_agile_data { pitch_angle,  yaw_angle,  roll_angle  }  # 单点姿态
}
```

注：`agile_data` 列表长度 == 窗口长度（按 `time_step` 离散，默认 1 s），逐元素是该时刻的姿态角。BCRD 候选动作的子窗口选择本质就是在这条序列上挑一个偏移 `k`。

---

## 4. Schedule JSON 字段结构

样例：`docs/scheduler_Scenario_S1_Sats20_M500_T0.5d_dist0_c1_mip_p0.25_c0.25_t0.25_b0.25.json`

顶层字段：

```
scenario_id        "Scenario_S1"
start_time         ISO8601
end_time           ISO8601
assignments        list[356]
unassigned_tasks   list[144]   全部是 task_id 字符串
metrics            dict
```

`len(assignments) + len(unassigned_tasks) == 500 == len(missions)` ✓

### assignment 字段（no-GS 场景；有 GS 时额外含 `ground_station_id/gs_start_time/gs_end_time`）

```
task_id          "M154"
satellite_id     "SCD_2_25504"
sat_start_time   ISO8601
sat_end_time     ISO8601
sensor_id        str
orbit_number     int
data_volume_GB   float    sensor.data_rate_Mbps × duration_s / (8*1024)
power_cost_W    float    sensor.power_consumption_W
sat_angles       agile: { pitch_angles, yaw_angles, roll_angles }（与执行子窗口等长的切片）
                 non_agile: { pitch_angle, yaw_angle, roll_angle }
priority         float    冗余字段，从 task 抄过来便于可视化
```

注意：`sat_window_id` 在内存中存在（`Assignment.sat_window_id`），但 `save_schedule_to_json` **没有写到磁盘**。复原候选索引需要按 `(task_id, satellite_id, sensor_id, sat_start_time, orbit_number)` 在 Scenario 的 `observation_windows` 中匹配。

### metrics 字段

```
TP   float        Task Profit         = Σ priority_i [task i 被调度]
TCR  float ∈[0,1] Task Completion Rate
BD   float        Balance Degree（基于工作时长方差）
TM   float        Timeliness Metric
RT   float (秒)    runtime（即算法 wall-clock）
RV   float|null   Robustness Variance（默认 null；除非传入多次 TP 样本）
```

MIP 还可能有 `mip_gap` 写进 `schedule.metadata`，但导出 JSON 时未持久化。

样例值：

```
{ "TP": 2276.0, "TCR": 0.712, "BD": 0.746, "TM": 0.469, "RT": 29986.617, "RV": null }
```

---

## 5. 关键观察（与 BCRD 设计相关）

1. **schedule JSON 完全可用作 IL 监督信号**：assignment 携带了 (task, sat, sensor, 时间区间, orbit, 姿态切片, priority)，结合 Scenario JSON 的 observation_windows，可以完整复现每个 task 的"被选中的候选动作"。
2. **每个 scenario 提供多份不同 solver 的 schedule**：MIP（接近最优）、SA/GA/ACO（元启发）、profit/completion-first（贪心）、PPO（学习器）。这天然适合 BCRD 中"多 expert 蒸馏 + counterfactual"。
3. **数据范围有限**：仓库只发布了 **1 个场景** 的全套 schedule（S1_Sats20_M500_T0.5d_dist0）。要扩到更多场景，必须自己跑 `main_generate.py + main_scheduler.py`。
4. **MIP 仅支持 ≤20 颗卫星**（`main_scheduler._sat_count_from_scenario_name` 处会跳过 >20）。所以大规模 expert 标签只能来自启发式 / 元启发。
5. **未发现已有 `output/` 目录**：默认输入扫描目录 `BASE_DIR/output/` 不存在；若想本地复跑得自己把 Scenario JSON 移过去，或在 `main_scheduler.py` 改 `SCENARIO_DIR`。

---

## 6. 现成可用资产清单（一句话版）

| 资产 | 路径 | BCRD 用途 |
|---|---|---|
| 1 个完整 Scenario JSON | `docs/Scenario_S1_Sats20_M500_T0.5d_dist0.json` | 唯一可即开即用的状态空间 |
| 15 个 schedule JSON | `docs/scheduler_*.json` | 多 expert 监督信号（MIP / SA / GA / ACO / PPO / profit / completion） |
| 候选生成器 | `algorithms/candidate_pool.py::enumerate_task_candidates` | 离线枚举每个 task 的可行动作集 |
| 合法性判断 | `schedulers/constraint_model.py::ConstraintModel.is_feasible_assignment` | counterfactual rollout 时检查 |
| 指标计算 | `schedulers/evaluation_metrics.py::compute_evaluation_metrics` | 评估学到的 scheduler |
| 批跑日志 | `docs/runlog_batch_20260427_191631.txt` | 看各 expert 的 TP/TCR/BD/TM/RT/Gap 对照 |
| 汇总表 | `Scenario_Level_Results.xlsx` | 整套场景级指标（不止 S1） |

