# BCRD: Budgeted Counterfactual Regret Distillation

> **基于 EOS-Bench 的预算化反事实遗憾蒸馏多卫星调度方法**
>
> 本仓库 fork 自 [Ethan19YQ/EOS-Bench](https://github.com/Ethan19YQ/EOS-Bench)，
> 在其上构建轻量神经调度器的 imitation learning 数据管线。

---

## 1. 项目目标

用 EOS-Bench 官方 solver（MIP / SA / GA / ACO / Profit-First / Completion-First / PPO）
产出的 schedule JSON 作为多 expert 监督信号，训练一个推理快、长期决策稳定的轻量
神经调度器。当前阶段（v1）仅做：

- ✅ 多 solver schedule 转 imitation learning trajectory
- ✅ Schema 校验 + streaming dataloader
- ✅ 最小可行的 BC（行为克隆）baseline

**不在 v1 范围内**：counterfactual regret label、learned scheduler 接入
`main_scheduler.py`、diffusion / 大模型 / 在线 MARL。

---

## 2. 仓库结构（BCRD 新增部分）

```
scripts/
├── batch_export_trajectories.py    # schedule -> trajectory 批量导出
├── validate_trajectory_schema.py   # 流式 schema 校验
└── shuffle_jsonl.py                # 一次性全局 shuffle JSONL

models/
└── bc_policy.py                    # 轻量 MLP candidate scorer (~20K 参数)

data_utils/
└── trajectory_dataset.py           # IterableDataset + 候选下采样 + 特征归一化

train_bc.py                         # BC 训练入口

docs/
├── data_inventory.md               # 数据资产盘点
├── eosbench_interface_report.md    # EOS-Bench 接口报告
├── batch_trajectory_export_report.md
└── trajectory_schema_validation_report.md
```

---

## 3. 复现 BCRD 训练数据

> ⚠️ `data/` 与 `outputs/` 被 `.gitignore` 排除，clone 之后需要本地生成。
> 完整 7500 条 trajectory 样本 ≈ 5GB，约 10 分钟跑完。

### 3.1 环境

```bash
pip install torch numpy
# EOS-Bench 求解器本身额外依赖（若需要重跑 solver）：
# pip install pulp matplotlib
```

### 3.2 导出 trajectory（必做）

```bash
# 把官方 15 个 schedule_*.json 转成 jsonl，每条样本含
# state_features / candidate_features / expert_action_index / reward / future_return_H
python scripts/batch_export_trajectories.py \
    --scenario docs/Scenario_S1_Sats20_M500_T0.5d_dist0.json \
    --schedule_dir docs \
    --out_dir data/trajectories \
    --horizon 10
```

预计产出：
- `data/trajectories/scheduler_*.jsonl` × 15 (每个 500 行)
- `data/trajectories/all_schedules_merged.jsonl` (7500 行, 2.6 GB)
- `docs/batch_trajectory_export_report.md` (验收报告)

### 3.3 Schema 校验（推荐）

```bash
python scripts/validate_trajectory_schema.py \
    --input data/trajectories/all_schedules_merged.jsonl \
    --max_rows 0
```

校验通过会输出 `docs/trajectory_schema_validation_report.md`，
Verdict 一栏应显示 **PASS**。

### 3.4 训练前全局 shuffle（推荐）

```bash
python scripts/shuffle_jsonl.py \
    --input data/trajectories/all_schedules_merged.jsonl \
    --output data/trajectories/all_schedules_shuffled.jsonl \
    --seed 0
```

合并文件按 schedule_file 分组排列，不 shuffle 会导致训练 batch 内 solver 分布
偏移、训练曲线震荡（实测 val_top1 会卡在 0.05~0.5 不稳定）。

### 3.5 训练 BC baseline

```bash
python train_bc.py \
    --train data/trajectories/all_schedules_shuffled.jsonl \
    --out outputs/bc_debug \
    --epochs 5 \
    --batch_size 16 \
    --max_train_candidates 256 \
    --lr 1e-3 \
    --use_objective_weights \
    --device cpu      # 或者 --device mps / --device cuda
```

CPU 上 5 个 epoch 约 3 分钟；产出：
- `outputs/bc_debug/checkpoint.pt`（val_loss 最佳的模型权重）
- `outputs/bc_debug/metrics.json`
- `outputs/bc_debug/train_log.jsonl`

支持 `--split_by_schedule` 把 3 个 schedule（一个 MIP / 一个 SA / 一个 ACO）整体
留出做 val，验证跨 solver 泛化能力。

---

## 4. v1 BC baseline 实测结果

| | random | val（epoch 4 模型） |
|---|---|---|
| top-1 accuracy | 0.004 (1/256) | **0.241**（60× 随机） |
| top-5 accuracy | 0.020 | **0.603**（30× 随机） |
| skip recall | — | 0.826 |
| skip precision | — | 0.112（模型过预测 SKIP）|
| non-skip accuracy | 0.004 | 0.174 |

模型 19K 参数，5 epoch / 3 min 即收敛。距离 expert 还很远，但
**loss 单调下降、各项指标稳定、无内存爆炸、无位置捷径**，作为 v1 baseline 已通过。

---

## 5. 下一步路线（按 brief §10）

- [ ] 阶段 4 — counterfactual regret label 构造（`scripts/build_counterfactual_regret.py`）
- [ ] 阶段 5 — BCRD 模型（policy_score + regret_score 双头）
- [ ] 阶段 6 — learned scheduler 接入 `main_scheduler.py`
- [ ] 扩到 S10 / S20 / S50 多场景训练

详细规划见 `CURSOR_AGENT_BCRD_PROJECT_BRIEF.md`。

---

## 6. 上游

EOS-Bench 原仓库：[Ethan19YQ/EOS-Bench](https://github.com/Ethan19YQ/EOS-Bench)
保留为 `upstream` 远程，便于跟踪上游更新。

```bash
git remote add upstream https://github.com/Ethan19YQ/EOS-Bench.git
git fetch upstream
```

---

## 7. License & Citation

继承上游 EOS-Bench 的 license 和 `CITATION.cff`。
