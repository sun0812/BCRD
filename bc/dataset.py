# -*- coding: utf-8 -*-
"""
bc/dataset.py

供 BC / BCRD 训练使用的流式数据集，读取
``scripts/batch_export_trajectories.py`` 产出的 JSONL。

主要特性：

* **完全流式**：一行一行读 JSONL，绝不会把 2.6 GB 数据一次性加载到内存。
* **候选下采样**：每条样本最多保留 ``max_train_candidates`` 个候选，
  剩余的截掉；保证 batch 内 candidate 维度可固定 padding。
* **必留候选**：SKIP 槽位（index 0）和 expert 候选永远保留。
* **重映射 label**：下采样 + 打乱顺序后会重新计算 expert 的位置。
* **自动检测 SKIP 槽位**：当前 exporter 已经把 SKIP 放在 index 0；
  本类会先验证这一点，若不成立则自动 prepend 一个合成 SKIP。
* **稳定切分**：支持基于 (schedule_file, timestep) 的哈希切分（默认）
  与按 schedule_file 留出（``split_method='by_schedule'``）两种验证集划分方式。
* **特征缩放**：candidate_features 各维量纲悬殊，本类用硬编码尺度做归一化，
  否则 ``sat_start_off_s`` ∈ [0, 43200] 会主导梯度。

每个样本输出的字典（已转成 torch tensor）::

    state_features    : (state_dim,)              float32
    candidate_features: (N_max, cand_dim)         float32  (zero-padded)
    valid_mask        : (N_max,)                  float32  (1=真实, 0=padding)
    expert_label      : ()                        int64    (下采样后的位置)
    objective_weights : (4,)                      float32
    is_skip_label     : ()                        int64    (1 if label == 0)
    num_cand_before   : ()                        int64    (下采样前候选数)
    num_cand_after    : ()                        int64    (下采样后真实候选数)
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

import torch
from torch.utils.data import IterableDataset, get_worker_info

# 维度常量，必须与 batch_export_trajectories 中的定义一致
STATE_FEAT_DIM = 5
CAND_FEAT_DIM = 10
OBJ_DIM = 4

# 候选特征各维度的硬编码缩放因子。exporter 写出的是原始量纲：
# sat_start_off_s ∈ [0, 43200]、data_volume_GB ≈ 0.001、power_cost_W ≈ 50 等，
# 量纲相差 5 个数量级以上；不归一化的话梯度会被 sat_start_off_s 主导，
# 模型几乎学不动。下方的尺度把每一维都拉到大致 [0, 1] 范围。
CAND_FEAT_SCALES: List[float] = [
    1.0,        # is_skip 标志位，已经是 0/1
    43200.0,    # sat_start_off_s （12 小时场景）
    43200.0,    # sat_end_off_s
    100.0,      # duration_s（大多数任务 11~30 秒）
    0.01,       # data_volume_GB  -> 除以 0.01 等价于乘以 100
    500.0,      # power_cost_W
    10.0,       # orbit_number
    10.0,       # task_priority
    43200.0,    # downlink_offset_s
    100.0,      # downlink_duration_s
]
assert len(CAND_FEAT_SCALES) == CAND_FEAT_DIM

# 当某行 JSONL 没有显式 SKIP 槽位时，临时合成的占位特征（is_skip=1，其余为 0）
_SYNTHETIC_SKIP_FEAT: List[float] = [1.0] + [0.0] * (CAND_FEAT_DIM - 1)


def _scale_candidate(feat: List[float]) -> List[float]:
    """对单个候选特征做逐维缩放。"""
    return [f / s for f, s in zip(feat, CAND_FEAT_SCALES)]


def _stable_split_bucket(record: Dict[str, Any]) -> int:
    """为单条样本计算一个 [0, 100) 的稳定桶号，用于哈希切分 train/val。

    选 (schedule_file, timestep) 作为切分键是因为它们在同一 scenario 下
    唯一确定一行，而且改种子可以重现完全相同的切分。
    """
    key = f"{record.get('schedule_file', '')}|{record.get('timestep', 0)}"
    h = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % 100


class TrajectoryStreamingDataset(IterableDataset):
    """对单个 JSONL 文件做流式迭代的 IterableDataset。"""

    def __init__(
        self,
        jsonl_path: str | Path,
        max_train_candidates: int = 256,
        split: str = "all",  # "train" | "val" | "all"
        split_method: str = "hash",  # "hash" | "by_schedule"
        train_fraction: float = 0.8,
        val_schedule_files: Optional[Sequence[str]] = None,
        seed: int = 0,
        shuffle_buffer: int = 1024,
        hard_neg_top_k: int = 0,
        warn_skip_prepended: bool = True,
    ) -> None:
        super().__init__()
        self.jsonl_path = Path(jsonl_path)
        self.max_train_candidates = int(max_train_candidates)
        self.split = str(split)
        self.split_method = str(split_method)
        self.train_fraction = float(train_fraction)
        self.val_schedule_files = set(val_schedule_files or [])
        self.seed = int(seed)
        self.shuffle_buffer = int(shuffle_buffer)
        self.hard_neg_top_k = int(hard_neg_top_k)
        self.warn_skip_prepended = bool(warn_skip_prepended)

        if self.split not in ("train", "val", "all"):
            raise ValueError(f"split must be one of train/val/all, got {self.split!r}")
        if self.split_method not in ("hash", "by_schedule"):
            raise ValueError(f"split_method must be hash or by_schedule, got {self.split_method!r}")

        # 哈希桶号 < 这个值就归入 train，其余进 val
        self._train_bucket_max = int(round(self.train_fraction * 100))
        # 第一次遇到 SKIP 槽位排布问题时打印一条 warn；之后不再重复
        self._notified_skip_layout = False

    # ------------------------------------------------------------- 数据切分
    def _accept_record(self, record: Dict[str, Any]) -> bool:
        """判断当前样本是否应该被当前 split 接受。"""
        if self.split == "all":
            return True
        if self.split_method == "by_schedule":
            # 按 schedule_file 列表硬切：val_schedule_files 里的全部归 val
            sf = record.get("schedule_file", "")
            is_val = sf in self.val_schedule_files
            return is_val if self.split == "val" else (not is_val)
        # 否则按哈希桶切分
        bucket = _stable_split_bucket(record)
        is_train = bucket < self._train_bucket_max
        return is_train if self.split == "train" else (not is_train)

    # ----------------------------------------------------------- 候选下采样
    def _subsample(self, rng: random.Random, record: Dict[str, Any]) -> Dict[str, Any]:
        """选出 ``max_train_candidates`` 个候选、重映射 expert label、做 padding。"""
        cf: List[List[float]] = record["candidate_features"]
        ck: List[Dict[str, Any]] = record["candidate_keys"]
        vm: List[int] = record["valid_mask"]
        n_before = len(cf)
        expert_idx = int(record["expert_action_index"])

        # ---------- 1) 自动检测 / 修复 SKIP 槽位
        has_explicit_skip = isinstance(ck[0], dict) and bool(ck[0].get("is_skip"))
        if not has_explicit_skip:
            # 在最前面 prepend 一个合成 SKIP 候选
            cf = [list(_SYNTHETIC_SKIP_FEAT)] + cf
            ck = [{"is_skip": True}] + ck
            vm = [1] + vm
            # 原 label 是相对于旧列表的，整体右移 1
            expert_idx = expert_idx + 1
            if self.warn_skip_prepended and not self._notified_skip_layout:
                print(
                    f"[trajectory_dataset] WARN: synthetic SKIP prepended (row schedule={record.get('schedule_file')})",
                    flush=True,
                )
                self._notified_skip_layout = True
        else:
            if not self._notified_skip_layout:
                # 只是把"已经检测过"标志位置 True，本身不再打印任何信息
                self._notified_skip_layout = True

        n_total = len(cf)

        # ---------- 2) 决定要保留哪些候选索引
        K = int(self.max_train_candidates)
        keep_set: List[int] = [0]  # SKIP 永远保留
        if expert_idx != 0 and 0 <= expert_idx < n_total:
            keep_set.append(expert_idx)

        # 除已选的之外，剩余可抽样的索引集合
        other_pool: List[int] = [i for i in range(1, n_total) if i not in keep_set]

        # 可选：hard negatives = 除 expert 之外 sat_start 最早的几个候选
        # （cand 特征布局：[is_skip, sat_start_off_s, sat_end_off_s, ...]）
        if self.hard_neg_top_k > 0 and other_pool:
            other_pool_sorted = sorted(
                other_pool, key=lambda i: cf[i][1]  # 按 sat_start_off_s 升序：越小代表越早
            )
            hard = other_pool_sorted[: self.hard_neg_top_k]
            for h in hard:
                if h not in keep_set:
                    keep_set.append(h)
            other_pool = [i for i in other_pool if i not in keep_set]

        # 用剩余预算随机补充负样本
        remaining_budget = max(0, K - len(keep_set))
        if remaining_budget > 0 and other_pool:
            if len(other_pool) > remaining_budget:
                random_picks = rng.sample(other_pool, remaining_budget)
            else:
                random_picks = list(other_pool)
            keep_set.extend(random_picks)

        # 去重并保持插入顺序：SKIP 永远在最前，其次是 expert，再之后是其它候选
        seen: set = set()
        ordered_first: List[int] = []
        for idx in keep_set:
            if idx not in seen:
                seen.add(idx)
                ordered_first.append(idx)

        # 关键：把非 SKIP 部分再洗牌一次，避免 expert 永远落在 index 1。
        # 之前调试时发现，如果不洗牌，模型会学到"永远预测位置 1"的位置捷径，
        # 训练曲线显示 91% 假准确率但其实啥也没学。
        # SKIP 槽位仍然钉在 index 0，因为下游用 label==0 表示 SKIP 类。
        skip_first = (ordered_first[0] == 0)
        head = [0] if skip_first else []
        tail = ordered_first[1:] if skip_first else ordered_first
        rng.shuffle(tail)
        ordered = head + tail

        # 重映射 expert label 到打乱之后的新位置
        new_label = ordered.index(expert_idx) if expert_idx in seen else 0

        # ---------- 3) 把保留下来的候选转成定长 padded 张量（带特征缩放）
        N_max = K
        kept_feats: List[List[float]] = [_scale_candidate(cf[i]) for i in ordered[:N_max]]
        kept_mask: List[int] = [vm[i] if i < len(vm) else 0 for i in ordered[:N_max]]
        n_kept = len(kept_feats)

        # 不够 N_max 的就 pad 到 N_max；padding 位置的 mask 设为 0
        if n_kept < N_max:
            pad_feat = [0.0] * CAND_FEAT_DIM
            kept_feats.extend([pad_feat] * (N_max - n_kept))
            kept_mask.extend([0] * (N_max - n_kept))

        return {
            "state": record["state_features"],
            "cands": kept_feats,
            "mask": kept_mask,
            "label": int(new_label),
            "obj_w": record.get("objective_weights", [0.0, 0.0, 0.0, 0.0]),
            "is_skip_label": int(new_label == 0),
            "n_before": int(n_before + (0 if has_explicit_skip else 1)),
            "n_after": int(n_kept),
        }

    # ------------------------------------------------------------- 迭代器
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        # 多 worker DataLoader 时，需要让每个 worker 只处理自己负责的行号
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1

        # 每个 worker 用独立 RNG，避免不同 worker 抽样完全同步
        rng = random.Random((self.seed + 1) * 1000003 + worker_id)
        buffer: List[Dict[str, Any]] = []

        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                # 简单的轮询切分：worker_id 决定本 worker 处理哪些行
                if num_workers > 1 and (i % num_workers) != worker_id:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # 不属于当前 split 就直接跳过，不计入 shuffle buffer
                if not self._accept_record(record):
                    continue

                buffer.append(record)
                if len(buffer) < self.shuffle_buffer:
                    continue

                # buffer 满了，shuffle 后逐个 yield
                rng.shuffle(buffer)
                for rec in buffer:
                    yield self._to_tensors(self._subsample(rng, rec))
                buffer.clear()

        # 文件读完后清空 buffer 中剩余样本
        if buffer:
            rng.shuffle(buffer)
            for rec in buffer:
                yield self._to_tensors(self._subsample(rng, rec))

    @staticmethod
    def _to_tensors(d: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """把 python 字典中的列表 / 标量统一转成 torch tensor。"""
        return {
            "state": torch.as_tensor(d["state"], dtype=torch.float32),
            "cands": torch.as_tensor(d["cands"], dtype=torch.float32),
            "mask": torch.as_tensor(d["mask"], dtype=torch.float32),
            "obj_w": torch.as_tensor(d["obj_w"], dtype=torch.float32),
            "label": torch.tensor(d["label"], dtype=torch.long),
            "is_skip_label": torch.tensor(d["is_skip_label"], dtype=torch.long),
            "n_before": torch.tensor(d["n_before"], dtype=torch.long),
            "n_after": torch.tensor(d["n_after"], dtype=torch.long),
        }
