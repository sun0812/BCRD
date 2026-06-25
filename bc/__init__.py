# -*- coding: utf-8 -*-
"""bc 包：BCRD 项目第一阶段的行为克隆（BC）模块。

聚合了 BC 训练所需的三个文件：

* ``policy.py``  — 轻量 MLP candidate scorer
* ``dataset.py`` — 流式 IterableDataset + 候选下采样 + 特征归一化
* ``train.py``   — 训练入口（含 hash / by-schedule 切分）

入口运行方式::

    python -m bc.train --train data/trajectories_combined/all_shuffled.jsonl \\
        --out outputs/bc_combined --epochs 5 --device cpu
"""

from .policy import BCPolicy
from .dataset import (
    TrajectoryStreamingDataset,
    CAND_FEAT_DIM,
    OBJ_DIM,
    STATE_FEAT_DIM,
    CAND_FEAT_SCALES,
)

__all__ = [
    "BCPolicy",
    "TrajectoryStreamingDataset",
    "CAND_FEAT_DIM",
    "OBJ_DIM",
    "STATE_FEAT_DIM",
    "CAND_FEAT_SCALES",
]
