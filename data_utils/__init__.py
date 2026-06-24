# BC / BCRD 数据流工具包；统一对外暴露训练用的 IterableDataset 与维度常量
from .trajectory_dataset import (
    TrajectoryStreamingDataset,
    CAND_FEAT_DIM,
    STATE_FEAT_DIM,
    OBJ_DIM,
)

__all__ = [
    "TrajectoryStreamingDataset",
    "CAND_FEAT_DIM",
    "STATE_FEAT_DIM",
    "OBJ_DIM",
]
