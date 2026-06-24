# BC / BCRD 相关模型模块；当前仅有 BCPolicy 一个轻量 MLP 候选打分器
from .bc_policy import BCPolicy

__all__ = ["BCPolicy"]
