# -*- coding: utf-8 -*-
"""
bc/policy.py

BC（行为克隆）阶段使用的轻量 candidate scorer。

模型本身是一个极小的 MLP（约 20K 参数），对每个候选独立打一个标量分，
推理时在合法候选集合（valid_mask=1 的位置）上 argmax 选动作。

前向接口::

    scores = policy(state, candidates, objective_weights, valid_mask)

Tensor 形状（B = batch, N = 候选个数，含 SKIP，已对齐到固定 N_max）：

    state              : (B, state_dim=5)
    candidates         : (B, N, cand_dim=10)
    objective_weights  : (B, 4)  或  None
    valid_mask         : (B, N)  1=真实候选，0=padding 占位
    -> scores          : (B, N)  padding 位置被设为 -inf
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class BCPolicy(nn.Module):
    """带可选目标权重条件的逐候选 MLP 打分模型。

    把 ``[state, candidate, objective_weights]`` 拼接后送入两层 MLP，
    输出每个候选一个标量分。模型故意做得很小，是为了能在 CPU/MPS 上
    几分钟内完成 debug 级别的训练。
    """

    def __init__(
        self,
        state_dim: int = 5,
        cand_dim: int = 10,
        obj_dim: int = 4,
        hidden: int = 128,
        num_hidden_layers: int = 2,
        dropout: float = 0.1,
        use_objective_weights: bool = True,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.cand_dim = int(cand_dim)
        self.obj_dim = int(obj_dim)
        self.use_objective_weights = bool(use_objective_weights)

        # 输入维度 = state + candidate + (可选) objective_weights
        in_dim = state_dim + cand_dim + (obj_dim if use_objective_weights else 0)

        # 多层 MLP：[Linear -> ReLU -> Dropout] * num_hidden_layers -> Linear(=>1)
        layers: list[nn.Module] = []
        prev = in_dim
        for _ in range(int(num_hidden_layers)):
            layers.append(nn.Linear(prev, hidden))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(float(dropout)))
            prev = hidden
        layers.append(nn.Linear(prev, 1))
        self.mlp = nn.Sequential(*layers)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """权重初始化。"""
        # 隐藏层用 PyTorch 默认风格的 Kaiming uniform 初始化
        for i, m in enumerate(self.mlp):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=5 ** 0.5, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # 最后一层用很小的 Xavier 初始化，让训练刚开始时 logits 接近 0，
        # 初始 CrossEntropy 接近 ln(N) 而不是几千；避免被一开始的大 logits 干扰。
        last = None
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                last = m
        if last is not None:
            nn.init.xavier_uniform_(last.weight, gain=0.01)
            if last.bias is not None:
                nn.init.zeros_(last.bias)

    def forward(
        self,
        state: torch.Tensor,
        candidates: torch.Tensor,
        objective_weights: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """对每个候选打分。padding 槽位会被强行设为 ``-inf``。"""
        # 兼容单样本（无 batch 维）的输入
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if candidates.dim() == 2:
            candidates = candidates.unsqueeze(0)
        B, N, _ = candidates.shape

        # state 是 per-sample 的 (B, state_dim)，把它扩展成 (B, N, state_dim)
        # 才能与 candidates 在最后一维拼接
        state_b = state.unsqueeze(1).expand(B, N, self.state_dim)

        feats = [state_b, candidates]
        if self.use_objective_weights:
            # 目标权重同样是 per-sample 的，需要广播到每个候选上
            if objective_weights is None:
                # 如果调用方没传，就给一个全 0 张量保持维度对齐
                obj = candidates.new_zeros((B, self.obj_dim))
            else:
                obj = objective_weights
                if obj.dim() == 1:
                    obj = obj.unsqueeze(0)
            obj_b = obj.unsqueeze(1).expand(B, N, self.obj_dim)
            feats.append(obj_b)

        x = torch.cat(feats, dim=-1)  # (B, N, in_dim)
        scores = self.mlp(x).squeeze(-1)  # (B, N)

        # padding 位置的分强行设为 -inf，让 argmax / softmax 自动跳过它们
        if valid_mask is not None:
            neg_inf = torch.full_like(scores, float("-inf"))
            scores = torch.where(valid_mask > 0, scores, neg_inf)
        return scores

    def predict(
        self,
        state: torch.Tensor,
        candidates: torch.Tensor,
        objective_weights: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """在合法候选上 argmax，返回 shape 为 ``(B,)`` 的 LongTensor。"""
        scores = self.forward(state, candidates, objective_weights, valid_mask)
        return scores.argmax(dim=-1)
