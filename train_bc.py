# -*- coding: utf-8 -*-
"""
train_bc.py

BCRD 项目的第一版 BC（行为克隆）训练入口。

把 ``scripts/batch_export_trajectories.py`` 产出的 JSONL 当成监督数据，
流式训练一个轻量 MLP candidate scorer（``models/bc_policy.BCPolicy``）。

按照项目 brief 的要求，本脚本只做 BC 基线，**不实现**：

* counterfactual regret label / rollout（BCRD 的核心模块，后续才做）
* 多 expert 蒸馏中的 oracle picking（避免数据泄漏）
* 接入 ``main_scheduler.py`` 跑 learned scheduler

输出物::

    <out>/checkpoint.pt   验证集 loss 最佳的模型权重 + 训练参数
    <out>/train_log.jsonl 每 N batch / 每 epoch 的训练日志
    <out>/metrics.json    汇总所有 epoch 的训练 / 验证指标
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_utils.trajectory_dataset import (
    TrajectoryStreamingDataset,
    CAND_FEAT_DIM,
    OBJ_DIM,
    STATE_FEAT_DIM,
)
from models.bc_policy import BCPolicy


# 当 --split_by_schedule 打开时，默认留出做 val 的 3 个 schedule
# （一个 MIP、一个 SA、一个 ACO，覆盖不同 solver 类型）
DEFAULT_VAL_SCHEDULES = (
    "scheduler_Scenario_S1_Sats20_M500_T0.5d_dist0_c1_mip_p1_c0_t0_b0.json",
    "scheduler_Scenario_S1_Sats20_M500_T0.5d_dist0_c3_sa_p0.25_c0.25_t0.25_b0.25.json",
    "scheduler_Scenario_S1_Sats20_M500_T0.5d_dist0_c3_aco_p0.25_c0.25_t0.25_b0.25.json",
)


# =============================================================================
# 工具函数
# =============================================================================


def _resolve_device(name: str) -> torch.device:
    """根据 CLI 字符串选择实际可用的设备，不可用时降级到 CPU 并打印提示。"""
    n = (name or "").strip().lower()
    if n == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        print("[warn] MPS unavailable, falling back to CPU", flush=True)
        return torch.device("cpu")
    if n == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("[warn] CUDA unavailable, falling back to CPU", flush=True)
        return torch.device("cpu")
    return torch.device("cpu")


class MetricAccumulator:
    """逐 batch 累计训练 / 验证指标，flush 时统一汇总。"""

    def __init__(self) -> None:
        # 主指标
        self.loss_sum: float = 0.0
        self.n_examples: int = 0
        self.top1_hits: int = 0
        self.top5_hits: int = 0
        # SKIP 相关
        self.skip_count: int = 0           # 真值标签 == 0 (SKIP) 的样本数
        self.skip_pred_count: int = 0      # 模型预测为 0 的样本数
        self.skip_true_pos: int = 0        # 真预测且为 SKIP
        # 非 SKIP 相关
        self.non_skip_count: int = 0
        self.non_skip_hits: int = 0
        # 候选数分布（采样前 / 采样后）
        self.n_before_sum: int = 0
        self.n_after_sum: int = 0
        self.n_steps: int = 0

    def update(self, logits: torch.Tensor, labels: torch.Tensor, loss: float,
               n_before: torch.Tensor, n_after: torch.Tensor) -> None:
        """传入一个 batch 的 logits + labels + loss，累加各项指标。"""
        B = labels.numel()
        self.n_examples += B
        self.loss_sum += float(loss) * B

        # top-1 / top-5：candidate 维上取最大若干个并对照 label
        topk = min(5, logits.size(-1))
        _, top1 = logits.topk(1, dim=-1)
        _, topk_idx = logits.topk(topk, dim=-1)
        labels_v = labels.view(-1, 1)
        self.top1_hits += int((top1 == labels_v).any(dim=-1).sum().item())
        self.top5_hits += int((topk_idx == labels_v).any(dim=-1).sum().item())

        # SKIP / non-SKIP 子集统计
        preds = top1.squeeze(-1)
        is_skip_truth = labels == 0
        is_skip_pred = preds == 0
        self.skip_count += int(is_skip_truth.sum().item())
        self.skip_pred_count += int(is_skip_pred.sum().item())
        self.skip_true_pos += int((is_skip_truth & is_skip_pred).sum().item())
        non_skip_truth = ~is_skip_truth
        self.non_skip_count += int(non_skip_truth.sum().item())
        self.non_skip_hits += int(((preds == labels) & non_skip_truth).sum().item())

        # 候选数分布
        self.n_before_sum += int(n_before.sum().item())
        self.n_after_sum += int(n_after.sum().item())
        self.n_steps += B

    def summary(self) -> Dict[str, float]:
        """生成可序列化的 dict，方便写日志 / 写 metrics.json。"""
        n = max(1, self.n_examples)
        skip_recall = (self.skip_true_pos / self.skip_count) if self.skip_count > 0 else float("nan")
        skip_precision = (self.skip_true_pos / self.skip_pred_count) if self.skip_pred_count > 0 else float("nan")
        skip_acc = skip_recall  # brief 中 skip_accuracy 与 skip_recall 同义
        non_skip_acc = (self.non_skip_hits / self.non_skip_count) if self.non_skip_count > 0 else float("nan")
        return {
            "loss": self.loss_sum / n,
            "top1_action_accuracy": self.top1_hits / n,
            "top5_action_accuracy": self.top5_hits / n,
            "skip_accuracy": skip_acc,
            "skip_precision": skip_precision,
            "skip_recall": skip_recall,
            "non_skip_accuracy": non_skip_acc,
            "mean_num_candidates_before_sampling": self.n_before_sum / max(1, self.n_steps),
            "mean_num_candidates_after_sampling": self.n_after_sum / max(1, self.n_steps),
            "examples": float(self.n_examples),
        }


# =============================================================================
# 单 epoch 训练 / 验证
# =============================================================================


def run_epoch(
    *,
    model: BCPolicy,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    criterion: nn.Module,
    device: torch.device,
    use_objective_weights: bool,
    is_train: bool,
    log_every: int = 50,
    epoch_idx: int = 0,
    train_log_fp=None,
) -> Dict[str, float]:
    """跑一遍完整的 train 或 val epoch，返回汇总指标。"""
    # 训练时 model.train() 打开 Dropout；验证时切到 eval 模式
    model.train(mode=is_train)
    acc = MetricAccumulator()
    batch_i = 0
    t0 = time.time()

    for batch in loader:
        # 把 batch 中的张量挪到目标 device 上
        state = batch["state"].to(device, non_blocking=True)
        cands = batch["cands"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        n_before = batch["n_before"]
        n_after = batch["n_after"]
        # 是否拼接目标权重作为模型输入
        obj_w = batch["obj_w"].to(device, non_blocking=True) if use_objective_weights else None

        # 前向：得到 (B, N_max) 的候选分数，padding 位置已经被设为 -inf
        logits = model(state=state, candidates=cands, objective_weights=obj_w, valid_mask=mask)
        # 防御性检查：如果整行都是 -inf 会让 CrossEntropy 产生 NaN。
        # 因为 SKIP 槽位始终 valid，此处理论上不会触发，加个 assertion 防止意外。
        if torch.isneginf(logits).all(dim=-1).any():
            raise RuntimeError("All-masked row detected; check valid_mask construction")

        loss = criterion(logits, label)

        # 训练才反传 + 梯度更新；验证只算指标
        if is_train:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        # 把 detached tensor 传入累加器，避免追踪不必要的计算图
        acc.update(logits.detach(), label.detach(), float(loss.item()), n_before, n_after)
        batch_i += 1

        # 训练中按 log_every 节奏打印阶段性指标
        if is_train and (batch_i % log_every == 0):
            sub = acc.summary()
            msg = {
                "phase": "train",
                "epoch": epoch_idx,
                "batch": batch_i,
                "loss": round(sub["loss"], 4),
                "top1": round(sub["top1_action_accuracy"], 4),
                "skip_acc": round(sub["skip_accuracy"], 4) if not math.isnan(sub["skip_accuracy"]) else None,
                "non_skip_acc": round(sub["non_skip_accuracy"], 4) if not math.isnan(sub["non_skip_accuracy"]) else None,
                "elapsed_s": round(time.time() - t0, 2),
            }
            print(f"[train] {json.dumps(msg)}", flush=True)
            if train_log_fp is not None:
                train_log_fp.write(json.dumps(msg) + "\n")
                train_log_fp.flush()

    return acc.summary()


# =============================================================================
# 主入口
# =============================================================================


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True, help="merged JSONL file")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_train_candidates", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--num_hidden_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--use_objective_weights", action="store_true")
    parser.add_argument("--no_objective_weights", action="store_true",
                        help="若同时给了 --use_objective_weights，此开关优先生效。")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--shuffle_buffer", type=int, default=1024)
    parser.add_argument("--hard_neg_top_k", type=int, default=0)
    parser.add_argument("--split_by_schedule", action="store_true",
                        help="按 schedule_file 把若干个 schedule 整体留出做 val，而不是哈希切分。")
    parser.add_argument("--val_schedules", nargs="*", default=None,
                        help="（可选）显式指定要做 val 的 schedule_file basename 列表。")
    parser.add_argument("--train_fraction", type=float, default=0.8)
    parser.add_argument("--log_every", type=int, default=50)
    args = parser.parse_args()

    # 两个开关都给的时候，--no_objective_weights 优先
    if args.no_objective_weights:
        use_objective_weights = False
    else:
        use_objective_weights = bool(args.use_objective_weights)

    args.out.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(args.device)
    torch.manual_seed(args.seed)

    # ---------- 构造训练 / 验证数据集
    split_method = "by_schedule" if args.split_by_schedule else "hash"
    val_schedules: Sequence[str]
    if args.split_by_schedule:
        val_schedules = tuple(args.val_schedules) if args.val_schedules else DEFAULT_VAL_SCHEDULES
        print(f"[info] val schedules ({len(val_schedules)}):", flush=True)
        for s in val_schedules:
            print(f"        {s}", flush=True)
    else:
        val_schedules = tuple()

    train_ds = TrajectoryStreamingDataset(
        args.train,
        max_train_candidates=args.max_train_candidates,
        split="train",
        split_method=split_method,
        train_fraction=args.train_fraction,
        val_schedule_files=val_schedules,
        seed=args.seed,
        shuffle_buffer=args.shuffle_buffer,
        hard_neg_top_k=args.hard_neg_top_k,
    )
    val_ds = TrajectoryStreamingDataset(
        args.train,
        max_train_candidates=args.max_train_candidates,
        split="val",
        split_method=split_method,
        train_fraction=args.train_fraction,
        val_schedule_files=val_schedules,
        seed=args.seed,
        shuffle_buffer=args.shuffle_buffer,
        hard_neg_top_k=args.hard_neg_top_k,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # ---------- 构造模型
    model = BCPolicy(
        state_dim=STATE_FEAT_DIM,
        cand_dim=CAND_FEAT_DIM,
        obj_dim=OBJ_DIM,
        hidden=args.hidden,
        num_hidden_layers=args.num_hidden_layers,
        dropout=args.dropout,
        use_objective_weights=use_objective_weights,
    ).to(device)
    print(f"[info] model parameter count = {sum(p.numel() for p in model.parameters()):,}", flush=True)
    print(f"[info] device = {device}, use_objective_weights = {use_objective_weights}", flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    train_log_path = args.out / "train_log.jsonl"
    train_log_fp = open(train_log_path, "w", encoding="utf-8")

    epoch_metrics: List[Dict[str, Any]] = []
    best_val_loss = float("inf")

    try:
        for ep in range(int(args.epochs)):
            tr_t0 = time.time()
            # 训练一个 epoch
            train_sum = run_epoch(
                model=model, loader=train_loader, optimizer=optimizer,
                criterion=criterion, device=device,
                use_objective_weights=use_objective_weights,
                is_train=True, log_every=args.log_every,
                epoch_idx=ep, train_log_fp=train_log_fp,
            )
            tr_elapsed = time.time() - tr_t0

            # 验证集评估（no_grad 节省显存 / 加速）
            with torch.no_grad():
                val_sum = run_epoch(
                    model=model, loader=val_loader, optimizer=None,
                    criterion=criterion, device=device,
                    use_objective_weights=use_objective_weights,
                    is_train=False,
                )

            row = {
                "epoch": ep,
                "train": train_sum,
                "val": val_sum,
                "train_elapsed_s": round(tr_elapsed, 2),
            }
            epoch_metrics.append(row)
            print(f"[epoch {ep}] {json.dumps(row, default=lambda x: round(x, 4) if isinstance(x, float) else x)}", flush=True)
            train_log_fp.write(json.dumps({"phase": "epoch_end", **row}) + "\n")
            train_log_fp.flush()

            # 用 val loss 决定是否保存 checkpoint（best-on-val 策略）
            if val_sum["loss"] < best_val_loss:
                best_val_loss = val_sum["loss"]
                ck = {
                    "model_state": model.state_dict(),
                    "args": vars(args),
                    "epoch": ep,
                    "val": val_sum,
                    "train": train_sum,
                    "best_val_loss": best_val_loss,
                }
                torch.save(ck, args.out / "checkpoint.pt")
                print(f"[info] saved checkpoint at epoch {ep}, val_loss={best_val_loss:.4f}", flush=True)
    finally:
        train_log_fp.close()

    # 把整轮训练结果写入 metrics.json
    metrics_blob = {
        "best_val_loss": best_val_loss,
        "epoch_metrics": epoch_metrics,
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    }
    with open(args.out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_blob, f, indent=2, default=lambda x: round(x, 6) if isinstance(x, float) else x)
    print(f"[info] metrics -> {args.out/'metrics.json'}", flush=True)
    print(f"[info] log     -> {train_log_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
