# -*- coding: utf-8 -*-
"""
scripts/validate_trajectory_schema.py

轨迹 JSONL 文件的流式 schema 校验脚本。

输入：scripts/batch_export_trajectories.py 产出的 jsonl 文件。
作用：在前 ``--max_rows`` 行（或 0 表示扫全文件）上验证以下内容：

    - 所有必填的顶层字段是否齐全
    - ``state_features`` 是长度为 5 的浮点列表
    - ``candidate_features`` 中每条候选向量长度为 10
    - ``len(candidate_features) == len(candidate_keys) == len(valid_mask)``
    - ``expert_action_index`` 是落在合法区间 [0, num_candidates) 内的整数
    - SKIP 约定：``candidate_keys[0].is_skip == True``，
      且 ``candidate_features[0][0] == 1.0``（is_skip 标志位）

校验结果以 Markdown 报告形式写入 ``--report`` 指定的路径；脚本只读，
不会修改任何数据。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# 每一行 jsonl 样本必须出现的顶层字段集合
REQUIRED_FIELDS = [
    "scenario_id",
    "schedule_file",
    "solver_name",
    "class_id",
    "objective_tag",
    "objective_weights",
    "timestep",
    "state_features",
    "candidate_features",
    "candidate_keys",
    "valid_mask",
    "expert_action_index",
    "reward",
    "future_return_H",
    "schedule_metrics",
]

# 维度约定，必须与 batch_export_trajectories.py 完全一致
STATE_DIM = 5
CAND_DIM = 10


@dataclass
class ValidationStats:
    """累计单次校验过程中收集到的所有计数 / 样例。"""

    rows_checked: int = 0  # 已检查的行数
    rows_with_all_fields: int = 0  # 必填字段齐全的行数
    rows_failing_state_dim: int = 0  # state_features 维度错误的行数
    rows_failing_cand_dim: int = 0  # 任意候选维度错误的行数
    rows_failing_length_consistency: int = 0  # cands / keys / mask 长度不一致的行数
    rows_failing_label_range: int = 0  # expert_action_index 越界的行数
    rows_failing_skip_convention: int = 0  # SKIP 约定不成立的行数
    missing_field_counts: Counter = field(default_factory=Counter)
    skip_index0_count: int = 0  # candidate_keys[0] 标识为 skip 的行数
    expert_at_skip_count: int = 0  # expert_action_index == 0 的行数
    expert_at_positive_count: int = 0  # expert_action_index >= 1 的行数
    # 进一步细分 SKIP 标签的来源（便于诊断）
    expert_at_skip_and_skipped_by_json: int = 0
    expert_at_skip_and_infeasible: int = 0
    # 候选集合的大小分布
    num_cand_min: int = 10 ** 9
    num_cand_max: int = 0
    num_cand_sum: int = 0
    # 前 20 条错误信息，写入报告便于人工排查
    first_few_errors: List[str] = field(default_factory=list)

    def note_error(self, msg: str) -> None:
        """登记一条错误信息，最多保留 20 条避免报告过大。"""
        if len(self.first_few_errors) < 20:
            self.first_few_errors.append(msg)


def validate_row(idx: int, row: Dict[str, Any], stats: ValidationStats) -> None:
    """对单行 jsonl 进行结构验证，并把统计结果累加到 ``stats``。"""
    stats.rows_checked += 1

    # ---------- 1) 必填字段
    missing = [k for k in REQUIRED_FIELDS if k not in row]
    if missing:
        for k in missing:
            stats.missing_field_counts[k] += 1
        stats.note_error(f"row {idx}: missing fields {missing}")
        return
    stats.rows_with_all_fields += 1

    # ---------- 2) state_features 维度
    sf = row["state_features"]
    if not isinstance(sf, list) or len(sf) != STATE_DIM:
        stats.rows_failing_state_dim += 1
        stats.note_error(f"row {idx}: state_features len={len(sf) if isinstance(sf, list) else 'NA'}, expected {STATE_DIM}")

    cf = row["candidate_features"]
    ck = row["candidate_keys"]
    vm = row["valid_mask"]

    # ---------- 3) 三个列表的类型 + 长度一致性
    if not (isinstance(cf, list) and isinstance(ck, list) and isinstance(vm, list)):
        stats.rows_failing_length_consistency += 1
        stats.note_error(f"row {idx}: candidate_*/valid_mask types {type(cf).__name__}/{type(ck).__name__}/{type(vm).__name__}")
        return

    if not (len(cf) == len(ck) == len(vm)):
        stats.rows_failing_length_consistency += 1
        stats.note_error(f"row {idx}: lens cand_feat={len(cf)} keys={len(ck)} mask={len(vm)}")

    # ---------- 4) 每个候选向量的维度
    bad_dim = next((i for i, row_ in enumerate(cf) if not isinstance(row_, list) or len(row_) != CAND_DIM), None)
    if bad_dim is not None:
        stats.rows_failing_cand_dim += 1
        stats.note_error(f"row {idx}: candidate_features[{bad_dim}] dim={len(cf[bad_dim])}, expected {CAND_DIM}")

    # 候选数分布统计（min / mean / max）
    n = len(cf)
    stats.num_cand_min = min(stats.num_cand_min, n)
    stats.num_cand_max = max(stats.num_cand_max, n)
    stats.num_cand_sum += n

    # ---------- 5) expert_action_index 必须落在合法范围
    label = row["expert_action_index"]
    if not isinstance(label, int) or not (0 <= label < n):
        stats.rows_failing_label_range += 1
        stats.note_error(f"row {idx}: expert_action_index={label}, num_cands={n}")
        return

    # ---------- 6) SKIP 约定：candidate_keys[0] 必须是 skip 槽位
    skip_marker_ok = False
    if isinstance(ck[0], dict) and ck[0].get("is_skip") is True:
        skip_marker_ok = True
    if not skip_marker_ok:
        stats.rows_failing_skip_convention += 1
        stats.note_error(f"row {idx}: candidate_keys[0]={ck[0]}, expected is_skip:True")
    else:
        stats.skip_index0_count += 1
        # 同时核对 candidate_features[0][0] 是否为 1.0（is_skip 标志位）
        if cf[0][0] != 1.0:
            stats.note_error(f"row {idx}: cand_features[0][0]={cf[0][0]}, expected 1.0 (is_skip flag)")

    # ---------- 7) 标签分布（SKIP vs positive）
    if label == 0:
        stats.expert_at_skip_count += 1
        dbg = row.get("_debug") or {}
        # 进一步区分两种 SKIP：JSON 原本就标注为 unassigned 的，
        # 与 replay 时被 is_feasible_assignment 拒绝才退化为 SKIP 的。
        if dbg.get("expert_skipped_by_json"):
            stats.expert_at_skip_and_skipped_by_json += 1
        if dbg.get("expert_infeasible_in_replay"):
            stats.expert_at_skip_and_infeasible += 1
    else:
        stats.expert_at_positive_count += 1


def write_report(report_path: Path, stats: ValidationStats, input_path: Path, max_rows: int) -> None:
    """根据累计的 stats 渲染一份 Markdown 报告。"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    n = stats.rows_checked or 1
    mean_cands = stats.num_cand_sum / n

    skip_share = stats.expert_at_skip_count / n
    pos_share = stats.expert_at_positive_count / n

    lines: List[str] = []
    lines.append("# Trajectory Schema Validation Report")
    lines.append("")
    lines.append("> Generated by `scripts/validate_trajectory_schema.py`.")
    lines.append("")
    lines.append("## 1. Run parameters")
    lines.append("")
    lines.append("```")
    lines.append(f"input    : {input_path}")
    lines.append(f"max_rows : {max_rows if max_rows > 0 else 'ALL'}")
    lines.append(f"checked  : {stats.rows_checked}")
    lines.append("```")
    lines.append("")

    lines.append("## 2. Required-field coverage")
    lines.append("")
    lines.append(f"- rows with all {len(REQUIRED_FIELDS)} required fields : **{stats.rows_with_all_fields} / {n}**")
    if stats.missing_field_counts:
        lines.append("- missing-field histogram:")
        for k, v in stats.missing_field_counts.most_common():
            lines.append(f"    - `{k}` missing in {v} rows")
    lines.append("")

    lines.append("## 3. Shape checks")
    lines.append("")
    lines.append(f"- rows failing `len(state_features) == {STATE_DIM}`           : **{stats.rows_failing_state_dim}**")
    lines.append(f"- rows failing per-candidate `len == {CAND_DIM}`              : **{stats.rows_failing_cand_dim}**")
    lines.append(f"- rows failing `len(cands)==len(keys)==len(mask)`             : **{stats.rows_failing_length_consistency}**")
    lines.append(f"- rows failing `0 <= expert_action_index < num_candidates`    : **{stats.rows_failing_label_range}**")
    lines.append("")

    lines.append("## 4. SKIP convention")
    lines.append("")
    lines.append(f"- rows with `candidate_keys[0].is_skip == True`               : **{stats.skip_index0_count} / {n}**")
    lines.append(f"- rows failing the SKIP convention                            : **{stats.rows_failing_skip_convention}**")
    if stats.rows_failing_skip_convention == 0 and stats.skip_index0_count == n:
        lines.append("")
        lines.append("✅ exporter **已经把 SKIP 候选放在 index 0**，并将 `candidate_features[0][0] == 1.0` 作为 `is_skip` 标志位。")
        lines.append("")
        lines.append("所以训练 dataloader **不应再次** prepend SKIP，做候选下采样时需保证：")
        lines.append("")
        lines.append("- index 0 的 SKIP 槽位永远保留；")
        lines.append("- expert 候选（`expert_action_index` 指向的那条）永远保留；")
        lines.append("- 下采样完成后把 expert label 重映射到新的位置。")
    else:
        lines.append("")
        lines.append("⚠️ 存在没有显式 SKIP 槽位的行。训练 dataloader 必须自动检测并 prepend 合成 SKIP（`is_skip=1.0`，其余维度全 0）。")
    lines.append("")

    lines.append("## 5. Label distribution")
    lines.append("")
    lines.append(f"- `expert_action_index == 0` (SKIP)            : **{stats.expert_at_skip_count} ({skip_share:.1%})**")
    lines.append(f"    - of which `_debug.expert_skipped_by_json` : {stats.expert_at_skip_and_skipped_by_json}")
    lines.append(f"    - of which `_debug.expert_infeasible_in_replay` : {stats.expert_at_skip_and_infeasible}")
    lines.append(f"- `expert_action_index >= 1` (positive action) : **{stats.expert_at_positive_count} ({pos_share:.1%})**")
    lines.append("")

    lines.append("## 6. Candidate-set size")
    lines.append("")
    lines.append(f"- `num_candidates_incl_skip` min  = {stats.num_cand_min}")
    lines.append(f"- `num_candidates_incl_skip` mean = {mean_cands:.1f}")
    lines.append(f"- `num_candidates_incl_skip` max  = {stats.num_cand_max}")
    lines.append("")

    if stats.first_few_errors:
        lines.append("## 7. First 20 errors / warnings")
        lines.append("")
        lines.append("```")
        for msg in stats.first_few_errors:
            lines.append(msg)
        lines.append("```")
        lines.append("")

    # 汇总错误数；任何一项 > 0 都视为 FAIL
    n_err = (
        stats.rows_failing_state_dim
        + stats.rows_failing_cand_dim
        + stats.rows_failing_length_consistency
        + stats.rows_failing_label_range
        + stats.rows_failing_skip_convention
        + sum(stats.missing_field_counts.values())
    )
    lines.append("## 8. Verdict")
    lines.append("")
    if n_err == 0:
        lines.append("**PASS** - schema 通过；可以进入 BC 训练。")
    else:
        lines.append(f"**FAIL** - 发现 {n_err} 项 schema 问题，详见上文。")
    lines.append("")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--max_rows", type=int, default=100, help="0 表示扫全文件")
    parser.add_argument("--report", type=Path, default=Path("docs/trajectory_schema_validation_report.md"))
    args = parser.parse_args()

    if not args.input.exists():
        print(f"[fatal] input not found: {args.input}", file=sys.stderr)
        return 2

    stats = ValidationStats()
    max_rows = int(args.max_rows)

    # 一行一行流式读取，避免一次性把 GB 级 jsonl 全部加载进内存
    with open(args.input, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_rows > 0 and i >= max_rows:
                break
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                stats.note_error(f"line {i}: json parse error: {e}")
                continue
            validate_row(i, row, stats)

    write_report(args.report, stats, args.input, max_rows)
    print(f"[info] rows_checked={stats.rows_checked}")
    print(f"[info] expert_at_skip={stats.expert_at_skip_count}, expert_at_positive={stats.expert_at_positive_count}")
    print(f"[info] candidates min/mean/max = {stats.num_cand_min}/{stats.num_cand_sum/(stats.rows_checked or 1):.1f}/{stats.num_cand_max}")
    print(f"[info] report -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
