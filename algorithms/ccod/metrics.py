# -*- coding: utf-8 -*-
"""CCOD signal diagnostic 的无第三方依赖指标。"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Sequence


class DiagnosticMetricError(ValueError):
    """诊断指标输入非法时抛出的异常。"""


def finite_float_hex(value: float) -> str:
    """把有限浮点数编码为可精确往返的十六进制字符串。"""
    if isinstance(value, bool):
        raise DiagnosticMetricError("浮点值必须是有限数值")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise DiagnosticMetricError("浮点值必须是有限数值") from exc
    if not math.isfinite(numeric):
        raise DiagnosticMetricError("浮点值必须是有限数值")
    return numeric.hex()


def linear_quantile(values: Sequence[float], probability: float) -> float:
    """使用线性插值计算分位数，定义与常用 R-7/NumPy linear 一致。"""
    if not values:
        raise DiagnosticMetricError("分位数输入不能为空")
    if isinstance(probability, bool):
        raise DiagnosticMetricError("分位概率必须位于 [0, 1]")
    try:
        normalized_probability = float(probability)
    except (TypeError, ValueError) as exc:
        raise DiagnosticMetricError("分位概率必须位于 [0, 1]") from exc
    if not (0.0 <= normalized_probability <= 1.0):
        raise DiagnosticMetricError("分位概率必须位于 [0, 1]")
    try:
        ordered = sorted(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise DiagnosticMetricError("分位数输入必须全部有限") from exc
    if not all(math.isfinite(value) for value in ordered):
        raise DiagnosticMetricError("分位数输入必须全部有限")
    if len(ordered) == 1:
        return ordered[0]
    position = normalized_probability * float(len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - float(lower)
    return math.fsum(
        (
            ordered[lower] * (1.0 - fraction),
            ordered[upper] * fraction,
        )
    )


def summarize_state_q_values(
    q_values: Iterable[float],
    *,
    spread_threshold: float = 0.01,
) -> Dict[str, Any]:
    """汇总状态查询标签，并判断是否具有非平凡双尾 spread。"""
    try:
        values: List[float] = [float(value) for value in q_values]
    except (TypeError, ValueError) as exc:
        raise DiagnosticMetricError("Q 值必须全部有限") from exc
    if not values:
        raise DiagnosticMetricError("状态至少需要一个 Q 值")
    if not all(math.isfinite(value) for value in values):
        raise DiagnosticMetricError("Q 值必须全部有限")
    if isinstance(spread_threshold, bool):
        raise DiagnosticMetricError("spread_threshold 必须有限且非负")
    try:
        normalized_threshold = float(spread_threshold)
    except (TypeError, ValueError) as exc:
        raise DiagnosticMetricError("spread_threshold 必须有限且非负") from exc
    if not math.isfinite(normalized_threshold) or normalized_threshold < 0.0:
        raise DiagnosticMetricError("spread_threshold 必须有限且非负")
    p10 = linear_quantile(values, 0.10)
    p50 = linear_quantile(values, 0.50)
    p90 = linear_quantile(values, 0.90)
    spread = math.fsum((p90, -p10))
    spread = 0.0 if spread == 0.0 else spread
    return {
        "query_count": len(values),
        "min_q": min(values),
        "p10_q": p10,
        "median_q": p50,
        "p90_q": p90,
        "max_q": max(values),
        "p90_minus_p10": spread,
        "nontrivial_spread": spread >= normalized_threshold,
        "spread_threshold": normalized_threshold,
    }
