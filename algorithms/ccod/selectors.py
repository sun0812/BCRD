# -*- coding: utf-8 -*-
"""CCOD 诊断阶段使用的确定性候选排序器。"""

from __future__ import annotations

from copy import deepcopy
import json
from typing import Any, Dict, Mapping, Sequence, Tuple, Union

from schedulers.state_replay import canonical_json_bytes, sha256_json


UNIFORM_RANK_SCHEMA_VERSION = "eosbench-ccod-uniform-rank-v1"


class SelectorError(ValueError):
    """候选排序输入不满足确定性契约时抛出的异常。"""


def _canonical_action_key(
    action_key: Mapping[str, Any],
) -> Tuple[bytes, Dict[str, Any]]:
    """归一为规范字节及仅由该字节恢复的防御性副本。"""
    if not isinstance(action_key, Mapping):
        raise SelectorError("ActionKey 必须是映射")
    try:
        encoded = canonical_json_bytes(dict(action_key))
        normalized = json.loads(encoded.decode("utf-8"))
    except (TypeError, ValueError, UnicodeError) as exc:
        raise SelectorError(f"ActionKey 无法规范序列化: {exc}") from exc
    if not isinstance(normalized, dict):
        raise SelectorError("ActionKey 的规范表示必须是对象")
    return encoded, normalized


def _normalized_action_keys(
    action_keys: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[str, Any], ...]:
    """按规范字节去重，并返回与输入顺序无关的 ActionKey 集合。"""
    if isinstance(action_keys, (str, bytes, bytearray)):
        raise SelectorError("候选集合必须是 ActionKey 序列")
    by_bytes: Dict[bytes, Dict[str, Any]] = {}
    for action_key in action_keys:
        encoded, normalized = _canonical_action_key(action_key)
        by_bytes.setdefault(encoded, normalized)
    return tuple(by_bytes[encoded] for encoded in sorted(by_bytes))


def _normalized_observed_action(
    observed_action_keys: Union[
        Mapping[str, Any], Sequence[Mapping[str, Any]]
    ],
) -> Tuple[bytes, Dict[str, Any]]:
    """读取唯一 observed 动作，并兼容旧版单元素序列接口。"""
    if isinstance(observed_action_keys, Mapping):
        raw_actions = (observed_action_keys,)
    else:
        if isinstance(observed_action_keys, (str, bytes, bytearray)):
            raise SelectorError("observed ActionKey 必须是映射或单元素序列")
        raw_actions = tuple(observed_action_keys)

    unique: Dict[bytes, Dict[str, Any]] = {}
    for action_key in raw_actions:
        encoded, normalized = _canonical_action_key(action_key)
        unique.setdefault(encoded, normalized)
    if not unique:
        raise SelectorError("必须提供 observed ActionKey")
    if len(unique) != 1:
        raise SelectorError("每个状态只能提供一个唯一 observed ActionKey")
    encoded = next(iter(unique))
    return encoded, unique[encoded]


def _validate_rank_inputs(state_hash: str, seed: int) -> None:
    """校验会进入稳定哈希载荷的公共字段。"""
    if not isinstance(state_hash, str) or not state_hash:
        raise SelectorError("state_hash 必须是非空字符串")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise SelectorError("seed 必须是严格整数")


def stable_uniform_rank(
    state_hash: str,
    action_keys: Sequence[Mapping[str, Any]],
    *,
    seed: int,
) -> Tuple[Dict[str, Any], ...]:
    """按内容哈希产生与输入顺序无关的 seeded-uniform 全排列。

    排名哈希的载荷固定为 ``schema_version``、``seed``、``state_hash`` 与
    ``action_key``。极小概率的哈希碰撞再由规范字节稳定破平局。
    """
    _validate_rank_inputs(state_hash, seed)
    normalized = _normalized_action_keys(action_keys)

    def rank_key(action_key: Mapping[str, Any]) -> Tuple[str, bytes]:
        encoded = canonical_json_bytes(dict(action_key))
        digest = sha256_json(
            {
                "schema_version": UNIFORM_RANK_SCHEMA_VERSION,
                "seed": seed,
                "state_hash": state_hash,
                "action_key": dict(action_key),
            }
        )
        return digest, encoded

    return tuple(
        deepcopy(action_key)
        for action_key in sorted(normalized, key=rank_key)
    )


def diagnostic_query_prefix(
    state_hash: str,
    action_keys: Sequence[Mapping[str, Any]],
    observed_action_keys: Union[
        Mapping[str, Any], Sequence[Mapping[str, Any]]
    ],
    *,
    budget: int,
    seed: int,
) -> Tuple[Dict[str, Any], ...]:
    """强制 observed 与 SKIP，再稳定补位至 ``min(B, C)``。

    ``C`` 是按 ActionKey 规范字节去重后的候选数。返回项包含
    ``action_key``、``selection_sources`` 与 ``query_rank``；当 observed
    本身就是 SKIP 时只查询一次，并按固定次序保留两个来源标签。
    """
    if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
        raise SelectorError("budget 必须是正整数")
    _validate_rank_inputs(state_hash, seed)
    normalized = _normalized_action_keys(action_keys)
    if not normalized:
        raise SelectorError("候选集合不能为空")
    by_bytes = {
        canonical_json_bytes(action_key): action_key for action_key in normalized
    }

    observed_bytes, _ = _normalized_observed_action(observed_action_keys)
    if observed_bytes not in by_bytes:
        raise SelectorError("observed ActionKey 不在当前候选集合")
    skip_matches = [
        action_key for action_key in normalized if action_key.get("kind") == "skip"
    ]
    if len(skip_matches) != 1:
        raise SelectorError("去重后的候选集合必须恰好包含一个 SKIP")
    skip_bytes = canonical_json_bytes(skip_matches[0])

    # 先固定锚点次序；同一动作的来源标签固定为 observed、skip。
    selected_order = [observed_bytes]
    sources_by_key: Dict[bytes, list[str]] = {observed_bytes: ["observed"]}
    if skip_bytes in sources_by_key:
        sources_by_key[skip_bytes].append("skip")
    else:
        selected_order.append(skip_bytes)
        sources_by_key[skip_bytes] = ["skip"]

    target = min(budget, len(normalized))
    if len(selected_order) > target:
        raise SelectorError("budget 小于 observed/SKIP 去重后的锚点数")

    # uniform 只负责补位，锚点进入其前缀也不重复标注来源。
    for action_key in stable_uniform_rank(state_hash, normalized, seed=seed):
        if len(selected_order) >= target:
            break
        encoded = canonical_json_bytes(action_key)
        if encoded in sources_by_key:
            continue
        selected_order.append(encoded)
        sources_by_key[encoded] = ["stable_uniform"]

    result = []
    for query_rank, encoded in enumerate(selected_order):
        result.append(
            {
                "action_key": deepcopy(by_bytes[encoded]),
                "selection_sources": list(sources_by_key[encoded]),
                "query_rank": query_rank,
            }
        )
    return tuple(result)
