# -*- coding: utf-8 -*-
"""CCOD 反事实标签的内容寻址缓存。"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, Mapping, Optional

from algorithms.ccod.continuation import (
    CONTINUATION_SCHEMA_VERSION,
    CounterfactualError,
    CounterfactualResult,
    ContinuationConfig,
    QUERY_IDENTITY_SCHEMA_VERSION,
    build_query_identity,
)
from schedulers.state_replay import (
    ConstraintConfig,
    EnumeratorConfig,
    ObjectiveConfig,
    canonical_json_bytes,
    sha256_json,
)


LABEL_CACHE_SCHEMA_VERSION = "eosbench-ccod-label-cache-v1"
COUNTERFACTUAL_RESULT_FIELDS = frozenset(
    {
        "schema_version",
        "query_key",
        "state_hash",
        "step",
        "task_id",
        "requested_horizon",
        "decisions_executed",
        "terminated_by_task_exhaustion",
        "forced_action_key",
        "rollout_action_keys",
        "rollout_action_keys_hash",
        "objective_score_hexes",
        "base_score_hex",
        "forced_score_hex",
        "final_score_hex",
        "q_h_hex",
        "final_schedule_hash",
        "final_schedule_runtime_hash",
        "continuation_hash",
        "continuation_implementation_hash",
        "constraint_hash",
        "enumerator_hash",
        "objective_hash",
        "problem_runtime_fingerprint",
        "result_hash",
    }
)


def _canonical_object_copy(value: Mapping[str, Any]) -> Dict[str, Any]:
    """用规范 JSON 往返生成深层快照，消除嵌套对象的 TOCTOU。"""
    loaded = json.loads(canonical_json_bytes(dict(value)).decode("utf-8"))
    if not isinstance(loaded, dict):
        raise CounterfactualError("缓存身份必须是 JSON object")
    return loaded


def _validate_result_identity(
    identity: Mapping[str, Any],
    result: Mapping[str, Any],
    key: str,
) -> None:
    """闭合校验查询身份、结果清单和派生数值。"""
    if identity.get("schema_version") != QUERY_IDENTITY_SCHEMA_VERSION:
        raise CounterfactualError("查询身份 schema_version 不受支持")
    if result.get("schema_version") != CONTINUATION_SCHEMA_VERSION:
        raise CounterfactualError("结果 schema_version 不受支持")
    if result.get("query_key") != key:
        raise CounterfactualError("结果 query_key 与缓存身份不一致")

    field_pairs = (
        ("state_hash", "state_hash"),
        ("forced_action_key", "action_key"),
        ("constraint_hash", "constraint_hash"),
        ("enumerator_hash", "enumerator_hash"),
        ("objective_hash", "objective_hash"),
        ("requested_horizon", "horizon"),
        ("continuation_hash", "continuation_hash"),
        (
            "continuation_implementation_hash",
            "continuation_implementation_hash",
        ),
    )
    for result_field, identity_field in field_pairs:
        if result.get(result_field) != identity.get(identity_field):
            raise CounterfactualError(
                f"结果 {result_field} 与缓存身份不一致"
            )

    rollout = result.get("rollout_action_keys")
    if not isinstance(rollout, list) or not rollout:
        raise CounterfactualError("缓存结果缺少 rollout 动作序列")
    if rollout[0] != result.get("forced_action_key"):
        raise CounterfactualError("缓存结果的强制动作与 rollout 首动作不一致")
    if result.get("rollout_action_keys_hash") != sha256_json(rollout):
        raise CounterfactualError("缓存结果的 rollout 动作哈希不一致")

    decisions = result.get("decisions_executed")
    horizon = result.get("requested_horizon")
    if (
        isinstance(decisions, bool)
        or not isinstance(decisions, int)
        or isinstance(horizon, bool)
        or not isinstance(horizon, int)
        or not (1 <= decisions <= horizon)
        or len(rollout) != decisions
    ):
        raise CounterfactualError("缓存结果的决策数或视野非法")

    score_hexes = result.get("objective_score_hexes")
    if not isinstance(score_hexes, list) or len(score_hexes) != decisions + 1:
        raise CounterfactualError("缓存结果的目标分数序列长度非法")
    endpoints = (
        (0, "base_score_hex"),
        (1, "forced_score_hex"),
        (-1, "final_score_hex"),
    )
    for index, field_name in endpoints:
        if score_hexes[index] != result.get(field_name):
            raise CounterfactualError(f"缓存结果的 {field_name} 与分数序列不一致")
    try:
        base_score = float.fromhex(str(result["base_score_hex"]))
        final_score = float.fromhex(str(result["final_score_hex"]))
        q_h = float.fromhex(str(result["q_h_hex"]))
    except (KeyError, ValueError) as exc:
        raise CounterfactualError("缓存结果包含非法浮点十六进制值") from exc
    if not all(math.isfinite(value) for value in (base_score, final_score, q_h)):
        raise CounterfactualError("缓存结果包含非有限目标值")
    expected_q_h = math.fsum((final_score, -base_score))
    expected_q_h = 0.0 if expected_q_h == 0.0 else expected_q_h
    if q_h.hex() != expected_q_h.hex():
        raise CounterfactualError("缓存结果的 q_h 与目标差值不一致")


def validate_counterfactual_result_manifest(
    identity: Mapping[str, Any],
    result: Mapping[str, Any],
) -> Dict[str, Any]:
    """规范化并闭合验证一个可独立审计的反事实结果清单。

    cache 读写层与执行汇总层必须复用同一校验入口，避免同一 ``result_hash``
    在两个模块中获得不同解释。返回值是与调用方嵌套对象隔离的普通 JSON
    快照，可安全写入缓存或结果审计文件。
    """
    normalized_identity = _canonical_object_copy(identity)
    normalized_result = _canonical_object_copy(result)
    if set(normalized_result) != COUNTERFACTUAL_RESULT_FIELDS:
        missing = sorted(COUNTERFACTUAL_RESULT_FIELDS - set(normalized_result))
        extra = sorted(set(normalized_result) - COUNTERFACTUAL_RESULT_FIELDS)
        raise CounterfactualError(
            f"结果字段集合不一致: missing={missing}, extra={extra}"
        )
    key = cache_key(normalized_identity)
    stored_hash = normalized_result.get("result_hash")
    unhashed = {
        name: value
        for name, value in normalized_result.items()
        if name != "result_hash"
    }
    if stored_hash != sha256_json(unhashed):
        raise CounterfactualError("结果 result_hash 不一致")
    _validate_result_identity(normalized_identity, normalized_result, key)
    return normalized_result


def build_cache_identity(
    *,
    state_hash: str,
    action_key: Mapping[str, Any],
    constraint_config: ConstraintConfig,
    enumerator_config: EnumeratorConfig,
    objective_config: ObjectiveConfig,
    continuation_config: ContinuationConfig,
) -> Dict[str, Any]:
    """构造与论文 cache key 一致的完整身份载荷。"""
    return build_query_identity(
        state_hash=state_hash,
        action_key=action_key,
        constraint_hash=constraint_config.hash,
        enumerator_hash=enumerator_config.hash,
        objective_hash=objective_config.hash,
        continuation_config=continuation_config,
    )


def cache_key(identity: Mapping[str, Any]) -> str:
    """返回身份载荷的内容寻址键。"""
    return sha256_json(dict(identity))


class CounterfactualLabelCache:
    """按 SHA-256 键原子写入并校验反事实标签。"""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def path_for_key(self, key: str) -> Path:
        digest = str(key).removeprefix("sha256:")
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise CounterfactualError(f"invalid cache key: {key!r}")
        return self.root / digest[:2] / f"{digest}.json"

    def load(self, identity: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        """读取并完整验证缓存；不存在时返回 ``None``。"""
        expected_identity = _canonical_object_copy(identity)
        key = cache_key(expected_identity)
        path = self.path_for_key(key)
        if not path.is_file():
            return None
        try:
            with path.open("r", encoding="utf-8") as handle:
                record = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise CounterfactualError(f"failed to read cache record {path}: {exc}") from exc
        if not isinstance(record, dict):
            raise CounterfactualError(f"cache record {path} must be a JSON object")
        stored_hash = record.get("record_hash")
        unhashed = {name: value for name, value in record.items() if name != "record_hash"}
        if stored_hash != sha256_json(unhashed):
            raise CounterfactualError(f"cache record hash mismatch: {path}")
        if record.get("schema_version") != LABEL_CACHE_SCHEMA_VERSION:
            raise CounterfactualError(f"cache schema mismatch: {path}")
        if record.get("cache_key") != key or record.get("identity") != expected_identity:
            raise CounterfactualError(f"cache identity mismatch: {path}")
        result = record.get("result")
        if not isinstance(result, dict):
            raise CounterfactualError(f"cache result must be an object: {path}")
        try:
            return validate_counterfactual_result_manifest(
                expected_identity,
                result,
            )
        except CounterfactualError as exc:
            raise CounterfactualError(f"cached result invalid {path}: {exc}") from exc

    def store(
        self,
        identity: Mapping[str, Any],
        result: CounterfactualResult,
    ) -> Path:
        """原子保存结果；同一身份已有不同内容时拒绝覆盖。"""
        normalized_identity = _canonical_object_copy(identity)
        key = cache_key(normalized_identity)
        result_payload = validate_counterfactual_result_manifest(
            normalized_identity,
            result.to_manifest(),
        )
        record: Dict[str, Any] = {
            "schema_version": LABEL_CACHE_SCHEMA_VERSION,
            "cache_key": key,
            "identity": normalized_identity,
            "result": result_payload,
        }
        record["record_hash"] = sha256_json(record)
        path = self.path_for_key(key)
        if path.is_file():
            existing = self.load(normalized_identity)
            if existing != result_payload:
                raise CounterfactualError(
                    f"cache key collision or non-deterministic result: {path}"
                )
            return path

        path.parent.mkdir(parents=True, exist_ok=True)
        payload = canonical_json_bytes(record) + b"\n"
        temporary: Optional[Path] = None
        try:
            # 临时文件必须与目标位于同一目录，且每个 writer 使用唯一名称。
            # fsync 完成后用 hard-link 原子发布；link 不会覆盖已经存在的键，
            # 因而同 key 的并发 writer 只能有一个获胜，其余必须校验获胜内容。
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=path.parent,
                prefix=f".{path.stem}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                existing = self.load(normalized_identity)
                if existing != result_payload:
                    raise CounterfactualError(
                        f"缓存键冲突或结果非确定: {path}"
                    )
        except OSError as exc:
            raise CounterfactualError(f"写入缓存记录失败 {path}: {exc}") from exc
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    # 目标文件已经原子发布；临时文件清理失败不影响记录正确性。
                    pass
        return path
