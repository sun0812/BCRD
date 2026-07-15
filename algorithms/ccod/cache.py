# -*- coding: utf-8 -*-
"""CCOD 反事实标签的内容寻址缓存。"""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, Mapping, Optional

from algorithms.ccod.continuation import (
    CounterfactualError,
    CounterfactualResult,
    ContinuationConfig,
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
        expected_identity = dict(identity)
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
        result_unhashed = {
            name: value for name, value in result.items() if name != "result_hash"
        }
        if result.get("result_hash") != sha256_json(result_unhashed):
            raise CounterfactualError(f"cached result hash mismatch: {path}")
        return result

    def store(
        self,
        identity: Mapping[str, Any],
        result: CounterfactualResult,
    ) -> Path:
        """原子保存结果；同一身份已有不同内容时拒绝覆盖。"""
        normalized_identity = dict(identity)
        key = cache_key(normalized_identity)
        result_payload = result.to_manifest()
        if result_payload.get("query_key") != key:
            raise CounterfactualError(
                "result query_key differs from cache identity"
            )
        if result_payload.get("state_hash") != normalized_identity.get("state_hash"):
            raise CounterfactualError("result state_hash differs from cache identity")
        if result_payload.get("forced_action_key") != normalized_identity.get("action_key"):
            raise CounterfactualError("result action key differs from cache identity")
        if result_payload.get("continuation_hash") != normalized_identity.get(
            "continuation_hash"
        ):
            raise CounterfactualError("result continuation hash differs from cache identity")
        if result_payload.get("continuation_implementation_hash") != normalized_identity.get(
            "continuation_implementation_hash"
        ):
            raise CounterfactualError(
                "result continuation implementation differs from cache identity"
            )
        field_pairs = (
            ("constraint_hash", "constraint_hash"),
            ("enumerator_hash", "enumerator_hash"),
            ("objective_hash", "objective_hash"),
            ("requested_horizon", "horizon"),
        )
        for result_field, identity_field in field_pairs:
            if result_payload.get(result_field) != normalized_identity.get(
                identity_field
            ):
                raise CounterfactualError(
                    f"result {result_field} differs from cache identity"
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
