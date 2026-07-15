# -*- coding: utf-8 -*-
"""CCOD 实例的可搬迁身份与确定性数据划分。

科学身份只由场景内容决定；本地绝对路径仅用于找到源文件，永不进入语义
哈希。``instance_key``、``instance_alias`` 与 ``source_stem`` 是便于追踪
文件的逻辑名称，同样不参与语义哈希。
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

from schedulers.state_replay import sha256_file, sha256_json


SEMANTIC_SCENARIO_SCHEMA_VERSION = "eosbench-semantic-instance-v1"
INSTANCE_IDENTITY_SCHEMA_VERSION = "eosbench-ccod-instance-v1"
INSTANCE_SPLIT_SCHEMA_VERSION = "eosbench-ccod-instance-split-v1"

# 该顺序是在查看任何 CCOD 标签之前由场景语义内容审计冻结的。常量保存公开的
# 逻辑别名而不是机器路径；后续 canonicalizer 升级也不能悄悄改写划分。
RECOMMENDED_SEMANTIC_ORDER: Tuple[str, ...] = (
    "cities_03",
    "cities_05",
    "cities_01",
    "cities_09",
    "cities_06",
    "cities_07",
    "cities_08",
    "cities_04",
    "cities_02",
    "cities_10",
)
RECOMMENDED_SPLIT: Mapping[str, Tuple[str, ...]] = {
    "train": RECOMMENDED_SEMANTIC_ORDER[:6],
    "dev": RECOMMENDED_SEMANTIC_ORDER[6:8],
    "test": RECOMMENDED_SEMANTIC_ORDER[8:],
}

_ALIAS_PATTERN = re.compile(r"(cities_[0-9]+)$")
class InstanceIdentityError(ValueError):
    """场景内容或实例清单无法形成唯一身份时抛出的异常。"""


def _mapping(value: Any, field: str) -> MutableMapping[str, Any]:
    if not isinstance(value, Mapping):
        raise InstanceIdentityError(f"{field} 必须是 JSON object")
    return dict(value)


def _list(value: Any, field: str) -> List[Any]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise InstanceIdentityError(f"{field} 必须是 JSON array")
    return list(value)


def _sort_unique_mappings(
    values: Any,
    field: str,
    key_fields: Sequence[str],
) -> List[Dict[str, Any]]:
    """按审计 v1 的业务键排序；重复键直接拒绝，禁止隐式改写规则。"""
    rows = [_mapping(item, f"{field}[]") for item in _list(values, field)]

    def key(row: Mapping[str, Any]) -> Tuple[Any, ...]:
        try:
            return tuple(row[name] for name in key_fields)
        except KeyError as exc:
            raise InstanceIdentityError(
                f"{field}[] 缺少排序字段 {exc.args[0]}"
            ) from exc

    keys = [key(row) for row in rows]
    if len(set(keys)) != len(keys):
        raise InstanceIdentityError(f"{field} 出现重复业务键")
    return sorted(rows, key=key)


def semantic_scenario_payload(scenario: Mapping[str, Any]) -> Dict[str, Any]:
    """复现标签生成前审计冻结的场景语义投影。

    v1 只服务当前 Sats1/M100 十实例预注册：排除展示用 ``scenario_id``
    以及 metadata 的 name/description/extra，保留会影响本批调度的卫星、任务和
    observation windows。领域无序的顶层集合按业务键排序；嵌套传感器与姿态
    角序列保持原样。若未来场景出现地面站或通信窗口，应升级 schema，而不是
    在 v1 下静默改变已有十个实例的身份。
    """
    root = _mapping(deepcopy(scenario), "scenario")
    metadata = _mapping(root.get("metadata"), "metadata")
    required_metadata = ("creation_time", "duration", "time_step")
    missing_metadata = [name for name in required_metadata if name not in metadata]
    if missing_metadata:
        raise InstanceIdentityError(f"metadata 缺少字段: {missing_metadata}")

    satellites = _sort_unique_mappings(
        root.get("satellites"), "satellites", ("id",)
    )
    missions = _sort_unique_mappings(
        root.get("missions"), "missions", ("id",)
    )
    observation_windows: List[Dict[str, Any]] = []
    for raw_group in _list(root.get("observation_windows"), "observation_windows"):
        group = _mapping(deepcopy(raw_group), "observation_windows[]")
        group["time_windows"] = _sort_unique_mappings(
            group.get("time_windows"),
            "observation_windows[].time_windows",
            ("start_time", "end_time", "orbit_number"),
        )
        observation_windows.append(group)
    group_keys = []
    for group in observation_windows:
        try:
            group_keys.append(
                (
                    group["satellite_id"],
                    group["sensor_id"],
                    group["mission_id"],
                )
            )
        except KeyError as exc:
            raise InstanceIdentityError(
                f"observation_windows[] 缺少排序字段 {exc.args[0]}"
            ) from exc
    if len(set(group_keys)) != len(group_keys):
        raise InstanceIdentityError("observation_windows 出现重复业务键")
    observation_windows.sort(
        key=lambda group: (
            group["satellite_id"],
            group["sensor_id"],
            group["mission_id"],
        )
    )

    return {
        "schema": SEMANTIC_SCENARIO_SCHEMA_VERSION,
        "scenario_type": root.get("scenario_type"),
        "metadata": {
            name: metadata[name]
            for name in required_metadata
        },
        "satellites": satellites,
        "missions": missions,
        "observation_windows": observation_windows,
    }


def semantic_scenario_hash(scenario: Mapping[str, Any]) -> str:
    """计算场景规范化语义内容的 SHA-256。"""
    return sha256_json(semantic_scenario_payload(scenario))


def infer_instance_alias(source_stem: str) -> str:
    """从标准场景文件 stem 提取 ``cities_XX`` 逻辑别名。"""
    match = _ALIAS_PATTERN.search(str(source_stem))
    if match is None:
        raise InstanceIdentityError(
            f"无法从 source_stem 提取 instance_alias: {source_stem!r}"
        )
    return match.group(1)


@dataclass(frozen=True)
class InstanceIdentity:
    """一个场景文件的内容身份与可读逻辑名称。"""

    raw_hash: str
    semantic_hash: str
    instance_alias: str
    instance_key: str
    source_stem: str

    def __post_init__(self) -> None:
        for name, value in (
            ("raw_hash", self.raw_hash),
            ("semantic_hash", self.semantic_hash),
            ("instance_alias", self.instance_alias),
            ("instance_key", self.instance_key),
            ("source_stem", self.source_stem),
        ):
            if not isinstance(value, str) or not value:
                raise InstanceIdentityError(f"{name} 必须是非空字符串")
        for name, value in (
            ("raw_hash", self.raw_hash),
            ("semantic_hash", self.semantic_hash),
        ):
            if not re.fullmatch(r"sha256:[0-9a-f]{64}", value):
                raise InstanceIdentityError(f"{name} 不是规范 SHA-256: {value!r}")
        if Path(self.instance_key).is_absolute():
            raise InstanceIdentityError("instance_key 不得包含绝对路径")

    def to_payload(self) -> Dict[str, str]:
        """返回不含绝对路径的可序列化身份记录。"""
        return {
            "schema_version": INSTANCE_IDENTITY_SCHEMA_VERSION,
            "raw_hash": self.raw_hash,
            "semantic_hash": self.semantic_hash,
            "instance_alias": self.instance_alias,
            "instance_key": self.instance_key,
            "source_stem": self.source_stem,
        }


def build_instance_identity(
    scenario_path: str | Path,
    *,
    instance_alias: str | None = None,
    collection_key: str | None = None,
) -> InstanceIdentity:
    """读取场景并构造身份，绝对路径不进入返回载荷或任何科学哈希。"""
    path = Path(scenario_path)
    if not path.is_file():
        raise FileNotFoundError(f"场景文件不存在: {path}")
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InstanceIdentityError(f"无法读取场景 JSON {path}: {exc}") from exc
    scenario = _mapping(loaded, "scenario")

    source_stem = path.stem
    alias = instance_alias or infer_instance_alias(source_stem)
    logical_collection = collection_key or path.parent.name
    if not logical_collection or Path(logical_collection).is_absolute():
        raise InstanceIdentityError("collection_key 必须是非空相对逻辑名称")
    instance_key = f"{logical_collection.rstrip('/')}/{alias}"
    return InstanceIdentity(
        raw_hash=sha256_file(path),
        semantic_hash=semantic_scenario_hash(scenario),
        instance_alias=alias,
        instance_key=instance_key,
        source_stem=source_stem,
    )


def discover_instance_identities(
    scenario_dir: str | Path,
    *,
    pattern: str = "*.json",
    collection_key: str | None = None,
) -> List[InstanceIdentity]:
    """发现目录中的场景，并按逻辑别名返回唯一身份。"""
    directory = Path(scenario_dir)
    paths = sorted(directory.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"未发现场景文件: {directory}/{pattern}")
    identities = [
        build_instance_identity(path, collection_key=collection_key)
        for path in paths
    ]
    aliases = [identity.instance_alias for identity in identities]
    if len(set(aliases)) != len(aliases):
        raise InstanceIdentityError(f"instance_alias 不唯一: {aliases!r}")
    semantic_hashes = [identity.semantic_hash for identity in identities]
    if len(set(semantic_hashes)) != len(semantic_hashes):
        raise InstanceIdentityError("发现语义内容重复的实例")
    return sorted(identities, key=lambda item: item.instance_alias)


def split_by_semantic_hash(
    identities: Iterable[InstanceIdentity],
    *,
    train_size: int = 6,
    dev_size: int = 2,
    test_size: int = 2,
) -> Dict[str, List[str]]:
    """按 semantic hash 升序生成确定性的 6/2/2（或指定大小）划分。"""
    sizes = (train_size, dev_size, test_size)
    if any(
        isinstance(size, bool) or not isinstance(size, int) or size < 0
        for size in sizes
    ):
        raise InstanceIdentityError("split 大小必须是非负整数")
    rows = list(identities)
    if len(rows) != sum(sizes):
        raise InstanceIdentityError(
            f"实例数 {len(rows)} 与 split 总大小 {sum(sizes)} 不一致"
        )
    if len({row.instance_alias for row in rows}) != len(rows):
        raise InstanceIdentityError("split 输入包含重复 instance_alias")
    if len({row.semantic_hash for row in rows}) != len(rows):
        raise InstanceIdentityError("split 输入包含重复 semantic_hash")
    ordered = sorted(rows, key=lambda row: (row.semantic_hash, row.instance_alias))
    train_end = train_size
    dev_end = train_end + dev_size
    return {
        "train": [row.instance_alias for row in ordered[:train_end]],
        "dev": [row.instance_alias for row in ordered[train_end:dev_end]],
        "test": [row.instance_alias for row in ordered[dev_end:]],
    }


def build_recommended_split_manifest(
    identities: Iterable[InstanceIdentity],
) -> Dict[str, Any]:
    """验证本地 10 实例库存，并返回预注册且不可漂移的推荐划分。

    推荐划分来自预先完成的 semantic-content SHA-256 审计；当前函数使用同一
    canonicalizer 重新计算并要求顺序完全相同，防止实现与预注册静默漂移。
    """
    rows = list(identities)
    by_alias = {row.instance_alias: row for row in rows}
    if len(by_alias) != len(rows):
        raise InstanceIdentityError("推荐划分输入包含重复 instance_alias")
    expected = set(RECOMMENDED_SEMANTIC_ORDER)
    actual = set(by_alias)
    if actual != expected:
        raise InstanceIdentityError(
            "本地实例库存与推荐 10 实例不一致: "
            f"missing={sorted(expected - actual)!r}, extra={sorted(actual - expected)!r}"
        )
    if len({row.semantic_hash for row in rows}) != len(rows):
        raise InstanceIdentityError("推荐划分输入包含重复 semantic_hash")

    split = {name: list(values) for name, values in RECOMMENDED_SPLIT.items()}
    folded = split["train"] + split["dev"] + split["test"]
    if len(folded) != len(set(folded)) or set(folded) != actual:
        raise AssertionError("内置推荐划分不是完整互斥划分")
    current_order = [
        row.instance_alias
        for row in sorted(
            rows,
            key=lambda row: (row.semantic_hash, row.instance_alias),
        )
    ]
    if current_order != list(RECOMMENDED_SEMANTIC_ORDER):
        raise InstanceIdentityError(
            "semantic hash 顺序与预注册划分不一致: "
            f"current={current_order!r}"
        )
    return {
        "schema_version": INSTANCE_SPLIT_SCHEMA_VERSION,
        "policy_version": "semantic-content-sha256-explicit-6-2-2-v1",
        "canonicalizer_version": SEMANTIC_SCENARIO_SCHEMA_VERSION,
        "train": split["train"],
        "dev": split["dev"],
        "test": split["test"],
        "sealed_test": True,
        "recommended_semantic_order": list(RECOMMENDED_SEMANTIC_ORDER),
        "semantic_order_matches_recommendation": True,
        "instances": [
            by_alias[alias].to_payload()
            for alias in RECOMMENDED_SEMANTIC_ORDER
        ],
        "current_semantic_hash_order": current_order,
    }
