# -*- coding: utf-8 -*-
"""CCOD 100-state 诊断的确定性 catalog、去重与选样逻辑。"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import math
from pathlib import PurePosixPath
import re
import sys
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from algorithms.ccod.instances import RECOMMENDED_SPLIT
from schedulers.state_replay import (
    ACTION_KEY_VERSION,
    MANIFEST_SCHEMA_VERSION,
    canonical_json_bytes,
    sha256_json,
)


STATE_CATALOG_SCHEMA_VERSION = "eosbench-ccod-state-catalog-v1"
SELECTION_SCHEMA_VERSION = "eosbench-ccod-state-selection-v1"
BALANCED_OBJECTIVE_NAME = "balanced"
DIAGNOSTIC_CONFIG_SCHEMA_VERSION = "eosbench-ccod-diagnostic-config-v2"

_FROZEN_SPLIT = {
    name: list(RECOMMENDED_SPLIT[name])
    for name in ("train", "dev", "test")
}
_SPLIT_NAMES = ("train", "dev", "test")
_STATE_INVARIANT_FIELDS = (
    "instance_alias",
    "physical_state_hash",
    "objective_hash",
    "constraint_hash",
    "enumerator_hash",
    "step",
    "task_id",
    "schedule_hash",
    "ordered_candidate_hash",
    "candidate_membership_hash",
    "candidate_count",
    "candidate_set_stats",
    "cap_reached",
)
_REQUIRED_SOURCE_FIELDS = (
    "instance_alias",
    "split",
    "objective_name",
    "source_family",
    "source_id",
    "trace_id",
    "trace_hash",
    "state_hash",
    "physical_state_hash",
    "objective_hash",
    "constraint_hash",
    "enumerator_hash",
    "step",
    "task_id",
    "schedule_hash",
    "ordered_candidate_hash",
    "candidate_membership_hash",
    "candidate_count",
    "candidate_set_stats",
    "cap_reached",
    "observed_action_key",
    "state_manifest_hash",
    "state_manifest",
    "scenario_ref",
    "trace_ref",
)


class DiagnosticConfigError(ValueError):
    """预注册配置违反内部一致性时抛出的异常。"""


class DiagnosticCatalogError(ValueError):
    """来源状态无法安全合并或选择时抛出的异常。"""


def objective_name_from_weights(weights: Sequence[float]) -> str:
    """只接受首轮诊断冻结的四目标均权配置。"""
    if isinstance(weights, (str, bytes, bytearray)):
        raise DiagnosticConfigError(f"非法目标权重: {weights!r}")
    values = tuple(weights)
    if (
        len(values) != 4
        or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in values
        )
    ):
        raise DiagnosticConfigError(f"非法目标权重: {weights!r}")
    numeric = tuple(float(value) for value in values)
    if any(not math.isfinite(value) or value < 0.0 for value in numeric):
        raise DiagnosticConfigError(f"目标权重必须有限且非负: {weights!r}")
    total = math.fsum(numeric)
    if total <= 0.0:
        raise DiagnosticConfigError("目标权重之和必须为正")
    normalized = tuple(value / total for value in numeric)
    if any(abs(value - 0.25) > 1e-12 for value in normalized):
        raise DiagnosticConfigError(
            f"diagnostic_v1 只接受 balanced objective: {normalized!r}"
        )
    return BALANCED_OBJECTIVE_NAME


def _strict_positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DiagnosticConfigError(f"{field_name} 必须为严格正整数")
    return value


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: Sequence[str],
    field_name: str,
) -> None:
    actual = set(value)
    expected_set = set(expected)
    if actual != expected_set:
        raise DiagnosticConfigError(
            f"{field_name} 字段不匹配: missing={sorted(expected_set - actual)}, "
            f"extra={sorted(actual - expected_set)}"
        )


def validate_diagnostic_config(config: Mapping[str, Any]) -> None:
    """在查看任何反事实标签前验证 split、来源、配额与 gate。"""
    if config.get("schema_version") != DIAGNOSTIC_CONFIG_SCHEMA_VERSION:
        raise DiagnosticConfigError("diagnostic config schema_version 不受支持")
    _require_exact_keys(
        config,
        (
            "schema_version",
            "seed",
            "runtime",
            "split",
            "sources",
            "source_inventory",
            "replay",
            "state_selection",
            "query",
            "signal_gate",
            "label_validity",
            "mac_training_gate",
            "gpu_gate",
        ),
        "diagnostic config",
    )
    seed = config.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed != 20260715:
        raise DiagnosticConfigError("diagnostic_v1 seed 必须冻结为 20260715")
    runtime = config.get("runtime")
    if not isinstance(runtime, Mapping) or dict(runtime) != {
        "python_implementation": "cpython",
        "python_version": "3.10.20",
    }:
        raise DiagnosticConfigError("diagnostic_v1 runtime 必须冻结为 CPython 3.10.20")
    split = config.get("split")
    sources = config.get("sources")
    replay = config.get("replay")
    selection = config.get("state_selection")
    query = config.get("query")
    gate = config.get("signal_gate")
    if not all(
        isinstance(value, Mapping)
        for value in (split, sources, replay, selection, query, gate)
    ):
        raise DiagnosticConfigError(
            "配置缺少 split/sources/replay/state_selection/query/signal_gate"
        )

    split_sets = {
        name: {str(value) for value in split.get(name, [])}
        for name in _SPLIT_NAMES
    }
    if any(not values for values in split_sets.values()):
        raise DiagnosticConfigError("train/dev/test split 均不能为空")
    if any(
        split_sets[left] & split_sets[right]
        for index, left in enumerate(_SPLIT_NAMES)
        for right in _SPLIT_NAMES[index + 1 :]
    ):
        raise DiagnosticConfigError("train/dev/test split 必须互斥")
    if split.get("sealed_test") is not True:
        raise DiagnosticConfigError("diagnostic_v1 要求 sealed_test=true")
    if len(split_sets["train"] | split_sets["dev"] | split_sets["test"]) != 10:
        raise DiagnosticConfigError("diagnostic_v1 必须恰好冻结 10 个实例")
    for name, expected in _FROZEN_SPLIT.items():
        if split.get(name) != expected:
            raise DiagnosticConfigError(f"diagnostic_v1 {name} split 已漂移")
    if split.get("policy_version") != "semantic-content-sha256-explicit-6-2-2-v1":
        raise DiagnosticConfigError("split policy_version 已漂移")
    if split.get("canonicalizer_version") != "eosbench-semantic-instance-v1":
        raise DiagnosticConfigError("split canonicalizer_version 已漂移")

    objective_name = str(sources.get("objective_name", ""))
    if objective_name != BALANCED_OBJECTIVE_NAME:
        raise DiagnosticConfigError("首轮诊断只允许 balanced objective")
    if objective_name_from_weights(sources.get("objective_weights", [])) != objective_name:
        raise DiagnosticConfigError("objective_name 与 objective_weights 不一致")
    solver_families = sources.get("solver_families")
    if solver_families != ["sa", "ga", "aco"]:
        raise DiagnosticConfigError("solver_families 必须冻结为 sa/ga/aco")
    if sources.get("schedule_suffix") != "p0.25_c0.25_t0.25_b0.25":
        raise DiagnosticConfigError("balanced schedule_suffix 已漂移")
    if dict(sources) != {
        "objective_name": "balanced",
        "objective_weights": [0.25, 0.25, 0.25, 0.25],
        "solver_families": ["sa", "ga", "aco"],
        "schedule_suffix": "p0.25_c0.25_t0.25_b0.25",
    }:
        raise DiagnosticConfigError("diagnostic_v1 sources 配置已漂移")
    source_inventory = config.get("source_inventory")
    if not isinstance(source_inventory, Mapping) or dict(source_inventory) != {
        "path": "algorithms/ccod/configs/diagnostic_v1_sources.json",
        "schema_version": "eosbench-ccod-source-inventory-v1",
        "file_hash": "sha256:27cb6a28fd94f7c0360f6dd351c1607f80424b92c5001efba8682b89915311c5",
    }:
        raise DiagnosticConfigError("source_inventory 指针已漂移")

    frozen_replay = {
        "max_candidates": 8192,
        "placement_mode": "earliest",
        "downlink_duration_ratio": 1.0,
        "agility_profile": "Standard-Agility",
        "non_agile_transition_s": 10.0,
        "max_step_inclusive": 95,
    }
    if dict(replay) != frozen_replay:
        raise DiagnosticConfigError("diagnostic_v1 replay 配置已漂移")

    total_states = _strict_positive_int(
        selection.get("total_states"), "state_selection.total_states"
    )
    per_instance = selection.get("per_instance")
    if not isinstance(per_instance, Mapping):
        raise DiagnosticConfigError("state_selection.per_instance 必须为 object")
    if set(per_instance) != split_sets["dev"]:
        raise DiagnosticConfigError("状态配额必须且只能覆盖 dev instances")
    quotas = {
        str(instance): _strict_positive_int(quota, f"per_instance.{instance}")
        for instance, quota in per_instance.items()
    }
    if sum(quotas.values()) != total_states:
        raise DiagnosticConfigError("per_instance 配额之和不等于 total_states")
    frozen_selection = {
        "total_states": 100,
        "per_instance": {"cities_08": 50, "cities_04": 50},
        "actionable_min_candidates": 2,
        "signal_min_candidates": 10,
        "selector_min_candidates": 17,
        "time_bins": 4,
        "candidate_count_bins": 4,
        "observed_skip_target_fraction": 0.0,
        "max_cap_hits": 0,
        "deduplicate_by": "state_hash",
        "canonical_source_rank": "sha256(seed,state_hash,source_family)",
        "selection_rank": "sha256(seed,instance_alias,state_hash)",
    }
    if dict(selection) != frozen_selection:
        raise DiagnosticConfigError("diagnostic_v1 state_selection 配置已漂移")
    if selection.get("deduplicate_by") != "state_hash":
        raise DiagnosticConfigError("diagnostic_v1 必须按完整 state_hash 去重")
    if selection.get("max_cap_hits") != 0:
        raise DiagnosticConfigError("100-state selection 不允许 candidate cap hit")
    if selection.get("canonical_source_rank") != (
        "sha256(seed,state_hash,source_family)"
    ):
        raise DiagnosticConfigError("canonical_source_rank 已漂移")
    if selection.get("selection_rank") != (
        "sha256(seed,instance_alias,state_hash)"
    ):
        raise DiagnosticConfigError("selection_rank 已漂移")
    actionable_min = _strict_positive_int(
        selection.get("actionable_min_candidates"),
        "state_selection.actionable_min_candidates",
    )
    signal_min = _strict_positive_int(
        selection.get("signal_min_candidates"),
        "state_selection.signal_min_candidates",
    )
    if actionable_min != 2 or signal_min < actionable_min:
        raise DiagnosticConfigError("actionable/signal candidate 下界非法")
    if selection.get("selector_min_candidates") != 17:
        raise DiagnosticConfigError("selector_min_candidates 必须冻结为 17")
    _strict_positive_int(selection.get("time_bins"), "state_selection.time_bins")
    _strict_positive_int(
        selection.get("candidate_count_bins"),
        "state_selection.candidate_count_bins",
    )
    skip_fraction = float(selection.get("observed_skip_target_fraction", -1.0))
    if not math.isfinite(skip_fraction) or not 0.0 <= skip_fraction <= 1.0:
        raise DiagnosticConfigError("observed_skip_target_fraction 必须位于 [0,1]")
    max_step = replay.get("max_step_inclusive")
    if isinstance(max_step, bool) or not isinstance(max_step, int) or max_step < 0:
        raise DiagnosticConfigError("replay.max_step_inclusive 必须为非负整数")

    budget = _strict_positive_int(query.get("budget_per_state"), "query.budget_per_state")
    horizon = _strict_positive_int(
        query.get("continuation_horizon"), "query.continuation_horizon"
    )
    if budget != 16 or horizon != 5:
        raise DiagnosticConfigError("diagnostic_v1 必须冻结 B=16、H=5")
    if query.get("mandatory_actions") != ["observed", "skip"]:
        raise DiagnosticConfigError("必须先查询 observed 与 SKIP")
    if query.get("fill_policy") != "eosbench-ccod-uniform-rank-v1":
        raise DiagnosticConfigError("query fill_policy 已漂移")
    if query.get("workers") != 1 or query.get("max_attempts") != 2:
        raise DiagnosticConfigError("Mac 诊断必须冻结 workers=1、max_attempts=2")
    for field_name in ("query_timeout_s", "state_timeout_s"):
        value = query.get(field_name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise DiagnosticConfigError(f"query.{field_name} 必须为正数")
    rss_limit = query.get("worker_peak_rss_limit_mib")
    if (
        isinstance(rss_limit, bool)
        or not isinstance(rss_limit, (int, float))
        or float(rss_limit) != 6144.0
    ):
        raise DiagnosticConfigError("worker_peak_rss_limit_mib 必须冻结为 6144")
    if dict(query) != {
        "budget_per_state": 16,
        "mandatory_actions": ["observed", "skip"],
        "fill_policy": "eosbench-ccod-uniform-rank-v1",
        "continuation_horizon": 5,
        "workers": 1,
        "max_attempts": 2,
        "query_timeout_s": 120.0,
        "state_timeout_s": 1200.0,
        "worker_peak_rss_limit_mib": 6144.0,
    }:
        raise DiagnosticConfigError("diagnostic_v1 query 配置已漂移")

    if gate.get("require_exact_selected_states") != total_states:
        raise DiagnosticConfigError("signal gate 的状态数必须等于选样状态数")
    minimum_signal = _strict_positive_int(
        gate.get("minimum_signal_states"), "signal_gate.minimum_signal_states"
    )
    if minimum_signal > total_states:
        raise DiagnosticConfigError("minimum_signal_states 不能超过 total_states")
    if gate.get("actionable_min_candidates") != actionable_min:
        raise DiagnosticConfigError("actionable candidate 门槛前后不一致")
    if gate.get("signal_min_candidates") != signal_min:
        raise DiagnosticConfigError("signal candidate 门槛前后不一致")
    minimum_by_instance = gate.get("minimum_signal_states_per_instance")
    if not isinstance(minimum_by_instance, Mapping):
        raise DiagnosticConfigError("缺少 per-instance signal 状态下界")
    if set(minimum_by_instance) != set(per_instance):
        raise DiagnosticConfigError("per-instance signal 下界必须覆盖全部 dev")
    normalized_minimum_by_instance = {
        str(instance): _strict_positive_int(
            value,
            f"signal_gate.minimum_signal_states_per_instance.{instance}",
        )
        for instance, value in minimum_by_instance.items()
    }
    if sum(normalized_minimum_by_instance.values()) != minimum_signal:
        raise DiagnosticConfigError("per-instance signal 下界之和必须等于全局下界")
    if any(
        normalized_minimum_by_instance[instance] > quotas[instance]
        for instance in quotas
    ):
        raise DiagnosticConfigError("per-instance signal 下界不能超过状态配额")
    spread_threshold = float(gate.get("spread_threshold", math.nan))
    if not math.isfinite(spread_threshold) or spread_threshold < 0.0:
        raise DiagnosticConfigError("signal_gate.spread_threshold 必须有限且非负")
    for field_name in (
        "minimum_signal_pass_fraction",
        "minimum_per_instance_pass_fraction",
    ):
        value = float(gate.get(field_name, math.nan))
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise DiagnosticConfigError(
                f"signal_gate.{field_name} 必须位于 [0,1]"
            )
    if gate.get("require_all_queries_successful") is not True:
        raise DiagnosticConfigError("signal gate 必须要求全部 query 成功")
    if gate.get("allow_confidence_interval_claim") is not False:
        raise DiagnosticConfigError("两个 dev 实例不允许置信区间结论")
    if dict(gate) != {
        "require_exact_selected_states": 100,
        "require_all_queries_successful": True,
        "actionable_min_candidates": 2,
        "signal_min_candidates": 10,
        "minimum_signal_states": 80,
        "minimum_signal_states_per_instance": {
            "cities_08": 40,
            "cities_04": 40,
        },
        "spread_metric": "Q_H_P90_minus_P10_type7",
        "spread_threshold": 0.01,
        "minimum_signal_pass_fraction": 0.60,
        "minimum_per_instance_pass_fraction": 0.50,
        "report_all_actionable_and_signal_separately": True,
        "allow_confidence_interval_claim": False,
    }:
        raise DiagnosticConfigError("diagnostic_v1 signal_gate 配置已漂移")

    label_validity = config.get("label_validity")
    if not isinstance(label_validity, Mapping) or dict(label_validity) != {
        "exhaustive_states": 20,
        "exhaustive_candidate_count_min": 17,
        "exhaustive_candidate_count_max": 128,
        "minimum_exhaustive_states_per_instance": {
            "cities_08": 10,
            "cities_04": 10,
        },
        "strong_states": 10,
        "exhaustive_selection": "stable_hash_balanced_instance_v1",
        "strong_selection": "stable_hash_balanced_instance_v1",
        "strong_candidate_count_min": 17,
        "strong_subset_of_exhaustive": True,
        "beam_width": 8,
        "median_spearman_threshold": 0.70,
    }:
        raise DiagnosticConfigError("label_validity 配置已漂移")
    mac_training_gate = config.get("mac_training_gate")
    if not isinstance(mac_training_gate, Mapping) or dict(mac_training_gate) != {
        "overfit_states": 64,
        "overfit_pairwise_accuracy_min": 0.95,
        "pilot_states_min": 500,
        "pilot_validation_pairwise_auc_min": 0.55,
        "pilot_validation_spearman_min": 0.10,
        "pilot_closed_loop_objective_delta_min": 0.0,
        "wall_time_limit_s": 3600.0,
        "peak_rss_limit_mib": 6144.0,
    }:
        raise DiagnosticConfigError("mac_training_gate 配置已漂移")
    gpu_gate = config.get("gpu_gate")
    if not isinstance(gpu_gate, Mapping) or dict(gpu_gate) != {
        "rent_3090_before_all_local_gates_pass": False,
        "required_local_gates": [
            "signal",
            "label_validity",
            "selector",
            "mac_overfit",
            "mac_one_seed_pilot",
        ],
    }:
        raise DiagnosticConfigError("3090 gate 必须保持关闭")


def _current_python_runtime() -> Dict[str, str]:
    """返回会进入诊断科学身份的解释器实现与精确版本。"""
    version = ".".join(str(part) for part in sys.version_info[:3])
    if sys.version_info.releaselevel != "final":
        # 3.10.20rc1 不得伪装成已冻结的 3.10.20 正式版。
        version += f"{sys.version_info.releaselevel}{sys.version_info.serial}"
    return {
        "python_implementation": sys.implementation.name,
        "python_version": version,
    }


def validate_diagnostic_runtime(config: Mapping[str, Any]) -> None:
    """在读取场景、schedule 或恢复状态前拒绝错误解释器。"""
    validate_diagnostic_config(config)
    expected = dict(config["runtime"])
    actual = _current_python_runtime()
    if actual != expected:
        raise DiagnosticConfigError(
            "diagnostic_v1 解释器不匹配: "
            f"expected={expected}, actual={actual}; "
            "请使用仓库 .venv/bin/python"
        )


def _canonical_source_rank(
    source: Mapping[str, Any],
    *,
    seed: int,
) -> Tuple[str, bytes]:
    """在合并 aliases 前选定唯一来源，避免利用 observed action 选样。"""
    payload = {
        "schema_version": "eosbench-ccod-canonical-source-rank-v1",
        "seed": int(seed),
        "state_hash": source["state_hash"],
        # 只使用预先冻结的 solver family 坐标；不得让 observed action、
        # schedule 内容或 trace/manifest hash 影响 canonical observed。
        "source_family": source["source_family"],
    }
    return sha256_json(payload), canonical_json_bytes(payload)


def _validate_source_record(
    source: Mapping[str, Any],
    *,
    dev_instances: set[str],
) -> None:
    missing = [field for field in _REQUIRED_SOURCE_FIELDS if field not in source]
    if missing:
        raise DiagnosticCatalogError(f"来源状态缺少字段: {missing}")
    if source["split"] != "dev" or source["instance_alias"] not in dev_instances:
        raise DiagnosticCatalogError("catalog 来源必须且只能属于 dev split")
    if source["objective_name"] != BALANCED_OBJECTIVE_NAME:
        raise DiagnosticCatalogError("catalog 混入非 balanced objective")
    if source["source_family"] not in {"sa", "ga", "aco"}:
        raise DiagnosticCatalogError("catalog 混入未冻结 solver family")
    if not isinstance(source["observed_action_key"], Mapping):
        raise DiagnosticCatalogError("observed_action_key 必须为 object")
    if not isinstance(source["state_manifest"], Mapping):
        raise DiagnosticCatalogError("state_manifest 必须为 object")
    manifest = dict(source["state_manifest"])
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise DiagnosticCatalogError("state_manifest schema_version 不受支持")
    expected_manifest_hash = sha256_json(
        {
            key: value
            for key, value in manifest.items()
            if key != "state_manifest_hash"
        }
    )
    if (
        manifest.get("state_manifest_hash") != expected_manifest_hash
        or source["state_manifest_hash"] != expected_manifest_hash
    ):
        raise DiagnosticCatalogError("state_manifest_hash 重算不一致")
    manifest_pairs = (
        ("state_hash", "state_hash"),
        ("physical_state_hash", "physical_state_hash"),
        ("step", "step"),
        ("task_id", "task_id"),
        ("schedule_hash", "schedule_hash"),
        ("ordered_candidate_hash", "ordered_candidate_hash"),
        ("candidate_membership_hash", "candidate_membership_hash"),
        ("candidate_count", "candidate_count"),
        ("candidate_set_stats", "candidate_set_stats"),
        ("observed_action_key", "observed_action_key"),
        ("trace_id", "trace_id"),
        ("trace_hash", "trace_hash"),
    )
    differences = [
        outer_name
        for manifest_name, outer_name in manifest_pairs
        if manifest.get(manifest_name) != source.get(outer_name)
    ]
    if differences:
        raise DiagnosticCatalogError(
            f"state_manifest 与外层字段不一致: {differences}"
        )
    stats = manifest.get("candidate_set_stats")
    if not isinstance(stats, Mapping) or bool(stats.get("cap_reached")) != source[
        "cap_reached"
    ]:
        raise DiagnosticCatalogError("candidate cap 状态与 manifest 不一致")
    count = source["candidate_count"]
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise DiagnosticCatalogError("candidate_count 必须为严格正整数")
    if not isinstance(source["cap_reached"], bool):
        raise DiagnosticCatalogError("cap_reached 必须为严格布尔值")

    observed = dict(source["observed_action_key"])
    if observed.get("version") != ACTION_KEY_VERSION:
        raise DiagnosticCatalogError("observed ActionKey version 不受支持")
    if observed.get("kind") not in {"skip", "assign"}:
        raise DiagnosticCatalogError("observed ActionKey kind 非法")
    if observed.get("task_id") != source["task_id"]:
        raise DiagnosticCatalogError("observed ActionKey task_id 不一致")

    for field_name in (
        "source_id",
        "trace_id",
        "trace_hash",
        "state_hash",
        "physical_state_hash",
        "objective_hash",
        "constraint_hash",
        "enumerator_hash",
        "schedule_hash",
        "ordered_candidate_hash",
        "candidate_membership_hash",
        "state_manifest_hash",
    ):
        value = source[field_name]
        if not isinstance(value, str) or re.fullmatch(
            r"sha256:[0-9a-f]{64}", value
        ) is None:
            raise DiagnosticCatalogError(f"{field_name} 不是规范 SHA-256")

    for ref_name in ("scenario_ref", "trace_ref"):
        reference = source[ref_name]
        if not isinstance(reference, Mapping):
            raise DiagnosticCatalogError(f"{ref_name} 必须为 object")
        relative_path = reference.get("relative_path")
        if not isinstance(relative_path, str) or not relative_path:
            raise DiagnosticCatalogError(f"{ref_name}.relative_path 不能为空")
        logical_path = PurePosixPath(relative_path)
        if logical_path.is_absolute() or ".." in logical_path.parts:
            raise DiagnosticCatalogError(f"{ref_name} 不得包含绝对或上级路径")
    scenario_ref = source["scenario_ref"]
    if scenario_ref.get("root_id") != "eosbench_output":
        raise DiagnosticCatalogError("scenario_ref.root_id 不受支持")
    scenario_hash = scenario_ref.get("sha256")
    if not isinstance(scenario_hash, str) or re.fullmatch(
        r"sha256:[0-9a-f]{64}", scenario_hash
    ) is None:
        raise DiagnosticCatalogError("scenario_ref.sha256 非法")
    trace_ref = source["trace_ref"]
    if (
        trace_ref.get("trace_id") != source["trace_id"]
        or trace_ref.get("trace_hash") != source["trace_hash"]
    ):
        raise DiagnosticCatalogError("trace_ref 与外层 trace 身份不一致")


def _compact_alias(source: Mapping[str, Any], rank: str) -> Dict[str, Any]:
    """保存恢复与审计所需的紧凑 alias，不复制完整 trajectory。"""
    return {
        "source_rank": rank,
        "source_id": source["source_id"],
        "source_family": source["source_family"],
        "trace_id": source["trace_id"],
        "trace_hash": source["trace_hash"],
        "state_manifest_hash": source["state_manifest_hash"],
        "observed_action_key": deepcopy(dict(source["observed_action_key"])),
        "scenario_ref": deepcopy(dict(source["scenario_ref"])),
        "trace_ref": deepcopy(dict(source["trace_ref"])),
    }


def merge_state_catalog(
    source_records: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    """按完整 state_hash 合并来源，并冻结 canonical observed action。"""
    validate_diagnostic_config(config)
    dev_instances = {str(value) for value in config["split"]["dev"]}
    seed = int(config["seed"])
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for raw_source in source_records:
        source = deepcopy(dict(raw_source))
        _validate_source_record(source, dev_instances=dev_instances)
        grouped.setdefault(str(source["state_hash"]), []).append(source)

    catalog: List[Dict[str, Any]] = []
    for state_hash in sorted(grouped):
        sources = grouped[state_hash]
        reference = sources[0]
        for source in sources[1:]:
            differences = [
                field
                for field in _STATE_INVARIANT_FIELDS
                if source.get(field) != reference.get(field)
            ]
            if differences:
                raise DiagnosticCatalogError(
                    f"相同 state_hash 的来源语义冲突: {state_hash}, {differences}"
                )

        # 完全相同的 alias 可安全去重；相同身份若携带不同 observed/ref，
        # 说明上游清单被污染，必须拒绝而不能依赖输入稳定排序。
        unique_sources: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        for source in sources:
            alias_identity = (
                str(source["source_id"]),
                str(source["trace_id"]),
                str(source["state_manifest_hash"]),
            )
            existing = unique_sources.get(alias_identity)
            if existing is not None:
                if canonical_json_bytes(existing) != canonical_json_bytes(source):
                    raise DiagnosticCatalogError(
                        f"相同 alias 身份携带冲突内容: {state_hash}"
                    )
                continue
            unique_sources[alias_identity] = source
        source_families = [
            str(source["source_family"])
            for source in unique_sources.values()
        ]
        if len(source_families) != len(set(source_families)):
            raise DiagnosticCatalogError(
                f"相同 state_hash 同时出现多个同 family 来源: {state_hash}"
            )

        ranked_sources = sorted(
            (
                (*_canonical_source_rank(source, seed=seed), source)
                for source in unique_sources.values()
            ),
            key=lambda item: (item[0], item[1]),
        )
        canonical_rank, _, canonical = ranked_sources[0]
        aliases = [
            _compact_alias(source, rank)
            for rank, _, source in ranked_sources
        ]
        row: Dict[str, Any] = {
            "record_type": "state",
            "schema_version": STATE_CATALOG_SCHEMA_VERSION,
            "state_hash": state_hash,
            "physical_state_hash": canonical["physical_state_hash"],
            "instance_alias": canonical["instance_alias"],
            "split": "dev",
            "objective_name": BALANCED_OBJECTIVE_NAME,
            "objective_hash": canonical["objective_hash"],
            "constraint_hash": canonical["constraint_hash"],
            "enumerator_hash": canonical["enumerator_hash"],
            "step": int(canonical["step"]),
            "task_id": canonical["task_id"],
            "schedule_hash": canonical["schedule_hash"],
            "candidate_count": int(canonical["candidate_count"]),
            "candidate_set_stats": deepcopy(
                dict(canonical["candidate_set_stats"])
            ),
            "cap_reached": bool(canonical["cap_reached"]),
            "ordered_candidate_hash": canonical["ordered_candidate_hash"],
            "candidate_membership_hash": canonical["candidate_membership_hash"],
            "observed_action_key": deepcopy(dict(canonical["observed_action_key"])),
            "observed_is_skip": canonical["observed_action_key"].get("kind") == "skip",
            "canonical_source": {
                **_compact_alias(canonical, canonical_rank),
                "state_manifest": deepcopy(dict(canonical["state_manifest"])),
            },
            "source_aliases": aliases,
            "selection": {
                "eligible": False,
                "selected": False,
            },
        }
        row["row_hash"] = sha256_json(row)
        catalog.append(row)
    return catalog


def _balanced_targets(categories: Sequence[Any], total: int) -> Dict[Any, int]:
    ordered = list(categories)
    if not ordered:
        raise DiagnosticCatalogError("无法为无类别维度分配目标")
    base, remainder = divmod(int(total), len(ordered))
    return {
        category: base + (1 if index < remainder else 0)
        for index, category in enumerate(ordered)
    }


def _candidate_bins(
    records: Sequence[Mapping[str, Any]],
    num_bins: int,
) -> Dict[str, int]:
    """按 candidate count 的稳定经验秩分箱，不读取任何标签。"""
    ordered = sorted(
        records,
        key=lambda row: (int(row["candidate_count"]), str(row["state_hash"])),
    )
    return {
        str(row["state_hash"]): min(
            num_bins - 1,
            index * num_bins // max(1, len(ordered)),
        )
        for index, row in enumerate(ordered)
    }


def _selection_rank(row: Mapping[str, Any], seed: int) -> str:
    return sha256_json(
        {
            "schema_version": "eosbench-ccod-state-selection-rank-v1",
            "seed": int(seed),
            "instance_alias": row["instance_alias"],
            "state_hash": row["state_hash"],
        }
    )


def select_preregistered_states(
    catalog: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    """以边际配额从去重 catalog 中确定性选择 100 个状态。"""
    validate_diagnostic_config(config)
    selection = config["state_selection"]
    gate = config["signal_gate"]
    validity = config["label_validity"]
    max_step = int(config["replay"]["max_step_inclusive"])
    actionable_min = int(selection["actionable_min_candidates"])
    signal_min = int(selection["signal_min_candidates"])
    time_bins = int(selection["time_bins"])
    count_bins = int(selection["candidate_count_bins"])
    seed = int(config["seed"])
    exhaustive_min = int(validity["exhaustive_candidate_count_min"])
    exhaustive_max = int(validity["exhaustive_candidate_count_max"])

    unique_hashes = {str(row["state_hash"]) for row in catalog}
    if len(unique_hashes) != len(catalog):
        raise DiagnosticCatalogError("输入 catalog 尚未按 state_hash 去重")
    eligible = [
        deepcopy(dict(row))
        for row in catalog
        if row.get("split") == "dev"
        and 0 <= int(row["step"]) <= max_step
        and int(row["candidate_count"]) >= actionable_min
        and not bool(row.get("cap_reached"))
    ]
    selected: List[Dict[str, Any]] = []
    for instance_alias in config["split"]["dev"]:
        quota = int(selection["per_instance"][instance_alias])
        pool = [
            row for row in eligible if row["instance_alias"] == instance_alias
        ]
        if len(pool) < quota:
            raise DiagnosticCatalogError(
                f"{instance_alias} actionable 状态不足: {len(pool)} < {quota}"
            )
        minimum_signal = int(
            gate["minimum_signal_states_per_instance"][instance_alias]
        )
        available_signal = sum(
            int(row["candidate_count"]) >= signal_min for row in pool
        )
        if available_signal < minimum_signal:
            raise DiagnosticCatalogError(
                f"{instance_alias} signal-eligible 库存不足: "
                f"{available_signal} < {minimum_signal}"
            )
        minimum_exhaustive = min(
            quota,
            int(
                validity["minimum_exhaustive_states_per_instance"][
                    instance_alias
                ]
            ),
        )
        available_exhaustive = sum(
            exhaustive_min
            <= int(row["candidate_count"])
            <= exhaustive_max
            for row in pool
        )
        if available_exhaustive < minimum_exhaustive:
            raise DiagnosticCatalogError(
                f"{instance_alias} exhaustive-eligible 库存不足: "
                f"{available_exhaustive} < {minimum_exhaustive}"
            )
        bin_by_hash = _candidate_bins(pool, count_bins)
        for row in pool:
            row["time_bin"] = min(
                time_bins - 1,
                int(row["step"]) * time_bins // (max_step + 1),
            )
            row["candidate_count_bin"] = bin_by_hash[str(row["state_hash"])]
            row["source_family"] = row["canonical_source"]["source_family"]
            row["observed_kind"] = (
                "skip" if bool(row["observed_is_skip"]) else "assign"
            )
            row["signal_eligible"] = int(row["candidate_count"]) >= signal_min
            row["exhaustive_eligible"] = (
                exhaustive_min
                <= int(row["candidate_count"])
                <= exhaustive_max
            )
            row["selection_rank"] = _selection_rank(row, seed)

        source_targets = _balanced_targets(
            list(config["sources"]["solver_families"]), quota
        )
        time_targets = _balanced_targets(list(range(time_bins)), quota)
        count_targets = _balanced_targets(list(range(count_bins)), quota)
        skip_target = int(
            round(quota * float(selection["observed_skip_target_fraction"]))
        )
        observed_targets = {"skip": skip_target, "assign": quota - skip_target}
        maximum_non_signal = quota - minimum_signal
        maximum_non_exhaustive = quota - minimum_exhaustive
        non_signal_selected = 0
        non_exhaustive_selected = 0
        counts = {
            "source_family": Counter(),
            "time_bin": Counter(),
            "candidate_count_bin": Counter(),
            "observed_kind": Counter(),
        }
        chosen_hashes: set[str] = set()
        for _slot in range(quota):
            choices: List[Tuple[float, str, str, Dict[str, Any]]] = []
            for row in pool:
                state_hash = str(row["state_hash"])
                if state_hash in chosen_hashes:
                    continue
                if not row["signal_eligible"] and non_signal_selected >= maximum_non_signal:
                    continue
                if (
                    not row["exhaustive_eligible"]
                    and non_exhaustive_selected >= maximum_non_exhaustive
                ):
                    continue
                score = 0.0
                dimensions = (
                    ("source_family", source_targets, 1.0),
                    ("time_bin", time_targets, 1.0),
                    ("candidate_count_bin", count_targets, 1.0),
                    ("observed_kind", observed_targets, 0.75),
                )
                for dimension, targets, weight in dimensions:
                    category = row[dimension]
                    current = counts[dimension][category]
                    target = targets[category]
                    score += weight * (
                        (current + 1 - target) ** 2 - (current - target) ** 2
                    )
                choices.append(
                    (score, row["selection_rank"], str(row["state_hash"]), row)
                )
            if not choices:
                raise DiagnosticCatalogError(
                    f"{instance_alias} 无法在预注册边际配额下填满 {quota} 个状态"
                )
            _, _, _, chosen = min(
                choices,
                key=lambda item: (item[0], item[1], item[2]),
            )
            chosen_hashes.add(str(chosen["state_hash"]))
            if not chosen["signal_eligible"]:
                non_signal_selected += 1
            if not chosen["exhaustive_eligible"]:
                non_exhaustive_selected += 1
            for dimension in (
                "source_family",
                "time_bin",
                "candidate_count_bin",
                "observed_kind",
            ):
                counts[dimension][chosen[dimension]] += 1
            selected.append(deepcopy(chosen))

    expected = int(selection["total_states"])
    if len(selected) != expected:
        raise DiagnosticCatalogError(
            f"选样数量错误: selected={len(selected)}, expected={expected}"
        )
    if len({row["state_hash"] for row in selected}) != expected:
        raise DiagnosticCatalogError("选样结果包含重复 state_hash")
    signal_count = sum(bool(row["signal_eligible"]) for row in selected)
    if signal_count < int(gate["minimum_signal_states"]):
        raise DiagnosticCatalogError("选样结果未达到 minimum_signal_states")
    for instance_alias, minimum in validity[
        "minimum_exhaustive_states_per_instance"
    ].items():
        actual = sum(
            row["instance_alias"] == instance_alias
            and bool(row["exhaustive_eligible"])
            for row in selected
        )
        effective_minimum = min(
            int(selection["per_instance"][instance_alias]),
            int(minimum),
        )
        if actual < effective_minimum:
            raise DiagnosticCatalogError(
                f"{instance_alias} exhaustive-eligible 状态不足: "
                f"{actual} < {effective_minimum}"
            )
    selected.sort(
        key=lambda row: (
            config["split"]["dev"].index(row["instance_alias"]),
            row["selection_rank"],
            row["state_hash"],
        )
    )
    for ordinal, row in enumerate(selected):
        row["selection"] = {
            "schema_version": SELECTION_SCHEMA_VERSION,
            "eligible": True,
            "selected": True,
            "selected_ordinal": ordinal,
            "quota_cell": row["instance_alias"],
            "selection_rank": row["selection_rank"],
        }
        row["row_hash"] = sha256_json(
            {key: value for key, value in row.items() if key != "row_hash"}
        )
    return selected


def annotate_catalog_selection(
    catalog: Sequence[Mapping[str, Any]],
    selected: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """将冻结的 selection 投影回完整 catalog，便于审计未选池。"""
    catalog_hashes = [str(row["state_hash"]) for row in catalog]
    if len(catalog_hashes) != len(set(catalog_hashes)):
        raise DiagnosticCatalogError("annotate 输入 catalog 包含重复 state_hash")
    selected_hashes = [str(row["state_hash"]) for row in selected]
    if len(selected_hashes) != len(set(selected_hashes)):
        raise DiagnosticCatalogError("annotate 输入 selection 包含重复 state_hash")
    unknown = set(selected_hashes) - set(catalog_hashes)
    if unknown:
        raise DiagnosticCatalogError(
            f"selection 引用了 catalog 外状态: {sorted(unknown)}"
        )
    selected_by_hash = {}
    for row in selected:
        selection = row.get("selection")
        if not isinstance(selection, Mapping) or selection.get("selected") is not True:
            raise DiagnosticCatalogError("selected row 缺少有效 selection 清单")
        selected_by_hash[str(row["state_hash"])] = deepcopy(dict(selection))
    output: List[Dict[str, Any]] = []
    for raw_row in catalog:
        row = deepcopy(dict(raw_row))
        state_hash = str(row["state_hash"])
        row["selection"] = {
            "eligible": False,
            "selected": False,
        }
        if state_hash in selected_by_hash:
            row["selection"] = selected_by_hash[state_hash]
        row["row_hash"] = sha256_json(
            {key: value for key, value in row.items() if key != "row_hash"}
        )
        output.append(row)
    return output


def selection_summary(selected: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """生成不含标签的确定性边际分布摘要。"""
    if not selected:
        raise DiagnosticCatalogError("selected states 不能为空")
    dimensions = {
        "instance": Counter(str(row["instance_alias"]) for row in selected),
        "source_family": Counter(str(row["source_family"]) for row in selected),
        "time_bin": Counter(str(row["time_bin"]) for row in selected),
        "candidate_count_bin": Counter(
            str(row["candidate_count_bin"]) for row in selected
        ),
        "observed_kind": Counter(str(row["observed_kind"]) for row in selected),
    }
    return {
        "states": len(selected),
        "unique_state_hashes": len({row["state_hash"] for row in selected}),
        "signal_eligible_states": sum(
            bool(row["signal_eligible"]) for row in selected
        ),
        "exhaustive_eligible_states": sum(
            bool(row["exhaustive_eligible"]) for row in selected
        ),
        "candidate_count_min": min(int(row["candidate_count"]) for row in selected),
        "candidate_count_max": max(int(row["candidate_count"]) for row in selected),
        "marginals": {
            name: dict(sorted(counter.items()))
            for name, counter in dimensions.items()
        },
    }


def catalog_prelabel_audit(
    catalog: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    """仅用标签前字段统计可行动、信号与 exhaustive 库存。"""
    validate_diagnostic_config(config)
    selection = config["state_selection"]
    validity = config["label_validity"]
    max_step = int(config["replay"]["max_step_inclusive"])
    actionable_min = int(selection["actionable_min_candidates"])
    signal_min = int(selection["signal_min_candidates"])
    exhaustive_min = int(validity["exhaustive_candidate_count_min"])
    exhaustive_max = int(validity["exhaustive_candidate_count_max"])

    def summarize(rows: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
        in_step = [row for row in rows if 0 <= int(row["step"]) <= max_step]
        uncapped = [row for row in in_step if not bool(row.get("cap_reached"))]
        actionable = [
            row
            for row in uncapped
            if int(row["candidate_count"]) >= actionable_min
        ]
        return {
            "catalog_states": len(rows),
            "observed_skip_states": sum(
                bool(row["observed_is_skip"]) for row in rows
            ),
            "within_step_states": len(in_step),
            "cap_hit_states": sum(bool(row.get("cap_reached")) for row in rows),
            "actionable_states": len(actionable),
            "actionable_observed_skip_states": sum(
                bool(row["observed_is_skip"]) for row in actionable
            ),
            "signal_eligible_states": sum(
                int(row["candidate_count"]) >= signal_min
                for row in actionable
            ),
            "exhaustive_eligible_states": sum(
                exhaustive_min
                <= int(row["candidate_count"])
                <= exhaustive_max
                for row in actionable
            ),
        }

    dev_rows = [
        row
        for row in catalog
        if row.get("split") == "dev"
        and row.get("instance_alias") in config["split"]["dev"]
    ]
    observed_skip_rows = [
        row for row in dev_rows if bool(row["observed_is_skip"])
    ]
    skip_counts = [int(row["candidate_count"]) for row in observed_skip_rows]
    payload: Dict[str, Any] = {
        "schema_version": "eosbench-ccod-prelabel-audit-v1",
        "thresholds": {
            "max_step_inclusive": max_step,
            "actionable_min_candidates": actionable_min,
            "signal_min_candidates": signal_min,
            "exhaustive_candidate_count_min": exhaustive_min,
            "exhaustive_candidate_count_max": exhaustive_max,
        },
        "totals": summarize(dev_rows),
        "per_instance": {
            instance_alias: summarize(
                [
                    row
                    for row in dev_rows
                    if row["instance_alias"] == instance_alias
                ]
            )
            for instance_alias in config["split"]["dev"]
        },
        "observed_skip_candidate_count": {
            "states": len(skip_counts),
            "minimum": min(skip_counts) if skip_counts else None,
            "maximum": max(skip_counts) if skip_counts else None,
            "below_actionable": sum(
                count < actionable_min for count in skip_counts
            ),
        },
    }
    payload["audit_hash"] = sha256_json(payload)
    return payload
