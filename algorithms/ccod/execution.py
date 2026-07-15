# -*- coding: utf-8 -*-
"""CCOD 标签执行器的纯计划、身份与 signal gate 核心。

本模块刻意不包含命令行、子进程、状态恢复或真实 oracle。调用方必须先用
外部 verifier 验收冻结包，再把本模块返回的不可变计划交给运行层。
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from algorithms.ccod.metrics import summarize_state_q_values
from schedulers.state_replay import sha256_json


EXECUTION_IDENTITY_SCHEMA_VERSION = "eosbench-ccod-execution-identity-v1"
SIGNAL_GATE_SUMMARY_SCHEMA_VERSION = "eosbench-ccod-signal-gate-summary-v1"
SCIENTIFIC_LABELS_SCHEMA_VERSION = "eosbench-ccod-scientific-labels-v1"

FROZEN_STATE_COUNT = 100
FROZEN_QUERY_COUNT = 1570
QUERY_BUDGET_PER_STATE = 16
SIGNAL_MIN_CANDIDATES = 10
SIGNAL_SPREAD_THRESHOLD = 0.01
MINIMUM_SIGNAL_STATES = 80
MINIMUM_SIGNAL_STATES_PER_INSTANCE = 40
MINIMUM_SIGNAL_PASS_FRACTION = 0.60
MINIMUM_PER_INSTANCE_PASS_FRACTION = 0.50

_SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")
_FROZEN_RUNTIME = {
    "python_implementation": "cpython",
    "python_version": "3.10.20",
}
_FROZEN_GUARDS = {
    "workers": 1,
    "max_attempts": 2,
    "query_timeout_s": 120.0,
    "state_timeout_s": 1200.0,
    "worker_peak_rss_limit_mib": 6144.0,
}


class CCODExecutionError(ValueError):
    """执行计划、标签记录或 gate 输入不闭合时抛出的异常。"""


class ExecutionIdentityError(CCODExecutionError):
    """执行身份与冻结计划或标签记录冲突时抛出的异常。"""


@dataclass(frozen=True)
class FrozenDiagnosticPlan:
    """已由外部 verifier 验收、并经本模块结构校验的冻结计划。"""

    frozen_dir: Path
    run_manifest: Mapping[str, Any]
    selected_states: Tuple[Mapping[str, Any], ...]
    query_header: Mapping[str, Any]
    query_rows: Tuple[Mapping[str, Any], ...]
    queries_by_state: Mapping[int, Tuple[Mapping[str, Any], ...]]

    @property
    def run_id(self) -> str:
        """返回冻结 run 的内容身份。"""
        return str(self.run_manifest["run_id"])

    @property
    def query_plan_hash(self) -> str:
        """返回有序查询计划的科学哈希。"""
        return str(self.run_manifest["query_plan_hash"])

    @property
    def selection_hash(self) -> str:
        """返回 100-state 选择的科学哈希。"""
        return str(self.run_manifest["selection_hash"])


def _strict_int(value: Any, field_name: str) -> int:
    """拒绝 bool、字符串与非整数，避免隐式强制转换掩盖漂移。"""
    if isinstance(value, bool) or not isinstance(value, int):
        raise CCODExecutionError(f"{field_name} 必须是整数")
    return value


def _require_sha256(value: Any, field_name: str) -> str:
    """校验统一的带算法前缀、小写 SHA-256 文本。"""
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise CCODExecutionError(f"{field_name} 必须是规范 SHA-256")
    return value


def _strict_json_object(path: Path) -> Dict[str, Any]:
    """读取单个 JSON object，并拒绝重复键。"""

    def reject_duplicate_keys(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise CCODExecutionError(f"{path} 含重复 JSON 键: {key}")
            result[key] = value
        return result

    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle, object_pairs_hook=reject_duplicate_keys)
    except CCODExecutionError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise CCODExecutionError(f"无法读取冻结 JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CCODExecutionError(f"{path} 必须包含 JSON object")
    return payload


def _strict_jsonl_objects(path: Path) -> List[Dict[str, Any]]:
    """读取 JSONL object；空行、残缺末行和重复键均视为错误。"""

    def reject_duplicate_keys(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise CCODExecutionError(f"{path} 含重复 JSON 键: {key}")
            result[key] = value
        return result

    rows: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.endswith("\n") or not line.strip():
                    raise CCODExecutionError(
                        f"{path}:{line_number} 不是完整 JSONL 记录"
                    )
                value = json.loads(
                    line,
                    object_pairs_hook=reject_duplicate_keys,
                )
                if not isinstance(value, dict):
                    raise CCODExecutionError(
                        f"{path}:{line_number} 必须是 JSON object"
                    )
                rows.append(value)
    except CCODExecutionError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise CCODExecutionError(f"无法读取冻结 JSONL {path}: {exc}") from exc
    if not rows:
        raise CCODExecutionError(f"冻结 JSONL 不能为空: {path}")
    return rows


def _verify_embedded_hash(
    payload: Mapping[str, Any],
    hash_field: str,
    *,
    label: str,
) -> None:
    """复核行级内容哈希，尽早截断意外编辑或错误目录。"""
    expected = _require_sha256(payload.get(hash_field), f"{label}.{hash_field}")
    unhashed = {key: value for key, value in payload.items() if key != hash_field}
    if sha256_json(unhashed) != expected:
        raise CCODExecutionError(f"{label} 的 {hash_field} 不一致")


def load_frozen_diagnostic_plan(frozen_dir: str | Path) -> FrozenDiagnosticPlan:
    """严格加载 100-state/1570-query 计划，不触发任何标签计算。

    ``frozen_dir`` 必须事先通过完整外部 verifier。本函数仍复核执行层依赖的
    数量、ordinal、内容哈希和交叉引用，防止调用方把错误目录交给 runner。
    """
    root = Path(frozen_dir).resolve()
    if not root.is_dir():
        raise CCODExecutionError(f"冻结目录不存在: {root}")

    run_manifest = _strict_json_object(root / "run_manifest.json")
    selected_rows = _strict_jsonl_objects(root / "selected_states.jsonl")
    query_records = _strict_jsonl_objects(root / "query_plan.jsonl")
    if len(selected_rows) != FROZEN_STATE_COUNT:
        raise CCODExecutionError("selected_states 必须恰好包含 100 个状态")
    if len(query_records) != FROZEN_QUERY_COUNT + 1:
        raise CCODExecutionError("query_plan 必须包含 1 个 header 和 1570 个查询")

    _verify_embedded_hash(run_manifest, "manifest_hash", label="run manifest")
    run_id = _require_sha256(run_manifest.get("run_id"), "run_manifest.run_id")
    selection_hash = _require_sha256(
        run_manifest.get("selection_hash"),
        "run_manifest.selection_hash",
    )
    query_plan_hash = _require_sha256(
        run_manifest.get("query_plan_hash"),
        "run_manifest.query_plan_hash",
    )
    if (
        run_manifest.get("schema_version")
        != "eosbench-ccod-diagnostic-run-v1"
    ):
        raise CCODExecutionError("run manifest schema_version 不受支持")
    for field_name, expected in (
        ("planned_states", FROZEN_STATE_COUNT),
        ("restore_count", FROZEN_STATE_COUNT),
        ("planned_queries", FROZEN_QUERY_COUNT),
        ("workers", 1),
    ):
        if _strict_int(run_manifest.get(field_name), field_name) != expected:
            raise CCODExecutionError(f"run manifest 的 {field_name} 已漂移")

    selected_ordinals: List[int] = []
    state_hashes: List[str] = []
    for row_index, row in enumerate(selected_rows):
        _verify_embedded_hash(row, "row_hash", label=f"selected[{row_index}]")
        selection = row.get("selection")
        if not isinstance(selection, Mapping):
            raise CCODExecutionError("selected state 缺少 selection object")
        ordinal = _strict_int(
            selection.get("selected_ordinal"),
            "selection.selected_ordinal",
        )
        selected_ordinals.append(ordinal)
        state_hashes.append(_require_sha256(row.get("state_hash"), "state_hash"))
        candidate_count = _strict_int(row.get("candidate_count"), "candidate_count")
        if candidate_count <= 0:
            raise CCODExecutionError("candidate_count 必须为严格正整数")
        if row.get("signal_eligible") is not (
            candidate_count >= SIGNAL_MIN_CANDIDATES
        ):
            raise CCODExecutionError(
                "signal_eligible 与 candidate_count 门槛不一致"
            )
        if not isinstance(row.get("instance_alias"), str):
            raise CCODExecutionError("selected state 缺少 instance_alias")
    if selected_ordinals != list(range(FROZEN_STATE_COUNT)):
        raise CCODExecutionError("selected_ordinal 不唯一连续或文件顺序漂移")
    if len(set(state_hashes)) != FROZEN_STATE_COUNT:
        raise CCODExecutionError("selected_states 含重复 state_hash")
    if sha256_json(selected_rows) != selection_hash:
        raise CCODExecutionError("selection_hash 与 selected_states 不一致")

    query_header, query_rows = query_records[0], query_records[1:]
    _verify_embedded_hash(query_header, "header_hash", label="query header")
    if query_header.get("record_type") != "header":
        raise CCODExecutionError("query_plan 首行必须是 header")
    for field_name, expected in (
        ("run_id", run_id),
        ("query_plan_hash", query_plan_hash),
        ("states", FROZEN_STATE_COUNT),
        ("queries", FROZEN_QUERY_COUNT),
    ):
        if query_header.get(field_name) != expected:
            raise CCODExecutionError(f"query header 的 {field_name} 引用不一致")

    grouped: Dict[int, List[Mapping[str, Any]]] = {
        ordinal: [] for ordinal in range(FROZEN_STATE_COUNT)
    }
    query_keys: List[str] = []
    query_ordinals: List[int] = []
    for row_index, row in enumerate(query_rows):
        _verify_embedded_hash(row, "row_hash", label=f"query[{row_index}]")
        query_ordinal = _strict_int(row.get("query_ordinal"), "query_ordinal")
        query_ordinals.append(query_ordinal)
        if row.get("run_id") != run_id:
            raise CCODExecutionError("query row 的 run_id 引用不一致")
        state_ordinal = _strict_int(row.get("state_ordinal"), "state_ordinal")
        if state_ordinal not in grouped:
            raise CCODExecutionError("query row 引用了 selection 外 state_ordinal")
        state = selected_rows[state_ordinal]
        if (
            row.get("state_hash") != state.get("state_hash")
            or row.get("instance_alias") != state.get("instance_alias")
        ):
            raise CCODExecutionError("query row 的 state 引用不一致")
        identity = row.get("query_identity")
        if not isinstance(identity, Mapping):
            raise CCODExecutionError("query row 缺少 query_identity object")
        if identity.get("state_hash") != state.get("state_hash"):
            raise CCODExecutionError("query_identity 的 state_hash 引用不一致")
        query_key = _require_sha256(row.get("query_key"), "query_key")
        if sha256_json(dict(identity)) != query_key:
            raise CCODExecutionError("query_key 与 query_identity 不一致")
        query_keys.append(query_key)
        grouped[state_ordinal].append(row)

    if query_ordinals != list(range(FROZEN_QUERY_COUNT)):
        raise CCODExecutionError("query_ordinal 不唯一连续或文件顺序漂移")
    if len(set(query_keys)) != FROZEN_QUERY_COUNT:
        raise CCODExecutionError("query plan 含重复 query_key")
    if sha256_json(query_rows) != query_plan_hash:
        raise CCODExecutionError("query_plan_hash 与 query rows 不一致")

    frozen_grouped: Dict[int, Tuple[Mapping[str, Any], ...]] = {}
    for state_ordinal, state_queries in grouped.items():
        state = selected_rows[state_ordinal]
        action_ordinals = [
            _strict_int(row.get("action_ordinal"), "action_ordinal")
            for row in state_queries
        ]
        if action_ordinals != list(range(len(state_queries))):
            raise CCODExecutionError(
                f"state[{state_ordinal}] action_ordinal 不唯一连续"
            )
        expected = min(QUERY_BUDGET_PER_STATE, int(state["candidate_count"]))
        if len(state_queries) != expected:
            raise CCODExecutionError(
                f"state[{state_ordinal}] query 数不等于 min(16, C)"
            )
        frozen_grouped[state_ordinal] = tuple(state_queries)

    return FrozenDiagnosticPlan(
        frozen_dir=root,
        run_manifest=run_manifest,
        selected_states=tuple(selected_rows),
        query_header=query_header,
        query_rows=tuple(query_rows),
        queries_by_state=frozen_grouped,
    )


def build_execution_identity(
    plan: FrozenDiagnosticPlan,
    *,
    runner_implementation_hash: str,
) -> Dict[str, Any]:
    """构造与机器路径、运行时间和 cache hit 无关的执行身份。"""
    implementation_hash = _require_sha256(
        runner_implementation_hash,
        "runner_implementation_hash",
    )
    return {
        "schema_version": EXECUTION_IDENTITY_SCHEMA_VERSION,
        "frozen_plan": {
            "run_id": plan.run_id,
            "query_plan_hash": plan.query_plan_hash,
            "selection_hash": plan.selection_hash,
        },
        "runner_implementation_hash": implementation_hash,
        "runtime": dict(_FROZEN_RUNTIME),
        "guards": dict(_FROZEN_GUARDS),
    }


def execution_identity_hash(identity: Mapping[str, Any]) -> str:
    """用规范 JSON 对 execution identity 计算 SHA-256。"""
    if not isinstance(identity, Mapping):
        raise ExecutionIdentityError("execution identity 必须是 object")
    return sha256_json(dict(identity))


def _validate_execution_identity(
    plan: FrozenDiagnosticPlan,
    identity: Mapping[str, Any],
) -> str:
    """确认调用方身份只含冻结字段，且全部绑定当前计划。"""
    if not isinstance(identity, Mapping):
        raise ExecutionIdentityError("execution identity 必须是 object")
    implementation_hash = identity.get("runner_implementation_hash")
    try:
        expected = build_execution_identity(
            plan,
            runner_implementation_hash=str(implementation_hash),
        )
    except CCODExecutionError as exc:
        raise ExecutionIdentityError(str(exc)) from exc
    if dict(identity) != expected:
        raise ExecutionIdentityError("execution identity 与冻结计划或护栏冲突")
    return execution_identity_hash(expected)


def _result_q_value(row: Mapping[str, Any]) -> float:
    """读取成功结果的 Q_H，并规范化为有限 float。"""
    source = row.get("result")
    payload = source if isinstance(source, Mapping) else row
    try:
        if "q_h_hex" in payload:
            value = float.fromhex(str(payload["q_h_hex"]))
        else:
            raw = payload["q_h"]
            if isinstance(raw, bool):
                raise TypeError
            value = float(raw)
    except (KeyError, TypeError, ValueError) as exc:
        raise CCODExecutionError("success query 缺少合法 q_h/q_h_hex") from exc
    if not math.isfinite(value):
        raise CCODExecutionError("Q_H 必须是有限数值")
    return 0.0 if value == 0.0 else value


def _index_query_results(
    plan: FrozenDiagnosticPlan,
    identity: Mapping[str, Any],
    query_results: Iterable[Mapping[str, Any]],
) -> Tuple[str, Dict[str, Tuple[Mapping[str, Any], float | None]]]:
    """校验运行结果对计划的引用，并按 query_key 建立唯一索引。"""
    expected_execution_id = _validate_execution_identity(plan, identity)
    planned_by_key = {
        str(row["query_key"]): row
        for row in plan.query_rows
    }
    indexed: Dict[str, Tuple[Mapping[str, Any], float | None]] = {}
    for result_index, result_row in enumerate(query_results):
        if not isinstance(result_row, Mapping):
            raise CCODExecutionError(f"query result[{result_index}] 必须是 object")
        if result_row.get("execution_id") != expected_execution_id:
            raise ExecutionIdentityError(
                f"query result[{result_index}] 的 execution_id 冲突"
            )
        if "run_id" in result_row and result_row.get("run_id") != plan.run_id:
            raise ExecutionIdentityError(
                f"query result[{result_index}] 的 run_id 冲突"
            )
        query_key = str(result_row.get("query_key", ""))
        planned = planned_by_key.get(query_key)
        if planned is None:
            raise CCODExecutionError("query result 引用了计划外 query_key")
        if query_key in indexed:
            raise CCODExecutionError("query results 含重复 query_key")
        if (
            result_row.get("query_ordinal") != planned.get("query_ordinal")
            or result_row.get("state_hash") != planned.get("state_hash")
        ):
            raise CCODExecutionError("query result 的 ordinal/state 引用不一致")
        status = result_row.get("status")
        q_value = _result_q_value(result_row) if status == "success" else None
        indexed[query_key] = (result_row, q_value)
    return expected_execution_id, indexed


def scientific_results_hash(
    plan: FrozenDiagnosticPlan,
    execution_identity: Mapping[str, Any],
    query_results: Iterable[Mapping[str, Any]],
) -> str:
    """计算冷缓存与热缓存运行应完全相同的科学标签哈希。

    只保留 query 的冻结引用和规范 Q_H；cache hit、绝对路径、attempt、
    时间戳、耗时、RSS 与输入顺序均不会进入载荷。
    """
    execution_id, indexed = _index_query_results(
        plan,
        execution_identity,
        query_results,
    )
    if len(indexed) != FROZEN_QUERY_COUNT or any(
        q_value is None for _, q_value in indexed.values()
    ):
        raise CCODExecutionError("scientific hash 只接受完整成功的 1570 个查询")
    labels = []
    for planned in plan.query_rows:
        query_key = str(planned["query_key"])
        q_value = indexed[query_key][1]
        assert q_value is not None
        labels.append(
            {
                "query_ordinal": planned["query_ordinal"],
                "query_key": query_key,
                "state_hash": planned["state_hash"],
                "q_h_hex": q_value.hex(),
            }
        )
    return sha256_json(
        {
            "schema_version": SCIENTIFIC_LABELS_SCHEMA_VERSION,
            "execution_id": execution_id,
            "labels": labels,
        }
    )


def _incomplete_signal_summary(
    plan: FrozenDiagnosticPlan,
    execution_id: str,
    indexed: Mapping[str, Tuple[Mapping[str, Any], float | None]],
) -> Dict[str, Any]:
    """构造不泄漏半成品 gate 结论的统一 incomplete 结果。"""
    successful_keys = {
        key for key, (_, q_value) in indexed.items() if q_value is not None
    }
    completed_states = sum(
        all(str(row["query_key"]) in successful_keys for row in state_queries)
        for state_queries in plan.queries_by_state.values()
    )
    return {
        "schema_version": SIGNAL_GATE_SUMMARY_SCHEMA_VERSION,
        "execution_id": execution_id,
        "execution_status": "incomplete",
        "signal_gate": "not_evaluated",
        "method_decision": None,
        "status": "incomplete",
        "decision": "not_evaluated",
        "passed": None,
        "required_states": FROZEN_STATE_COUNT,
        "completed_states": completed_states,
        "required_queries": FROZEN_QUERY_COUNT,
        "completed_queries": len(successful_keys),
        "scientific_results_hash": None,
        "overall": {
            "eligible_states": None,
            "passing_states": None,
            "pass_fraction": None,
            "passed": None,
        },
        "per_instance": None,
    }


def _fraction_at_least(
    numerator: int,
    denominator: int,
    *,
    threshold_numerator: int,
    threshold_denominator: int,
) -> bool:
    """用整数交叉相乘判定比例门槛，避免浮点边界漂移。"""
    if denominator <= 0:
        return False
    return (
        numerator * threshold_denominator
        >= denominator * threshold_numerator
    )


def _describe_population(
    rows: Sequence[Mapping[str, Any]],
    aliases: Sequence[str],
) -> Dict[str, Any]:
    """分别报告 all/actionable/signal 的连续 spread 描述统计。"""
    passing = [row for row in rows if row["spread_nontrivial"]]
    per_instance: Dict[str, Dict[str, Any]] = {}
    for alias in aliases:
        instance_rows = [row for row in rows if row["instance_alias"] == alias]
        instance_passing = [
            row for row in instance_rows if row["spread_nontrivial"]
        ]
        per_instance[alias] = {
            "states": len(instance_rows),
            "passing_states": len(instance_passing),
            "pass_fraction": (
                len(instance_passing) / len(instance_rows)
                if instance_rows
                else None
            ),
        }
    return {
        "states": len(rows),
        "passing_states": len(passing),
        "pass_fraction": len(passing) / len(rows) if rows else None,
        "per_instance": per_instance,
    }


def summarize_signal_gate(
    plan: FrozenDiagnosticPlan,
    execution_identity: Mapping[str, Any],
    query_results: Iterable[Mapping[str, Any]],
) -> Dict[str, Any]:
    """用 Type-7 P90-P10 汇总预注册 signal gate。

    只有 100 个状态和 1570 个查询全部成功时才给出 pass/fail。缺查询或失败
    查询统一返回 ``incomplete/not_evaluated/None``，身份冲突则直接抛错。
    """
    result_rows = list(query_results)
    execution_id, indexed = _index_query_results(
        plan,
        execution_identity,
        result_rows,
    )
    if len(indexed) != FROZEN_QUERY_COUNT or any(
        q_value is None for _, q_value in indexed.values()
    ):
        return _incomplete_signal_summary(plan, execution_id, indexed)

    state_summaries: List[Dict[str, Any]] = []
    for state_ordinal, state in enumerate(plan.selected_states):
        q_values = [
            indexed[str(query["query_key"])][1]
            for query in plan.queries_by_state[state_ordinal]
        ]
        if any(value is None for value in q_values):
            raise AssertionError("完整性分支不应包含缺失 Q_H")
        metric = summarize_state_q_values(
            [float(value) for value in q_values if value is not None],
            spread_threshold=SIGNAL_SPREAD_THRESHOLD,
        )
        signal_eligible = (
            bool(state.get("signal_eligible"))
            and int(state["candidate_count"]) >= SIGNAL_MIN_CANDIDATES
        )
        state_summaries.append(
            {
                "state_ordinal": state_ordinal,
                "state_hash": state["state_hash"],
                "instance_alias": state["instance_alias"],
                "signal_eligible": signal_eligible,
                "query_count": metric["query_count"],
                "p10_q": metric["p10_q"],
                "p10_q_hex": float(metric["p10_q"]).hex(),
                "median_q": metric["median_q"],
                "median_q_hex": float(metric["median_q"]).hex(),
                "p90_q": metric["p90_q"],
                "p90_q_hex": float(metric["p90_q"]).hex(),
                "p90_minus_p10": metric["p90_minus_p10"],
                "p90_minus_p10_hex": float(
                    metric["p90_minus_p10"]
                ).hex(),
                "spread_nontrivial": bool(metric["nontrivial_spread"]),
                "spread_passed": (
                    bool(metric["nontrivial_spread"])
                    if signal_eligible
                    else None
                ),
            }
        )

    eligible = [row for row in state_summaries if row["signal_eligible"]]
    passing = [row for row in eligible if row["spread_passed"] is True]
    aliases = sorted(
        {str(row["instance_alias"]) for row in plan.selected_states}
    )
    per_instance: Dict[str, Dict[str, Any]] = {}
    for alias in aliases:
        instance_eligible = [
            row for row in eligible if row["instance_alias"] == alias
        ]
        instance_passing = [
            row for row in instance_eligible if row["spread_passed"] is True
        ]
        fraction = (
            len(instance_passing) / len(instance_eligible)
            if instance_eligible
            else None
        )
        eligibility_passed = (
            len(instance_eligible) >= MINIMUM_SIGNAL_STATES_PER_INSTANCE
        )
        fraction_passed = (
            _fraction_at_least(
                len(instance_passing),
                len(instance_eligible),
                threshold_numerator=1,
                threshold_denominator=2,
            )
        )
        per_instance[alias] = {
            "eligible_states": len(instance_eligible),
            "minimum_eligible_states": MINIMUM_SIGNAL_STATES_PER_INSTANCE,
            "passing_states": len(instance_passing),
            "pass_fraction": fraction,
            "minimum_pass_fraction": MINIMUM_PER_INSTANCE_PASS_FRACTION,
            "eligibility_passed": eligibility_passed,
            "fraction_passed": fraction_passed,
            "passed": eligibility_passed and fraction_passed,
        }

    overall_fraction = len(passing) / len(eligible) if eligible else None
    overall_eligibility_passed = len(eligible) >= MINIMUM_SIGNAL_STATES
    overall_fraction_passed = (
        _fraction_at_least(
            len(passing),
            len(eligible),
            threshold_numerator=3,
            threshold_denominator=5,
        )
    )
    passed = (
        overall_eligibility_passed
        and overall_fraction_passed
        and bool(per_instance)
        and all(summary["passed"] for summary in per_instance.values())
    )
    return {
        "schema_version": SIGNAL_GATE_SUMMARY_SCHEMA_VERSION,
        "execution_id": execution_id,
        "execution_status": "complete",
        "signal_gate": "pass" if passed else "fail",
        "method_decision": "go" if passed else "no_go",
        "status": "complete",
        "decision": "passed" if passed else "failed",
        "passed": passed,
        "required_states": FROZEN_STATE_COUNT,
        "completed_states": FROZEN_STATE_COUNT,
        "required_queries": FROZEN_QUERY_COUNT,
        "completed_queries": FROZEN_QUERY_COUNT,
        "scientific_results_hash": scientific_results_hash(
            plan,
            execution_identity,
            result_rows,
        ),
        "spread_metric": "Q_H_P90_minus_P10_type7",
        "spread_threshold": SIGNAL_SPREAD_THRESHOLD,
        "populations": {
            "all_selected": _describe_population(state_summaries, aliases),
            "actionable": _describe_population(
                [
                    row
                    for row, state in zip(
                        state_summaries,
                        plan.selected_states,
                    )
                    if int(state["candidate_count"]) >= 2
                ],
                aliases,
            ),
            "signal": _describe_population(eligible, aliases),
        },
        "overall": {
            "eligible_states": len(eligible),
            "minimum_eligible_states": MINIMUM_SIGNAL_STATES,
            "passing_states": len(passing),
            "pass_fraction": overall_fraction,
            "minimum_pass_fraction": MINIMUM_SIGNAL_PASS_FRACTION,
            "eligibility_passed": overall_eligibility_passed,
            "fraction_passed": overall_fraction_passed,
            "passed": overall_eligibility_passed and overall_fraction_passed,
        },
        "per_instance": per_instance,
        "states": state_summaries,
    }
