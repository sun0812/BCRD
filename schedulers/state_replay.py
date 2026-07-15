# -*- coding: utf-8 -*-
"""为 EOS 调度反事实分析提供确定性的状态重建。

旧版轨迹 JSONL 只保存聚合特征，无法重建影响可行性判断的部分调度。本模块
定义紧凑的旁路清单，并根据原始场景与稳定动作键的精确序列重建每个状态。

物理状态标识的哈希会刻意排除求解器来源，使不同求解器生成的相同前缀能够
共享缓存的反事实标签。
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from algorithms.candidate_pool import enumerate_task_candidates
from algorithms.objectives import ObjectiveModel, ObjectiveWeights
from schedulers.constraint_model import Assignment, ConstraintModel, Schedule
from schedulers.scenario_loader import (
    SchedulingProblem,
    load_scheduling_problem_from_json,
)


MANIFEST_SCHEMA_VERSION = "eosbench-state-replay-v1"
ACTION_KEY_VERSION = "eosbench-action-key-v1"


class StateReplayError(ValueError):
    """清单无法被精确重建时抛出的异常。"""


def canonical_json_bytes(value: Any) -> bytes:
    """使用唯一且稳定的 JSON 表示序列化哈希载荷。"""
    try:
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise StateReplayError(f"value is not canonical-JSON serializable: {exc}") from exc
    return text.encode("utf-8")


def sha256_json(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _without_hash(payload: Mapping[str, Any], hash_field: str) -> Dict[str, Any]:
    return {key: value for key, value in payload.items() if key != hash_field}


def _verify_payload_hash(
    payload: Mapping[str, Any],
    hash_field: str,
    *,
    label: str,
) -> None:
    expected = payload.get(hash_field)
    actual = sha256_json(_without_hash(payload, hash_field))
    if expected != actual:
        raise StateReplayError(
            f"{label} hash mismatch: manifest={expected!r}, reconstructed={actual!r}"
        )


def _float_payload(value: float) -> str:
    return float(value).hex()


def _float_from_payload(value: Any, field: str) -> float:
    if not isinstance(value, str):
        raise StateReplayError(f"{field} must be a float.hex() string")
    try:
        return float.fromhex(value)
    except ValueError as exc:
        raise StateReplayError(f"invalid {field}: {value!r}") from exc


@dataclass(frozen=True)
class ConstraintConfig:
    placement_mode: str = "earliest"
    downlink_duration_ratio: float = 1.0
    agility_profile: str = "Standard-Agility"
    non_agile_transition_s: float = 10.0

    def to_payload(self) -> Dict[str, Any]:
        return {
            "placement_mode": str(self.placement_mode),
            "downlink_duration_ratio": _float_payload(self.downlink_duration_ratio),
            "agility_profile": str(self.agility_profile),
            "non_agile_transition_s": _float_payload(self.non_agile_transition_s),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ConstraintConfig":
        return cls(
            placement_mode=str(payload["placement_mode"]),
            downlink_duration_ratio=_float_from_payload(
                payload["downlink_duration_ratio"], "downlink_duration_ratio"
            ),
            agility_profile=str(payload["agility_profile"]),
            non_agile_transition_s=_float_from_payload(
                payload["non_agile_transition_s"], "non_agile_transition_s"
            ),
        )

    @property
    def hash(self) -> str:
        return sha256_json(self.to_payload())

    def build_model(self, problem: SchedulingProblem) -> ConstraintModel:
        return ConstraintModel(
            problem=problem,
            placement_mode=self.placement_mode,
            downlink_duration_ratio=self.downlink_duration_ratio,
            agility_profile=self.agility_profile,
            non_agile_transition_s=self.non_agile_transition_s,
        )


@dataclass(frozen=True)
class EnumeratorConfig:
    max_candidates: int = 8192
    random_samples_per_window: int = 0
    ordering_version: str = "canonical_v1"
    seed: int = 0

    def __post_init__(self) -> None:
        if int(self.max_candidates) <= 0:
            raise ValueError("max_candidates must be positive")
        if int(self.random_samples_per_window) < 0:
            raise ValueError("random_samples_per_window must be non-negative")

    def to_payload(self) -> Dict[str, Any]:
        return {
            "max_candidates": int(self.max_candidates),
            "random_samples_per_window": int(self.random_samples_per_window),
            "ordering_version": str(self.ordering_version),
            "seed": int(self.seed),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "EnumeratorConfig":
        return cls(
            max_candidates=int(payload["max_candidates"]),
            random_samples_per_window=int(payload["random_samples_per_window"]),
            ordering_version=str(payload["ordering_version"]),
            seed=int(payload["seed"]),
        )

    @property
    def hash(self) -> str:
        return sha256_json(self.to_payload())


@dataclass(frozen=True)
class ObjectiveConfig:
    weights: Tuple[float, float, float, float]

    def normalized(self) -> "ObjectiveConfig":
        w = ObjectiveWeights(*self.weights).normalized()
        return ObjectiveConfig(
            (w.w_profit, w.w_completion, w.w_timeliness, w.w_balance)
        )

    def to_payload(self) -> Dict[str, Any]:
        normalized = self.normalized()
        return {
            "weights": [_float_payload(value) for value in normalized.weights],
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ObjectiveConfig":
        raw = payload.get("weights")
        if not isinstance(raw, list) or len(raw) != 4:
            raise StateReplayError("objective weights must contain four values")
        return cls(
            tuple(
                _float_from_payload(value, f"objective.weights[{index}]")
                for index, value in enumerate(raw)
            )  # type: ignore[arg-type]
        )

    @property
    def hash(self) -> str:
        return sha256_json(self.to_payload())

    def build_model(self, problem: SchedulingProblem) -> ObjectiveModel:
        return ObjectiveModel(problem, ObjectiveWeights(*self.normalized().weights))


@dataclass
class ReplayedState:
    """已验证部分调度状态的防御性副本。"""

    schedule: Schedule
    task_id: str
    candidates: Tuple[Optional[Assignment], ...]
    state_manifest: Dict[str, Any]
    objective_score: float


def canonical_task_ids(problem: SchedulingProblem) -> Tuple[str, ...]:
    return tuple(
        task.id
        for task in sorted(
            problem.tasks.values(),
            key=lambda task: (-float(task.priority), str(task.id)),
        )
    )


def _relative_microseconds(reference: datetime, value: Optional[datetime]) -> Optional[int]:
    if value is None:
        return None
    delta = value - reference
    return (
        (delta.days * 24 * 60 * 60 + delta.seconds) * 1_000_000
        + delta.microseconds
    )


def candidate_action_key(
    problem: SchedulingProblem,
    task_id: str,
    candidate: Optional[Assignment],
) -> Dict[str, Any]:
    """返回 SKIP 或指派相对于场景起点的身份键。"""
    if candidate is None:
        return {
            "version": ACTION_KEY_VERSION,
            "kind": "skip",
            "task_id": str(task_id),
        }
    if candidate.task_id != task_id:
        raise StateReplayError(
            f"candidate task mismatch: expected {task_id!r}, got {candidate.task_id!r}"
        )
    return {
        "version": ACTION_KEY_VERSION,
        "kind": "assign",
        "task_id": str(task_id),
        "satellite_id": str(candidate.satellite_id),
        "sensor_id": str(candidate.sensor_id or ""),
        "orbit_number": int(candidate.orbit_number or 0),
        "sat_window_id": int(candidate.sat_window_id),
        "sat_start_us": _relative_microseconds(
            problem.start_time, candidate.sat_start_time
        ),
        "sat_end_us": _relative_microseconds(problem.start_time, candidate.sat_end_time),
        "ground_station_id": candidate.ground_station_id,
        "gs_window_id": (
            int(candidate.gs_window_id)
            if candidate.gs_window_id is not None
            else None
        ),
        "gs_start_us": _relative_microseconds(problem.start_time, candidate.gs_start_time),
        "gs_end_us": _relative_microseconds(problem.start_time, candidate.gs_end_time),
    }


def _action_key_index(
    problem: SchedulingProblem,
    task_id: str,
    candidates: Sequence[Optional[Assignment]],
) -> Dict[bytes, Optional[Assignment]]:
    index: Dict[bytes, Optional[Assignment]] = {}
    for candidate in candidates:
        key = canonical_json_bytes(candidate_action_key(problem, task_id, candidate))
        if key in index:
            raise StateReplayError(
                f"duplicate action key for task {task_id}: {key.decode('utf-8')}"
            )
        index[key] = candidate
    return index


def enumerate_feasible_actions(
    problem: SchedulingProblem,
    schedule: Schedule,
    task_id: str,
    constraint_config: ConstraintConfig,
    enumerator_config: EnumeratorConfig,
) -> List[Optional[Assignment]]:
    actions, _ = _enumerate_feasible_action_set(
        problem,
        schedule,
        task_id,
        constraint_config,
        enumerator_config,
    )
    return actions


def _enumerate_feasible_action_set(
    problem: SchedulingProblem,
    schedule: Schedule,
    task_id: str,
    constraint_config: ConstraintConfig,
    enumerator_config: EnumeratorConfig,
) -> Tuple[List[Optional[Assignment]], Dict[str, Any]]:
    task = problem.tasks.get(task_id)
    if task is None:
        raise StateReplayError(f"unknown task id: {task_id}")
    candidates = enumerate_task_candidates(
        problem=problem,
        task=task,
        placement_mode=constraint_config.placement_mode,
        downlink_duration_ratio=constraint_config.downlink_duration_ratio,
        max_candidates=enumerator_config.max_candidates,
        random_samples_per_window=enumerator_config.random_samples_per_window,
        seed=enumerator_config.seed,
        ordering_version=enumerator_config.ordering_version,
    )
    model = constraint_config.build_model(problem)
    feasible = [
        candidate
        for candidate in candidates
        if model.is_feasible_assignment(candidate, schedule)
    ]
    actions: List[Optional[Assignment]] = [None]
    actions.extend(feasible)
    _action_key_index(problem, task_id, actions)
    stats = {
        "cap": enumerator_config.max_candidates,
        "enumerated_before_feasibility": len(candidates),
        "feasible_after_cap": len(feasible),
        "cap_reached": len(candidates) >= enumerator_config.max_candidates,
    }
    return actions, stats


def _ordered_candidate_keys(
    problem: SchedulingProblem,
    task_id: str,
    candidates: Sequence[Optional[Assignment]],
) -> List[Dict[str, Any]]:
    return [
        candidate_action_key(problem, task_id, candidate)
        for candidate in candidates
    ]


def _schedule_keys(
    problem: SchedulingProblem,
    schedule: Schedule,
) -> List[Dict[str, Any]]:
    keys = [
        candidate_action_key(problem, assignment.task_id, assignment)
        for assignment in schedule.assignments
    ]
    return sorted(keys, key=canonical_json_bytes)


def schedule_hash(problem: SchedulingProblem, schedule: Schedule) -> str:
    return sha256_json(_schedule_keys(problem, schedule))


def _candidate_hashes(keys: Sequence[Mapping[str, Any]]) -> Tuple[str, str]:
    ordered = sha256_json(list(keys))
    sorted_keys = sorted(
        (dict(key) for key in keys),
        key=canonical_json_bytes,
    )
    membership = sha256_json(sorted_keys)
    return ordered, membership


def _parse_iso_time(value: Any) -> datetime:
    if not isinstance(value, str):
        raise StateReplayError(f"expected ISO datetime string, got {value!r}")
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def _raw_assignment_matches(
    raw: Mapping[str, Any],
    candidate: Assignment,
) -> bool:
    if str(raw.get("task_id")) != str(candidate.task_id):
        return False
    if str(raw.get("satellite_id")) != str(candidate.satellite_id):
        return False
    raw_sensor = str(raw.get("sensor_id") or "")
    if raw_sensor and raw_sensor != str(candidate.sensor_id or ""):
        return False
    if raw.get("orbit_number") is not None:
        if int(raw.get("orbit_number") or 0) != int(candidate.orbit_number or 0):
            return False
    if _parse_iso_time(raw.get("sat_start_time")) != candidate.sat_start_time:
        return False
    if _parse_iso_time(raw.get("sat_end_time")) != candidate.sat_end_time:
        return False

    raw_gs = raw.get("ground_station_id")
    if raw_gs != candidate.ground_station_id:
        return False
    raw_gs_start = raw.get("gs_start_time")
    raw_gs_end = raw.get("gs_end_time")
    if raw_gs_start is None:
        if candidate.gs_start_time is not None:
            return False
    elif _parse_iso_time(raw_gs_start) != candidate.gs_start_time:
        return False
    if raw_gs_end is None:
        if candidate.gs_end_time is not None:
            return False
    elif _parse_iso_time(raw_gs_end) != candidate.gs_end_time:
        return False
    return True


def _match_raw_assignment(
    raw: Mapping[str, Any],
    candidates: Sequence[Optional[Assignment]],
) -> Assignment:
    matches = [
        candidate
        for candidate in candidates
        if candidate is not None and _raw_assignment_matches(raw, candidate)
    ]
    if len(matches) != 1:
        task_id = raw.get("task_id")
        raise StateReplayError(
            f"raw assignment for task {task_id!r} matched {len(matches)} "
            "canonical feasible candidates; expected exactly one"
        )
    return matches[0]


def _state_snapshot(
    problem: SchedulingProblem,
    schedule: Schedule,
    *,
    step: int,
    task_ids: Sequence[str],
    prefix_actions: Sequence[Mapping[str, Any]],
    scenario_hash: str,
    task_order_hash: str,
    constraint_config: ConstraintConfig,
    enumerator_config: EnumeratorConfig,
    objective_config: ObjectiveConfig,
) -> Tuple[Dict[str, Any], List[Optional[Assignment]]]:
    if not (0 <= step < len(task_ids)):
        raise StateReplayError(
            f"state step {step} outside [0, {len(task_ids)})"
        )
    if len(prefix_actions) != step:
        raise StateReplayError(
            f"prefix length {len(prefix_actions)} does not equal step {step}"
        )
    task_id = task_ids[step]
    candidates, candidate_stats = _enumerate_feasible_action_set(
        problem,
        schedule,
        task_id,
        constraint_config,
        enumerator_config,
    )
    candidate_keys = _ordered_candidate_keys(problem, task_id, candidates)
    ordered_hash, membership_hash = _candidate_hashes(candidate_keys)
    current_schedule_hash = schedule_hash(problem, schedule)
    physical_state_hash = sha256_json(
        {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "scenario_hash": scenario_hash,
            "constraint_hash": constraint_config.hash,
            "task_order_hash": task_order_hash,
            "step": int(step),
            "schedule_hash": current_schedule_hash,
        }
    )
    state_hash = sha256_json(
        {
            "physical_state_hash": physical_state_hash,
            "objective_hash": objective_config.hash,
            "enumerator_hash": enumerator_config.hash,
        }
    )
    objective_score = objective_config.build_model(problem).score(schedule)
    state = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "step": int(step),
        "task_id": task_id,
        "prefix_length": int(step),
        "prefix_hash": sha256_json(list(prefix_actions)),
        "schedule_hash": current_schedule_hash,
        "physical_state_hash": physical_state_hash,
        "state_hash": state_hash,
        "ordered_candidate_hash": ordered_hash,
        "candidate_membership_hash": membership_hash,
        "candidate_count": len(candidates),
        "candidate_set_stats": candidate_stats,
        "objective_score_hex": float(objective_score).hex(),
    }
    return state, candidates


def build_trace_manifests(
    problem: SchedulingProblem,
    scenario_path: str | Path,
    schedule_path: str | Path,
    *,
    constraint_config: ConstraintConfig,
    enumerator_config: EnumeratorConfig,
    objective_config: ObjectiveConfig,
    code_provenance: Optional[Mapping[str, Any]] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """构建一份轨迹清单及所有动作执行前的状态清单。"""
    scenario_path = Path(scenario_path)
    schedule_path = Path(schedule_path)
    scenario_digest = sha256_file(scenario_path)
    schedule_digest = sha256_file(schedule_path)
    with schedule_path.open("r", encoding="utf-8") as handle:
        source = json.load(handle)

    source_scenario_id = source.get("scenario_id")
    if (
        source_scenario_id is not None
        and str(source_scenario_id) != str(problem.scenario_id)
    ):
        raise StateReplayError(
            "schedule scenario_id does not match the loaded problem: "
            f"schedule={source_scenario_id!r}, problem={problem.scenario_id!r}"
        )

    raw_assignments = source.get("assignments") or []
    if not isinstance(raw_assignments, list):
        raise StateReplayError("schedule assignments must be a list")
    raw_by_task: Dict[str, Mapping[str, Any]] = {}
    for raw in raw_assignments:
        if not isinstance(raw, dict):
            raise StateReplayError("each schedule assignment must be an object")
        task_id = str(raw.get("task_id"))
        if task_id in raw_by_task:
            raise StateReplayError(f"duplicate schedule assignment for task {task_id}")
        if task_id not in problem.tasks:
            raise StateReplayError(f"schedule contains unknown task {task_id}")
        raw_by_task[task_id] = raw

    task_ids = canonical_task_ids(problem)
    task_order_hash = sha256_json(list(task_ids))
    schedule = Schedule()
    observed_actions: List[Dict[str, Any]] = []
    states: List[Dict[str, Any]] = []

    for step, task_id in enumerate(task_ids):
        state, candidates = _state_snapshot(
            problem,
            schedule,
            step=step,
            task_ids=task_ids,
            prefix_actions=observed_actions,
            scenario_hash=scenario_digest,
            task_order_hash=task_order_hash,
            constraint_config=constraint_config,
            enumerator_config=enumerator_config,
            objective_config=objective_config,
        )
        raw = raw_by_task.get(task_id)
        chosen: Optional[Assignment]
        if raw is None:
            chosen = None
        else:
            chosen = _match_raw_assignment(raw, candidates)
        action_key = candidate_action_key(problem, task_id, chosen)
        state["observed_action_key"] = action_key
        states.append(state)
        observed_actions.append(action_key)
        if chosen is not None:
            # 追加重建后的候选对象，不使用根据调度 JSON 手工拼出的有损对象。
            schedule.assignments.append(deepcopy(chosen))

    if len(schedule.assignments) != len(raw_assignments):
        raise StateReplayError(
            "reconstructed assignment count does not match source schedule"
        )

    normalized_code_provenance = json.loads(
        canonical_json_bytes(
            dict(code_provenance)
            if code_provenance is not None
            else {"commit_id": "unknown"}
        ).decode("utf-8")
    )
    provenance = {
        "scenario": {
            "path": scenario_path.as_posix(),
            "sha256": scenario_digest,
            "scenario_id": problem.scenario_id,
            "semantic_instance_id": scenario_path.stem,
        },
        "source_schedule": {
            "path": schedule_path.as_posix(),
            "sha256": schedule_digest,
        },
        "code": normalized_code_provenance,
    }
    trace_identity = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "scenario_hash": scenario_digest,
        "source_schedule_hash": schedule_digest,
        "constraint_hash": constraint_config.hash,
        "enumerator_hash": enumerator_config.hash,
        "objective_hash": objective_config.hash,
        "task_order_hash": task_order_hash,
    }
    trace = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "trace_id": sha256_json(trace_identity),
        **provenance,
        "task_ids": list(task_ids),
        "task_order_hash": task_order_hash,
        "constraint_config": constraint_config.to_payload(),
        "constraint_hash": constraint_config.hash,
        "enumerator_config": enumerator_config.to_payload(),
        "enumerator_hash": enumerator_config.hash,
        "objective_config": objective_config.to_payload(),
        "objective_hash": objective_config.hash,
        "observed_action_keys": observed_actions,
        "final_schedule_hash": schedule_hash(problem, schedule),
    }
    trace["trace_hash"] = sha256_json(trace)

    final_states: List[Dict[str, Any]] = []
    for state in states:
        record = {
            **state,
            "trace_id": trace["trace_id"],
            "trace_hash": trace["trace_hash"],
        }
        record["state_manifest_hash"] = sha256_json(record)
        final_states.append(record)
    return trace, final_states


def _candidate_for_action_key(
    problem: SchedulingProblem,
    task_id: str,
    candidates: Sequence[Optional[Assignment]],
    action_key: Mapping[str, Any],
) -> Optional[Assignment]:
    index = _action_key_index(problem, task_id, candidates)
    encoded = canonical_json_bytes(dict(action_key))
    if encoded not in index:
        raise StateReplayError(
            f"action key for task {task_id} is absent from reconstructed candidates"
        )
    return index[encoded]


def _compare_state_manifest(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    fields = (
        "schema_version",
        "step",
        "task_id",
        "prefix_length",
        "prefix_hash",
        "schedule_hash",
        "physical_state_hash",
        "state_hash",
        "ordered_candidate_hash",
        "candidate_membership_hash",
        "candidate_count",
        "candidate_set_stats",
        "objective_score_hex",
        "observed_action_key",
        "trace_id",
        "trace_hash",
    )
    differences = [
        field
        for field in fields
        if actual.get(field) != expected.get(field)
    ]
    if differences:
        details = ", ".join(
            f"{field}: expected={expected.get(field)!r}, actual={actual.get(field)!r}"
            for field in differences
        )
        raise StateReplayError(f"state manifest mismatch ({details})")


def restore_state(
    problem: SchedulingProblem,
    trace_manifest: Mapping[str, Any],
    state_manifest: Mapping[str, Any],
    *,
    scenario_path: Optional[str | Path] = None,
    verify: bool = True,
) -> ReplayedState:
    """重建部分调度并验证所有状态指纹。"""
    trace = dict(trace_manifest)
    state = dict(state_manifest)
    if verify:
        _verify_payload_hash(trace, "trace_hash", label="trace manifest")
        _verify_payload_hash(state, "state_manifest_hash", label="state manifest")
    if trace.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise StateReplayError(
            f"unsupported trace schema: {trace.get('schema_version')!r}"
        )
    if state.get("trace_id") != trace.get("trace_id"):
        raise StateReplayError("state references a different trace_id")
    if state.get("trace_hash") != trace.get("trace_hash"):
        raise StateReplayError("state references a different trace_hash")

    scenario_digest = str(trace["scenario"]["sha256"])
    if scenario_path is not None:
        actual_digest = sha256_file(scenario_path)
        if actual_digest != scenario_digest:
            raise StateReplayError(
                f"scenario hash mismatch: manifest={scenario_digest}, file={actual_digest}"
            )

    constraint_config = ConstraintConfig.from_payload(trace["constraint_config"])
    enumerator_config = EnumeratorConfig.from_payload(trace["enumerator_config"])
    objective_config = ObjectiveConfig.from_payload(trace["objective_config"])
    if constraint_config.hash != trace.get("constraint_hash"):
        raise StateReplayError("constraint configuration hash mismatch")
    if enumerator_config.hash != trace.get("enumerator_hash"):
        raise StateReplayError("enumerator configuration hash mismatch")
    if objective_config.hash != trace.get("objective_hash"):
        raise StateReplayError("objective configuration hash mismatch")

    task_ids = canonical_task_ids(problem)
    if list(task_ids) != trace.get("task_ids"):
        raise StateReplayError("canonical task order differs from trace manifest")
    task_order_hash = sha256_json(list(task_ids))
    if task_order_hash != trace.get("task_order_hash"):
        raise StateReplayError("task order hash mismatch")

    step = int(state["step"])
    actions = trace.get("observed_action_keys")
    if not isinstance(actions, list) or len(actions) != len(task_ids):
        raise StateReplayError("trace observed_action_keys length mismatch")
    schedule = Schedule()
    for index in range(step):
        task_id = task_ids[index]
        candidates = enumerate_feasible_actions(
            problem,
            schedule,
            task_id,
            constraint_config,
            enumerator_config,
        )
        chosen = _candidate_for_action_key(
            problem,
            task_id,
            candidates,
            actions[index],
        )
        if chosen is not None:
            model = constraint_config.build_model(problem)
            if not model.is_feasible_assignment(chosen, schedule):
                raise StateReplayError(
                    f"reconstructed action for task {task_id} is infeasible"
                )
            schedule.assignments.append(deepcopy(chosen))

    actual, candidates = _state_snapshot(
        problem,
        schedule,
        step=step,
        task_ids=task_ids,
        prefix_actions=actions[:step],
        scenario_hash=scenario_digest,
        task_order_hash=task_order_hash,
        constraint_config=constraint_config,
        enumerator_config=enumerator_config,
        objective_config=objective_config,
    )
    actual["observed_action_key"] = actions[step]
    actual["trace_id"] = trace["trace_id"]
    actual["trace_hash"] = trace["trace_hash"]
    actual["state_manifest_hash"] = sha256_json(actual)
    if verify:
        _compare_state_manifest(actual, state)

    score = objective_config.build_model(problem).score(schedule)
    return ReplayedState(
        schedule=deepcopy(schedule),
        task_id=task_ids[step],
        candidates=tuple(deepcopy(candidates)),
        state_manifest=actual,
        objective_score=float(score),
    )


def load_and_restore_state(
    scenario_path: str | Path,
    trace_manifest: Mapping[str, Any],
    state_manifest: Mapping[str, Any],
    *,
    verify: bool = True,
) -> ReplayedState:
    problem = load_scheduling_problem_from_json(scenario_path)
    return restore_state(
        problem,
        trace_manifest,
        state_manifest,
        scenario_path=scenario_path,
        verify=verify,
    )
