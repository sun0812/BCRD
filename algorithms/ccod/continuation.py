# -*- coding: utf-8 -*-
"""CCOD 的确定性强制动作与有限视野 continuation。

本模块只实现论文方案中的标签 oracle，不负责候选查询分配或模型训练。
强制动作计作视野中的第一个决策；随后使用统一调度目标下的即时贪心策略，
直到总共执行 ``H`` 个决策或任务耗尽。所有平局均由稳定 ActionKey 打破。
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from functools import lru_cache
import math
import platform
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

from algorithms.objectives import ObjectiveModel
from schedulers.constraint_model import Assignment, Schedule
from schedulers.scenario_loader import SchedulingProblem
from schedulers.state_replay import (
    ConstraintConfig,
    EnumeratorConfig,
    MANIFEST_SCHEMA_VERSION,
    ObjectiveConfig,
    ReplayedState,
    assignment_runtime_payload,
    candidate_action_key,
    canonical_json_bytes,
    canonical_task_ids,
    enumerate_feasible_actions,
    problem_runtime_fingerprint,
    replayed_state_runtime_fingerprint,
    schedule_hash,
    schedule_runtime_fingerprint,
    sha256_file,
    sha256_json,
)


CONTINUATION_SCHEMA_VERSION = "eosbench-ccod-continuation-v1"
QUERY_IDENTITY_SCHEMA_VERSION = "eosbench-ccod-query-v1"
ActionLike = Union[None, Assignment, Mapping[str, Any]]


@lru_cache(maxsize=1)
def continuation_implementation_hash() -> str:
    """返回 continuation 及其完整直接/间接依赖的启动时集合哈希。

    运行期间源码必须保持只读；热更新源码后应重启进程，或在测试中显式调用
    ``continuation_implementation_hash.cache_clear()``，避免复用旧指纹。
    """
    repository_root = Path(__file__).resolve().parents[2]
    paths = (
        Path(__file__).resolve(),
        repository_root / "algorithms" / "candidate_pool.py",
        repository_root / "algorithms" / "objectives.py",
        repository_root / "algorithms" / "random_utils.py",
        repository_root / "schedulers" / "balance_utils.py",
        repository_root / "schedulers" / "constraint_model.py",
        repository_root / "schedulers" / "scenario_loader.py",
        repository_root / "schedulers" / "state_replay.py",
        repository_root / "schedulers" / "timeliness_utils.py",
        repository_root / "schedulers" / "transition_utils.py",
    )
    payload = {
        "schema_version": "eosbench-ccod-implementation-v1",
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "sources": {
            path.relative_to(repository_root).as_posix(): sha256_file(path)
            for path in paths
        },
    }
    return sha256_json(payload)


class CounterfactualError(ValueError):
    """反事实动作或 continuation 无法按规范执行时抛出的异常。"""


@dataclass(frozen=True)
class ContinuationConfig:
    """确定性 continuation 的可哈希配置。"""

    horizon: int = 5
    policy_version: str = "objective_greedy_v1"
    tie_break_version: str = "action_key_lexicographic_v1"
    forced_action_counts_toward_horizon: bool = True

    def __post_init__(self) -> None:
        if (
            isinstance(self.horizon, bool)
            or not isinstance(self.horizon, int)
            or self.horizon <= 0
        ):
            raise ValueError("horizon must be a strict positive integer")
        if not self.forced_action_counts_toward_horizon:
            raise ValueError(
                "CCOD v1 requires the forced action to count toward the horizon"
            )
        if self.policy_version != "objective_greedy_v1":
            raise ValueError(f"unsupported policy_version: {self.policy_version!r}")
        if self.tie_break_version != "action_key_lexicographic_v1":
            raise ValueError(
                f"unsupported tie_break_version: {self.tie_break_version!r}"
            )

    def to_payload(self) -> Dict[str, Any]:
        return {
            "horizon": int(self.horizon),
            "policy_version": self.policy_version,
            "tie_break_version": self.tie_break_version,
            "forced_action_counts_toward_horizon": bool(
                self.forced_action_counts_toward_horizon
            ),
        }

    @property
    def hash(self) -> str:
        return sha256_json(self.to_payload())


def build_query_identity(
    *,
    state_hash: str,
    action_key: Mapping[str, Any],
    constraint_hash: str,
    enumerator_hash: str,
    objective_hash: str,
    continuation_config: ContinuationConfig,
    implementation_hash: Optional[str] = None,
) -> Dict[str, Any]:
    """构造不含求解器来源的规范查询身份。"""
    return {
        "schema_version": QUERY_IDENTITY_SCHEMA_VERSION,
        "state_hash": str(state_hash),
        "action_key": deepcopy(dict(action_key)),
        "constraint_hash": str(constraint_hash),
        "enumerator_hash": str(enumerator_hash),
        "objective_hash": str(objective_hash),
        "horizon": int(continuation_config.horizon),
        "continuation_hash": continuation_config.hash,
        "continuation_implementation_hash": (
            str(implementation_hash)
            if implementation_hash is not None
            else continuation_implementation_hash()
        ),
    }


@dataclass
class GreedyChoice:
    """一个确定性贪心决策及其执行后的防御性调度副本。"""

    action: Optional[Assignment]
    action_key: Dict[str, Any]
    schedule: Schedule
    objective_score: float


@dataclass(frozen=True)
class CounterfactualResult:
    """一次强制动作加有限视野 continuation 的结果。"""

    final_schedule: Schedule
    query_key: str
    state_hash: str
    step: int
    task_id: str
    requested_horizon: int
    decisions_executed: int
    terminated_by_task_exhaustion: bool
    forced_action_key: Dict[str, Any]
    rollout_action_keys: Tuple[Dict[str, Any], ...]
    rollout_action_keys_hash: str
    objective_score_hexes: Tuple[str, ...]
    base_score: float
    forced_score: float
    final_score: float
    q_h: float
    final_schedule_hash: str
    final_schedule_runtime_hash: str
    continuation_hash: str
    continuation_implementation_hash: str
    constraint_hash: str
    enumerator_hash: str
    objective_hash: str
    problem_runtime_hash: str
    _task_ids: Tuple[str, ...] = field(repr=False, compare=False)
    _creation_fingerprint: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_creation_fingerprint",
            self._current_fingerprint(),
        )

    def _current_fingerprint(self) -> str:
        """绑定创建时的完整结果语义，检测浅层冻结内部的后续修改。"""
        return sha256_json(
            {
                "query_key": self.query_key,
                "state_hash": self.state_hash,
                "step": int(self.step),
                "task_id": self.task_id,
                "requested_horizon": int(self.requested_horizon),
                "decisions_executed": int(self.decisions_executed),
                "terminated_by_task_exhaustion": bool(
                    self.terminated_by_task_exhaustion
                ),
                "forced_action_key": self.forced_action_key,
                "rollout_action_keys": list(self.rollout_action_keys),
                "objective_score_hexes": list(self.objective_score_hexes),
                "base_score_hex": _stable_float(self.base_score).hex(),
                "forced_score_hex": _stable_float(self.forced_score).hex(),
                "final_score_hex": _stable_float(self.final_score).hex(),
                "q_h_hex": _stable_float(self.q_h).hex(),
                "final_schedule_hash": self.final_schedule_hash,
                "final_schedule_runtime_hash": self.final_schedule_runtime_hash,
                "current_schedule_runtime_hash": schedule_runtime_fingerprint(
                    self.final_schedule
                ),
                "continuation_hash": self.continuation_hash,
                "continuation_implementation_hash": (
                    self.continuation_implementation_hash
                ),
                "constraint_hash": self.constraint_hash,
                "enumerator_hash": self.enumerator_hash,
                "objective_hash": self.objective_hash,
                "problem_runtime_fingerprint": self.problem_runtime_hash,
                "task_ids": list(self._task_ids),
            }
        )

    def to_manifest(self) -> Dict[str, Any]:
        """生成不包含可变 Schedule 对象的规范结果清单。"""
        if self._current_fingerprint() != self._creation_fingerprint:
            raise CounterfactualError("结果对象的 creation_fingerprint 校验失败")
        if self.decisions_executed != len(self.rollout_action_keys):
            raise CounterfactualError("decisions_executed 与动作序列长度不一致")
        if not (1 <= self.decisions_executed <= self.requested_horizon):
            raise CounterfactualError("decisions_executed 超出请求视野")
        if len(self.objective_score_hexes) != self.decisions_executed + 1:
            raise CounterfactualError("目标分数序列长度与决策数不一致")
        if not self.rollout_action_keys:
            raise CounterfactualError("结果缺少强制动作")
        if self.rollout_action_keys[0] != self.forced_action_key:
            raise CounterfactualError("强制动作与 rollout 首动作不一致")
        if self.forced_action_key.get("task_id") != self.task_id:
            raise CounterfactualError("强制动作的 task_id 与结果任务不一致")
        current_rollout_hash = sha256_json(list(self.rollout_action_keys))
        if current_rollout_hash != self.rollout_action_keys_hash:
            raise CounterfactualError("rollout_action_keys_hash 与动作序列不一致")

        task_ids = self._task_ids
        stop = min(len(task_ids), self.step + self.requested_horizon)
        expected_tasks = task_ids[self.step:stop]
        if self.decisions_executed != len(expected_tasks):
            raise CounterfactualError("决策数与任务耗尽边界不一致")
        if self.terminated_by_task_exhaustion != (stop == len(task_ids)):
            raise CounterfactualError("任务耗尽标志与决策边界不一致")
        rollout_tasks = tuple(
            str(action_key.get("task_id", ""))
            for action_key in self.rollout_action_keys
        )
        if rollout_tasks != expected_tasks:
            raise CounterfactualError("rollout 动作的任务顺序不一致")

        base_score_hex = _stable_float(self.base_score).hex()
        forced_score_hex = _stable_float(self.forced_score).hex()
        final_score_hex = _stable_float(self.final_score).hex()
        if self.objective_score_hexes[0] != base_score_hex:
            raise CounterfactualError("目标分数序列的基线分数不一致")
        if self.objective_score_hexes[1] != forced_score_hex:
            raise CounterfactualError("目标分数序列的强制动作分数不一致")
        if self.objective_score_hexes[-1] != final_score_hex:
            raise CounterfactualError("目标分数序列的最终分数不一致")
        expected_q_h = _stable_float(
            math.fsum((self.final_score, -self.base_score))
        )
        if _stable_float(self.q_h).hex() != expected_q_h.hex():
            raise CounterfactualError("q_h 与最终分数减基线分数不一致")

        current_runtime_hash = schedule_runtime_fingerprint(self.final_schedule)
        if current_runtime_hash != self.final_schedule_runtime_hash:
            raise CounterfactualError(
                "final_schedule_runtime_hash 与当前结果调度不一致"
            )
        expected_config = ContinuationConfig(horizon=self.requested_horizon)
        if self.continuation_hash != expected_config.hash:
            raise CounterfactualError("continuation_hash 与请求视野不一致")
        expected_identity = build_query_identity(
            state_hash=self.state_hash,
            action_key=self.forced_action_key,
            constraint_hash=self.constraint_hash,
            enumerator_hash=self.enumerator_hash,
            objective_hash=self.objective_hash,
            continuation_config=expected_config,
            implementation_hash=self.continuation_implementation_hash,
        )
        if sha256_json(expected_identity) != self.query_key:
            raise CounterfactualError("query_key 与结果身份字段不一致")
        payload: Dict[str, Any] = {
            "schema_version": CONTINUATION_SCHEMA_VERSION,
            "query_key": self.query_key,
            "state_hash": self.state_hash,
            "step": int(self.step),
            "task_id": self.task_id,
            "requested_horizon": int(self.requested_horizon),
            "decisions_executed": int(self.decisions_executed),
            "terminated_by_task_exhaustion": bool(
                self.terminated_by_task_exhaustion
            ),
            "forced_action_key": deepcopy(self.forced_action_key),
            "rollout_action_keys": deepcopy(list(self.rollout_action_keys)),
            "rollout_action_keys_hash": self.rollout_action_keys_hash,
            "objective_score_hexes": list(self.objective_score_hexes),
            "base_score_hex": base_score_hex,
            "forced_score_hex": forced_score_hex,
            "final_score_hex": final_score_hex,
            "q_h_hex": expected_q_h.hex(),
            "final_schedule_hash": self.final_schedule_hash,
            "final_schedule_runtime_hash": self.final_schedule_runtime_hash,
            "continuation_hash": self.continuation_hash,
            "continuation_implementation_hash": (
                self.continuation_implementation_hash
            ),
            "constraint_hash": self.constraint_hash,
            "enumerator_hash": self.enumerator_hash,
            "objective_hash": self.objective_hash,
            "problem_runtime_fingerprint": self.problem_runtime_hash,
        }
        payload["result_hash"] = sha256_json(payload)
        return payload


def _stable_float(value: float) -> float:
    """拒绝非有限值，并把负零规范为正零。"""
    value = float(value)
    if not math.isfinite(value):
        raise CounterfactualError(f"objective produced a non-finite value: {value!r}")
    return 0.0 if value == 0.0 else value


def _action_key_bytes(
    problem: SchedulingProblem,
    task_id: str,
    action: Optional[Assignment],
) -> bytes:
    return canonical_json_bytes(candidate_action_key(problem, task_id, action))


def _requested_action_key_bytes(
    problem: SchedulingProblem,
    task_id: str,
    action: ActionLike,
) -> bytes:
    """把候选对象或显式 ActionKey 统一编码为比较键。"""
    try:
        if isinstance(action, Mapping):
            return canonical_json_bytes(dict(action))
        return _action_key_bytes(problem, task_id, action)
    except (KeyError, TypeError, ValueError) as exc:
        raise CounterfactualError(
            f"任务 {task_id!r} 的请求动作无法编码为规范 ActionKey"
        ) from exc


def _resolve_allowed_action(
    problem: SchedulingProblem,
    task_id: str,
    requested_action: ActionLike,
    allowed_actions: Sequence[Optional[Assignment]],
) -> Optional[Assignment]:
    """按 ActionKey 在规范候选集合中唯一解析动作。"""
    requested_key = _requested_action_key_bytes(problem, task_id, requested_action)
    matches = [
        action
        for action in allowed_actions
        if _action_key_bytes(problem, task_id, action) == requested_key
    ]
    if len(matches) != 1:
        raise CounterfactualError(
            f"requested action for task {task_id!r} matched {len(matches)} "
            "allowed actions; expected exactly one"
        )
    return deepcopy(matches[0])


def _force_resolved_action(
    problem: SchedulingProblem,
    schedule: Schedule,
    task_id: str,
    action: Optional[Assignment],
    *,
    constraint_config: ConstraintConfig,
) -> Schedule:
    """执行已经从规范候选集中唯一解析出的动作。"""
    if task_id not in problem.tasks:
        raise CounterfactualError(f"unknown task id: {task_id!r}")
    if task_id in schedule.assigned_task_ids:
        raise CounterfactualError(f"task {task_id!r} is already assigned")
    output = deepcopy(schedule)
    if action is None:
        return output
    if action.task_id != task_id:
        raise CounterfactualError(
            f"action task mismatch: expected {task_id!r}, "
            f"got {action.task_id!r}"
        )

    model = constraint_config.build_model(problem)
    if not model.is_feasible_assignment(action, output):
        raise CounterfactualError(
            f"action for task {task_id!r} became infeasible before force"
        )
    output.assignments.append(deepcopy(action))
    return output


def _schedule_after_enumerated_action(
    schedule: Schedule,
    action: Optional[Assignment],
) -> Schedule:
    """执行已由同一状态枚举器验证的动作，避免逐候选重复约束检查。"""
    assignments = list(schedule.assignments)
    if action is not None:
        assignments.append(action)
    return Schedule(assignments=assignments, metadata=deepcopy(schedule.metadata))


def force_action(
    problem: SchedulingProblem,
    schedule: Schedule,
    task_id: str,
    action: ActionLike,
    *,
    constraint_config: ConstraintConfig,
    enumerator_config: EnumeratorConfig,
) -> Schedule:
    """只从当前规范 capped 候选集解析并强制执行动作。"""
    actions = enumerate_feasible_actions(
        problem,
        schedule,
        task_id,
        constraint_config,
        enumerator_config,
    )
    canonical_action = _resolve_allowed_action(
        problem,
        task_id,
        action,
        actions,
    )
    return _force_resolved_action(
        problem,
        schedule,
        task_id,
        canonical_action,
        constraint_config=constraint_config,
    )


def _choose_objective_greedy_action(
    problem: SchedulingProblem,
    schedule: Schedule,
    task_id: str,
    *,
    constraint_config: ConstraintConfig,
    enumerator_config: EnumeratorConfig,
    objective_model: ObjectiveModel,
) -> GreedyChoice:
    actions = enumerate_feasible_actions(
        problem,
        schedule,
        task_id,
        constraint_config,
        enumerator_config,
    )
    best_action: Optional[Assignment] = None
    best_key: Optional[bytes] = None
    best_schedule: Optional[Schedule] = None
    best_score: Optional[float] = None

    for action in actions:
        # actions 已在当前 schedule 上完成 capped 枚举与可行性过滤；循环内
        # 只构造短命调度用于评分，避免重复建模和重复约束扫描。
        next_schedule = _schedule_after_enumerated_action(schedule, action)
        score = _stable_float(objective_model.score(next_schedule))
        key = _action_key_bytes(problem, task_id, action)
        if (
            best_score is None
            or score > best_score
            or (score == best_score and (best_key is None or key < best_key))
        ):
            best_action = deepcopy(action)
            best_key = key
            best_schedule = next_schedule
            best_score = score

    if best_schedule is None or best_score is None or best_key is None:
        raise CounterfactualError(f"task {task_id!r} produced no legal actions")
    return GreedyChoice(
        action=deepcopy(best_action),
        action_key=candidate_action_key(problem, task_id, best_action),
        schedule=deepcopy(best_schedule),
        objective_score=best_score,
    )


def choose_objective_greedy_action(
    problem: SchedulingProblem,
    schedule: Schedule,
    task_id: str,
    *,
    constraint_config: ConstraintConfig,
    enumerator_config: EnumeratorConfig,
    objective_config: ObjectiveConfig,
) -> GreedyChoice:
    """选择使一步后统一目标最大的动作，并用 ActionKey 稳定破平局。"""
    return _choose_objective_greedy_action(
        problem,
        schedule,
        task_id,
        constraint_config=constraint_config,
        enumerator_config=enumerator_config,
        objective_model=objective_config.build_model(problem),
    )


def _candidate_hashes(
    problem: SchedulingProblem,
    task_id: str,
    candidates: Sequence[Optional[Assignment]],
) -> Tuple[str, str, str, Tuple[Dict[str, Any], ...]]:
    """计算候选 ActionKey 哈希、运行时载荷哈希及规范键。"""
    keys = tuple(
        candidate_action_key(problem, task_id, candidate)
        for candidate in candidates
    )
    ordered_hash = sha256_json(list(keys))
    membership_hash = sha256_json(
        sorted((dict(key) for key in keys), key=canonical_json_bytes)
    )
    runtime_hash = sha256_json(
        [
            None if candidate is None else assignment_runtime_payload(candidate)
            for candidate in candidates
        ]
    )
    return ordered_hash, membership_hash, runtime_hash, keys


def _validate_replayed_state(
    problem: SchedulingProblem,
    state: ReplayedState,
    *,
    constraint_config: ConstraintConfig,
    enumerator_config: EnumeratorConfig,
    objective_config: ObjectiveConfig,
    known_problem_fingerprint: Optional[str] = None,
) -> Tuple[
    Tuple[str, ...],
    int,
    str,
    Sequence[Optional[Assignment]],
    str,
]:
    """重新验证回放对象，禁止过期状态或配置混入同一查询键。"""
    identity = state.replay_identity
    expected_hashes = {
        "constraint_hash": constraint_config.hash,
        "enumerator_hash": enumerator_config.hash,
        "objective_hash": objective_config.hash,
    }
    for field_name, expected in expected_hashes.items():
        if identity.get(field_name) != expected:
            raise CounterfactualError(f"重放状态的 {field_name} 与查询配置不一致")

    current_problem_fingerprint = (
        str(known_problem_fingerprint)
        if known_problem_fingerprint is not None
        else problem_runtime_fingerprint(problem)
    )
    if state.problem_runtime_fingerprint != current_problem_fingerprint:
        raise CounterfactualError("重放对象的 problem_runtime_fingerprint 已失配")
    if identity.get("problem_runtime_fingerprint") != current_problem_fingerprint:
        raise CounterfactualError("重放身份与当前 SchedulingProblem 不一致")

    manifest = state.state_manifest
    stored_manifest_hash = manifest.get("state_manifest_hash")
    unhashed_manifest = {
        key: value
        for key, value in manifest.items()
        if key != "state_manifest_hash"
    }
    if stored_manifest_hash != sha256_json(unhashed_manifest):
        raise CounterfactualError("重放状态的 state_manifest_hash 校验失败")

    task_ids = canonical_task_ids(problem)
    expected_task_order_hash = sha256_json(list(task_ids))
    if identity.get("task_order_hash") != expected_task_order_hash:
        raise CounterfactualError("重放状态的 task_order_hash 与场景不一致")
    try:
        step = int(manifest["step"])
        state_hash = str(manifest["state_hash"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CounterfactualError("重放状态缺少 step 或 state_hash") from exc
    if not (0 <= step < len(task_ids)):
        raise CounterfactualError(f"重放状态 step {step} 越界")
    task_id = task_ids[step]
    if state.task_id != task_id or manifest.get("task_id") != task_id:
        raise CounterfactualError("重放状态的任务游标与规范顺序不一致")

    current_schedule_hash = schedule_hash(problem, state.schedule)
    if manifest.get("schedule_hash") != current_schedule_hash:
        raise CounterfactualError("重放状态的 schedule_hash 与当前调度不一致")
    objective_model = objective_config.build_model(problem)
    current_score = _stable_float(objective_model.score(state.schedule))
    if manifest.get("objective_score_hex") != current_score.hex():
        raise CounterfactualError("重放状态的 objective_score 与当前调度不一致")
    if _stable_float(state.objective_score).hex() != current_score.hex():
        raise CounterfactualError("重放对象保存的 objective_score 已失配")

    state_ordered, state_membership, state_runtime, state_keys = _candidate_hashes(
        problem,
        task_id,
        state.candidates,
    )
    if manifest.get("candidate_count") != len(state.candidates):
        raise CounterfactualError("重放状态的 candidate_count 不一致")
    if manifest.get("ordered_candidate_hash") != state_ordered:
        raise CounterfactualError("重放状态的 ordered_candidate_hash 不一致")
    if manifest.get("candidate_membership_hash") != state_membership:
        raise CounterfactualError("重放状态的 candidate_membership_hash 不一致")

    fresh_candidates = enumerate_feasible_actions(
        problem,
        state.schedule,
        task_id,
        constraint_config,
        enumerator_config,
    )
    fresh_ordered, fresh_membership, fresh_runtime, fresh_keys = _candidate_hashes(
        problem,
        task_id,
        fresh_candidates,
    )
    if (
        fresh_ordered != state_ordered
        or fresh_membership != state_membership
        or fresh_runtime != state_runtime
        or fresh_keys != state_keys
    ):
        raise CounterfactualError("重新枚举的候选集合与回放状态不一致")

    scenario_hash = str(identity.get("scenario_hash", ""))
    if not scenario_hash:
        raise CounterfactualError("重放状态缺少 scenario_hash")
    physical_state_hash = sha256_json(
        {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "scenario_hash": scenario_hash,
            "constraint_hash": constraint_config.hash,
            "task_order_hash": expected_task_order_hash,
            "step": step,
            "schedule_hash": current_schedule_hash,
        }
    )
    if manifest.get("physical_state_hash") != physical_state_hash:
        raise CounterfactualError("重放状态的 physical_state_hash 无法重建")
    reconstructed_state_hash = sha256_json(
        {
            "physical_state_hash": physical_state_hash,
            "objective_hash": objective_config.hash,
            "enumerator_hash": enumerator_config.hash,
        }
    )
    if state_hash != reconstructed_state_hash:
        raise CounterfactualError("重放状态的 state_hash 无法重建")

    runtime_fingerprint = replayed_state_runtime_fingerprint(
        state.schedule,
        state.task_id,
        state.candidates,
        state.state_manifest,
        state.replay_identity,
        state.problem_runtime_fingerprint,
    )
    if runtime_fingerprint != state.runtime_fingerprint:
        raise CounterfactualError("重放状态的 runtime_fingerprint 校验失败")

    return (
        task_ids,
        step,
        state_hash,
        fresh_candidates,
        current_problem_fingerprint,
    )


def _prepared_snapshot_fingerprint(
    state: ReplayedState,
    current_actions: Sequence[Optional[Assignment]],
    task_ids: Sequence[str],
    step: int,
    state_hash: str,
    oracle_hash: str,
) -> str:
    """计算批量查询快照的完整创建时指纹。"""
    return sha256_json(
        {
            "oracle_hash": str(oracle_hash),
            "task_ids": [str(task_id) for task_id in task_ids],
            "step": int(step),
            "state_hash": str(state_hash),
            "state_runtime_fingerprint": replayed_state_runtime_fingerprint(
                state.schedule,
                state.task_id,
                state.candidates,
                state.state_manifest,
                state.replay_identity,
                state.problem_runtime_fingerprint,
            ),
            "current_actions": [
                None
                if action is None
                else assignment_runtime_payload(action)
                for action in current_actions
            ],
        }
    )


@dataclass(frozen=True)
class PreparedReplayedState:
    """由 ``ContinuationOracle`` 完整验证后生成的只读查询快照。"""

    state: ReplayedState = field(repr=False, compare=False)
    task_ids: Tuple[str, ...]
    step: int
    state_hash: str
    current_actions: Tuple[Optional[Assignment], ...] = field(
        repr=False,
        compare=False,
    )
    oracle_hash: str
    _creation_fingerprint: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_creation_fingerprint",
            _prepared_snapshot_fingerprint(
                self.state,
                self.current_actions,
                self.task_ids,
                self.step,
                self.state_hash,
                self.oracle_hash,
            ),
        )

    def verify(self) -> None:
        """检测快照内部的嵌套对象是否在准备后被修改。"""
        current = _prepared_snapshot_fingerprint(
            self.state,
            self.current_actions,
            self.task_ids,
            self.step,
            self.state_hash,
            self.oracle_hash,
        )
        if current != self._creation_fingerprint:
            raise CounterfactualError("PreparedReplayedState 创建后被修改")


class ContinuationOracle:
    """为同一问题上的多状态、多动作查询复用只读语义验证。

    初始化时深拷贝完整问题并计算一次问题指纹；每个回放状态只需调用一次
    ``prepare``，之后可以对该快照查询多个动作，而不会重复枚举当前候选集。
    """

    def __init__(
        self,
        problem: SchedulingProblem,
        *,
        constraint_config: ConstraintConfig,
        enumerator_config: EnumeratorConfig,
        objective_config: ObjectiveConfig,
    ) -> None:
        self._problem = deepcopy(problem)
        self.constraint_config = deepcopy(constraint_config)
        self.enumerator_config = deepcopy(enumerator_config)
        self.objective_config = deepcopy(objective_config)
        self.problem_fingerprint = problem_runtime_fingerprint(self._problem)
        self.oracle_hash = sha256_json(
            {
                "schema_version": "eosbench-ccod-oracle-v1",
                "problem_runtime_fingerprint": self.problem_fingerprint,
                "constraint_hash": self.constraint_config.hash,
                "enumerator_hash": self.enumerator_config.hash,
                "objective_hash": self.objective_config.hash,
                "continuation_implementation_hash": (
                    continuation_implementation_hash()
                ),
            }
        )

    def prepare(self, state: ReplayedState) -> PreparedReplayedState:
        """完整验证一次状态，并冻结批量动作查询所需的防御性副本。"""
        snapshot = deepcopy(state)
        (
            task_ids,
            step,
            state_hash,
            current_actions,
            problem_fingerprint,
        ) = _validate_replayed_state(
            self._problem,
            snapshot,
            constraint_config=self.constraint_config,
            enumerator_config=self.enumerator_config,
            objective_config=self.objective_config,
            known_problem_fingerprint=self.problem_fingerprint,
        )
        if problem_fingerprint != self.problem_fingerprint:
            raise CounterfactualError("批量 Oracle 的问题指纹不一致")
        return PreparedReplayedState(
            state=snapshot,
            task_ids=tuple(task_ids),
            step=step,
            state_hash=state_hash,
            current_actions=tuple(deepcopy(current_actions)),
            oracle_hash=self.oracle_hash,
        )

    def evaluate(
        self,
        prepared: PreparedReplayedState,
        forced_action: ActionLike,
        *,
        continuation_config: ContinuationConfig = ContinuationConfig(),
    ) -> CounterfactualResult:
        """在已准备快照上计算一个动作的有限视野标签。"""
        if prepared.oracle_hash != self.oracle_hash:
            raise CounterfactualError("PreparedReplayedState 属于另一个 Oracle")
        prepared.verify()
        return _evaluate_verified_state(
            self._problem,
            prepared.state,
            prepared.task_ids,
            prepared.step,
            prepared.state_hash,
            self.problem_fingerprint,
            forced_action,
            prepared.current_actions,
            constraint_config=self.constraint_config,
            enumerator_config=self.enumerator_config,
            objective_config=self.objective_config,
            continuation_config=continuation_config,
        )


def _evaluate_verified_state(
    problem: SchedulingProblem,
    state: ReplayedState,
    task_ids: Sequence[str],
    step: int,
    state_hash: str,
    problem_fingerprint: str,
    forced_action: ActionLike,
    current_actions: Sequence[Optional[Assignment]],
    *,
    constraint_config: ConstraintConfig,
    enumerator_config: EnumeratorConfig,
    objective_config: ObjectiveConfig,
    continuation_config: ContinuationConfig,
) -> CounterfactualResult:
    """在已通过完整身份验证的状态上执行有限视野查询。"""
    task_id = str(task_ids[step])
    objective_model = objective_config.build_model(problem)
    base_schedule = deepcopy(state.schedule)
    base_score = _stable_float(objective_model.score(base_schedule))
    canonical_forced = _resolve_allowed_action(
        problem,
        task_id,
        forced_action,
        current_actions,
    )
    current_schedule = _force_resolved_action(
        problem,
        base_schedule,
        task_id,
        canonical_forced,
        constraint_config=constraint_config,
    )
    forced_score = _stable_float(objective_model.score(current_schedule))
    forced_key = candidate_action_key(problem, task_id, canonical_forced)
    implementation_hash = continuation_implementation_hash()
    query_identity = build_query_identity(
        state_hash=state_hash,
        action_key=forced_key,
        constraint_hash=constraint_config.hash,
        enumerator_hash=enumerator_config.hash,
        objective_hash=objective_config.hash,
        continuation_config=continuation_config,
        implementation_hash=implementation_hash,
    )
    action_keys = [forced_key]
    score_hexes = [base_score.hex(), forced_score.hex()]

    stop = min(len(task_ids), step + continuation_config.horizon)
    for next_step in range(step + 1, stop):
        next_task_id = str(task_ids[next_step])
        choice = _choose_objective_greedy_action(
            problem,
            current_schedule,
            next_task_id,
            constraint_config=constraint_config,
            enumerator_config=enumerator_config,
            objective_model=objective_model,
        )
        current_schedule = deepcopy(choice.schedule)
        action_keys.append(deepcopy(choice.action_key))
        score_hexes.append(_stable_float(choice.objective_score).hex())

    final_score = _stable_float(objective_model.score(current_schedule))
    q_h = _stable_float(math.fsum((final_score, -base_score)))
    final_schedule = deepcopy(current_schedule)
    return CounterfactualResult(
        final_schedule=final_schedule,
        query_key=sha256_json(query_identity),
        state_hash=state_hash,
        step=step,
        task_id=task_id,
        requested_horizon=continuation_config.horizon,
        decisions_executed=len(action_keys),
        terminated_by_task_exhaustion=(stop == len(task_ids)),
        forced_action_key=deepcopy(forced_key),
        rollout_action_keys=tuple(deepcopy(action_keys)),
        rollout_action_keys_hash=sha256_json(action_keys),
        objective_score_hexes=tuple(score_hexes),
        base_score=base_score,
        forced_score=forced_score,
        final_score=final_score,
        q_h=q_h,
        final_schedule_hash=schedule_hash(problem, final_schedule),
        final_schedule_runtime_hash=schedule_runtime_fingerprint(final_schedule),
        continuation_hash=continuation_config.hash,
        continuation_implementation_hash=implementation_hash,
        constraint_hash=constraint_config.hash,
        enumerator_hash=enumerator_config.hash,
        objective_hash=objective_config.hash,
        problem_runtime_hash=str(problem_fingerprint),
        _task_ids=tuple(str(task_id) for task_id in task_ids),
    )


def evaluate_replayed_state(
    problem: SchedulingProblem,
    state: ReplayedState,
    forced_action: ActionLike,
    *,
    constraint_config: ConstraintConfig,
    enumerator_config: EnumeratorConfig,
    objective_config: ObjectiveConfig,
    continuation_config: ContinuationConfig = ContinuationConfig(),
) -> CounterfactualResult:
    """验证回放身份后计算强制动作的 $Q_H$。"""
    (
        task_ids,
        step,
        state_hash,
        current_actions,
        problem_fingerprint,
    ) = _validate_replayed_state(
        problem,
        state,
        constraint_config=constraint_config,
        enumerator_config=enumerator_config,
        objective_config=objective_config,
    )
    return _evaluate_verified_state(
        problem,
        state,
        task_ids,
        step,
        state_hash,
        problem_fingerprint,
        forced_action,
        current_actions,
        constraint_config=constraint_config,
        enumerator_config=enumerator_config,
        objective_config=objective_config,
        continuation_config=continuation_config,
    )
