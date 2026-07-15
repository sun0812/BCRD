# -*- coding: utf-8 -*-
"""
scripts/batch_export_trajectories.py

Batch exporter for EOS-Bench constructive imitation-learning trajectories.

For every scheduler_*.json found under --schedule_dir, this script replays
the schedule against the given Scenario JSON using EOS-Bench's own
candidate_pool + constraint_model, and writes one (state, candidate set,
expert action, objective_delta, observed_return_H) sample per scheduling step.

Decision protocol (documented for downstream training):

    - Tasks are processed in one of two orders, controlled by ``--replay_order``:

        * ``priority_desc`` (default): match `schedulers/rl_env.py::RLSchedulingEnv`
          and `ConstraintModel.build_initial_schedule`. This is what the future
          BCRD inference loop uses. May produce SKIP labels when the expert's
          chosen window is occupied by a higher-priority task in the
          replayed partial schedule (typical for MIP / SA / GA / ACO).

        * ``sat_start_asc``: walk the schedule JSON's ``assignments`` sorted
          by ``sat_start_time``, then append tasks listed in
          ``unassigned_tasks``. This is faithful to MIP-style output order
          and almost guarantees expert label resolution, but the resulting
          intermediate states do not match the priority-desc inference loop.

    - At each step we call
      `algorithms.candidate_pool.enumerate_task_candidates` with a *stable*
      seed derived from (scenario_id, task_id, schedule_stem) so the
      candidate order is reproducible.

    - The action space at step t is [SKIP] + feasible_candidates, where
      "feasible" is judged by `ConstraintModel.is_feasible_assignment`
      against the schedule built so far.

    - The expert action label is found by matching
      (satellite_id, sensor_id, orbit_number, sat_start_time) of the
      schedule's assignment for this task against candidate keys.
      Tasks listed in `unassigned_tasks` get label = SKIP (index 0).
      If the expert assignment is no longer feasible in the replayed
      partial schedule (a known mismatch case for MIP / SA / GA / ACO),
      the label also collapses to SKIP and we count it in the stats.

    - Every observed action, including SKIP, uses one objective convention:
      ``objective_delta = ObjectiveModel.score(after) -
      ObjectiveModel.score(before)``.  SKIP leaves the schedule unchanged, so
      its immediate delta is naturally zero.  No procedural reward scaling or
      unassigned-task penalty is mixed into this target.

    - ``observed_return_H`` is the sum of observed ``objective_delta`` values
      over ``[t, t + H)``.  It describes the replayed trace only; it is not a
      candidate-specific counterfactual target.

Outputs:

    data/trajectories/<schedule_stem>.jsonl
    data/trajectories/all_schedules_merged.jsonl
    docs/batch_trajectory_export_report.md

Does NOT modify EOS-Bench core code, does NOT train anything.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Make sibling EOS-Bench packages importable
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from schedulers.scenario_loader import (
    SchedulingProblem,
    SchedulingTask,
    TaskWindow,
    load_scheduling_problem_from_json,
)
from schedulers.constraint_model import ConstraintModel, Schedule, Assignment
from algorithms.candidate_pool import enumerate_task_candidates
from algorithms.objectives import ObjectiveWeights, ObjectiveModel


# =============================================================================
# Filename parsing
# =============================================================================

_C12_RE = re.compile(
    r"_c(?P<cid>[12])_(?P<algo>[a-zA-Z_]+?)_(?P<obj>p[\d.]+_c[\d.]+_t[\d.]+_b[\d.]+|implicit)$"
)
_C3_RE = re.compile(
    r"_c3_(?P<algo>sa|ga|aco)_(?P<obj>p[\d.]+_c[\d.]+_t[\d.]+_b[\d.]+)$"
)
_C4_RE = re.compile(
    r"_c4_ppo_(?P<model_stem>[A-Za-z0-9._-]+?)(?:_(?P<obj>p[\d.]+_c[\d.]+_t[\d.]+_b[\d.]+))?$"
)
_OBJ_RE = re.compile(
    r"^p(?P<p>[\d.]+)_c(?P<c>[\d.]+)_t(?P<t>[\d.]+)_b(?P<b>[\d.]+)$"
)

# Class-2 heuristic schedules encode their objective in the algorithm name and
# therefore use the literal ``implicit`` objective tag in the filename.  Keep
# this mapping next to the filename parser so replay objective deltas use the intended
# metric instead of passing all-zero weights to ObjectiveWeights.normalized(),
# whose documented fallback is profit-only.
_IMPLICIT_OBJECTIVE_WEIGHTS: Dict[str, Tuple[float, float, float, float]] = {
    "profit_first": (1.0, 0.0, 0.0, 0.0),
    "completion_first": (0.0, 1.0, 0.0, 0.0),
    "timeliness_first": (0.0, 0.0, 1.0, 0.0),
    "balance_first": (0.0, 0.0, 0.0, 1.0),
}


def parse_schedule_filename(stem: str) -> Dict[str, Any]:
    """Parse ``scheduler_<scenario>_c<id>_<algo>[_<obj_tag>].json`` -> dict.

    Returns a dict with: solver_name, class_id, objective_tag, objective_weights.
    """
    info: Dict[str, Any] = {
        "solver_name": "unknown",
        "class_id": -1,
        "objective_tag": "unknown",
        "objective_weights": [0.0, 0.0, 0.0, 0.0],
        "ppo_model_stem": None,
    }

    m = _C12_RE.search(stem)
    if m:
        info["class_id"] = int(m.group("cid"))
        info["solver_name"] = m.group("algo")
        info["objective_tag"] = m.group("obj")
        info["objective_weights"] = _parse_obj_weights(
            m.group("obj"), solver_name=m.group("algo")
        )
        return info

    m = _C3_RE.search(stem)
    if m:
        info["class_id"] = 3
        info["solver_name"] = m.group("algo")
        info["objective_tag"] = m.group("obj")
        info["objective_weights"] = _parse_obj_weights(m.group("obj"))
        return info

    m = _C4_RE.search(stem)
    if m:
        info["class_id"] = 4
        info["solver_name"] = "ppo"
        info["ppo_model_stem"] = m.group("model_stem")
        obj_tag = m.group("obj")
        if obj_tag is None:
            # PPO model_stem often embeds the weights, e.g. ``ppo_model_p1_c0_t0_b0``
            sub = re.search(r"p[\d.]+_c[\d.]+_t[\d.]+_b[\d.]+", info["ppo_model_stem"] or "")
            obj_tag = sub.group(0) if sub else "implicit"
        info["objective_tag"] = obj_tag
        info["objective_weights"] = _parse_obj_weights(obj_tag)
        return info

    return info


def _parse_obj_weights(obj_tag: str, solver_name: Optional[str] = None) -> List[float]:
    if obj_tag == "implicit" or obj_tag is None:
        implicit = _IMPLICIT_OBJECTIVE_WEIGHTS.get((solver_name or "").lower())
        if implicit is not None:
            return list(implicit)
        return [0.0, 0.0, 0.0, 0.0]
    m = _OBJ_RE.match(obj_tag)
    if not m:
        return [0.0, 0.0, 0.0, 0.0]
    return [float(m.group("p")), float(m.group("c")), float(m.group("t")), float(m.group("b"))]


# =============================================================================
# Time helpers
# =============================================================================


def _parse_iso_utc_naive(s: str) -> datetime:
    """Same convention as ``schedulers/scenario_loader._parse_iso_time``."""
    s = (s or "").strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def _stable_seed(parts: List[str]) -> int:
    h = hashlib.md5("|".join(parts).encode("utf-8"), usedforsecurity=False).hexdigest()
    return int(h[:8], 16) & 0x7FFFFFFF


# =============================================================================
# Window index (for matching expert assignments to scenario windows)
# =============================================================================


@dataclass(frozen=True)
class WindowKey:
    sat_id: str
    sensor_id: str
    orbit: int


def build_window_index(problem: SchedulingProblem) -> Dict[WindowKey, List[TaskWindow]]:
    """Group all TaskWindows by ``(sat_id, sensor_id, orbit)`` for fast lookup."""
    idx: Dict[WindowKey, List[TaskWindow]] = {}
    for task in problem.tasks.values():
        for w in task.windows:
            key = WindowKey(w.satellite_id, w.sensor_id, int(w.orbit_number or 0))
            idx.setdefault(key, []).append(w)
    return idx


def match_assignment_to_window(
    asg: Dict[str, Any],
    window_index: Dict[WindowKey, List[TaskWindow]],
    task_id: str,
) -> Tuple[Optional[TaskWindow], Optional[datetime]]:
    """Find the TaskWindow that this schedule-JSON assignment came from.

    Returns ``(window, sat_start_dt)`` or ``(None, None)`` when no window
    on the same ``(sat, sensor, orbit)`` contains ``sat_start_time``.
    """
    key = WindowKey(
        asg["satellite_id"],
        asg.get("sensor_id", "") or "",
        int(asg.get("orbit_number", 0) or 0),
    )
    sat_start = _parse_iso_utc_naive(asg["sat_start_time"])
    sat_end = _parse_iso_utc_naive(asg["sat_end_time"])

    candidates = window_index.get(key, [])
    # The expert task_id must also be the window's mission_id; this filters
    # away the rare case of two missions sharing the same (sat,sensor,orbit)
    # whose windows could otherwise overlap.
    filtered = [w for w in candidates if w.mission_id == task_id]
    for w in filtered:
        # sat_start must lie inside [w.start_time, w.end_time)
        if w.start_time <= sat_start <= w.end_time and sat_end <= w.end_time + timedelta(seconds=1):
            return w, sat_start

    # Fallback: drop the mission_id filter; some schedulers may not respect it
    for w in candidates:
        if w.start_time <= sat_start <= w.end_time and sat_end <= w.end_time + timedelta(seconds=1):
            return w, sat_start
    return None, None


# =============================================================================
# Feature extraction (state + candidate features)
# =============================================================================


TRAJECTORY_SCHEMA_VERSION = "eosbench-trajectory-v2"


CAND_FEAT_NAMES = (
    "is_skip",
    "sat_start_off_s",
    "sat_end_off_s",
    "duration_s",
    "data_volume_GB",
    "power_cost_W",
    "orbit_number",
    "task_priority",
    "downlink_offset_s",  # 0 when no GS or no downlink
    "downlink_duration_s",
)


@dataclass
class StepStats:
    expert_task_found_in_candidates: bool = False
    expert_action_index: int = 0  # 0 = skip
    expert_skipped_by_json: bool = False  # task was in unassigned_tasks
    expert_infeasible_in_replay: bool = False  # enumerated but is_feasible_assignment rejected
    expert_window_not_enumerated: bool = False  # missing from scenario window index
    expert_not_in_candidate_list: bool = False  # enumerated set truncated by max_candidates
    num_candidates_incl_skip: int = 0


def _state_features(
    problem: SchedulingProblem,
    task: SchedulingTask,
    schedule: Schedule,
    task_index: int,
    total_tasks: int,
) -> List[float]:
    """RLSchedulingEnv-compatible state vector + a few extras."""
    T = (problem.end_time - problem.start_time).total_seconds() or 1.0

    # Per-satellite assignment counts
    sat_ids = list(problem.satellites.keys())
    counts = [0] * len(sat_ids)
    for i, sid in enumerate(sat_ids):
        for a in schedule.assignments:
            if a.satellite_id == sid:
                counts[i] += 1
    if not counts:
        counts = [0]
    n = max(total_tasks, 1)
    mean = sum(counts) / len(counts)
    var = sum((c - mean) ** 2 for c in counts) / len(counts)
    std = var ** 0.5

    feats = [
        float(task.priority) / 10.0,
        float(task.required_duration) / float(T),
        1.0 - float(task_index) / float(n),
        mean / float(n),
        std / float(n),
    ]

    if len(problem.ground_stations) > 0:
        gs_ids = list(problem.ground_stations.keys())
        gs_counts = [0] * len(gs_ids)
        for i, gid in enumerate(gs_ids):
            for a in schedule.assignments:
                if a.ground_station_id == gid:
                    gs_counts[i] += 1
        if not gs_counts:
            gs_counts = [0]
        gs_mean = sum(gs_counts) / len(gs_counts)
        gs_var = sum((c - gs_mean) ** 2 for c in gs_counts) / len(gs_counts)
        feats.append(gs_mean / float(n))
        feats.append((gs_var ** 0.5) / float(n))

    return feats


def _candidate_features(
    cand: Optional[Assignment],
    task: SchedulingTask,
    problem: SchedulingProblem,
) -> List[float]:
    """Single feature row aligned with ``CAND_FEAT_NAMES``."""
    ref = problem.start_time
    if cand is None:
        return [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float(task.priority), 0.0, 0.0]

    sat_start_off = (cand.sat_start_time - ref).total_seconds()
    sat_end_off = (cand.sat_end_time - ref).total_seconds()
    dur = sat_end_off - sat_start_off
    dl_off = 0.0
    dl_dur = 0.0
    if cand.gs_start_time is not None and cand.gs_end_time is not None:
        dl_off = (cand.gs_start_time - ref).total_seconds()
        dl_dur = (cand.gs_end_time - cand.gs_start_time).total_seconds()
    return [
        0.0,  # is_skip
        float(sat_start_off),
        float(sat_end_off),
        float(dur),
        float(cand.data_volume_GB or 0.0),
        float(cand.power_cost_W or 0.0),
        float(cand.orbit_number or 0),
        float(task.priority),
        float(dl_off),
        float(dl_dur),
    ]


def _candidate_key(cand: Optional[Assignment]) -> Dict[str, Any]:
    """Stable identifier per candidate, useful for debugging and label matching."""
    if cand is None:
        return {"is_skip": True}
    return {
        "is_skip": False,
        "satellite_id": cand.satellite_id,
        "sensor_id": cand.sensor_id or "",
        "orbit_number": int(cand.orbit_number or 0),
        "window_id": int(cand.sat_window_id or 0),
        "sat_start_time": cand.sat_start_time.isoformat(),
        "sat_end_time": cand.sat_end_time.isoformat(),
        "ground_station_id": cand.ground_station_id,
        "gs_start_time": cand.gs_start_time.isoformat() if cand.gs_start_time else None,
        "gs_end_time": cand.gs_end_time.isoformat() if cand.gs_end_time else None,
    }


# =============================================================================
# Per-schedule replay
# =============================================================================


@dataclass
class ReplayConfig:
    horizon: int = 10
    max_candidates: int = 8192
    placement_mode: str = "earliest"
    downlink_duration_ratio: float = 1.0
    agility_profile: str = "Standard-Agility"
    non_agile_transition_s: float = 10.0
    replay_order: str = "priority_desc"  # one of: priority_desc | sat_start_asc


@dataclass
class ScheduleStats:
    schedule_file: str
    solver_name: str
    class_id: int
    objective_tag: str
    num_assignments_in_json: int
    num_unassigned_in_json: int
    num_assignments_matched_to_windows: int
    num_assignments_unmatched: int
    total_steps: int
    expert_action_found_steps: int
    expert_action_skip_steps_json: int
    expert_action_skip_steps_infeasible: int
    expert_action_skip_steps_window_missing: int
    avg_num_candidates: float
    min_num_candidates: int
    max_num_candidates: int
    elapsed_s: float
    metrics: Dict[str, Any]

    @property
    def match_rate(self) -> float:
        denom = self.num_assignments_in_json or 1
        return self.num_assignments_matched_to_windows / float(denom)

    @property
    def expert_action_found_rate(self) -> float:
        # Either the assignment was found in candidate set (positive label),
        # or the expert correctly skipped (label = SKIP from JSON's
        # unassigned_tasks). Both are valid IL supervision.
        denom = self.total_steps or 1
        good = self.expert_action_found_steps + self.expert_action_skip_steps_json
        return good / float(denom)


def _require_objective_weights(parsed: Dict[str, Any], source: str) -> ObjectiveWeights:
    """Return validated objective weights or fail before replay starts.

    ``ObjectiveWeights.normalized`` deliberately falls back to profit-only for
    a zero vector.  That fallback is useful at the generic objective layer but
    is unsafe for trajectory export: an unparsed filename would silently label
    data with the wrong objective.  Export therefore requires a parsed,
    finite, non-negative, non-zero four-vector.
    """
    import math

    raw = parsed.get("objective_weights")
    if parsed.get("class_id", -1) < 0 or parsed.get("objective_tag") == "unknown":
        raise ValueError(
            f"cannot parse solver/objective metadata from schedule filename: {source}"
        )
    if not isinstance(raw, list) or len(raw) != 4:
        raise ValueError(f"invalid objective weight vector for {source}: {raw!r}")
    weights = [float(value) for value in raw]
    if any(not math.isfinite(value) or value < 0.0 for value in weights):
        raise ValueError(f"objective weights must be finite and non-negative for {source}: {raw!r}")
    if sum(weights) <= 0.0:
        raise ValueError(
            f"objective weights are all zero for {source}; refusing implicit profit fallback"
        )
    return ObjectiveWeights(*weights)


def _objective_delta(before_score: float, after_score: float) -> float:
    """Unified immediate target used by both assignment and SKIP actions."""
    return float(after_score) - float(before_score)


def _candidate_set_stats(cap: int, enumerated: int, feasible: int) -> Dict[str, Any]:
    """Describe the exporter's cap-before-feasibility candidate protocol."""
    return {
        "cap": int(cap),
        "enumerated": int(enumerated),
        "feasible": int(feasible),
        # Equality cannot prove truncation, hence the deliberately conservative
        # name ``cap_reached`` rather than ``truncated``.
        "cap_reached": bool(int(enumerated) >= int(cap)),
    }


def replay_schedule(
    problem: SchedulingProblem,
    schedule_path: Path,
    window_index: Dict[WindowKey, List[TaskWindow]],
    cfg: ReplayConfig,
    verbose: bool = True,
) -> Tuple[List[Dict[str, Any]], ScheduleStats]:
    t0 = time.time()
    with open(schedule_path, "r", encoding="utf-8") as f:
        sched_data = json.load(f)

    stem = schedule_path.stem
    parsed = parse_schedule_filename(stem)
    weights = _require_objective_weights(parsed, schedule_path.name)
    obj_model = ObjectiveModel(problem, weights)

    cm = ConstraintModel(
        problem=problem,
        placement_mode=cfg.placement_mode,
        downlink_duration_ratio=cfg.downlink_duration_ratio,
        agility_profile=cfg.agility_profile,
        non_agile_transition_s=cfg.non_agile_transition_s,
    )

    # ----- Step 1: index expert assignments and match them to scenario windows
    expert_by_task: Dict[str, Dict[str, Any]] = {}
    matched = 0
    unmatched_examples: List[Tuple[str, str]] = []
    for asg in sched_data.get("assignments", []):
        tid = asg["task_id"]
        w, sat_start = match_assignment_to_window(asg, window_index, tid)
        if w is None:
            unmatched_examples.append((tid, asg["satellite_id"]))
            expert_by_task[tid] = {"raw": asg, "window": None, "sat_start": None}
            continue
        matched += 1
        expert_by_task[tid] = {"raw": asg, "window": w, "sat_start": sat_start}

    unassigned_set = set(sched_data.get("unassigned_tasks", []))
    num_assignments_json = len(sched_data.get("assignments", []))
    num_unassigned_json = len(unassigned_set)

    # ----- Step 2: choose replay order
    if cfg.replay_order == "sat_start_asc":
        # Walk JSON assignments in chronological order, then append unassigned
        sorted_asg = sorted(
            sched_data.get("assignments", []),
            key=lambda a: (_parse_iso_utc_naive(a["sat_start_time"]), a["task_id"]),
        )
        task_order_ids: List[str] = [a["task_id"] for a in sorted_asg]
        # Append unassigned tasks in stable order
        seen = set(task_order_ids)
        for tid in sorted(problem.tasks.keys()):
            if tid not in seen and tid not in unassigned_set:
                # Tasks that are neither in assignments nor in unassigned (shouldn't happen but be safe)
                task_order_ids.append(tid)
                seen.add(tid)
        for tid in sorted(unassigned_set):
            if tid not in seen:
                task_order_ids.append(tid)
                seen.add(tid)
        tasks_sorted = [problem.tasks[tid] for tid in task_order_ids if tid in problem.tasks]
    else:
        # Default: priority-desc, matches RLSchedulingEnv
        tasks_sorted = sorted(
            problem.tasks.values(),
            key=lambda t: (-float(t.priority), t.id),
        )
    total_tasks = len(tasks_sorted)

    schedule = Schedule()
    prev_score = float(obj_model.score(schedule))

    samples: List[Dict[str, Any]] = []
    cand_counts: List[int] = []
    found_steps = 0
    skip_json_steps = 0
    skip_infeasible_steps = 0
    skip_window_missing_steps = 0

    schedule_metrics = sched_data.get("metrics", {})

    for ti, task in enumerate(tasks_sorted):
        # ----- Step 2a: enumerate candidates with a stable per-task seed
        seed = _stable_seed([problem.scenario_id, stem, task.id])
        cand_assigns = enumerate_task_candidates(
            problem=problem,
            task=task,
            placement_mode=cfg.placement_mode,
            downlink_duration_ratio=cfg.downlink_duration_ratio,
            max_candidates=cfg.max_candidates,
            random_samples_per_window=0,
            seed=seed,
        )

        feasible: List[Assignment] = [c for c in cand_assigns if cm.is_feasible_assignment(c, schedule)]
        candidate_set_stats = _candidate_set_stats(
            cap=cfg.max_candidates,
            enumerated=len(cand_assigns),
            feasible=len(feasible),
        )
        enumerated_keys = {
            (int(c.sat_window_id or 0), c.sat_start_time) for c in cand_assigns
        }
        # Action 0 = SKIP, actions 1.. = feasible_candidates
        action_set: List[Optional[Assignment]] = [None] + feasible

        # ----- Step 2b: determine expert action label
        step_stats = StepStats(num_candidates_incl_skip=len(action_set))
        expert_info = expert_by_task.get(task.id)

        if (task.id in unassigned_set) or (expert_info is None):
            step_stats.expert_skipped_by_json = True
            step_stats.expert_action_index = 0  # SKIP
            skip_json_steps += 1
        elif expert_info["window"] is None:
            # Assignment couldn't be matched to any scenario window: degenerate to SKIP
            step_stats.expert_window_not_enumerated = True
            step_stats.expert_action_index = 0
            skip_window_missing_steps += 1
        else:
            exp_w = expert_info["window"]
            exp_start = expert_info["sat_start"]
            # Find matching candidate by (window_id, sat_start_time)
            idx_found = -1
            for i, c in enumerate(feasible, start=1):
                if int(c.sat_window_id or 0) == int(exp_w.window_id) and c.sat_start_time == exp_start:
                    idx_found = i
                    break
            if idx_found >= 0:
                step_stats.expert_task_found_in_candidates = True
                step_stats.expert_action_index = idx_found
                found_steps += 1
            else:
                # Two possible reasons:
                #  (a) candidate was enumerated but is_feasible_assignment
                #      rejected it against the current partial schedule
                #      (typical for global solvers like MIP/SA/GA/ACO under
                #      priority-desc replay);
                #  (b) candidate was never enumerated, e.g. truncated by
                #      max_candidates because the task has very many
                #      windows × time_step combinations.
                if (int(exp_w.window_id), exp_start) in enumerated_keys:
                    step_stats.expert_infeasible_in_replay = True
                    skip_infeasible_steps += 1
                else:
                    step_stats.expert_not_in_candidate_list = True
                    # Re-use the same bucket name "skip_window_missing"
                    # is semantically different (no window at all). We
                    # store a distinct flag for this case.
                    skip_window_missing_steps += 1
                step_stats.expert_action_index = 0

        # ----- Step 2c: build features
        state_feats = _state_features(problem, task, schedule, ti, total_tasks)
        cand_feats = [_candidate_features(c, task, problem) for c in action_set]
        cand_keys = [_candidate_key(c) for c in action_set]
        valid_mask = [1] * len(action_set)  # everything in action_set passed feasibility

        # ----- Step 2d: greedy-replay the expert action so future steps see
        # the same state distribution as the inference loop
        chosen = action_set[step_stats.expert_action_index]
        if chosen is not None:
            # Defensive guard - should always be True since we built ``feasible`` already
            if cm.is_feasible_assignment(chosen, schedule):
                schedule.assignments.append(chosen)

        # ----- Step 2e: one objective semantics for every observed action.
        # SKIP follows exactly the same score-after minus score-before path;
        # because it does not mutate ``schedule``, its delta is naturally 0.
        new_score = float(obj_model.score(schedule))
        objective_delta = _objective_delta(prev_score, new_score)
        prev_score = new_score

        cand_counts.append(len(action_set))

        # Expert window UID (string, scenario-stable) for downstream debugging
        if expert_info is not None and expert_info["window"] is not None:
            w = expert_info["window"]
            expert_window_uid = f"{w.satellite_id}|{w.sensor_id}|{w.orbit_number}|{w.window_id}"
        else:
            expert_window_uid = None

        samples.append({
            # Optional, self-describing metadata.  It is deliberately not a
            # required field so existing JSONL exports remain readable.
            "_schema": {
                "version": TRAJECTORY_SCHEMA_VERSION,
                "state_dim": len(state_feats),
                "candidate_dim": len(CAND_FEAT_NAMES),
            },
            "scenario_id": problem.scenario_id,
            "schedule_file": str(schedule_path.name),
            "solver_name": parsed["solver_name"],
            "class_id": parsed["class_id"],
            "objective_tag": parsed["objective_tag"],
            "objective_weights": parsed["objective_weights"],
            "timestep": ti,
            "expert_task_id": task.id,
            "expert_satellite_id": (
                expert_info["raw"]["satellite_id"] if expert_info and expert_info.get("raw") else None
            ),
            "expert_window_uid": expert_window_uid,
            "state_features": state_feats,
            "candidate_features": cand_feats,
            "candidate_keys": cand_keys,
            "valid_mask": valid_mask,
            "candidate_set_stats": candidate_set_stats,
            "expert_action_index": int(step_stats.expert_action_index),
            "objective_delta": float(objective_delta),
            # Filled in a second pass once all observed deltas are known.
            "observed_return_H": None,
            "schedule_metrics": schedule_metrics,
            "_debug": {
                "expert_skipped_by_json": step_stats.expert_skipped_by_json,
                "expert_window_not_enumerated": step_stats.expert_window_not_enumerated,
                "expert_infeasible_in_replay": step_stats.expert_infeasible_in_replay,
                "expert_not_in_candidate_list": step_stats.expert_not_in_candidate_list,
                "expert_task_found_in_candidates": step_stats.expert_task_found_in_candidates,
            },
        })

    # ----- Step 3: observed_return_H = sum objective_delta over [t, t+H)
    H = int(cfg.horizon)
    objective_deltas = [s["objective_delta"] for s in samples]
    for i in range(len(samples)):
        samples[i]["observed_return_H"] = float(sum(objective_deltas[i:i + H]))

    elapsed = time.time() - t0
    if verbose:
        if unmatched_examples:
            print(
                f"[warn] {schedule_path.name}: {len(unmatched_examples)} assignments could not be matched to any scenario window; "
                f"first 3 examples: {unmatched_examples[:3]}",
                flush=True,
            )

    stats = ScheduleStats(
        schedule_file=str(schedule_path.name),
        solver_name=parsed["solver_name"],
        class_id=parsed["class_id"],
        objective_tag=parsed["objective_tag"],
        num_assignments_in_json=num_assignments_json,
        num_unassigned_in_json=num_unassigned_json,
        num_assignments_matched_to_windows=matched,
        num_assignments_unmatched=num_assignments_json - matched,
        total_steps=len(samples),
        expert_action_found_steps=found_steps,
        expert_action_skip_steps_json=skip_json_steps,
        expert_action_skip_steps_infeasible=skip_infeasible_steps,
        expert_action_skip_steps_window_missing=skip_window_missing_steps,
        avg_num_candidates=float(sum(cand_counts)) / float(len(cand_counts) or 1),
        min_num_candidates=int(min(cand_counts)) if cand_counts else 0,
        max_num_candidates=int(max(cand_counts)) if cand_counts else 0,
        elapsed_s=elapsed,
        metrics=schedule_metrics,
    )
    return samples, stats


# =============================================================================
# Batch driver
# =============================================================================


def write_jsonl(path: Path, samples: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False))
            f.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Export eosbench-trajectory-v2 constructive traces with unified "
            "ObjectiveModel deltas (SKIP delta = 0)."
        ),
    )
    parser.add_argument("--scenario", required=True, type=Path, help="Path to Scenario_*.json")
    parser.add_argument("--schedule_dir", required=True, type=Path, help="Directory containing scheduler_*.json files")
    parser.add_argument("--out_dir", required=True, type=Path, help="Output directory for jsonl + merged jsonl")
    parser.add_argument(
        "--horizon",
        type=int,
        default=10,
        help="Horizon H for observed_return_H=sum objective_delta[t:t+H] (default: 10)",
    )
    parser.add_argument(
        "--max_candidates",
        type=int,
        default=8192,
        help=(
            "Enumeration cap applied before feasibility filtering (default: 8192); "
            "each row records cap/enumerated/feasible/cap_reached."
        ),
    )
    parser.add_argument(
        "--replay_order",
        choices=["priority_desc", "sat_start_asc"],
        default="priority_desc",
        help="Task iteration order during replay (default: priority_desc; sat_start_asc improves match for global solvers like MIP/SA/GA/ACO).",
    )
    parser.add_argument(
        "--include_only",
        nargs="*",
        default=None,
        help="Optional list of substrings; only schedules whose filename contains any of them are processed.",
    )
    parser.add_argument("--report", type=Path, default=None, help="Optional path for the Markdown report")
    parser.add_argument("--limit", type=int, default=0, help="Process only the first N schedules (0 = all)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    scenario_path: Path = args.scenario
    schedule_dir: Path = args.schedule_dir
    out_dir: Path = args.out_dir
    horizon: int = int(args.horizon)
    max_candidates: int = int(args.max_candidates)
    report_path: Path = args.report or (REPO_ROOT / "docs" / "batch_trajectory_export_report.md")
    verbose: bool = not args.quiet

    if horizon <= 0:
        parser.error("--horizon must be a positive integer")
    if max_candidates <= 0:
        parser.error("--max_candidates must be a positive integer")

    if not scenario_path.exists():
        print(f"[fatal] Scenario JSON not found: {scenario_path}", file=sys.stderr)
        return 2
    if not schedule_dir.exists():
        print(f"[fatal] Schedule dir not found: {schedule_dir}", file=sys.stderr)
        return 2

    out_dir.mkdir(parents=True, exist_ok=True)

    schedule_paths = sorted(schedule_dir.glob("scheduler_*.json"))
    if args.include_only:
        schedule_paths = [p for p in schedule_paths if any(sub in p.name for sub in args.include_only)]
    if args.limit > 0:
        schedule_paths = schedule_paths[: args.limit]
    if not schedule_paths:
        print(f"[fatal] No scheduler_*.json files found under {schedule_dir}", file=sys.stderr)
        return 2

    # Validate the complete batch before opening the merged output.  A typo in
    # a late filename must not leave a seemingly valid but partial v2 export.
    for schedule_path in schedule_paths:
        try:
            _require_objective_weights(
                parse_schedule_filename(schedule_path.stem),
                schedule_path.name,
            )
        except (TypeError, ValueError) as exc:
            print(f"[fatal] {exc}", file=sys.stderr)
            return 2

    print(f"[info] Loading scenario: {scenario_path}", flush=True)
    problem = load_scheduling_problem_from_json(scenario_path)
    print(
        f"[info] Scenario {problem.scenario_id}: "
        f"satellites={len(problem.satellites)}, tasks={len(problem.tasks)}, "
        f"comm_windows={len(problem.comm_windows)}, "
        f"ground_stations={len(problem.ground_stations)}",
        flush=True,
    )

    window_index = build_window_index(problem)
    print(f"[info] Built window index: {len(window_index)} unique (sat,sensor,orbit) groups", flush=True)

    cfg = ReplayConfig(horizon=horizon, max_candidates=max_candidates, replay_order=args.replay_order)
    print(f"[info] replay_order = {cfg.replay_order}", flush=True)

    per_schedule_stats: List[ScheduleStats] = []
    merged_path = out_dir / "all_schedules_merged.jsonl"
    merged_total = 0

    with open(merged_path, "w", encoding="utf-8") as merged_f:
        for i, sched_path in enumerate(schedule_paths, start=1):
            print(f"[info] ({i}/{len(schedule_paths)}) replaying {sched_path.name} ...", flush=True)
            samples, stats = replay_schedule(problem, sched_path, window_index, cfg, verbose=verbose)

            per_schedule_path = out_dir / f"{sched_path.stem}.jsonl"
            write_jsonl(per_schedule_path, samples)

            for s in samples:
                merged_f.write(json.dumps(s, ensure_ascii=False))
                merged_f.write("\n")
            merged_total += len(samples)

            per_schedule_stats.append(stats)
            print(
                f"[info]    -> steps={stats.total_steps}, "
                f"match_rate={stats.match_rate:.3f}, "
                f"expert_found_rate={stats.expert_action_found_rate:.3f}, "
                f"avg_cands={stats.avg_num_candidates:.1f}, elapsed={stats.elapsed_s:.1f}s",
                flush=True,
            )

    # ----- Report
    write_report(
        report_path=report_path,
        per_schedule_stats=per_schedule_stats,
        merged_path=merged_path,
        merged_total=merged_total,
        cfg=cfg,
        scenario_id=problem.scenario_id,
        scenario_path=scenario_path,
        schedule_dir=schedule_dir,
        out_dir=out_dir,
    )
    print(f"[info] merged jsonl rows = {merged_total}, written to {merged_path}", flush=True)
    print(f"[info] report written to {report_path}", flush=True)

    return 0


def write_report(
    report_path: Path,
    per_schedule_stats: List[ScheduleStats],
    merged_path: Path,
    merged_total: int,
    cfg: ReplayConfig,
    scenario_id: str,
    scenario_path: Path,
    schedule_dir: Path,
    out_dir: Path,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)

    by_solver: Dict[str, int] = {}
    by_class: Dict[int, int] = {}
    by_obj: Dict[str, int] = {}
    for s in per_schedule_stats:
        by_solver[s.solver_name] = by_solver.get(s.solver_name, 0) + 1
        by_class[s.class_id] = by_class.get(s.class_id, 0) + 1
        by_obj[s.objective_tag] = by_obj.get(s.objective_tag, 0) + 1

    total_steps = sum(s.total_steps for s in per_schedule_stats) or 1
    found_steps = sum(s.expert_action_found_steps for s in per_schedule_stats)
    skip_json = sum(s.expert_action_skip_steps_json for s in per_schedule_stats)
    skip_infeas = sum(s.expert_action_skip_steps_infeasible for s in per_schedule_stats)
    skip_winmiss = sum(s.expert_action_skip_steps_window_missing for s in per_schedule_stats)

    all_cands_min = min((s.min_num_candidates for s in per_schedule_stats), default=0)
    all_cands_max = max((s.max_num_candidates for s in per_schedule_stats), default=0)
    all_cands_mean = (
        sum(s.avg_num_candidates * s.total_steps for s in per_schedule_stats) / float(total_steps)
    )

    lines: List[str] = []
    lines.append("# Batch Trajectory Export Report")
    lines.append("")
    lines.append("> Generated by `scripts/batch_export_trajectories.py`.")
    lines.append(
        f"> Replay protocol: `{cfg.replay_order}` "
        "(`priority_desc` matches the constructive inference order)."
    )
    lines.append("> Candidate seeding: `_stable_seed(scenario_id, schedule_stem, task_id)` for reproducibility.")
    lines.append(f"> Schema: `{TRAJECTORY_SCHEMA_VERSION}`; labels use unscaled `ObjectiveModel.score` deltas.")
    lines.append("")
    lines.append("## 1. Run parameters")
    lines.append("")
    lines.append("```")
    lines.append(f"scenario           : {scenario_path}")
    lines.append(f"schedule_dir       : {schedule_dir}")
    lines.append(f"out_dir            : {out_dir}")
    lines.append(f"horizon (H)        : {cfg.horizon}")
    lines.append(f"max_candidates     : {cfg.max_candidates}")
    lines.append("target             : objective_delta = F(after) - F(before)")
    lines.append("skip_delta         : 0 (schedule unchanged)")
    lines.append("return             : observed_return_H = sum objective_delta[t:t+H]")
    lines.append(f"placement_mode     : {cfg.placement_mode}")
    lines.append(f"downlink_ratio     : {cfg.downlink_duration_ratio}")
    lines.append(f"agility_profile    : {cfg.agility_profile}")
    lines.append(f"non_agile_trans_s  : {cfg.non_agile_transition_s}")
    lines.append(f"replay_order       : {cfg.replay_order}")
    lines.append("```")
    lines.append("")

    lines.append("## 2. Top-level numbers")
    lines.append("")
    lines.append(f"- `num_schedules`               = **{len(per_schedule_stats)}**")
    lines.append(f"- `scenario_id`                 = `{scenario_id}`")
    lines.append(f"- `total_trajectory_steps`      = **{total_steps}**")
    lines.append(f"- `merged_jsonl`                = `{merged_path}` (rows = {merged_total})")
    lines.append(f"- `num_candidates_per_step`     = min {all_cands_min} / mean {all_cands_mean:.1f} / max {all_cands_max}")
    lines.append("- expert label breakdown (over all steps):")
    lines.append(f"    - `found_in_candidates`     = {found_steps} ({found_steps/total_steps:.1%})")
    lines.append(f"    - `skip (json unassigned)`  = {skip_json} ({skip_json/total_steps:.1%})")
    lines.append(f"    - `skip (infeasible at replay)` = {skip_infeas} ({skip_infeas/total_steps:.1%})")
    lines.append(f"    - `skip (window missing)`   = {skip_winmiss} ({skip_winmiss/total_steps:.1%})")
    lines.append("")

    lines.append("## 3. Distributions")
    lines.append("")
    lines.append("### solver_name")
    lines.append("")
    lines.append("| solver | #schedules |")
    lines.append("|---|---|")
    for k in sorted(by_solver):
        lines.append(f"| `{k}` | {by_solver[k]} |")
    lines.append("")
    lines.append("### class_id")
    lines.append("")
    lines.append("| class_id | #schedules |")
    lines.append("|---|---|")
    for k in sorted(by_class):
        lines.append(f"| {k} | {by_class[k]} |")
    lines.append("")
    lines.append("### objective_tag")
    lines.append("")
    lines.append("| objective | #schedules |")
    lines.append("|---|---|")
    for k in sorted(by_obj):
        lines.append(f"| `{k}` | {by_obj[k]} |")
    lines.append("")

    lines.append("## 4. Per-schedule details")
    lines.append("")
    lines.append("| schedule | solver | class | obj | #assigned | #unassigned | match_rate | found_rate | steps | avg cands | TP | TCR | RT(s) | wall(s) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for s in per_schedule_stats:
        m = s.metrics or {}
        tp = m.get("TP")
        tcr = m.get("TCR")
        rt = m.get("RT")
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{s.schedule_file}`",
                    s.solver_name,
                    str(s.class_id),
                    f"`{s.objective_tag}`",
                    str(s.num_assignments_in_json),
                    str(s.num_unassigned_in_json),
                    f"{s.match_rate:.3f}",
                    f"{s.expert_action_found_rate:.3f}",
                    str(s.total_steps),
                    f"{s.avg_num_candidates:.1f}",
                    f"{tp:.0f}" if isinstance(tp, (int, float)) else "-",
                    f"{tcr:.3f}" if isinstance(tcr, (int, float)) else "-",
                    f"{rt:.1f}" if isinstance(rt, (int, float)) else "-",
                    f"{s.elapsed_s:.1f}",
                ]
            )
            + " |"
        )
    lines.append("")

    lines.append("## 5. Acceptance check")
    lines.append("")
    n_ok_match = sum(1 for s in per_schedule_stats if s.match_rate >= 0.95)
    n_ok_found = sum(1 for s in per_schedule_stats if s.expert_action_found_rate >= 0.95)
    lines.append(f"- schedules processed              : **{len(per_schedule_stats)}** (target = 15)")
    lines.append(f"- schedules with match_rate ≥ 0.95 : **{n_ok_match} / {len(per_schedule_stats)}**")
    lines.append(f"- schedules with found_rate ≥ 0.95 : **{n_ok_found} / {len(per_schedule_stats)}**")
    lines.append(f"- merged trajectory steps          : **{merged_total}** (target ≥ 5000)")
    lines.append("")
    lines.append("Notes:")
    lines.append("")
    lines.append("- `match_rate` = fraction of `assignments` whose `(sat,sensor,orbit,sat_start_time)` resolved to a unique scenario window.")
    lines.append("- `found_rate` = fraction of replay steps whose label is *not* `skip-by-infeasibility-mismatch` or `skip-by-window-missing`. Steps labeled SKIP because the JSON listed the task as unassigned are *counted as valid* (the expert correctly chose to skip).")
    lines.append("- `objective_delta` always uses the report objective `F`; no PPO reward scale or procedural unassigned penalty is included.")
    lines.append("- `observed_return_H` is an observed-trace sum, not a candidate-specific counterfactual target.")
    lines.append("- `candidate_set_stats` records the cap-before-feasibility protocol: `cap`, cap-limited `enumerated`, `feasible`, and `cap_reached`.")
    lines.append("- For solvers without a strict task-by-task semantics (SA / GA / ACO / MIP under priority-desc replay), some steps may collapse to SKIP due to ordering mismatch; this is documented in `eosbench_interface_report.md` §8.3.")
    lines.append("")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(main())
