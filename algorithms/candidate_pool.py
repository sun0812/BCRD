# algorithms/candidate_pool.py
# -*- coding: utf-8 -*-

"""
Main functionality:
This module generates diversified candidate assignments for individual scheduling
tasks, including observation placement and optional downlink placement. It also
provides helper functions for interval extraction and conflict detection between
assignments.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional
import hashlib
import json
import random

from schedulers.scenario_loader import SchedulingProblem, SchedulingTask, CommWindow
from schedulers.constraint_model import Assignment, TimePlacementStrategy

from algorithms.random_utils import make_rng


_SUPPORTED_ORDERING_VERSIONS = {"legacy_v1", "canonical_v1"}


def _stable_order_atom(value: object) -> Tuple[str, str]:
    """返回可比较且跨进程稳定的标量表示。"""
    if value is None:
        return ("none", "")
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        return (type(value).__qualname__, str(isoformat()))
    return (type(value).__qualname__, repr(value))


def _stable_payload_digest(value: object) -> str:
    """哈希类 JSON 姿态数据，避免在排序键中保留大对象。"""
    if value is None:
        return ""
    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=repr,
        )
    except (TypeError, ValueError):
        payload = repr(value)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _observation_window_order_key(window: object) -> tuple:
    """按所有可区分观测窗口的字段生成全序键。"""
    return (
        _stable_order_atom(getattr(window, "start_time", None)),
        _stable_order_atom(getattr(window, "end_time", None)),
        _stable_order_atom(getattr(window, "satellite_id", None)),
        _stable_order_atom(getattr(window, "mission_id", None)),
        _stable_order_atom(getattr(window, "sensor_id", None)),
        _stable_order_atom(getattr(window, "orbit_number", None)),
        _stable_order_atom(getattr(window, "window_id", None)),
        _stable_order_atom(getattr(window, "time_step", None)),
        _stable_payload_digest(getattr(window, "agile_data", None)),
        _stable_payload_digest(getattr(window, "non_agile_data", None)),
    )


def _comm_window_order_key(window: CommWindow) -> tuple:
    """按通信窗口的全部字段生成全序键。"""
    return (
        _stable_order_atom(window.start_time),
        _stable_order_atom(window.end_time),
        _stable_order_atom(window.satellite_id),
        _stable_order_atom(window.ground_station_id),
        _stable_order_atom(window.window_id),
    )


def _estimate_data_volume_GB(problem: SchedulingProblem, w, sat_start, sat_end) -> float:
    """Estimate the data volume consumption of this candidate in GB.
    Convert using data_rate_Mbps * duration_s."""
    try:
        dur = (sat_end - sat_start).total_seconds()
    except Exception:
        dur = getattr(problem.tasks.get(w.mission_id, None), "required_duration", 0.0)
    sat = problem.satellites.get(w.satellite_id)
    if sat is None:
        return 0.0
    sensor_id = getattr(w, "sensor_id", "") or ""
    spec = sat.sensors.get(sensor_id)
    if spec is None:
        return 0.0
    mbits = float(spec.data_rate_Mbps) * float(dur)
    # Mb -> GB (1 GB = 1024 MB, 1 MB = 8 Mb)
    return mbits / (8.0 * 1024.0)


def _estimate_power_cost_W(problem: SchedulingProblem, w) -> float:
    """Estimate the energy or power consumption of this candidate.
    It is counted per task, and the unit follows the W field."""
    sat = problem.satellites.get(w.satellite_id)
    if sat is None:
        return 0.0
    sensor_id = getattr(w, "sensor_id", "") or ""
    spec = sat.sensors.get(sensor_id)
    if spec is None:
        return 0.0
    return float(spec.power_consumption_W)


def _overlap(a_start, a_end, b_start, b_end) -> bool:
    """Check whether two time intervals overlap."""
    return not (a_end <= b_start or a_start >= b_end)


def _place_latest(window_start, window_end, required_duration):
    """Try to place the task at the latest feasible time within the window."""
    total = (window_end - window_start).total_seconds()
    if required_duration > total:
        return None
    end = window_end
    start = end - (window_end - window_start).__class__(seconds=required_duration)  # timedelta
    return start, end


def _place_random(window_start, window_end, required_duration, rng: random.Random):
    """Try to place the task randomly within the window."""
    total = (window_end - window_start).total_seconds()
    if required_duration > total:
        return None
    slack = total - required_duration
    offset = rng.random() * slack
    start = window_start + (window_end - window_start).__class__(seconds=offset)
    end = start + (window_end - window_start).__class__(seconds=required_duration)
    return start, end


def enumerate_task_candidates(
    problem: SchedulingProblem,
    task: SchedulingTask,
    placement_mode: str = "earliest",
    downlink_duration_ratio: float = 1.0,
    max_candidates: int = 256,
    random_samples_per_window: int = 1,
    seed: Optional[int] = None,
    prefer_must_first: bool = True,
    ordering_version: str = "legacy_v1",
) -> List[Assignment]:
    """
    Generate candidate Assignments for a single task.
    Conflicts with other tasks are ignored, and only window validity is enforced.

    This version provides diversified candidates:
    earliest, center, latest, plus several random samples.

    ``legacy_v1`` 保留历史上依赖随机种子的排序方式。``canonical_v1``
    使用完整且确定性的窗口平局判定规则；敏捷卫星先按最早、居中、最晚位置
    排序，再按偏移量升序排列。当 ``random_samples_per_window=0`` 时，动作顺序
    与 ``seed`` 和输入窗口顺序无关；启用随机采样后，同一种子仍可复现。
    """
    ordering_version = str(ordering_version)
    if ordering_version not in _SUPPORTED_ORDERING_VERSIONS:
        supported = ", ".join(sorted(_SUPPORTED_ORDERING_VERSIONS))
        raise ValueError(
            f"Unsupported ordering_version={ordering_version!r}; expected one of: {supported}"
        )

    if task.required_duration <= 0 or not task.windows:
        return []

    rng = make_rng(seed)

    if ordering_version == "canonical_v1":
        obs_windows = sorted(task.windows, key=_observation_window_order_key)
    else:
        obs_windows = sorted(task.windows, key=lambda w: w.start_time)
    has_gs = len(problem.ground_stations) > 0

    comm_by_sat: Dict[str, List[CommWindow]] = {}
    if has_gs:
        for cw in problem.comm_windows:
            comm_by_sat.setdefault(cw.satellite_id, []).append(cw)
        for sid in comm_by_sat:
            if ordering_version == "canonical_v1":
                comm_by_sat[sid].sort(key=_comm_window_order_key)
            else:
                comm_by_sat[sid].sort(key=lambda x: x.start_time)

    cands: List[Assignment] = []
    downlink_dur = task.required_duration * float(downlink_duration_ratio)

    def push_candidate(w, sat_start, sat_end, sat_angles=None, cw=None, gs_start=None, gs_end=None):
        """Push the generated candidate configuration into the list."""
        cands.append(
            Assignment(
                task_id=task.id,
                satellite_id=w.satellite_id,
                sat_start_time=sat_start,
                sat_end_time=sat_end,
                sat_window_id=w.window_id,
                sensor_id=getattr(w, 'sensor_id', ''),
                orbit_number=int(getattr(w, 'orbit_number', 0) or 0),
                data_volume_GB=_estimate_data_volume_GB(problem, w, sat_start, sat_end),
                power_cost_W=_estimate_power_cost_W(problem, w),
                sat_angles=sat_angles,
                ground_station_id=(None if cw is None else cw.ground_station_id),
                gs_start_time=gs_start,
                gs_end_time=gs_end,
                gs_window_id=(None if cw is None else cw.window_id),
            )
        )

    # Multiple placements within the observation window
    for w in obs_windows:
        if len(cands) >= max_candidates:
            break

        placements: List[tuple] = []

        sat = problem.satellites.get(w.satellite_id)
        sat_type = getattr(sat, "maneuverability_type", "agile") if sat is not None else "agile"

        total = (w.end_time - w.start_time).total_seconds()
        dur = task.required_duration
        step = float(getattr(w, "time_step", 1.0) or 1.0)

        if dur <= 0 or dur > total:
            continue

        max_offset = total - dur
        import math
        max_k = int(math.floor((max_offset + 1e-9) / step))
        if max_k < 0:
            continue

        def _slice_angles_payload(payload, start_idx: int, count: int):
            """Slice angle data by time steps.

            - list: slice directly
            - dict: recursively slice lists within it
              common case: {roll:[...], pitch:[...]}
            - otherwise: return as is
            """
            if payload is None:
                return None
            if isinstance(payload, list):
                return payload[start_idx: start_idx + count]
            if isinstance(payload, dict):
                out = {}
                for kk, vv in payload.items():
                    if isinstance(vv, (list, dict)):
                        out[kk] = _slice_angles_payload(vv, start_idx, count)
                    else:
                        out[kk] = vv
                return out
            return payload

        def _angles_for(k: int):
            # Angle data:
            # Agile -> take the slice corresponding to the sub-window
            # Non-agile -> use one single data set
            if sat_type == "non_agile":
                return getattr(w, "non_agile_data", None)

            ad = getattr(w, "agile_data", None)
            # Use ceil to ensure full duration coverage
            # and avoid being one step short because of round
            n = max(1, int(math.ceil((dur / step) - 1e-9)))
            return _slice_angles_payload(ad, k, n)

        from datetime import timedelta

        # Non-agile:
        # only allow the middle segment that satisfies the task duration
        # aligned with time_step
        if sat_type == "non_agile":
            center_k = int(round((max_offset / 2.0) / step))
            center_k = max(0, min(max_k, center_k))
            sat_start = w.start_time + timedelta(seconds=center_k * step)
            sat_end = sat_start + timedelta(seconds=dur)
            placements.append((sat_start, sat_end, _angles_for(center_k)))
        else:
            # Agile:
            # enumerate all feasible sub-window start positions by time_step
            # optionally shuffle the order to provide randomness
            ks = list(range(0, max_k + 1))

            # Make sure earliest, center, and latest are prioritized first
            # to increase the chance of getting a reasonable solution
            must = [0, int(round((max_offset / 2.0) / step)), max_k]
            must = [k for k in must if 0 <= k <= max_k]
            must_uniq = []
            for k in must:
                if k not in must_uniq:
                    must_uniq.append(k)

            rest = [k for k in ks if k not in must_uniq]
            if ordering_version == "canonical_v1":
                ks = must_uniq + rest
            elif prefer_must_first:
                rng.shuffle(rest)
                ks = must_uniq + rest
            else:
                ks = must_uniq + rest
                rng.shuffle(ks)

            # Generate one by one
            # the list will be truncated by max_candidates if needed
            for k in ks:
                sat_start = w.start_time + timedelta(seconds=k * step)
                sat_end = sat_start + timedelta(seconds=dur)
                placements.append((sat_start, sat_end, _angles_for(k)))

        # Deduplicate because the same window may produce repeated placements
        uniq = []
        seen = set()
        for s, e, ang in placements:
            key = (s, e)
            if key in seen:
                continue
            seen.add(key)
            uniq.append((s, e, ang))

        for sat_start, sat_end, sat_angles in uniq:
            if len(cands) >= max_candidates:
                break

            if not has_gs:
                push_candidate(w, sat_start, sat_end, sat_angles)
                continue

            for cw in comm_by_sat.get(w.satellite_id, []):
                if len(cands) >= max_candidates:
                    break

                earliest = max(cw.start_time, sat_end)
                avail = (cw.end_time - earliest).total_seconds()
                if avail < downlink_dur:
                    continue

                # Diverse downlink placements
                dl_places = []
                p = TimePlacementStrategy.place(earliest, cw.end_time, downlink_dur, mode="earliest")
                if p:
                    dl_places.append(p)
                p = TimePlacementStrategy.place(earliest, cw.end_time, downlink_dur, mode="center")
                if p:
                    dl_places.append(p)

                total_dl = (cw.end_time - earliest).total_seconds()
                if downlink_dur <= total_dl:
                    gs_end = cw.end_time
                    gs_start = gs_end - timedelta(seconds=downlink_dur)
                    if gs_start >= earliest:
                        dl_places.append((gs_start, gs_end))

                for _ in range(max(0, int(random_samples_per_window))):
                    slack_dl = total_dl - downlink_dur
                    if slack_dl <= 1e-9:
                        break
                    offset = rng.random() * slack_dl
                    gs_start = earliest + timedelta(seconds=offset)
                    gs_end = gs_start + timedelta(seconds=downlink_dur)
                    dl_places.append((gs_start, gs_end))

                # Deduplicate
                dl_seen = set()
                for gs_start, gs_end in dl_places:
                    key = (gs_start, gs_end)
                    if key in dl_seen:
                        continue
                    dl_seen.add(key)
                    push_candidate(w, sat_start, sat_end, sat_angles, cw=cw, gs_start=gs_start, gs_end=gs_end)

                    if len(cands) >= max_candidates:
                        break

    return cands


def assignment_sat_intervals(a: Assignment) -> List[Tuple[str, object, object]]:
    """Extract all satellite intervals occupied by an Assignment,
    including observation and communication."""
    res = [(a.satellite_id, a.sat_start_time, a.sat_end_time)]
    if a.gs_start_time is not None and a.gs_end_time is not None:
        res.append((a.satellite_id, a.gs_start_time, a.gs_end_time))
    return res


def assignments_conflict(a: Assignment, b: Assignment) -> bool:
    """Check whether two Assignments have resource conflicts."""
    if a.satellite_id == b.satellite_id:
        for _, s1, e1 in assignment_sat_intervals(a):
            for _, s2, e2 in assignment_sat_intervals(b):
                if _overlap(s1, e1, s2, e2):
                    return True

    if a.ground_station_id and b.ground_station_id and a.ground_station_id == b.ground_station_id:
        if a.gs_start_time and a.gs_end_time and b.gs_start_time and b.gs_end_time:
            if _overlap(a.gs_start_time, a.gs_end_time, b.gs_start_time, b.gs_end_time):
                return True

    return False
