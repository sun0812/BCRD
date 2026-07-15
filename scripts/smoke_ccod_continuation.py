#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在真实状态上跨进程验证 CCOD 强制动作与 H 步 continuation。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, Iterable, List, Mapping, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from algorithms.ccod.cache import CounterfactualLabelCache, build_cache_identity
from algorithms.ccod.continuation import (
    ContinuationOracle,
    ContinuationConfig,
    CounterfactualError,
    continuation_implementation_hash,
    force_action,
)
from schedulers.scenario_loader import load_scheduling_problem_from_json
from schedulers.state_replay import (
    ConstraintConfig,
    EnumeratorConfig,
    ObjectiveConfig,
    StateReplayError,
    candidate_action_key,
    canonical_json_bytes,
    replayed_state_runtime_fingerprint,
    restore_state,
    schedule_hash,
)
from scripts.build_replay_manifests import build_for_paths


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise CounterfactualError(
                    f"{path}:{line_number} 必须包含一个 JSON object"
                )
            rows.append(value)
    return rows


def _parse_steps(text: str) -> List[int]:
    result: List[int] = []
    for token in text.split(","):
        token = token.strip()
        if token:
            value = int(token)
            if value not in result:
                result.append(value)
    return result


def _evenly_spaced_steps(num_states: int, count: int) -> List[int]:
    if num_states <= 0:
        return []
    count = max(1, min(int(count), num_states))
    if count == 1:
        return [0]
    return sorted(
        {
            int(round(index * (num_states - 1) / float(count - 1)))
            for index in range(count)
        }
    )


def _parse_action_modes(text: str) -> Tuple[str, ...]:
    allowed = {"observed", "skip"}
    modes: List[str] = []
    for token in text.split(","):
        mode = token.strip().lower()
        if not mode:
            continue
        if mode not in allowed:
            raise ValueError(f"不支持的动作模式: {mode!r}")
        if mode not in modes:
            modes.append(mode)
    if not modes:
        raise ValueError("至少需要一个动作模式")
    return tuple(modes)


def _peak_rss_mib() -> float:
    """把当前 worker 的 ru_maxrss 统一换算为 MiB。"""
    raw = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return raw / (1024.0 * 1024.0)
    return raw / 1024.0


def _trace_configs(
    trace: Mapping[str, Any],
) -> Tuple[ConstraintConfig, EnumeratorConfig, ObjectiveConfig]:
    return (
        ConstraintConfig.from_payload(trace["constraint_config"]),
        EnumeratorConfig.from_payload(trace["enumerator_config"]),
        ObjectiveConfig.from_payload(trace["objective_config"]),
    )


def _action_specs(
    problem,
    state,
    state_manifest: Mapping[str, Any],
    modes: Iterable[str],
) -> List[Tuple[List[str], Any]]:
    grouped: Dict[bytes, Tuple[List[str], Any]] = {}
    for mode in modes:
        action = state_manifest["observed_action_key"] if mode == "observed" else None
        key = (
            dict(action)
            if isinstance(action, Mapping)
            else candidate_action_key(problem, state.task_id, None)
        )
        encoded = canonical_json_bytes(key)
        if encoded in grouped:
            grouped[encoded][0].append(mode)
        else:
            grouped[encoded] = ([mode], action)
    return list(grouped.values())


def _worker(
    scenario_path: Path,
    trace_path: Path,
    states_path: Path,
    cache_dir: Path,
    steps: Iterable[int],
    modes: Tuple[str, ...],
    horizon: int,
) -> int:
    problem = load_scheduling_problem_from_json(scenario_path)
    with trace_path.open("r", encoding="utf-8") as handle:
        trace = json.load(handle)
    states = {int(row["step"]): row for row in _read_jsonl(states_path)}
    constraint_config, enumerator_config, objective_config = _trace_configs(trace)
    continuation_config = ContinuationConfig(horizon=horizon)
    oracle = ContinuationOracle(
        problem,
        constraint_config=constraint_config,
        enumerator_config=enumerator_config,
        objective_config=objective_config,
    )
    cache = CounterfactualLabelCache(cache_dir)

    summaries: List[Dict[str, Any]] = []
    query_timings: List[float] = []
    prepare_timings: List[float] = []
    restore_timings: List[float] = []
    cache_timings: List[float] = []
    force_timings: List[float] = []
    cache_hits_before_store = 0
    for step in steps:
        if step not in states:
            raise CounterfactualError(f"状态清单中不存在 step={step}")
        state_manifest = states[step]
        restore_started = time.perf_counter()
        replayed = restore_state(
            problem,
            trace,
            state_manifest,
            scenario_path=scenario_path,
            verify=True,
        )
        restore_timings.append(time.perf_counter() - restore_started)
        prepare_started = time.perf_counter()
        prepared = oracle.prepare(replayed)
        prepare_timings.append(time.perf_counter() - prepare_started)
        for action_modes, action in _action_specs(
            problem,
            replayed,
            state_manifest,
            modes,
        ):
            observed_prefix_verified = None
            if "observed" in action_modes:
                force_started = time.perf_counter()
                forced_schedule = force_action(
                    problem,
                    replayed.schedule,
                    replayed.task_id,
                    action,
                    constraint_config=constraint_config,
                    enumerator_config=enumerator_config,
                )
                force_timings.append(time.perf_counter() - force_started)
                if step + 1 in states:
                    observed_prefix_verified = (
                        schedule_hash(problem, forced_schedule)
                        == states[step + 1]["schedule_hash"]
                    )
                    if not observed_prefix_verified:
                        raise CounterfactualError(
                            f"step={step} 的 observed action 未复现下一前缀"
                        )

            started = time.perf_counter()
            result = oracle.evaluate(
                prepared,
                action,
                continuation_config=continuation_config,
            )
            query_timings.append(time.perf_counter() - started)
            current_runtime_fingerprint = replayed_state_runtime_fingerprint(
                replayed.schedule,
                replayed.task_id,
                replayed.candidates,
                replayed.state_manifest,
                replayed.replay_identity,
                replayed.problem_runtime_fingerprint,
            )
            if current_runtime_fingerprint != replayed.runtime_fingerprint:
                raise CounterfactualError(
                    f"step={step} 的查询修改了输入回放状态"
                )
            identity = build_cache_identity(
                state_hash=result.state_hash,
                action_key=result.forced_action_key,
                constraint_config=constraint_config,
                enumerator_config=enumerator_config,
                objective_config=objective_config,
                continuation_config=continuation_config,
            )
            cache_started = time.perf_counter()
            if cache.load(identity) is not None:
                cache_hits_before_store += 1
            cache_path = cache.store(identity, result)
            cached = cache.load(identity)
            result_manifest = result.to_manifest()
            cache_timings.append(time.perf_counter() - cache_started)
            if cached != result_manifest:
                raise CounterfactualError("缓存往返校验失败")
            summaries.append(
                {
                    "step": int(step),
                    "task_id": replayed.task_id,
                    "modes": action_modes,
                    "query_key": result.query_key,
                    "result_hash": result_manifest["result_hash"],
                    "forced_action_kind": result.forced_action_key["kind"],
                    "decisions_executed": result.decisions_executed,
                    "terminated_by_task_exhaustion": (
                        result.terminated_by_task_exhaustion
                    ),
                    "q_h_hex": result.q_h.hex(),
                    "final_schedule_hash": result.final_schedule_hash,
                    "final_schedule_runtime_hash": (
                        result.final_schedule_runtime_hash
                    ),
                    "candidate_count": int(state_manifest["candidate_count"]),
                    "cap_reached": bool(
                        state_manifest["candidate_set_stats"]["cap_reached"]
                    ),
                    "observed_prefix_verified": observed_prefix_verified,
                    "cache_file": cache_path.name,
                }
            )
    print(
        json.dumps(
            {
                "summaries": summaries,
                "timings": {
                    "query": query_timings,
                    "prepare": prepare_timings,
                    "restore": restore_timings,
                    "cache": cache_timings,
                    "force": force_timings,
                },
                "cache_hits_before_store": cache_hits_before_store,
                "peak_rss_mib": _peak_rss_mib(),
                "continuation_implementation_hash": (
                    continuation_implementation_hash()
                ),
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "构建 exact replay sidecars，并在不同 PYTHONHASHSEED 的新进程中 "
            "比较 observed/SKIP 的确定性 H 步 continuation。"
        )
    )
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--schedule", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--steps", default="")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--actions", default="observed,skip")
    parser.add_argument("--max-candidates", type=int, default=8192)
    parser.add_argument("--worker-timeout-s", type=float, default=1200.0)
    parser.add_argument("--max-worker-rss-mib", type=float, default=6144.0)
    parser.add_argument("--placement-mode", default="earliest")
    parser.add_argument("--downlink-ratio", type=float, default=1.0)
    parser.add_argument("--agility-profile", default="Standard-Agility")
    parser.add_argument("--non-agile-transition-s", type=float, default=10.0)

    # 新进程使用的内部模式。
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--trace", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--states", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--cache-dir", type=Path, help=argparse.SUPPRESS)
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    try:
        modes = _parse_action_modes(args.actions)
    except ValueError as exc:
        parser.error(str(exc))

    if args.worker:
        if args.trace is None or args.states is None or args.cache_dir is None:
            parser.error("--worker 需要 --trace、--states 与 --cache-dir")
        try:
            return _worker(
                args.scenario,
                args.trace,
                args.states,
                args.cache_dir,
                _parse_steps(args.steps),
                modes,
                args.horizon,
            )
        except (CounterfactualError, StateReplayError, ValueError, KeyError) as exc:
            print(f"[fatal] continuation worker 失败: {exc}", file=sys.stderr)
            return 1

    if args.schedule is None:
        parser.error("--schedule 是必需参数")
    if args.repeat < 2:
        parser.error("--repeat 至少为 2")
    if (
        args.count <= 0
        or args.horizon <= 0
        or args.max_candidates <= 0
        or args.worker_timeout_s <= 0
        or args.max_worker_rss_mib <= 0
    ):
        parser.error("count、horizon、candidate、timeout 与 RSS 上限必须为正")
    if not args.scenario.is_file() or not args.schedule.is_file():
        parser.error("scenario 或 schedule 不存在")

    temporary: tempfile.TemporaryDirectory[str] | None = None
    if args.out_dir is None:
        temporary = tempfile.TemporaryDirectory(prefix="eosbench-ccod-smoke-")
        out_dir = Path(temporary.name)
    else:
        out_dir = args.out_dir

    try:
        trace_path, states_path, _, states = build_for_paths(
            args.scenario,
            args.schedule,
            out_dir,
            max_candidates=args.max_candidates,
            placement_mode=args.placement_mode,
            downlink_duration_ratio=args.downlink_ratio,
            agility_profile=args.agility_profile,
            non_agile_transition_s=args.non_agile_transition_s,
        )
        selected_steps = (
            _parse_steps(args.steps)
            if args.steps
            else _evenly_spaced_steps(len(states), args.count)
        )
        if not selected_steps or any(
            step < 0 or step >= len(states) for step in selected_steps
        ):
            raise CounterfactualError(f"非法状态集合: {selected_steps}")

        deterministic_results: List[List[Dict[str, Any]]] = []
        expected_implementation_hash = continuation_implementation_hash()
        timing_groups: Dict[str, List[float]] = {
            "query": [],
            "prepare": [],
            "restore": [],
            "cache": [],
            "force": [],
        }
        process_wall_times: List[float] = []
        worker_peak_rss_mib: List[float] = []
        cache_hits_before_store: List[int] = []
        cache_dir = out_dir / "label_cache"
        for repeat_index in range(args.repeat):
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--scenario",
                str(args.scenario.resolve()),
                "--trace",
                str(trace_path.resolve()),
                "--states",
                str(states_path.resolve()),
                "--cache-dir",
                str(cache_dir.resolve()),
                "--steps",
                ",".join(str(step) for step in selected_steps),
                "--actions",
                ",".join(modes),
                "--horizon",
                str(args.horizon),
            ]
            environment = dict(os.environ)
            environment["PYTHONHASHSEED"] = str(3000 + repeat_index)
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            started = time.perf_counter()
            try:
                completed = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                    env=environment,
                    cwd=out_dir.resolve(),
                    timeout=args.worker_timeout_s,
                )
            except subprocess.TimeoutExpired as exc:
                raise CounterfactualError(
                    f"新进程 {repeat_index} 超过 {args.worker_timeout_s:.1f}s"
                ) from exc
            process_wall_times.append(time.perf_counter() - started)
            if completed.returncode != 0:
                raise CounterfactualError(
                    f"新进程 {repeat_index} 失败: {completed.stderr.strip()}"
                )
            payload = json.loads(completed.stdout)
            if (
                payload["continuation_implementation_hash"]
                != expected_implementation_hash
            ):
                raise CounterfactualError(
                    f"新进程 {repeat_index} 使用了不同的实现源码"
                )
            continuation_implementation_hash.cache_clear()
            if continuation_implementation_hash() != expected_implementation_hash:
                raise CounterfactualError(
                    "smoke 运行期间关键源码发生变化，请在源码稳定后重跑"
                )
            deterministic_results.append(payload["summaries"])
            for timing_name, values in payload["timings"].items():
                timing_groups[timing_name].extend(float(value) for value in values)
            peak_rss = float(payload["peak_rss_mib"])
            worker_peak_rss_mib.append(peak_rss)
            if peak_rss > args.max_worker_rss_mib:
                raise CounterfactualError(
                    f"worker peak RSS {peak_rss:.1f} MiB 超过护栏 "
                    f"{args.max_worker_rss_mib:.1f} MiB"
                )
            cache_hits_before_store.append(
                int(payload["cache_hits_before_store"])
            )

        baseline = deterministic_results[0]
        for repeat_index, result in enumerate(
            deterministic_results[1:],
            start=1,
        ):
            if result != baseline:
                raise CounterfactualError(
                    f"新进程 {repeat_index} 的 continuation 指纹不同"
                )

        def timing_summary(values: List[float]) -> Dict[str, float]:
            if not values:
                return {"count": 0, "median_s": 0.0, "p90_s": 0.0, "max_s": 0.0}
            ordered = sorted(values)
            p90_index = max(0, int(round(0.9 * (len(ordered) - 1))))
            return {
                "count": len(ordered),
                "median_s": statistics.median(ordered),
                "p90_s": ordered[p90_index],
                "max_s": max(ordered),
            }

        timing_summaries = {
            name: timing_summary(values)
            for name, values in timing_groups.items()
        }
        query_summary = timing_summaries["query"]
        prepare_summary = timing_summaries["prepare"]
        smoke_summary = {
            "schema_version": "eosbench-ccod-smoke-v1",
            "scenario": str(args.scenario.resolve()),
            "schedule": str(args.schedule.resolve()),
            "steps": selected_steps,
            "modes": list(modes),
            "horizon": int(args.horizon),
            "repeat": int(args.repeat),
            "unique_queries": len(baseline),
            "cross_process_hash_match_rate": 1.0,
            "failures": 0,
            "continuation_implementation_hash": expected_implementation_hash,
            "timings": timing_summaries,
            "process_wall_s": process_wall_times,
            "worker_peak_rss_mib": worker_peak_rss_mib,
            "cache_hits_before_store": cache_hits_before_store,
            "queries": baseline,
        }
        summary_path = out_dir / "smoke_summary.json"
        summary_path.write_text(
            json.dumps(
                smoke_summary,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        print(
            f"[pass] {len(baseline)} 个唯一 queries 在 {args.repeat} 个新进程中一致"
        )
        print(f"[pass] steps={selected_steps}, modes={list(modes)}, H={args.horizon}")
        print(
            "[time] "
            f"median_query={query_summary['median_s']:.6f}s "
            f"p90_query={query_summary['p90_s']:.6f}s "
            f"max_query={query_summary['max_s']:.6f}s "
            f"median_prepare={prepare_summary['median_s']:.6f}s "
            f"p90_prepare={prepare_summary['p90_s']:.6f}s "
            f"process_wall={process_wall_times}"
        )
        print(
            f"[memory] worker_peak_rss_mib={worker_peak_rss_mib} "
            f"limit={args.max_worker_rss_mib:.1f} MiB"
        )
        for row in baseline:
            print(
                "[query] "
                f"step={row['step']} modes={row['modes']} "
                f"kind={row['forced_action_kind']} decisions={row['decisions_executed']} "
                f"q_h={row['q_h_hex']} result_hash={row['result_hash']}"
            )
        if args.out_dir is not None:
            print(f"[keep] artifacts -> {out_dir}; summary={summary_path}")
        return 0
    except (
        CounterfactualError,
        StateReplayError,
        ValueError,
        KeyError,
        json.JSONDecodeError,
    ) as exc:
        print(f"[fatal] CCOD continuation smoke 失败: {exc}", file=sys.stderr)
        return 1
    finally:
        if temporary is not None:
            temporary.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
