#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在使用不同哈希种子的新进程中验证精确状态重放。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Dict, Iterable, List

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from schedulers.scenario_loader import load_scheduling_problem_from_json
from schedulers.state_replay import StateReplayError, restore_state
from scripts.build_replay_manifests import build_for_paths


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise StateReplayError(
                    f"{path}:{line_number} must contain one JSON object"
                )
            rows.append(value)
    return rows


def _parse_steps(text: str) -> List[int]:
    result: List[int] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
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


def _worker(
    scenario_path: Path,
    trace_path: Path,
    states_path: Path,
    steps: Iterable[int],
) -> int:
    problem = load_scheduling_problem_from_json(scenario_path)
    with trace_path.open("r", encoding="utf-8") as handle:
        trace = json.load(handle)
    state_by_step = {
        int(row["step"]): row
        for row in _read_jsonl(states_path)
    }
    summaries: List[Dict[str, Any]] = []
    for step in steps:
        if step not in state_by_step:
            raise StateReplayError(f"requested step {step} is absent from state manifest")
        replayed = restore_state(
            problem,
            trace,
            state_by_step[step],
            scenario_path=scenario_path,
            verify=True,
        )
        manifest = replayed.state_manifest
        summaries.append(
            {
                "step": step,
                "task_id": replayed.task_id,
                "prefix_assignments": len(replayed.schedule.assignments),
                "candidate_count": len(replayed.candidates),
                "prefix_hash": manifest["prefix_hash"],
                "schedule_hash": manifest["schedule_hash"],
                "ordered_candidate_hash": manifest["ordered_candidate_hash"],
                "candidate_membership_hash": manifest["candidate_membership_hash"],
                "physical_state_hash": manifest["physical_state_hash"],
                "state_hash": manifest["state_hash"],
                "objective_score_hex": manifest["objective_score_hex"],
            }
        )
    print(
        json.dumps(
            summaries,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build exact replay sidecars and compare selected states across "
            "fresh Python processes with different PYTHONHASHSEED values."
        )
    )
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--schedule", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--steps", default="")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--max-candidates", type=int, default=8192)
    parser.add_argument("--placement-mode", default="earliest")
    parser.add_argument("--downlink-ratio", type=float, default=1.0)
    parser.add_argument("--agility-profile", default="Standard-Agility")
    parser.add_argument("--non-agile-transition-s", type=float, default=10.0)

    # 供新进程调用的内部模式。
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--trace", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--states", type=Path, help=argparse.SUPPRESS)
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    if args.worker:
        if args.trace is None or args.states is None:
            parser.error("--worker requires --trace and --states")
        try:
            return _worker(
                args.scenario,
                args.trace,
                args.states,
                _parse_steps(args.steps),
            )
        except (StateReplayError, ValueError, KeyError) as exc:
            print(f"[fatal] worker replay failed: {exc}", file=sys.stderr)
            return 1

    if args.schedule is None:
        parser.error("--schedule is required")
    if args.repeat < 2:
        parser.error("--repeat must be at least 2")
    if args.count <= 0:
        parser.error("--count must be positive")
    if not args.scenario.is_file():
        parser.error(f"scenario not found: {args.scenario}")
    if not args.schedule.is_file():
        parser.error(f"schedule not found: {args.schedule}")

    temporary: tempfile.TemporaryDirectory[str] | None = None
    if args.out_dir is None:
        temporary = tempfile.TemporaryDirectory(prefix="eosbench-replay-")
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
        if not selected_steps:
            raise StateReplayError("no states selected")
        if any(step < 0 or step >= len(states) for step in selected_steps):
            raise StateReplayError(
                f"selected steps outside [0, {len(states)}): {selected_steps}"
            )

        results: List[List[Dict[str, Any]]] = []
        worker_cwd = out_dir.resolve()
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
                "--steps",
                ",".join(str(step) for step in selected_steps),
            ]
            environment = dict(os.environ)
            environment["PYTHONHASHSEED"] = str(1000 + repeat_index)
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                env=environment,
                # 故意从输出目录启动，以验证回放不依赖仓库当前工作目录。
                cwd=worker_cwd,
            )
            if completed.returncode != 0:
                raise StateReplayError(
                    f"fresh process {repeat_index} failed: {completed.stderr.strip()}"
                )
            results.append(json.loads(completed.stdout))

        baseline = results[0]
        for repeat_index, result in enumerate(results[1:], start=1):
            if result != baseline:
                raise StateReplayError(
                    f"fresh process {repeat_index} produced different state fingerprints"
                )

        print(
            f"[pass] {len(selected_steps)}/{len(selected_steps)} states matched "
            f"across {args.repeat} fresh processes"
        )
        print(f"[pass] steps={selected_steps}")
        for row in baseline:
            print(
                "[state] "
                f"step={row['step']} task={row['task_id']} "
                f"prefix={row['prefix_assignments']} candidates={row['candidate_count']} "
                f"state_hash={row['state_hash']}"
            )
        if args.out_dir is not None:
            print(f"[keep] trace -> {trace_path}")
            print(f"[keep] states -> {states_path}")
        return 0
    except (StateReplayError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"[fatal] exact replay smoke failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if temporary is not None:
            temporary.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
