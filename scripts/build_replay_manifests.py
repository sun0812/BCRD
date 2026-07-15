#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""根据场景与最终调度构建紧凑的精确重放旁路清单。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from schedulers.scenario_loader import load_scheduling_problem_from_json
from schedulers.state_replay import (
    ConstraintConfig,
    EnumeratorConfig,
    ObjectiveConfig,
    StateReplayError,
    build_trace_manifests,
    canonical_json_bytes,
    sha256_file,
)
from scripts.batch_export_trajectories import (
    _require_objective_weights,
    parse_schedule_filename,
)


REPLAY_IMPLEMENTATION_FILES = (
    "algorithms/candidate_pool.py",
    "algorithms/objectives.py",
    "schedulers/constraint_model.py",
    "schedulers/scenario_loader.py",
    "schedulers/state_replay.py",
)


def collect_code_provenance(repo_root: Path = REPO_ROOT) -> Dict[str, Any]:
    """记录 Git 版本及重放关键源文件的哈希。"""
    override = os.environ.get("EOSBENCH_CODE_COMMIT_ID", "").strip()
    commit_id = override or "unknown"
    if not override:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode == 0 and completed.stdout.strip():
            commit_id = completed.stdout.strip()

    status = subprocess.run(
        ["git", "status", "--porcelain", "--", *REPLAY_IMPLEMENTATION_FILES],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    implementation_files = {
        relative_path: sha256_file(repo_root / relative_path)
        for relative_path in REPLAY_IMPLEMENTATION_FILES
        if (repo_root / relative_path).is_file()
    }
    return {
        "commit_id": commit_id,
        "implementation_dirty": bool(status.stdout.strip())
        if status.returncode == 0
        else None,
        "implementation_files": implementation_files,
    }


def objective_config_from_schedule_path(schedule_path: Path) -> ObjectiveConfig:
    parsed = parse_schedule_filename(schedule_path.stem)
    weights = _require_objective_weights(parsed, schedule_path.name).normalized()
    return ObjectiveConfig(
        (
            weights.w_profit,
            weights.w_completion,
            weights.w_timeliness,
            weights.w_balance,
        )
    )


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
    temporary.replace(path)


def write_replay_manifests(
    out_dir: Path,
    schedule_stem: str,
    trace: Dict[str, Any],
    states: List[Dict[str, Any]],
) -> Tuple[Path, Path]:
    trace_path = out_dir / f"{schedule_stem}.trace.json"
    states_path = out_dir / f"{schedule_stem}.states.jsonl"
    trace_bytes = json.dumps(
        trace,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    state_lines = b"".join(
        canonical_json_bytes(state) + b"\n"
        for state in states
    )
    _atomic_write_bytes(trace_path, trace_bytes)
    _atomic_write_bytes(states_path, state_lines)
    return trace_path, states_path


def build_for_paths(
    scenario_path: Path,
    schedule_path: Path,
    out_dir: Path,
    *,
    max_candidates: int,
    placement_mode: str,
    downlink_duration_ratio: float,
    agility_profile: str,
    non_agile_transition_s: float,
) -> Tuple[Path, Path, Dict[str, Any], List[Dict[str, Any]]]:
    problem = load_scheduling_problem_from_json(scenario_path)
    constraint_config = ConstraintConfig(
        placement_mode=placement_mode,
        downlink_duration_ratio=downlink_duration_ratio,
        agility_profile=agility_profile,
        non_agile_transition_s=non_agile_transition_s,
    )
    enumerator_config = EnumeratorConfig(
        max_candidates=max_candidates,
        random_samples_per_window=0,
        ordering_version="canonical_v1",
        seed=0,
    )
    objective_config = objective_config_from_schedule_path(schedule_path)
    trace, states = build_trace_manifests(
        problem,
        scenario_path,
        schedule_path,
        constraint_config=constraint_config,
        enumerator_config=enumerator_config,
        objective_config=objective_config,
        code_provenance=collect_code_provenance(),
    )
    trace_path, states_path = write_replay_manifests(
        out_dir,
        schedule_path.stem,
        trace,
        states,
    )
    return trace_path, states_path, trace, states


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build one compact trace manifest plus per-state fingerprints. "
            "The sidecars reference the original files and do not duplicate raw JSONL."
        )
    )
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-candidates", type=int, default=8192)
    parser.add_argument("--placement-mode", default="earliest")
    parser.add_argument("--downlink-ratio", type=float, default=1.0)
    parser.add_argument("--agility-profile", default="Standard-Agility")
    parser.add_argument("--non-agile-transition-s", type=float, default=10.0)
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    if args.max_candidates <= 0:
        parser.error("--max-candidates must be positive")
    if not args.scenario.is_file():
        parser.error(f"scenario not found: {args.scenario}")
    if not args.schedule.is_file():
        parser.error(f"schedule not found: {args.schedule}")

    try:
        trace_path, states_path, trace, states = build_for_paths(
            args.scenario,
            args.schedule,
            args.out_dir,
            max_candidates=args.max_candidates,
            placement_mode=args.placement_mode,
            downlink_duration_ratio=args.downlink_ratio,
            agility_profile=args.agility_profile,
            non_agile_transition_s=args.non_agile_transition_s,
        )
    except (StateReplayError, ValueError, KeyError) as exc:
        print(f"[fatal] exact replay manifest build failed: {exc}", file=sys.stderr)
        return 1

    cap_hits = sum(
        bool(state["candidate_set_stats"]["cap_reached"])
        for state in states
    )
    print(f"[ok] trace_id={trace['trace_id']}")
    print(f"[ok] trace_hash={trace['trace_hash']}")
    print(f"[ok] states={len(states)}, candidate_cap_hits={cap_hits}")
    print(f"[ok] trace -> {trace_path}")
    print(f"[ok] states -> {states_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
