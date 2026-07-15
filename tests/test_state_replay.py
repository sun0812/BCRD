from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta
import json
from pathlib import Path
import tempfile
import unittest

from schedulers.scenario_loader import load_scheduling_problem_from_json
from schedulers.state_replay import (
    ConstraintConfig,
    EnumeratorConfig,
    ObjectiveConfig,
    StateReplayError,
    build_trace_manifests,
    canonical_json_bytes,
    restore_state,
    sha256_json,
)


class ExactStateReplayTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.scenario_path = root / "Scenario_exact_replay.json"
        self.schedule_path = root / "scheduler_exact_replay_c2_profit_first_implicit.json"

        start = datetime(2026, 1, 1)
        missions = []
        observation_windows = []
        assignments = []
        for index, (task_id, priority, offset) in enumerate(
            (
                ("TASK-1", 3.0, 10),
                ("TASK-2", 2.0, 40),
                ("TASK-3", 1.0, 70),
            ),
            start=1,
        ):
            missions.append(
                {
                    "id": task_id,
                    "priority": priority,
                    "observation_requirement": {"duration_s": 5.0},
                }
            )
            window_start = start + timedelta(seconds=offset)
            observation_windows.append(
                {
                    "satellite_id": "SAT-1",
                    "mission_id": task_id,
                    "sensor_id": "SENSOR-1",
                    "time_windows": [
                        {
                            "start_time": window_start.isoformat(),
                            "end_time": (window_start + timedelta(seconds=20)).isoformat(),
                            "orbit_number": 1,
                            "time_step": 5.0,
                        }
                    ],
                }
            )
            if task_id != "TASK-2":
                assignments.append(
                    {
                        "task_id": task_id,
                        "satellite_id": "SAT-1",
                        "sat_start_time": window_start.isoformat(),
                        "sat_end_time": (window_start + timedelta(seconds=5)).isoformat(),
                        "sensor_id": "SENSOR-1",
                        "orbit_number": 1,
                    }
                )

        scenario = {
            "scenario_id": "exact-replay-test",
            "metadata": {
                "creation_time": start.isoformat(),
                "duration": 3600,
                "time_step": 5.0,
            },
            "satellites": [
                {
                    "id": "SAT-1",
                    "maneuverability_type": "agile",
                    "satellite_specs": {
                        "max_data_storage_GB": 0.0,
                        "max_power_W": 0.0,
                    },
                    "observation_capability": {
                        "sensors": [
                            {
                                "sensor_id": "SENSOR-1",
                                "data_rate_Mbps": 1.0,
                                "power_consumption_W": 2.0,
                            }
                        ]
                    },
                }
            ],
            "ground_stations": [],
            "missions": missions,
            "observation_windows": observation_windows,
            "communication_windows": [],
        }
        schedule = {
            "scenario_id": "exact-replay-test",
            "start_time": start.isoformat(),
            "end_time": (start + timedelta(hours=1)).isoformat(),
            "assignments": assignments,
            "unassigned_tasks": ["TASK-2"],
        }
        self.scenario_path.write_text(
            json.dumps(scenario, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.schedule_path.write_text(
            json.dumps(schedule, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        self.problem = load_scheduling_problem_from_json(self.scenario_path)
        self.constraint_config = ConstraintConfig()
        self.enumerator_config = EnumeratorConfig(max_candidates=64)
        self.objective_config = ObjectiveConfig((1.0, 0.0, 0.0, 0.0))
        self.trace, self.states = build_trace_manifests(
            self.problem,
            self.scenario_path,
            self.schedule_path,
            constraint_config=self.constraint_config,
            enumerator_config=self.enumerator_config,
            objective_config=self.objective_config,
            code_provenance={
                "commit_id": "test-commit",
                "implementation_dirty": False,
            },
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_canonical_json_ignores_dictionary_insertion_order(self) -> None:
        left = {"b": 2, "a": {"y": 4, "x": 3}}
        right = {"a": {"x": 3, "y": 4}, "b": 2}
        self.assertEqual(canonical_json_bytes(left), canonical_json_bytes(right))
        self.assertEqual(sha256_json(left), sha256_json(right))

    def test_trace_reconstructs_assign_skip_assign_prefixes(self) -> None:
        self.assertEqual(len(self.states), 3)
        self.assertEqual(self.trace["code"]["commit_id"], "test-commit")
        self.assertEqual(
            [key["kind"] for key in self.trace["observed_action_keys"]],
            ["assign", "skip", "assign"],
        )

        restored_0 = restore_state(
            self.problem,
            self.trace,
            self.states[0],
            scenario_path=self.scenario_path,
        )
        restored_1 = restore_state(
            self.problem,
            self.trace,
            self.states[1],
            scenario_path=self.scenario_path,
        )
        restored_2 = restore_state(
            self.problem,
            self.trace,
            self.states[2],
            scenario_path=self.scenario_path,
        )

        self.assertEqual(restored_0.schedule.assigned_task_ids, [])
        self.assertEqual(restored_1.schedule.assigned_task_ids, ["TASK-1"])
        self.assertEqual(restored_2.schedule.assigned_task_ids, ["TASK-1"])
        self.assertEqual(restored_2.task_id, "TASK-3")
        self.assertEqual(
            restored_2.state_manifest["ordered_candidate_hash"],
            self.states[2]["ordered_candidate_hash"],
        )

    def test_skip_keeps_schedule_hash_but_advances_state_identity(self) -> None:
        # 第 1 步执行前与第 2 步执行前的物理调度相同，因为中间动作是 SKIP；
        # 决策游标不同，因此二者必须拥有不同的状态身份。
        self.assertEqual(
            self.states[1]["schedule_hash"],
            self.states[2]["schedule_hash"],
        )
        self.assertNotEqual(
            self.states[1]["physical_state_hash"],
            self.states[2]["physical_state_hash"],
        )
        self.assertNotEqual(
            self.states[1]["state_hash"],
            self.states[2]["state_hash"],
        )

    def test_restore_returns_defensive_copies(self) -> None:
        first = restore_state(
            self.problem,
            self.trace,
            self.states[2],
            scenario_path=self.scenario_path,
        )
        first.schedule.assignments.clear()
        second = restore_state(
            self.problem,
            self.trace,
            self.states[2],
            scenario_path=self.scenario_path,
        )
        self.assertEqual(second.schedule.assigned_task_ids, ["TASK-1"])

    def test_tampered_state_fails_even_with_rehashed_manifest(self) -> None:
        tampered = deepcopy(self.states[1])
        tampered["ordered_candidate_hash"] = "sha256:" + ("0" * 64)
        tampered["state_manifest_hash"] = sha256_json(
            {
                key: value
                for key, value in tampered.items()
                if key != "state_manifest_hash"
            }
        )
        with self.assertRaisesRegex(StateReplayError, "state manifest mismatch"):
            restore_state(
                self.problem,
                self.trace,
                tampered,
                scenario_path=self.scenario_path,
            )

    def test_tampered_scenario_file_fails_before_replay(self) -> None:
        original = self.scenario_path.read_text(encoding="utf-8")
        self.scenario_path.write_text(original + "\n", encoding="utf-8")
        with self.assertRaisesRegex(StateReplayError, "scenario hash mismatch"):
            restore_state(
                self.problem,
                self.trace,
                self.states[0],
                scenario_path=self.scenario_path,
            )


if __name__ == "__main__":
    unittest.main()
