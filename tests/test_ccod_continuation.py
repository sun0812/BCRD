from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta
import inspect
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from algorithms.ccod.cache import (
    CounterfactualLabelCache,
    build_cache_identity,
    cache_key,
)
from algorithms.ccod.continuation import (
    ContinuationOracle,
    ContinuationConfig,
    CounterfactualError,
    choose_objective_greedy_action,
    continuation_implementation_hash,
    evaluate_replayed_state,
    force_action,
)
from schedulers.scenario_loader import load_scheduling_problem_from_json
from schedulers.constraint_model import Schedule
from schedulers.state_replay import (
    ConstraintConfig,
    EnumeratorConfig,
    ObjectiveConfig,
    build_trace_manifests,
    candidate_action_key,
    canonical_json_bytes,
    enumerate_feasible_actions,
    restore_state,
    schedule_hash,
    sha256_json,
)


class CCODContinuationTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.scenario_path = root / "Scenario_ccod_continuation.json"
        self.schedule_path = root / "scheduler_ccod_c2_profit_first_implicit.json"
        start = datetime(2026, 1, 1)

        missions = []
        observation_windows = []
        assignments = []
        for index in range(4):
            task_id = f"TASK-{index + 1}"
            priority = float(4 - index)
            window_start = start + timedelta(seconds=10 + index * 30)
            window_duration = 30 if index == 0 else 10
            missions.append(
                {
                    "id": task_id,
                    "priority": priority,
                    "observation_requirement": {"duration_s": 5.0},
                }
            )
            observation_windows.append(
                {
                    "satellite_id": "SAT-1",
                    "mission_id": task_id,
                    "sensor_id": "SENSOR-1",
                    "time_windows": [
                        {
                            "start_time": window_start.isoformat(),
                            "end_time": (
                                window_start + timedelta(seconds=window_duration)
                            ).isoformat(),
                            "orbit_number": 1,
                            "time_step": 5.0,
                        }
                    ],
                }
            )
            if task_id in {"TASK-1", "TASK-4"}:
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
            "scenario_id": "ccod-continuation-test",
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
                                "power_consumption_W": 1.0,
                            }
                        ]
                    },
                },
                {
                    "id": "SAT-2",
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
                                "power_consumption_W": 1.0,
                            }
                        ]
                    },
                },
            ],
            "ground_stations": [],
            "missions": missions,
            "observation_windows": observation_windows,
            "communication_windows": [],
        }
        schedule = {
            "scenario_id": "ccod-continuation-test",
            "start_time": start.isoformat(),
            "end_time": (start + timedelta(hours=1)).isoformat(),
            "assignments": assignments,
            "unassigned_tasks": ["TASK-2", "TASK-3"],
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
        self.continuation_config = ContinuationConfig(horizon=5)
        self.trace, self.states = build_trace_manifests(
            self.problem,
            self.scenario_path,
            self.schedule_path,
            constraint_config=self.constraint_config,
            enumerator_config=self.enumerator_config,
            objective_config=self.objective_config,
            code_provenance={"commit_id": "continuation-test"},
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _state(self, step: int):
        return restore_state(
            self.problem,
            self.trace,
            self.states[step],
            scenario_path=self.scenario_path,
        )

    def _evaluate(self, step: int, action, horizon: int = 5):
        return evaluate_replayed_state(
            self.problem,
            self._state(step),
            action,
            constraint_config=self.constraint_config,
            enumerator_config=self.enumerator_config,
            objective_config=self.objective_config,
            continuation_config=ContinuationConfig(horizon=horizon),
        )

    def test_horizon_must_be_positive(self) -> None:
        for invalid in (0, -1, 5.7, "5", True):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "horizon"):
                    ContinuationConfig(horizon=invalid)  # type: ignore[arg-type]

    def test_enumerator_integer_fields_are_strict(self) -> None:
        cases = (
            {"max_candidates": 1.9},
            {"max_candidates": True},
            {"random_samples_per_window": 0.5},
            {"random_samples_per_window": False},
            {"seed": 1.2},
            {"seed": True},
        )
        for arguments in cases:
            with self.subTest(arguments=arguments):
                with self.assertRaisesRegex(ValueError, "strict integer"):
                    EnumeratorConfig(**arguments)  # type: ignore[arg-type]

        with self.assertRaisesRegex(ValueError, "ordering_version"):
            EnumeratorConfig(ordering_version="unknown")

    def test_constraint_numeric_fields_are_canonical_and_valid(self) -> None:
        canonical = ConstraintConfig(
            downlink_duration_ratio=1,
            non_agile_transition_s=10,
        )
        floating = ConstraintConfig(
            downlink_duration_ratio=1.0,
            non_agile_transition_s=10.0,
        )
        self.assertEqual(canonical, floating)
        self.assertEqual(canonical.hash, floating.hash)

        for arguments in (
            {"downlink_duration_ratio": "1.0"},
            {"downlink_duration_ratio": 0.0},
            {"non_agile_transition_s": float("inf")},
            {"placement_mode": 1},
            {"agility_profile": ""},
        ):
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    ConstraintConfig(**arguments)  # type: ignore[arg-type]

    def test_force_skip_returns_an_unchanged_defensive_copy(self) -> None:
        state = self._state(0)
        forced = force_action(
            self.problem,
            state.schedule,
            state.task_id,
            None,
            constraint_config=self.constraint_config,
            enumerator_config=self.enumerator_config,
        )
        self.assertEqual(len(forced.assignments), 0)
        forced.assignments.append(state.candidates[1])
        self.assertEqual(len(state.schedule.assignments), 0)

    def test_force_observed_action_matches_next_manifest_schedule(self) -> None:
        state = self._state(0)
        forced = force_action(
            self.problem,
            state.schedule,
            state.task_id,
            self.states[0]["observed_action_key"],
            constraint_config=self.constraint_config,
            enumerator_config=self.enumerator_config,
        )
        self.assertEqual(
            schedule_hash(self.problem, forced),
            self.states[1]["schedule_hash"],
        )
        self.assertEqual(len(state.schedule.assignments), 0)

    def test_force_rejects_action_outside_current_candidate_set(self) -> None:
        state = self._state(0)
        foreign = dict(self.states[0]["observed_action_key"])
        foreign["task_id"] = "TASK-2"
        with self.assertRaises(CounterfactualError):
            force_action(
                self.problem,
                state.schedule,
                state.task_id,
                foreign,
                constraint_config=self.constraint_config,
                enumerator_config=self.enumerator_config,
            )

    def test_public_force_action_cannot_bypass_candidate_cap(self) -> None:
        self.assertNotIn("allowed_actions", inspect.signature(force_action).parameters)
        state = self._state(0)
        low_cap = EnumeratorConfig(max_candidates=1)
        low_actions = enumerate_feasible_actions(
            self.problem,
            state.schedule,
            state.task_id,
            self.constraint_config,
            low_cap,
        )
        low_keys = {
            canonical_json_bytes(
                candidate_action_key(self.problem, state.task_id, action)
            )
            for action in low_actions
        }
        outside = next(
            action
            for action in state.candidates
            if canonical_json_bytes(
                candidate_action_key(self.problem, state.task_id, action)
            )
            not in low_keys
        )
        with self.assertRaises(CounterfactualError):
            force_action(
                self.problem,
                state.schedule,
                state.task_id,
                outside,
                constraint_config=self.constraint_config,
                enumerator_config=low_cap,
            )

    def test_horizon_one_counts_only_the_forced_action(self) -> None:
        result = self._evaluate(
            0,
            self.states[0]["observed_action_key"],
            horizon=1,
        )
        self.assertEqual(result.decisions_executed, 1)
        self.assertEqual(len(result.final_schedule.assignments), 1)
        self.assertAlmostEqual(result.q_h, 0.4)

    def test_skip_horizon_one_has_zero_return(self) -> None:
        result = self._evaluate(0, None, horizon=1)
        self.assertEqual(result.decisions_executed, 1)
        self.assertEqual(result.q_h, 0.0)
        self.assertEqual(result.forced_action_key["kind"], "skip")

    def test_negative_q_is_preserved_in_result_and_manifest(self) -> None:
        balance_objective = ObjectiveConfig((0.0, 0.0, 0.0, 1.0))
        trace, states = build_trace_manifests(
            self.problem,
            self.scenario_path,
            self.schedule_path,
            constraint_config=self.constraint_config,
            enumerator_config=self.enumerator_config,
            objective_config=balance_objective,
            code_provenance={"commit_id": "negative-q-test"},
        )
        state = restore_state(
            self.problem,
            trace,
            states[0],
            scenario_path=self.scenario_path,
        )
        result = evaluate_replayed_state(
            self.problem,
            state,
            states[0]["observed_action_key"],
            constraint_config=self.constraint_config,
            enumerator_config=self.enumerator_config,
            objective_config=balance_objective,
            continuation_config=ContinuationConfig(horizon=1),
        )
        self.assertLess(result.q_h, 0.0)
        self.assertEqual(result.q_h, result.final_score - result.base_score)
        self.assertEqual(
            float.fromhex(result.to_manifest()["q_h_hex"]),
            result.q_h,
        )

    def test_exact_tie_uses_minimum_stable_action_key(self) -> None:
        state = self._state(0)
        assignment_keys = [
            candidate_action_key(self.problem, state.task_id, action)
            for action in state.candidates
            if action is not None
        ]
        self.assertGreater(len(assignment_keys), 1)
        expected = min(assignment_keys, key=canonical_json_bytes)
        first = choose_objective_greedy_action(
            self.problem,
            state.schedule,
            state.task_id,
            constraint_config=self.constraint_config,
            enumerator_config=self.enumerator_config,
            objective_config=self.objective_config,
        )
        with patch(
            "algorithms.ccod.continuation.enumerate_feasible_actions",
            return_value=list(reversed(state.candidates)),
        ):
            reversed_choice = choose_objective_greedy_action(
                self.problem,
                state.schedule,
                state.task_id,
                constraint_config=self.constraint_config,
                enumerator_config=self.enumerator_config,
                objective_config=self.objective_config,
            )
        self.assertEqual(first.action_key, expected)
        self.assertEqual(reversed_choice.action_key, expected)

    def test_cap_before_feasibility_can_leave_only_skip(self) -> None:
        state = self._state(0)
        first_assignment = next(
            action for action in state.candidates if action is not None
        )
        blocker = deepcopy(first_assignment)
        blocker.task_id = "BLOCKER"
        blocked_schedule = Schedule(assignments=[blocker])
        low_cap = EnumeratorConfig(max_candidates=1)
        actions = enumerate_feasible_actions(
            self.problem,
            blocked_schedule,
            state.task_id,
            self.constraint_config,
            low_cap,
        )
        self.assertEqual(actions, [None])
        choice = choose_objective_greedy_action(
            self.problem,
            blocked_schedule,
            state.task_id,
            constraint_config=self.constraint_config,
            enumerator_config=low_cap,
            objective_config=self.objective_config,
        )
        self.assertIsNone(choice.action)
        self.assertEqual(choice.action_key["kind"], "skip")

        high_cap_actions = enumerate_feasible_actions(
            self.problem,
            blocked_schedule,
            state.task_id,
            self.constraint_config,
            self.enumerator_config,
        )
        outside = next(action for action in high_cap_actions if action is not None)
        with self.assertRaises(CounterfactualError):
            force_action(
                self.problem,
                blocked_schedule,
                state.task_id,
                outside,
                constraint_config=self.constraint_config,
                enumerator_config=low_cap,
            )

    def test_horizon_stops_when_tasks_are_exhausted(self) -> None:
        result = self._evaluate(0, None, horizon=10)
        self.assertEqual(result.decisions_executed, 4)
        self.assertTrue(result.terminated_by_task_exhaustion)
        self.assertEqual(len(result.final_schedule.assignments), 3)
        self.assertAlmostEqual(result.q_h, 0.6)

    def test_last_task_executes_only_one_decision(self) -> None:
        result = self._evaluate(
            3,
            self.states[3]["observed_action_key"],
            horizon=5,
        )
        self.assertEqual(result.decisions_executed, 1)
        self.assertTrue(result.terminated_by_task_exhaustion)

    def test_continuation_is_deterministic_and_does_not_mutate_state(self) -> None:
        state = self._state(0)
        state.schedule.metadata["nested"] = {"values": [1]}
        before = deepcopy(state)
        first = evaluate_replayed_state(
            self.problem,
            state,
            self.states[0]["observed_action_key"],
            constraint_config=self.constraint_config,
            enumerator_config=self.enumerator_config,
            objective_config=self.objective_config,
            continuation_config=self.continuation_config,
        )
        second = evaluate_replayed_state(
            self.problem,
            state,
            self.states[0]["observed_action_key"],
            constraint_config=self.constraint_config,
            enumerator_config=self.enumerator_config,
            objective_config=self.objective_config,
            continuation_config=self.continuation_config,
        )
        self.assertEqual(first.query_key, second.query_key)
        self.assertEqual(first.to_manifest(), second.to_manifest())
        self.assertEqual(before, state)

        manifest = first.to_manifest()
        manifest["rollout_action_keys"][0]["task_id"] = "tampered"
        self.assertEqual(first.rollout_action_keys[0]["task_id"], "TASK-1")
        first.final_schedule.metadata["nested"]["values"].append(2)
        self.assertEqual(state.schedule.metadata["nested"]["values"], [1])

    def test_prepared_oracle_matches_strict_api_and_reuses_validation(self) -> None:
        state = self._state(0)
        action = self.states[0]["observed_action_key"]
        strict = evaluate_replayed_state(
            self.problem,
            state,
            action,
            constraint_config=self.constraint_config,
            enumerator_config=self.enumerator_config,
            objective_config=self.objective_config,
            continuation_config=self.continuation_config,
        )
        oracle = ContinuationOracle(
            self.problem,
            constraint_config=self.constraint_config,
            enumerator_config=self.enumerator_config,
            objective_config=self.objective_config,
        )
        with patch(
            "algorithms.ccod.continuation.problem_runtime_fingerprint",
            side_effect=AssertionError("批量查询不应重复计算完整问题指纹"),
        ):
            prepared = oracle.prepare(state)
            batched = oracle.evaluate(
                prepared,
                action,
                continuation_config=self.continuation_config,
            )
            skipped = oracle.evaluate(
                prepared,
                None,
                continuation_config=self.continuation_config,
            )
            batched_manifest = batched.to_manifest()
            skipped.to_manifest()
        self.assertEqual(strict.to_manifest(), batched_manifest)

    def test_prepared_oracle_owns_a_defensive_state_snapshot(self) -> None:
        state = self._state(0)
        oracle = ContinuationOracle(
            self.problem,
            constraint_config=self.constraint_config,
            enumerator_config=self.enumerator_config,
            objective_config=self.objective_config,
        )
        prepared = oracle.prepare(state)
        state.candidates[1].power_cost_W += 100.0
        result = oracle.evaluate(
            prepared,
            None,
            continuation_config=self.continuation_config,
        )
        result.to_manifest()

        prepared_candidate = next(
            action for action in prepared.current_actions if action is not None
        )
        prepared_candidate.power_cost_W += 1.0
        with self.assertRaisesRegex(CounterfactualError, "创建后被修改"):
            oracle.evaluate(
                prepared,
                None,
                continuation_config=self.continuation_config,
            )

    def test_replayed_state_rejects_tampered_schedule_and_state_hash(self) -> None:
        tampered_schedule = self._state(1)
        tampered_schedule.schedule.assignments[0].sat_start_time += timedelta(
            seconds=1
        )
        with self.assertRaisesRegex(CounterfactualError, "schedule_hash"):
            evaluate_replayed_state(
                self.problem,
                tampered_schedule,
                None,
                constraint_config=self.constraint_config,
                enumerator_config=self.enumerator_config,
                objective_config=self.objective_config,
            )

        tampered_hash = self._state(0)
        tampered_hash.state_manifest["state_hash"] = "sha256:" + "0" * 64
        payload = {
            key: value
            for key, value in tampered_hash.state_manifest.items()
            if key != "state_manifest_hash"
        }
        tampered_hash.state_manifest["state_manifest_hash"] = sha256_json(payload)
        with self.assertRaisesRegex(CounterfactualError, "state_hash"):
            evaluate_replayed_state(
                self.problem,
                tampered_hash,
                None,
                constraint_config=self.constraint_config,
                enumerator_config=self.enumerator_config,
                objective_config=self.objective_config,
            )

    def test_replayed_state_rejects_candidate_order_and_nested_mutation(self) -> None:
        state = self._state(0)
        reversed_state = replace(state, candidates=tuple(reversed(state.candidates)))
        with self.assertRaisesRegex(CounterfactualError, "ordered_candidate_hash"):
            evaluate_replayed_state(
                self.problem,
                reversed_state,
                None,
                constraint_config=self.constraint_config,
                enumerator_config=self.enumerator_config,
                objective_config=self.objective_config,
            )

        nested_state = self._state(0)
        candidate = next(item for item in nested_state.candidates if item is not None)
        candidate.sat_angles = {
            "pitch_angles": [0.0],
            "yaw_angles": [0.0],
            "roll_angles": [0.0],
        }
        with self.assertRaisesRegex(
            CounterfactualError,
            "候选集合|runtime_fingerprint",
        ):
            evaluate_replayed_state(
                self.problem,
                nested_state,
                None,
                constraint_config=self.constraint_config,
                enumerator_config=self.enumerator_config,
                objective_config=self.objective_config,
            )

    def test_replayed_state_rejects_configuration_identity_mismatch(self) -> None:
        cases = (
            (
                "constraint_hash",
                ConstraintConfig(non_agile_transition_s=11.0),
                self.enumerator_config,
                self.objective_config,
            ),
            (
                "enumerator_hash",
                self.constraint_config,
                EnumeratorConfig(max_candidates=63),
                self.objective_config,
            ),
            (
                "objective_hash",
                self.constraint_config,
                self.enumerator_config,
                ObjectiveConfig((0.0, 1.0, 0.0, 0.0)),
            ),
        )
        for expected, constraint, enumerator, objective in cases:
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(CounterfactualError, expected):
                    evaluate_replayed_state(
                        self.problem,
                        self._state(0),
                        None,
                        constraint_config=constraint,
                        enumerator_config=enumerator,
                        objective_config=objective,
                    )

    def test_replayed_state_rejects_problem_object_mutation(self) -> None:
        state = self._state(0)
        self.problem.tasks["TASK-4"].windows[0].end_time += timedelta(seconds=1)
        with self.assertRaisesRegex(
            CounterfactualError,
            "problem_runtime_fingerprint",
        ):
            evaluate_replayed_state(
                self.problem,
                state,
                None,
                constraint_config=self.constraint_config,
                enumerator_config=self.enumerator_config,
                objective_config=self.objective_config,
            )

    def test_result_hash_covers_the_manifest(self) -> None:
        manifest = self._evaluate(0, None).to_manifest()
        payload = {
            key: value
            for key, value in manifest.items()
            if key != "result_hash"
        }
        self.assertEqual(manifest["result_hash"], sha256_json(payload))

    def test_result_rejects_nested_rollout_mutation(self) -> None:
        result = self._evaluate(0, None)
        self.assertGreater(len(result.rollout_action_keys), 1)
        result.rollout_action_keys[1]["task_id"] = "tampered"
        with self.assertRaisesRegex(CounterfactualError, "creation_fingerprint"):
            result.to_manifest()

    def test_implementation_hash_covers_transitive_dependencies(self) -> None:
        seen = []

        def fake_hash(path):
            seen.append(Path(path).name)
            return "sha256:" + "1" * 64

        continuation_implementation_hash.cache_clear()
        try:
            with patch(
                "algorithms.ccod.continuation.sha256_file",
                side_effect=fake_hash,
            ):
                continuation_implementation_hash()
        finally:
            continuation_implementation_hash.cache_clear()
        required = {
            "random_utils.py",
            "balance_utils.py",
            "timeliness_utils.py",
            "transition_utils.py",
            "scenario_loader.py",
        }
        self.assertTrue(required.issubset(set(seen)))

    def test_objective_weights_must_be_finite_and_non_negative(self) -> None:
        for weights in (
            (-1.0, 2.0, 0.0, 0.0),
            (float("nan"), 1.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 0.0),
        ):
            with self.subTest(weights=weights):
                with self.assertRaises(ValueError):
                    ObjectiveConfig(weights)

    def test_cache_round_trip_and_identity_match(self) -> None:
        result = self._evaluate(0, None)
        identity = build_cache_identity(
            state_hash=result.state_hash,
            action_key=result.forced_action_key,
            constraint_config=self.constraint_config,
            enumerator_config=self.enumerator_config,
            objective_config=self.objective_config,
            continuation_config=self.continuation_config,
        )
        self.assertEqual(result.query_key, cache_key(identity))
        cache = CounterfactualLabelCache(Path(self._tmp.name) / "cache")
        cache.store(identity, result)
        self.assertEqual(cache.load(identity), result.to_manifest())

    def test_cache_concurrent_same_key_writes_are_atomic(self) -> None:
        result = self._evaluate(0, None)
        identity = build_cache_identity(
            state_hash=result.state_hash,
            action_key=result.forced_action_key,
            constraint_config=self.constraint_config,
            enumerator_config=self.enumerator_config,
            objective_config=self.objective_config,
            continuation_config=self.continuation_config,
        )
        cache = CounterfactualLabelCache(
            Path(self._tmp.name) / "concurrent-cache"
        )
        workers = 8
        barrier = threading.Barrier(workers)

        def store_once(_index: int):
            barrier.wait()
            return cache.store(identity, result)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            paths = list(executor.map(store_once, range(workers)))
        self.assertEqual(len(set(paths)), 1)
        self.assertEqual(cache.load(identity), result.to_manifest())
        self.assertEqual(
            list((Path(self._tmp.name) / "concurrent-cache").rglob("*.tmp")),
            [],
        )

    def test_cache_concurrent_different_payloads_detect_collision(self) -> None:
        result = self._evaluate(0, None)
        changed_scores = list(result.objective_score_hexes)
        changed_scores[2] = changed_scores[1]
        different_result = replace(
            result,
            objective_score_hexes=tuple(changed_scores),
        )
        self.assertNotEqual(result.to_manifest(), different_result.to_manifest())
        identity = build_cache_identity(
            state_hash=result.state_hash,
            action_key=result.forced_action_key,
            constraint_config=self.constraint_config,
            enumerator_config=self.enumerator_config,
            objective_config=self.objective_config,
            continuation_config=self.continuation_config,
        )
        cache = CounterfactualLabelCache(
            Path(self._tmp.name) / "collision-cache"
        )
        barrier = threading.Barrier(2)

        def store_once(candidate):
            barrier.wait()
            try:
                cache.store(identity, candidate)
                return "stored"
            except CounterfactualError:
                return "collision"

        with ThreadPoolExecutor(max_workers=2) as executor:
            outcomes = list(executor.map(store_once, (result, different_result)))
        self.assertCountEqual(outcomes, ["stored", "collision"])
        self.assertIn(
            cache.load(identity),
            (result.to_manifest(), different_result.to_manifest()),
        )

    def test_cache_load_rejects_self_consistent_identity_mismatch(self) -> None:
        result = self._evaluate(0, None)
        identity = build_cache_identity(
            state_hash=result.state_hash,
            action_key=result.forced_action_key,
            constraint_config=self.constraint_config,
            enumerator_config=self.enumerator_config,
            objective_config=self.objective_config,
            continuation_config=self.continuation_config,
        )
        cache = CounterfactualLabelCache(Path(self._tmp.name) / "tampered-cache")
        path = cache.store(identity, result)
        record = json.loads(path.read_text(encoding="utf-8"))
        record["result"]["state_hash"] = "sha256:" + "0" * 64
        result_unhashed = {
            key: value
            for key, value in record["result"].items()
            if key != "result_hash"
        }
        record["result"]["result_hash"] = sha256_json(result_unhashed)
        record_unhashed = {
            key: value
            for key, value in record.items()
            if key != "record_hash"
        }
        record["record_hash"] = sha256_json(record_unhashed)
        path.write_bytes(canonical_json_bytes(record) + b"\n")
        with self.assertRaisesRegex(CounterfactualError, "state_hash"):
            cache.load(identity)

    def test_query_key_is_sensitive_to_every_configuration(self) -> None:
        result = self._evaluate(0, None)

        def identity(
            constraint: ConstraintConfig,
            enumerator: EnumeratorConfig,
            objective: ObjectiveConfig,
            continuation: ContinuationConfig,
        ):
            return build_cache_identity(
                state_hash=result.state_hash,
                action_key=result.forced_action_key,
                constraint_config=constraint,
                enumerator_config=enumerator,
                objective_config=objective,
                continuation_config=continuation,
            )

        identities = (
            identity(
                self.constraint_config,
                self.enumerator_config,
                self.objective_config,
                self.continuation_config,
            ),
            identity(
                ConstraintConfig(placement_mode="center"),
                self.enumerator_config,
                self.objective_config,
                self.continuation_config,
            ),
            identity(
                self.constraint_config,
                EnumeratorConfig(max_candidates=63),
                self.objective_config,
                self.continuation_config,
            ),
            identity(
                self.constraint_config,
                self.enumerator_config,
                ObjectiveConfig((0.0, 1.0, 0.0, 0.0)),
                self.continuation_config,
            ),
            identity(
                self.constraint_config,
                self.enumerator_config,
                self.objective_config,
                ContinuationConfig(horizon=4),
            ),
        )
        keys = {cache_key(item) for item in identities}
        self.assertEqual(len(keys), len(identities))
        self.assertEqual(result.query_key, cache_key(identities[0]))

        scaled = identity(
            self.constraint_config,
            self.enumerator_config,
            ObjectiveConfig((2.0, 0.0, 0.0, 0.0)),
            self.continuation_config,
        )
        self.assertEqual(cache_key(identities[0]), cache_key(scaled))

    def test_cache_rejects_mutated_final_schedule(self) -> None:
        result = self._evaluate(0, self.states[0]["observed_action_key"])
        identity = build_cache_identity(
            state_hash=result.state_hash,
            action_key=result.forced_action_key,
            constraint_config=self.constraint_config,
            enumerator_config=self.enumerator_config,
            objective_config=self.objective_config,
            continuation_config=self.continuation_config,
        )
        result.final_schedule.assignments.clear()
        with self.assertRaisesRegex(
            CounterfactualError,
            "creation_fingerprint|final_schedule_hash",
        ):
            CounterfactualLabelCache(Path(self._tmp.name) / "cache").store(
                identity,
                result,
            )

    def test_skip_and_observed_actions_have_distinct_cache_keys(self) -> None:
        skipped = self._evaluate(0, None)
        observed = self._evaluate(0, self.states[0]["observed_action_key"])
        self.assertNotEqual(skipped.query_key, observed.query_key)


if __name__ == "__main__":
    unittest.main()
