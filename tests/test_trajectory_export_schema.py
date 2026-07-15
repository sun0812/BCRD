from __future__ import annotations

import unittest
from typing import Any, Dict, Optional

from scripts.batch_export_trajectories import (
    TRAJECTORY_SCHEMA_VERSION,
    _candidate_set_stats,
    _objective_delta,
    _require_objective_weights,
    parse_schedule_filename,
)
from scripts.validate_trajectory_schema import ValidationStats, validate_row


class ImplicitObjectiveParsingTest(unittest.TestCase):
    def test_class2_implicit_objectives_are_one_hot_by_solver(self) -> None:
        expected = {
            "profit_first": [1.0, 0.0, 0.0, 0.0],
            "completion_first": [0.0, 1.0, 0.0, 0.0],
            "timeliness_first": [0.0, 0.0, 1.0, 0.0],
            "balance_first": [0.0, 0.0, 0.0, 1.0],
        }

        for solver_name, weights in expected.items():
            with self.subTest(solver_name=solver_name):
                parsed = parse_schedule_filename(
                    f"scheduler_Scenario_X_c2_{solver_name}_implicit"
                )
                self.assertEqual(parsed["solver_name"], solver_name)
                self.assertEqual(parsed["objective_tag"], "implicit")
                self.assertEqual(parsed["objective_weights"], weights)

    def test_explicit_objective_parsing_is_unchanged(self) -> None:
        parsed = parse_schedule_filename(
            "scheduler_Scenario_X_c3_sa_p0.25_c0.25_t0.25_b0.25"
        )
        self.assertEqual(parsed["objective_weights"], [0.25, 0.25, 0.25, 0.25])

    def test_unknown_implicit_mapping_fails_fast(self) -> None:
        parsed = parse_schedule_filename(
            "scheduler_Scenario_X_c1_unmapped_solver_implicit"
        )
        with self.assertRaisesRegex(ValueError, "all zero"):
            _require_objective_weights(parsed, "unmapped.json")

    def test_unparseable_filename_fails_fast(self) -> None:
        parsed = parse_schedule_filename("not-a-scheduler-filename")
        with self.assertRaisesRegex(ValueError, "cannot parse"):
            _require_objective_weights(parsed, "not-a-scheduler-filename.json")

    def test_explicit_all_zero_objective_fails_fast(self) -> None:
        parsed = parse_schedule_filename(
            "scheduler_Scenario_X_c3_sa_p0_c0_t0_b0"
        )
        with self.assertRaisesRegex(ValueError, "all zero"):
            _require_objective_weights(parsed, "all-zero.json")


class UnifiedObjectiveDeltaTest(unittest.TestCase):
    def test_deltas_telescope_to_final_minus_initial_score(self) -> None:
        scores = [0.1, 0.25, 0.25, 0.4, 0.35]
        deltas = [
            _objective_delta(before, after)
            for before, after in zip(scores, scores[1:])
        ]
        self.assertAlmostEqual(sum(deltas), scores[-1] - scores[0])

    def test_skip_is_the_same_transition_and_has_zero_delta(self) -> None:
        before = 0.375
        self.assertEqual(_objective_delta(before, before), 0.0)

    def test_candidate_stats_describe_cap_before_feasibility(self) -> None:
        self.assertEqual(
            _candidate_set_stats(cap=16, enumerated=16, feasible=7),
            {"cap": 16, "enumerated": 16, "feasible": 7, "cap_reached": True},
        )
        self.assertFalse(
            _candidate_set_stats(cap=16, enumerated=8, feasible=3)["cap_reached"]
        )


class DynamicTrajectorySchemaTest(unittest.TestCase):
    @staticmethod
    def _row(
        state_dim: int,
        candidate_dim: int,
        schema: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        row: Dict[str, Any] = {
            "scenario_id": "schema-test",
            "schedule_file": "scheduler_schema_test.json",
            "solver_name": "profit_first",
            "class_id": 2,
            "objective_tag": "implicit",
            "objective_weights": [1.0, 0.0, 0.0, 0.0],
            "timestep": 0,
            "state_features": [0.0] * state_dim,
            "candidate_features": [[1.0] + [0.0] * (candidate_dim - 1)],
            "candidate_keys": [{"is_skip": True}],
            "valid_mask": [1],
            "expert_action_index": 0,
            "schedule_metrics": {},
        }
        if schema is not None:
            row["_schema"] = schema
        if schema is not None and schema.get("version") == "eosbench-trajectory-v2":
            row["objective_delta"] = 0.0
            row["observed_return_H"] = 0.0
        else:
            row["reward"] = -1.0
            row["future_return_H"] = -1.0
        return row

    def test_legacy_5_by_10_row_remains_compatible(self) -> None:
        stats = ValidationStats()
        validate_row(0, self._row(5, 10), stats)

        self.assertEqual(stats.expected_state_dim, 5)
        self.assertEqual(stats.expected_cand_dim, 10)
        self.assertEqual(stats.state_dim_source, "first valid row")
        self.assertEqual(stats.rows_failing_state_dim, 0)
        self.assertEqual(stats.rows_failing_cand_dim, 0)

    def test_dimensions_are_inferred_instead_of_assuming_5_by_10(self) -> None:
        stats = ValidationStats()
        validate_row(0, self._row(7, 12), stats)
        validate_row(1, self._row(7, 12), stats)

        self.assertEqual(stats.expected_state_dim, 7)
        self.assertEqual(stats.expected_cand_dim, 12)
        self.assertEqual(stats.rows_failing_state_dim, 0)
        self.assertEqual(stats.rows_failing_cand_dim, 0)

    def test_dimension_change_after_first_row_is_rejected(self) -> None:
        stats = ValidationStats()
        validate_row(0, self._row(5, 10), stats)
        validate_row(1, self._row(7, 10), stats)

        self.assertEqual(stats.expected_state_dim, 5)
        self.assertEqual(stats.rows_failing_state_dim, 1)

    def test_declared_schema_is_preferred_and_version_is_consistent(self) -> None:
        stats = ValidationStats()
        schema = {
            "version": "eosbench-trajectory-v1",
            "state_dim": 7,
            "candidate_dim": 10,
        }
        validate_row(0, self._row(7, 10, schema), stats)

        changed_version = dict(schema, version="eosbench-trajectory-v2")
        validate_row(1, self._row(7, 10, changed_version), stats)

        self.assertEqual(stats.state_dim_source, "_schema")
        self.assertEqual(stats.cand_dim_source, "_schema")
        self.assertEqual(stats.expected_schema_version, "eosbench-trajectory-v1")
        self.assertEqual(stats.rows_failing_schema_consistency, 1)

    def test_v2_uses_new_target_fields_without_legacy_reward_fields(self) -> None:
        stats = ValidationStats()
        schema = {
            "version": TRAJECTORY_SCHEMA_VERSION,
            "state_dim": 7,
            "candidate_dim": 10,
        }
        row = self._row(7, 10, schema)

        self.assertNotIn("reward", row)
        self.assertNotIn("future_return_H", row)
        validate_row(0, row, stats)

        self.assertEqual(stats.rows_with_all_fields, 1)
        self.assertEqual(stats.missing_field_counts, {})
        self.assertEqual(stats.rows_failing_target_semantics, 0)

    def test_v2_missing_observed_return_is_rejected_conditionally(self) -> None:
        stats = ValidationStats()
        row = self._row(
            7,
            10,
            {
                "version": TRAJECTORY_SCHEMA_VERSION,
                "state_dim": 7,
                "candidate_dim": 10,
            },
        )
        del row["observed_return_H"]
        validate_row(0, row, stats)

        self.assertEqual(stats.rows_with_all_fields, 0)
        self.assertEqual(stats.missing_field_counts["observed_return_H"], 1)
        self.assertEqual(stats.missing_field_counts["future_return_H"], 0)

    def test_v1_still_requires_legacy_target_fields(self) -> None:
        stats = ValidationStats()
        schema = {
            "version": "eosbench-trajectory-v1",
            "state_dim": 5,
            "candidate_dim": 10,
        }
        row = self._row(5, 10, schema)
        del row["future_return_H"]
        validate_row(0, row, stats)

        self.assertEqual(stats.missing_field_counts["future_return_H"], 1)
        self.assertEqual(stats.missing_field_counts["observed_return_H"], 0)

    def test_v2_skip_must_have_zero_objective_delta(self) -> None:
        stats = ValidationStats()
        row = self._row(
            7,
            10,
            {
                "version": TRAJECTORY_SCHEMA_VERSION,
                "state_dim": 7,
                "candidate_dim": 10,
            },
        )
        row["objective_delta"] = -1.0
        validate_row(0, row, stats)

        self.assertEqual(stats.rows_failing_target_semantics, 1)

    def test_declared_dimension_must_match_row(self) -> None:
        stats = ValidationStats()
        schema = {
            "version": "eosbench-trajectory-v1",
            "state_dim": 7,
            "candidate_dim": 12,
        }
        validate_row(0, self._row(5, 10, schema), stats)

        self.assertEqual(stats.expected_state_dim, 7)
        self.assertEqual(stats.expected_cand_dim, 12)
        self.assertEqual(stats.rows_failing_state_dim, 1)
        self.assertEqual(stats.rows_failing_cand_dim, 1)


if __name__ == "__main__":
    unittest.main()
