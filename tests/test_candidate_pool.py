from __future__ import annotations

import unittest
from datetime import datetime, timedelta
from typing import Iterable, Tuple

from algorithms.candidate_pool import enumerate_task_candidates
from schedulers.constraint_model import Assignment
from schedulers.scenario_loader import (
    CommWindow,
    SchedulingGroundStation,
    SchedulingProblem,
    SchedulingSatellite,
    SchedulingTask,
    TaskWindow,
)


class CanonicalCandidateOrderingTest(unittest.TestCase):
    @staticmethod
    def _problem(
        observation_order: Iterable[int] = (0, 1, 2),
        communication_order: Iterable[int] = (0, 1, 2),
    ) -> Tuple[SchedulingProblem, SchedulingTask]:
        start = datetime(2026, 1, 1)
        satellite = SchedulingSatellite(id="SAT-1", maneuverability_type="agile")
        station = SchedulingGroundStation(id="GS-1")

        observation_windows = [
            TaskWindow(
                window_id=30,
                satellite_id=satellite.id,
                mission_id="TASK-1",
                sensor_id="SENSOR-1",
                orbit_number=2,
                start_time=start + timedelta(seconds=40),
                end_time=start + timedelta(seconds=60),
                time_step=5.0,
                agile_data=[{"roll": value} for value in range(4)],
            ),
            TaskWindow(
                window_id=20,
                satellite_id=satellite.id,
                mission_id="TASK-1",
                sensor_id="SENSOR-1",
                orbit_number=1,
                start_time=start,
                end_time=start + timedelta(seconds=20),
                time_step=5.0,
                agile_data=[{"roll": value} for value in range(4, 8)],
            ),
            # 与前一个窗口使用相同时间戳，用于验证完整的规范化平局判定规则，
            # 避免依赖输入顺序的稳定性。
            TaskWindow(
                window_id=10,
                satellite_id=satellite.id,
                mission_id="TASK-1",
                sensor_id="SENSOR-1",
                orbit_number=1,
                start_time=start,
                end_time=start + timedelta(seconds=20),
                time_step=5.0,
                agile_data=[{"roll": value} for value in range(8, 12)],
            ),
        ]
        task = SchedulingTask(
            id="TASK-1",
            priority=1.0,
            required_duration=5.0,
            windows=[observation_windows[index] for index in observation_order],
        )

        communication_windows = [
            CommWindow(
                window_id=300,
                satellite_id=satellite.id,
                ground_station_id=station.id,
                start_time=start + timedelta(seconds=100),
                end_time=start + timedelta(seconds=120),
            ),
            CommWindow(
                window_id=200,
                satellite_id=satellite.id,
                ground_station_id=station.id,
                start_time=start + timedelta(seconds=80),
                end_time=start + timedelta(seconds=100),
            ),
            CommWindow(
                window_id=100,
                satellite_id=satellite.id,
                ground_station_id=station.id,
                start_time=start + timedelta(seconds=80),
                end_time=start + timedelta(seconds=100),
            ),
        ]
        problem = SchedulingProblem(
            scenario_id="canonical-candidate-test",
            start_time=start,
            end_time=start + timedelta(hours=1),
            satellites={satellite.id: satellite},
            ground_stations={station.id: station},
            tasks={task.id: task},
            comm_windows=[communication_windows[index] for index in communication_order],
        )
        return problem, task

    @staticmethod
    def _signatures(assignments: Iterable[Assignment]) -> list[tuple]:
        return [
            (
                assignment.task_id,
                assignment.satellite_id,
                assignment.sat_window_id,
                assignment.sat_start_time,
                assignment.sat_end_time,
                assignment.ground_station_id,
                assignment.gs_window_id,
                assignment.gs_start_time,
                assignment.gs_end_time,
            )
            for assignment in assignments
        ]

    def test_canonical_order_ignores_seed_and_input_window_order(self) -> None:
        problem_a, task_a = self._problem()
        problem_b, task_b = self._problem(
            observation_order=(2, 0, 1),
            communication_order=(1, 2, 0),
        )

        candidates_a = enumerate_task_candidates(
            problem_a,
            task_a,
            max_candidates=256,
            random_samples_per_window=0,
            seed=11,
            ordering_version="canonical_v1",
        )
        candidates_b = enumerate_task_candidates(
            problem_b,
            task_b,
            max_candidates=256,
            random_samples_per_window=0,
            seed=999,
            ordering_version="canonical_v1",
        )

        self.assertGreater(len(candidates_a), 0)
        self.assertEqual(self._signatures(candidates_a), self._signatures(candidates_b))

    def test_canonical_cap_is_a_stable_prefix(self) -> None:
        problem, task = self._problem(
            observation_order=(1, 0, 2),
            communication_order=(2, 0, 1),
        )

        full = enumerate_task_candidates(
            problem,
            task,
            max_candidates=256,
            random_samples_per_window=0,
            seed=7,
            ordering_version="canonical_v1",
        )
        capped = enumerate_task_candidates(
            problem,
            task,
            max_candidates=11,
            random_samples_per_window=0,
            seed=1234,
            ordering_version="canonical_v1",
        )

        self.assertEqual(self._signatures(capped), self._signatures(full)[:11])

    def test_canonical_low_cap_ignores_input_window_order(self) -> None:
        problem_a, task_a = self._problem()
        problem_b, task_b = self._problem(
            observation_order=(2, 0, 1),
            communication_order=(1, 2, 0),
        )

        capped_a = enumerate_task_candidates(
            problem_a,
            task_a,
            max_candidates=7,
            random_samples_per_window=0,
            seed=1,
            ordering_version="canonical_v1",
        )
        capped_b = enumerate_task_candidates(
            problem_b,
            task_b,
            max_candidates=7,
            random_samples_per_window=0,
            seed=999,
            ordering_version="canonical_v1",
        )

        self.assertEqual(self._signatures(capped_a), self._signatures(capped_b))

    def test_default_ordering_remains_legacy_v1(self) -> None:
        problem, task = self._problem()

        default = enumerate_task_candidates(
            problem,
            task,
            max_candidates=40,
            random_samples_per_window=1,
            seed=17,
        )
        explicit = enumerate_task_candidates(
            problem,
            task,
            max_candidates=40,
            random_samples_per_window=1,
            seed=17,
            ordering_version="legacy_v1",
        )

        self.assertEqual(self._signatures(default), self._signatures(explicit))


if __name__ == "__main__":
    unittest.main()
