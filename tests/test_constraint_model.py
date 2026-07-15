from __future__ import annotations

import unittest
from datetime import datetime, timedelta

from schedulers.constraint_model import Assignment, ConstraintModel
from schedulers.scenario_loader import SchedulingProblem, SchedulingSatellite
from schedulers.transition_utils import compute_transition_time_agile


class ConstraintModelTransitionConfigTest(unittest.TestCase):
    @staticmethod
    def _problem(maneuverability_type: str) -> SchedulingProblem:
        start = datetime(2026, 1, 1)
        satellite = SchedulingSatellite(
            id="SAT-1",
            maneuverability_type=maneuverability_type,
        )
        return SchedulingProblem(
            scenario_id="transition-config-test",
            start_time=start,
            end_time=start + timedelta(hours=1),
            satellites={satellite.id: satellite},
            ground_stations={},
            tasks={},
            comm_windows=[],
        )

    @staticmethod
    def _assignment(task_id: str, start: datetime, angles: object) -> Assignment:
        return Assignment(
            task_id=task_id,
            satellite_id="SAT-1",
            sat_start_time=start,
            sat_end_time=start + timedelta(seconds=5),
            sat_window_id=0,
            sat_angles=angles,
        )

    def test_non_agile_transition_uses_configured_constant(self) -> None:
        model = ConstraintModel(
            self._problem("non_agile"),
            non_agile_transition_s=37.5,
        )
        start = model.problem.start_time
        previous = self._assignment("T-1", start, None)
        following = self._assignment("T-2", start + timedelta(seconds=60), None)

        self.assertEqual(model.non_agile_transition_s, 37.5)
        self.assertEqual(model._transition_time_s("SAT-1", previous, following), 37.5)

    def test_agile_transition_uses_configured_profile(self) -> None:
        model = ConstraintModel(
            self._problem("agile"),
            agility_profile="High-Agility",
        )
        start = model.problem.start_time
        previous = self._assignment(
            "T-1",
            start,
            [{"roll": 0.0, "pitch": 0.0, "yaw": 0.0}],
        )
        following = self._assignment(
            "T-2",
            start + timedelta(seconds=60),
            [{"roll": 20.0, "pitch": 0.0, "yaw": 0.0}],
        )

        self.assertEqual(model.agility_profile, "High-Agility")
        self.assertAlmostEqual(
            model._transition_time_s("SAT-1", previous, following),
            compute_transition_time_agile(20.0, "High-Agility"),
        )


if __name__ == "__main__":
    unittest.main()
