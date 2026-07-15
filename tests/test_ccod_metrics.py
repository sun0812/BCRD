from __future__ import annotations

import math
import unittest

from algorithms.ccod.metrics import (
    DiagnosticMetricError,
    finite_float_hex,
    linear_quantile,
    summarize_state_q_values,
)


class CCODMetricTest(unittest.TestCase):
    def test_linear_quantile_matches_type7_reference_points(self) -> None:
        values = (5.0, 1.0, 4.0, 2.0, 3.0)
        self.assertEqual(linear_quantile(values, 0.0), 1.0)
        self.assertAlmostEqual(linear_quantile(values, 0.1), 1.4)
        self.assertEqual(linear_quantile(values, 0.5), 3.0)
        self.assertAlmostEqual(linear_quantile(values, 0.9), 4.6)
        self.assertEqual(linear_quantile(values, 1.0), 5.0)

    def test_linear_quantile_handles_singleton_and_rejects_empty(self) -> None:
        self.assertEqual(linear_quantile((2.5,), 0.37), 2.5)
        with self.assertRaisesRegex(DiagnosticMetricError, "不能为空"):
            linear_quantile((), 0.5)

    def test_linear_quantile_rejects_invalid_probability(self) -> None:
        for invalid in (-0.01, 1.01, math.nan, math.inf, True, "not-a-number"):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(DiagnosticMetricError, "位于"):
                    linear_quantile((0.0, 1.0), invalid)

    def test_linear_quantile_rejects_non_finite_values(self) -> None:
        for invalid in (math.nan, math.inf, -math.inf, "not-a-number"):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(
                    DiagnosticMetricError, "必须全部有限"
                ):
                    linear_quantile((0.0, invalid), 0.5)

    def test_summary_uses_preregistered_p90_minus_p10_gate(self) -> None:
        summary = summarize_state_q_values(
            (0.00, 0.01, 0.02, 0.03),
            spread_threshold=0.02,
        )
        self.assertEqual(summary["query_count"], 4)
        self.assertAlmostEqual(summary["p10_q"], 0.003)
        self.assertAlmostEqual(summary["median_q"], 0.015)
        self.assertAlmostEqual(summary["p90_q"], 0.027)
        self.assertAlmostEqual(summary["p90_minus_p10"], 0.024)
        self.assertTrue(summary["nontrivial_spread"])

    def test_summary_rejects_invalid_threshold(self) -> None:
        for invalid in (-0.01, math.nan, math.inf, True, "not-a-number"):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(
                    DiagnosticMetricError, "必须有限且非负"
                ):
                    summarize_state_q_values((0.0,), spread_threshold=invalid)

    def test_summary_accepts_iterable_and_uses_inclusive_gate(self) -> None:
        summary = summarize_state_q_values(
            (value for value in (0.0, 0.01)),
            spread_threshold=0.008,
        )
        self.assertAlmostEqual(summary["p10_q"], 0.001)
        self.assertAlmostEqual(summary["p90_q"], 0.009)
        self.assertTrue(summary["nontrivial_spread"])

    def test_summary_rejects_non_finite_q_values(self) -> None:
        for invalid in (math.nan, math.inf, -math.inf, "not-a-number"):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(DiagnosticMetricError, "Q 值"):
                    summarize_state_q_values((0.0, invalid))

    def test_finite_float_hex_round_trips_exactly(self) -> None:
        for value in (-0.0, 0.1, -123.5):
            with self.subTest(value=value):
                encoded = finite_float_hex(value)
                self.assertEqual(float.fromhex(encoded).hex(), float(value).hex())
        for invalid in (math.nan, math.inf, -math.inf, True, "not-a-number"):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(DiagnosticMetricError, "有限数值"):
                    finite_float_hex(invalid)


if __name__ == "__main__":
    unittest.main()
