# -*- coding: utf-8 -*-
"""CCOD 标签执行纯核心的无 oracle 单元测试。"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from algorithms.ccod.execution import (
    CCODExecutionError,
    ExecutionIdentityError,
    build_execution_identity,
    execution_identity_hash,
    load_frozen_diagnostic_plan,
    scientific_results_hash,
    summarize_signal_gate,
)
from schedulers.state_replay import canonical_json_bytes, sha256_json


RUNNER_HASH = "sha256:" + "de" * 32
RUN_ID = "sha256:" + "ab" * 32


def _with_hash(payload, hash_field):
    """为合成冻结记录补上与生产格式一致的嵌入哈希。"""
    result = dict(payload)
    result[hash_field] = sha256_json(result)
    return result


def _build_fixture_rows():
    """生成 97 个 signal state、总计 1570 个 query 的小字段夹具。"""
    selected = []
    queries = []
    for state_ordinal in range(100):
        instance_alias = "cities_08" if state_ordinal < 50 else "cities_04"
        signal_eligible = state_ordinal < 47 or state_ordinal >= 50
        candidate_count = 16 if signal_eligible else 6
        state_hash = f"sha256:{state_ordinal:064x}"
        selected.append(
            _with_hash(
                {
                    "schema_version": "eosbench-ccod-state-catalog-v1",
                    "record_type": "state",
                    "state_hash": state_hash,
                    "instance_alias": instance_alias,
                    "candidate_count": candidate_count,
                    "signal_eligible": signal_eligible,
                    "selection": {
                        "selected": True,
                        "selected_ordinal": state_ordinal,
                    },
                },
                "row_hash",
            )
        )
        for action_ordinal in range(min(16, candidate_count)):
            query_identity = {
                "schema_version": "eosbench-ccod-query-v1",
                "state_hash": state_hash,
                "action_key": {
                    "kind": "synthetic",
                    "ordinal": action_ordinal,
                },
            }
            queries.append(
                _with_hash(
                    {
                        "schema_version": "eosbench-ccod-query-plan-v1",
                        "run_id": RUN_ID,
                        "query_ordinal": len(queries),
                        "state_ordinal": state_ordinal,
                        "state_hash": state_hash,
                        "instance_alias": instance_alias,
                        "action_ordinal": action_ordinal,
                        "query_identity": query_identity,
                        "query_key": sha256_json(query_identity),
                    },
                    "row_hash",
                )
            )
    assert len(queries) == 1570
    return selected, queries


def _write_fixture(root: Path, mutate=None) -> None:
    """把可选篡改后的合成冻结包写入临时目录。"""
    root.mkdir(parents=True)
    selected, queries = _build_fixture_rows()
    if mutate is not None:
        mutate(selected, queries)
        # 篡改结构后重签行哈希，使测试只命中目标不变量。
        selected = [
            _with_hash(
                {key: value for key, value in row.items() if key != "row_hash"},
                "row_hash",
            )
            for row in selected
        ]
        queries = [
            _with_hash(
                {key: value for key, value in row.items() if key != "row_hash"},
                "row_hash",
            )
            for row in queries
        ]
    selection_hash = sha256_json(selected)
    query_plan_hash = sha256_json(queries)
    header = _with_hash(
        {
            "record_type": "header",
            "schema_version": "eosbench-ccod-query-plan-header-v1",
            "run_id": RUN_ID,
            "query_plan_hash": query_plan_hash,
            "states": 100,
            "queries": 1570,
        },
        "header_hash",
    )
    run_manifest = _with_hash(
        {
            "schema_version": "eosbench-ccod-diagnostic-run-v1",
            "run_id": RUN_ID,
            "selection_hash": selection_hash,
            "query_plan_hash": query_plan_hash,
            "planned_states": 100,
            "planned_queries": 1570,
            "restore_count": 100,
            "workers": 1,
        },
        "manifest_hash",
    )
    (root / "run_manifest.json").write_bytes(
        json.dumps(run_manifest, ensure_ascii=False, indent=2).encode("utf-8")
        + b"\n"
    )
    (root / "selected_states.jsonl").write_bytes(
        b"".join(canonical_json_bytes(row) + b"\n" for row in selected)
    )
    (root / "query_plan.jsonl").write_bytes(
        b"".join(
            canonical_json_bytes(row) + b"\n"
            for row in (header, *queries)
        )
    )


class FrozenDiagnosticPlanTest(unittest.TestCase):
    """验证纯加载器只接受闭合的 100/1570 冻结引用。"""

    def test_loads_and_groups_exact_min_16_candidate_budget(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "frozen"
            _write_fixture(root)
            plan = load_frozen_diagnostic_plan(root)

        self.assertEqual(len(plan.selected_states), 100)
        self.assertEqual(len(plan.query_rows), 1570)
        self.assertEqual(len(plan.queries_by_state[0]), 16)
        self.assertEqual(len(plan.queries_by_state[47]), 6)
        self.assertEqual(plan.query_rows[-1]["query_ordinal"], 1569)

    def test_rejects_duplicate_selected_ordinal(self) -> None:
        def duplicate_ordinal(selected, _queries):
            selected[1]["selection"]["selected_ordinal"] = 0

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "frozen"
            _write_fixture(root, duplicate_ordinal)
            with self.assertRaisesRegex(CCODExecutionError, "selected_ordinal"):
                load_frozen_diagnostic_plan(root)

    def test_rejects_missing_query_ordinal(self) -> None:
        def skip_ordinal(_selected, queries):
            queries[-1]["query_ordinal"] = 1570

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "frozen"
            _write_fixture(root, skip_ordinal)
            with self.assertRaisesRegex(CCODExecutionError, "query_ordinal"):
                load_frozen_diagnostic_plan(root)

    def test_absolute_path_never_enters_execution_identity(self) -> None:
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            first_root = Path(first) / "location-a"
            second_root = Path(second) / "location-b"
            _write_fixture(first_root)
            _write_fixture(second_root)
            first_plan = load_frozen_diagnostic_plan(first_root)
            second_plan = load_frozen_diagnostic_plan(second_root)

        first_identity = build_execution_identity(
            first_plan,
            runner_implementation_hash=RUNNER_HASH,
        )
        second_identity = build_execution_identity(
            second_plan,
            runner_implementation_hash=RUNNER_HASH,
        )
        self.assertEqual(first_identity, second_identity)
        identity_text = json.dumps(first_identity, sort_keys=True)
        self.assertNotIn(str(first_root), identity_text)
        self.assertNotIn(str(second_root), identity_text)
        self.assertEqual(first_identity["runtime"]["python_version"], "3.10.20")
        self.assertEqual(
            first_identity["guards"],
            {
                "workers": 1,
                "max_attempts": 2,
                "query_timeout_s": 120.0,
                "state_timeout_s": 1200.0,
                "worker_peak_rss_limit_mib": 6144.0,
            },
        )


class SignalGateTest(unittest.TestCase):
    """验证 97-state signal 分母及预注册比例边界。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary = tempfile.TemporaryDirectory()
        root = Path(cls._temporary.name) / "frozen"
        _write_fixture(root)
        cls.plan = load_frozen_diagnostic_plan(root)
        cls.identity = build_execution_identity(
            cls.plan,
            runner_implementation_hash=RUNNER_HASH,
        )
        cls.execution_id = execution_identity_hash(cls.identity)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary.cleanup()

    def _results(self, passing_states):
        """为指定 state ordinal 构造 spread 通过标签。"""
        passing = set(passing_states)
        rows = []
        for query in self.plan.query_rows:
            action_ordinal = int(query["action_ordinal"])
            q_h = (
                0.02
                if int(query["state_ordinal"]) in passing and action_ordinal % 2
                else 0.0
            )
            rows.append(
                {
                    "execution_id": self.execution_id,
                    "run_id": self.plan.run_id,
                    "query_key": query["query_key"],
                    "query_ordinal": query["query_ordinal"],
                    "state_hash": query["state_hash"],
                    "status": "success",
                    "q_h_hex": q_h.hex(),
                }
            )
        return rows

    def test_overall_boundary_is_59_of_97(self) -> None:
        passing_59 = set(range(29)) | set(range(50, 80))
        summary = summarize_signal_gate(
            self.plan,
            self.identity,
            self._results(passing_59),
        )
        self.assertEqual(summary["overall"]["passing_states"], 59)
        self.assertEqual(summary["overall"]["eligible_states"], 97)
        self.assertTrue(summary["passed"])
        self.assertEqual(summary["execution_status"], "complete")
        self.assertEqual(summary["signal_gate"], "pass")
        self.assertEqual(summary["method_decision"], "go")
        self.assertEqual(summary["populations"]["all_selected"]["states"], 100)
        self.assertEqual(summary["populations"]["actionable"]["states"], 100)
        self.assertEqual(summary["populations"]["signal"]["states"], 97)

        passing_58 = set(range(29)) | set(range(50, 79))
        failed = summarize_signal_gate(
            self.plan,
            self.identity,
            self._results(passing_58),
        )
        self.assertFalse(failed["overall"]["fraction_passed"])
        self.assertFalse(failed["passed"])

    def test_cities_08_boundary_is_24_of_47(self) -> None:
        passing = set(range(24)) | set(range(50, 85))
        summary = summarize_signal_gate(
            self.plan,
            self.identity,
            self._results(passing),
        )
        self.assertEqual(
            summary["per_instance"]["cities_08"]["passing_states"],
            24,
        )
        self.assertTrue(summary["per_instance"]["cities_08"]["passed"])

        below = set(range(23)) | set(range(50, 86))
        failed = summarize_signal_gate(
            self.plan,
            self.identity,
            self._results(below),
        )
        self.assertFalse(failed["per_instance"]["cities_08"]["passed"])
        self.assertFalse(failed["passed"])

    def test_cities_04_boundary_is_25_of_50(self) -> None:
        passing = set(range(34)) | set(range(50, 75))
        summary = summarize_signal_gate(
            self.plan,
            self.identity,
            self._results(passing),
        )
        self.assertEqual(
            summary["per_instance"]["cities_04"]["passing_states"],
            25,
        )
        self.assertTrue(summary["per_instance"]["cities_04"]["passed"])

        below = set(range(35)) | set(range(50, 74))
        failed = summarize_signal_gate(
            self.plan,
            self.identity,
            self._results(below),
        )
        self.assertFalse(failed["per_instance"]["cities_04"]["passed"])
        self.assertFalse(failed["passed"])

    def test_incomplete_results_never_evaluate_gate(self) -> None:
        rows = self._results(range(100))[:-1]
        summary = summarize_signal_gate(self.plan, self.identity, rows)
        self.assertEqual(summary["status"], "incomplete")
        self.assertEqual(summary["decision"], "not_evaluated")
        self.assertEqual(summary["execution_status"], "incomplete")
        self.assertEqual(summary["signal_gate"], "not_evaluated")
        self.assertIsNone(summary["method_decision"])
        self.assertIsNone(summary["passed"])
        self.assertIsNone(summary["scientific_results_hash"])

    def test_identity_conflict_raises_instead_of_mixing_runs(self) -> None:
        rows = self._results(range(100))
        rows[31]["execution_id"] = "sha256:" + "ff" * 32
        with self.assertRaisesRegex(ExecutionIdentityError, "冲突"):
            summarize_signal_gate(self.plan, self.identity, rows)

    def test_scientific_hash_ignores_cold_hot_operational_metadata(self) -> None:
        cold = self._results(range(59))
        hot = list(reversed([dict(row) for row in cold]))
        for index, row in enumerate(cold):
            row.update(
                {
                    "cache_hit": False,
                    "cache_path": f"/cold/cache/{index}",
                    "elapsed_s": 1.5,
                    "completed_at": "2026-07-15T00:00:00Z",
                }
            )
        for index, row in enumerate(hot):
            row.update(
                {
                    "cache_hit": True,
                    "cache_path": f"/hot/cache/{index}",
                    "elapsed_s": 0.0,
                    "completed_at": "2099-01-01T00:00:00Z",
                }
            )
        self.assertEqual(
            scientific_results_hash(self.plan, self.identity, cold),
            scientific_results_hash(self.plan, self.identity, hot),
        )


if __name__ == "__main__":
    unittest.main()
