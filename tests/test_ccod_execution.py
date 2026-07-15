# -*- coding: utf-8 -*-
"""CCOD 标签执行纯核心的无 oracle 单元测试。"""

from __future__ import annotations

import json
import operator
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from algorithms.ccod.execution import (
    CCODExecutionError,
    ExecutionIdentityError,
    build_execution_identity,
    execution_identity_hash,
    load_frozen_diagnostic_plan,
    mutable_json_copy,
    scientific_results_hash,
    summarize_signal_gate,
)
from schedulers.state_replay import canonical_json_bytes, sha256_json


RUNNER_HASH = "sha256:" + "de" * 32
RUN_ID = "sha256:" + "ab" * 32
CONSTRAINT_HASH = "sha256:" + "c1" * 32
ENUMERATOR_HASH = "sha256:" + "e2" * 32
OBJECTIVE_HASH = "sha256:" + "f3" * 32
CONTINUATION_IMPLEMENTATION_HASH = "sha256:" + "d4" * 32
CONTINUATION_CONFIG = {
    "horizon": 5,
    "policy_version": "objective_greedy_v1",
    "tie_break_version": "action_key_lexicographic_v1",
    "forced_action_counts_toward_horizon": True,
}
CONTINUATION_HASH = sha256_json(CONTINUATION_CONFIG)
FORMAL_PACKAGE = (
    Path(__file__).resolve().parents[1]
    / "output/ccod_diagnostic_v1_py310_final_9d74e46"
)


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
                    "split": "dev",
                    "state_hash": state_hash,
                    "instance_alias": instance_alias,
                    "candidate_count": candidate_count,
                    "signal_eligible": signal_eligible,
                    "constraint_hash": CONSTRAINT_HASH,
                    "enumerator_hash": ENUMERATOR_HASH,
                    "objective_hash": OBJECTIVE_HASH,
                    "selection": {
                        "selected": True,
                        "selected_ordinal": state_ordinal,
                    },
                },
                "row_hash",
            )
        )
        for action_ordinal in range(min(16, candidate_count)):
            action_key = {
                "kind": "synthetic",
                "ordinal": action_ordinal,
            }
            query_identity = {
                "schema_version": "eosbench-ccod-query-v1",
                "state_hash": state_hash,
                "action_key": action_key,
                "constraint_hash": CONSTRAINT_HASH,
                "enumerator_hash": ENUMERATOR_HASH,
                "objective_hash": OBJECTIVE_HASH,
                "horizon": 5,
                "continuation_hash": CONTINUATION_HASH,
                "continuation_implementation_hash": (
                    CONTINUATION_IMPLEMENTATION_HASH
                ),
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
                        "action_key": action_key,
                        "action_key_hash": sha256_json(action_key),
                        "roles": ["stable_uniform"],
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
            "status": "planned",
            "continuation_config": CONTINUATION_CONFIG,
            "continuation_implementation_hash": (
                CONTINUATION_IMPLEMENTATION_HASH
            ),
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


def _cache_result(query, q_h):
    """构造通过执行核心闭合校验的最小缓存结果清单。"""
    q_value = float(q_h)
    base_value = 0.0
    final_value = q_value
    action_key = mutable_json_copy(query["action_key"])
    payload = {
        "schema_version": "eosbench-ccod-continuation-v1",
        "query_key": query["query_key"],
        "state_hash": query["state_hash"],
        "step": 0,
        "task_id": "synthetic-task",
        "requested_horizon": 5,
        "decisions_executed": 1,
        "terminated_by_task_exhaustion": False,
        "forced_action_key": action_key,
        "rollout_action_keys": [action_key],
        "rollout_action_keys_hash": sha256_json([action_key]),
        "objective_score_hexes": [base_value.hex(), final_value.hex()],
        "base_score_hex": base_value.hex(),
        "forced_score_hex": final_value.hex(),
        "final_score_hex": final_value.hex(),
        "q_h_hex": q_value.hex(),
        "final_schedule_hash": "sha256:" + "11" * 32,
        "final_schedule_runtime_hash": "sha256:" + "22" * 32,
        "continuation_hash": query["query_identity"]["continuation_hash"],
        "continuation_implementation_hash": query["query_identity"][
            "continuation_implementation_hash"
        ],
        "constraint_hash": query["query_identity"]["constraint_hash"],
        "enumerator_hash": query["query_identity"]["enumerator_hash"],
        "objective_hash": query["query_identity"]["objective_hash"],
        "problem_runtime_fingerprint": "sha256:" + "33" * 32,
    }
    payload["result_hash"] = sha256_json(payload)
    return payload


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

    def test_loaded_plan_is_recursively_immutable(self) -> None:
        """嵌套动作不得在旧 query-plan hash 下被原地改写。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "frozen"
            _write_fixture(root)
            plan = load_frozen_diagnostic_plan(root)
        with self.assertRaises(TypeError):
            plan.query_rows[0]["query_identity"]["action_key"][
                "ordinal"
            ] = 999
        plan.assert_pristine()

    def test_loaded_plan_is_deeply_immutable(self) -> None:
        """dataclass 的嵌套 JSON 也必须拒绝原位修改。"""
        mutators = (
            lambda plan: operator.setitem(
                plan.selected_states[0], "signal_eligible", False
            ),
            lambda plan: operator.setitem(
                plan.query_rows[0]["action_key"], "ordinal", 999
            ),
            lambda plan: operator.setitem(
                plan.run_manifest, "status", "complete"
            ),
        )
        for mutator in mutators:
            with self.subTest(mutator=mutator):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory) / "frozen"
                    _write_fixture(root)
                    plan = load_frozen_diagnostic_plan(root)
                    with self.assertRaises(TypeError):
                        mutator(plan)
                    build_execution_identity(
                        plan,
                        runner_implementation_hash=RUNNER_HASH,
                    )

    def test_identity_builder_rejects_disguised_wrong_runtime(self) -> None:
        """执行身份不得在错误解释器下谎称 CPython 3.10.20。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "frozen"
            _write_fixture(root)
            plan = load_frozen_diagnostic_plan(root)
            with mock.patch(
                "algorithms.ccod.execution._current_runtime",
                return_value={
                    "python_implementation": "cpython",
                    "python_version": "3.14.6",
                },
            ):
                with self.assertRaisesRegex(ExecutionIdentityError, "解释器不匹配"):
                    build_execution_identity(
                        plan,
                        runner_implementation_hash=RUNNER_HASH,
                    )

    @unittest.skipUnless(FORMAL_PACKAGE.is_dir(), "本地未保留正式冻结包")
    def test_formal_package_matches_registered_execution_contract(self) -> None:
        """轻量集成检查正式包的关键身份、分母和非整齐 prefix。"""
        plan = load_frozen_diagnostic_plan(FORMAL_PACKAGE)
        self.assertEqual(
            plan.run_id,
            "sha256:56c0525bd200a22fc92c89c0ffb498b6cc6290f709f2d7640f29783482d1b1f3",
        )
        self.assertEqual(
            plan.query_plan_hash,
            "sha256:cfec7cd0c952e9aa1085733990a5ec4c67461d3a9c92fe7cfe5a44de49157396",
        )
        self.assertEqual(
            plan.selection_hash,
            "sha256:410f8255cf86d056f0f87f1ee9a88857a6a4fcbd2e583835099edcc004fcb296",
        )
        self.assertEqual(len(plan.selected_states), 100)
        self.assertEqual(len(plan.query_rows), 1570)
        self.assertEqual(
            sum(len(plan.queries_by_state[index]) for index in range(12)),
            179,
        )
        signal_by_instance = {
            alias: sum(
                state["instance_alias"] == alias and state["signal_eligible"]
                for state in plan.selected_states
            )
            for alias in ("cities_08", "cities_04")
        }
        self.assertEqual(signal_by_instance, {"cities_08": 47, "cities_04": 50})


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
                    "result": _cache_result(query, q_h),
                }
            )
        return rows

    def _single_success(self, query_index=0, q_h=0.0):
        """构造一条严格绑定计划的成功结果。"""
        query = self.plan.query_rows[query_index]
        return {
            "execution_id": self.execution_id,
            "run_id": self.plan.run_id,
            "query_key": query["query_key"],
            "query_ordinal": query["query_ordinal"],
            "state_hash": query["state_hash"],
            "status": "success",
            "result": _cache_result(query, q_h),
        }

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
        self.assertEqual(failed["execution_status"], "complete")
        self.assertEqual(failed["signal_gate"], "fail")
        self.assertEqual(failed["method_decision"], "no_go")

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

    def test_exhausted_resource_failures_invalidate_the_run(self) -> None:
        """资源故障可在中间重试，但耗尽后写入 failed 即作废本轮。"""
        recoverable = (
            "query_timeout",
            "state_timeout",
            "rss_exceeded",
            "worker_exit",
            "worker_error",
            "attempt_exhausted",
            "interrupted",
        )
        for failure_kind in recoverable:
            with self.subTest(failure_kind=failure_kind):
                rows = self._results(range(100))
                rows[0].pop("result")
                rows[0]["status"] = "failed"
                rows[0]["failure_kind"] = failure_kind
                summary = summarize_signal_gate(self.plan, self.identity, rows)
                self.assertEqual(summary["execution_status"], "invalid")
                self.assertEqual(summary["signal_gate"], "not_evaluated")
                self.assertIsNone(summary["method_decision"])
                self.assertEqual(summary["failure_counts"], {failure_kind: 1})

    def test_identity_failure_is_invalid_and_unknown_failure_is_rejected(self) -> None:
        invalid_kinds = (
            "identity_mismatch",
            "cache_corrupt",
            "hash_mismatch",
            "frozen_drift",
            "runner_drift",
        )
        for failure_kind in invalid_kinds:
            with self.subTest(failure_kind=failure_kind):
                rows = self._results(range(100))
                rows[0].pop("result")
                rows[0]["status"] = "failed"
                rows[0]["failure_kind"] = failure_kind
                summary = summarize_signal_gate(self.plan, self.identity, rows)
                self.assertEqual(summary["execution_status"], "invalid")
                self.assertEqual(summary["signal_gate"], "not_evaluated")

        rows[0]["failure_kind"] = "invented_failure"
        with self.assertRaisesRegex(CCODExecutionError, "failure_kind"):
            summarize_signal_gate(self.plan, self.identity, rows)

    def test_unknown_status_and_corrupt_nested_result_are_rejected(self) -> None:
        rows = self._results(range(100))
        rows[0]["status"] = "typo"
        with self.assertRaisesRegex(CCODExecutionError, "status"):
            summarize_signal_gate(self.plan, self.identity, rows)

        rows = self._results(range(100))
        result = rows[0]["result"]
        result["state_hash"] = "sha256:" + "ff" * 32
        result["result_hash"] = sha256_json(
            {
                name: value
                for name, value in result.items()
                if name != "result_hash"
            }
        )
        with self.assertRaisesRegex(CCODExecutionError, "result"):
            summarize_signal_gate(self.plan, self.identity, rows)

        rows = self._results(range(100))
        rows[0]["result"]["result_hash"] = "sha256:" + "00" * 32
        with self.assertRaisesRegex(CCODExecutionError, "result"):
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

    def test_success_requires_full_identity_closed_cache_result(self) -> None:
        """正确外层不得为错误、扁平或损坏的内层标签背书。"""
        flat = self._single_success()
        flat.pop("result")
        flat["q_h_hex"] = 0.0.hex()
        with self.assertRaisesRegex(CCODExecutionError, "完整 cache result"):
            summarize_signal_gate(self.plan, self.identity, [flat])

        bad_hash = self._single_success()
        bad_hash["result"]["result_hash"] = "sha256:" + "ff" * 32
        with self.assertRaisesRegex(CCODExecutionError, "result_hash"):
            summarize_signal_gate(self.plan, self.identity, [bad_hash])

        wrong_query = self._single_success()
        wrong_query["result"]["query_key"] = self.plan.query_rows[1]["query_key"]
        wrong_query["result"]["result_hash"] = sha256_json(
            {
                key: value
                for key, value in wrong_query["result"].items()
                if key != "result_hash"
            }
        )
        with self.assertRaisesRegex(CCODExecutionError, "query_key"):
            summarize_signal_gate(self.plan, self.identity, [wrong_query])

        wrong_q = self._single_success(q_h=0.0)
        wrong_q["result"]["q_h_hex"] = 0.25.hex()
        wrong_q["result"]["result_hash"] = sha256_json(
            {
                key: value
                for key, value in wrong_q["result"].items()
                if key != "result_hash"
            }
        )
        with self.assertRaisesRegex(CCODExecutionError, "q_h 与目标差值"):
            summarize_signal_gate(self.plan, self.identity, [wrong_q])

    def test_terminal_failure_is_invalid_but_missing_is_incomplete(self) -> None:
        """尚未运行可恢复；两次尝试耗尽后的 failed row 作废本轮。"""
        missing = summarize_signal_gate(self.plan, self.identity, [])
        self.assertEqual(missing["execution_status"], "incomplete")
        self.assertEqual(missing["failed_queries"], 0)

        query = self.plan.query_rows[0]
        failed = summarize_signal_gate(
            self.plan,
            self.identity,
            [
                {
                    "execution_id": self.execution_id,
                    "run_id": self.plan.run_id,
                    "query_key": query["query_key"],
                    "query_ordinal": query["query_ordinal"],
                    "state_hash": query["state_hash"],
                    "status": "failed",
                    "failure_kind": "query_timeout",
                }
            ],
        )
        self.assertEqual(failed["execution_status"], "invalid")
        self.assertEqual(failed["signal_gate"], "not_evaluated")
        self.assertIsNone(failed["passed"])
        self.assertEqual(failed["failed_queries"], 1)
        self.assertEqual(failed["failure_counts"], {"query_timeout": 1})

    def test_nested_operational_metadata_is_rejected(self) -> None:
        """运行元数据不得混入稳定 result_hash 并污染科学哈希。"""
        row = self._single_success()
        row["result"]["elapsed_s"] = 1.25
        row["result"]["result_hash"] = sha256_json(
            {
                key: value
                for key, value in row["result"].items()
                if key != "result_hash"
            }
        )
        with self.assertRaisesRegex(CCODExecutionError, "字段集合"):
            summarize_signal_gate(self.plan, self.identity, [row])

    def test_scientific_hash_binds_stable_result_hash(self) -> None:
        """同一 Q_H 但不同稳定结果证据必须产生不同科学哈希。"""
        first = self._results(range(59))
        second = json.loads(json.dumps(first))
        second[0]["result"]["final_schedule_hash"] = "sha256:" + "44" * 32
        second[0]["result"]["result_hash"] = sha256_json(
            {
                key: value
                for key, value in second[0]["result"].items()
                if key != "result_hash"
            }
        )
        self.assertNotEqual(
            scientific_results_hash(self.plan, self.identity, first),
            scientific_results_hash(self.plan, self.identity, second),
        )


if __name__ == "__main__":
    unittest.main()
