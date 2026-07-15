# -*- coding: utf-8 -*-
"""CCOD 单机 runner 的 cache-first、checkpoint 与护栏单元测试。"""

from __future__ import annotations

from contextlib import ExitStack, redirect_stderr
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import time
import unittest
from unittest import mock

from schedulers.state_replay import sha256_json
from scripts.run_ccod_diagnostic import (
    CCODRunnerError,
    INVALID_FAILURE_KINDS,
    QueryDeadlineExceeded,
    RunnerAttemptError,
    _acquire_run_lock,
    _heartbeat_journal,
    _launch_state_attempt,
    _result_rows_from_cache,
    _restore_state_progress,
    _terminal_missing_queries,
    _verify_worker_report,
    _worker_main,
    _worker_report_path,
    build_parser,
    execute_state_worker,
    execution_prefix,
    monitor_worker,
    publish_state_checkpoint_if_complete,
    query_deadline,
    run_diagnostic,
    validate_external_paths,
    _worker_command,
)


RUN_ID = "sha256:" + "12" * 32
EXECUTION_ID = "sha256:" + "34" * 32


def _query(ordinal: int, state_hash: str, state_ordinal: int = 0) -> dict:
    """构造 runner 协调层所需的最小冻结 query。"""
    action_key = {
        "version": "synthetic-v1",
        "kind": "synthetic",
        "task_id": "task-0",
        "ordinal": ordinal,
    }
    identity = {
        "schema_version": "eosbench-ccod-query-v1",
        "state_hash": state_hash,
        "action_key": action_key,
        "horizon": 5,
    }
    return {
        "query_ordinal": ordinal,
        "state_ordinal": state_ordinal,
        "action_ordinal": ordinal,
        "state_hash": state_hash,
        "action_key": action_key,
        "query_identity": identity,
        "query_key": sha256_json(identity),
    }


def _plan(query_count: int = 2):
    """生成只含一个 state 的轻量 plan 替身。"""
    state_hash = "sha256:" + "56" * 32
    queries = tuple(_query(index, state_hash) for index in range(query_count))
    return SimpleNamespace(
        run_id=RUN_ID,
        queries_by_state={0: queries},
        selected_states=({"state_hash": state_hash},),
        query_rows=queries,
    )


def _multi_plan(state_count: int = 2):
    """生成每个 state 含一个 query 的父进程编排替身。"""
    grouped = {}
    selected = []
    rows = []
    for state_ordinal in range(state_count):
        state_hash = "sha256:" + f"{state_ordinal + 1:02x}" * 32
        query = _query(state_ordinal, state_hash, state_ordinal)
        grouped[state_ordinal] = (query,)
        selected.append({"state_hash": state_hash})
        rows.append(query)
    return SimpleNamespace(
        run_id=RUN_ID,
        queries_by_state=grouped,
        selected_states=tuple(selected),
        query_rows=tuple(rows),
    )


class _FakeResult:
    """只实现 worker/cache 握手所需接口的标签结果。"""

    def __init__(self, query_key: str, q_h: float = 0.25) -> None:
        self.query_key = query_key
        self._manifest = {
            "query_key": query_key,
            "q_h_hex": q_h.hex(),
            "result_hash": "sha256:" + "78" * 32,
        }

    def to_manifest(self) -> dict:
        return dict(self._manifest)


class _FakeCache:
    """记录 store 副作用的内存缓存，用于验证中断后可恢复性。"""

    def __init__(self, values=None) -> None:
        self.values = dict(values or {})

    def load(self, identity):
        value = self.values.get(sha256_json(dict(identity)))
        return dict(value) if value is not None else None

    def store(self, identity, result) -> Path:
        self.values[sha256_json(dict(identity))] = result.to_manifest()
        return Path("synthetic-cache-record.json")


class _FakeProcess:
    """不会自行退出的进程替身，让 watchdog 用毫秒级时限终止。"""

    def __init__(self) -> None:
        self.pid = 12345
        self.returncode = None
        self.terminated = False

    def poll(self):
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -15

    def kill(self) -> None:
        self.returncode = -9

    def wait(self, timeout=None):
        return self.returncode

    def communicate(self):
        return "", ""


class CacheFirstWorkerTest(unittest.TestCase):
    """验证全 hit 不恢复，以及 partial failure 不丢已发布 cache。"""

    def test_all_hits_never_restore_or_prepare(self) -> None:
        plan = _plan()
        values = {
            query["query_key"]: _FakeResult(query["query_key"]).to_manifest()
            for query in plan.queries_by_state[0]
        }
        cache = _FakeCache(values)
        heartbeats = []

        def forbidden_factory(_plan, _ordinal):
            raise AssertionError("全 cache hit 不得恢复 state")

        report = execute_state_worker(
            plan,
            0,
            cache,
            lambda event, details: heartbeats.append((event, details)),
            query_timeout_s=0.1,
            runtime_factory=forbidden_factory,
        )

        self.assertFalse(report["restored"])
        self.assertEqual(report["evaluated_queries"], 0)
        self.assertEqual(heartbeats[-1][0], "state_complete")

    def test_mid_state_failure_preserves_prior_cache_record(self) -> None:
        plan = _plan()
        cache = _FakeCache()

        class FailingOracle:
            def __init__(self) -> None:
                self.calls = 0

            def evaluate(self, _prepared, _action_key, *, continuation_config):
                del continuation_config
                query = plan.queries_by_state[0][self.calls]
                self.calls += 1
                if self.calls == 2:
                    raise RuntimeError("合成中途故障")
                return _FakeResult(query["query_key"])

        oracle = FailingOracle()
        with self.assertRaisesRegex(RuntimeError, "合成中途故障"):
            execute_state_worker(
                plan,
                0,
                cache,
                lambda _event, _details: None,
                query_timeout_s=0.1,
                runtime_factory=lambda _plan, _ordinal: (oracle, object()),
            )

        first = plan.queries_by_state[0][0]
        second = plan.queries_by_state[0][1]
        self.assertIsNotNone(cache.load(first["query_identity"]))
        self.assertIsNone(cache.load(second["query_identity"]))

    def test_query_deadline_covers_store_and_reload(self) -> None:
        plan = _plan(query_count=1)

        class OneQueryOracle:
            def evaluate(self, _prepared, _action_key, *, continuation_config):
                del _prepared, _action_key, continuation_config
                return _FakeResult(plan.queries_by_state[0][0]["query_key"])

        class SlowStoreCache(_FakeCache):
            def store(self, identity, result) -> Path:
                time.sleep(0.20)
                return super().store(identity, result)

        with self.assertRaises(QueryDeadlineExceeded):
            execute_state_worker(
                plan,
                0,
                SlowStoreCache(),
                lambda _event, _details: None,
                query_timeout_s=0.02,
                runtime_factory=lambda _plan, _ordinal: (
                    OneQueryOracle(),
                    object(),
                ),
            )

        class SlowReloadCache(_FakeCache):
            def __init__(self) -> None:
                super().__init__()
                self.load_calls = 0

            def load(self, identity):
                self.load_calls += 1
                # 前两次分别是 state 初查和 evaluate 紧邻复查，第三次是 store 后复载。
                if self.load_calls >= 3:
                    time.sleep(0.20)
                return super().load(identity)

        with self.assertRaises(QueryDeadlineExceeded):
            execute_state_worker(
                plan,
                0,
                SlowReloadCache(),
                lambda _event, _details: None,
                query_timeout_s=0.02,
                runtime_factory=lambda _plan, _ordinal: (
                    OneQueryOracle(),
                    object(),
                ),
            )


class CheckpointTest(unittest.TestCase):
    """验证父进程只有在全 state cache 完整时才发布 checkpoint。"""

    def test_checkpoint_is_published_only_after_all_queries_load(self) -> None:
        plan = _plan()
        first, second = plan.queries_by_state[0]
        cache = _FakeCache(
            {first["query_key"]: _FakeResult(first["query_key"]).to_manifest()}
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_dir = Path(directory) / "checkpoints"
            self.assertFalse(
                publish_state_checkpoint_if_complete(
                    plan,
                    EXECUTION_ID,
                    0,
                    cache,
                    checkpoint_dir,
                )
            )
            checkpoint = checkpoint_dir / "state_000.json"
            self.assertFalse(checkpoint.exists())

            cache.values[second["query_key"]] = _FakeResult(
                second["query_key"]
            ).to_manifest()
            self.assertTrue(
                publish_state_checkpoint_if_complete(
                    plan,
                    EXECUTION_ID,
                    0,
                    cache,
                    checkpoint_dir,
                )
            )
            payload = json.loads(checkpoint.read_text(encoding="utf-8"))

        self.assertEqual(payload["query_count"], 2)
        self.assertEqual(
            payload["query_keys"], [first["query_key"], second["query_key"]]
        )

    def test_stale_checkpoint_is_removed_when_cache_becomes_incomplete(self) -> None:
        plan = _plan()
        values = {
            query["query_key"]: _FakeResult(query["query_key"]).to_manifest()
            for query in plan.queries_by_state[0]
        }
        cache = _FakeCache(values)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_dir = Path(directory) / "checkpoints"
            self.assertTrue(
                publish_state_checkpoint_if_complete(
                    plan,
                    EXECUTION_ID,
                    0,
                    cache,
                    checkpoint_dir,
                )
            )
            checkpoint = checkpoint_dir / "state_000.json"
            self.assertTrue(checkpoint.exists())
            cache.values.pop(plan.queries_by_state[0][1]["query_key"])
            self.assertFalse(
                publish_state_checkpoint_if_complete(
                    plan,
                    EXECUTION_ID,
                    0,
                    cache,
                    checkpoint_dir,
                )
            )
            self.assertFalse(checkpoint.exists())


class GuardTest(unittest.TestCase):
    """验证目录、解释器命令与短 watchdog。"""

    def test_run_and_cache_must_stay_outside_frozen_tree(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frozen = root / "frozen"
            frozen.mkdir()
            with self.assertRaisesRegex(CCODRunnerError, "完全分离"):
                validate_external_paths(frozen, frozen / "run", root / "cache")
            with self.assertRaisesRegex(CCODRunnerError, "完全分离"):
                validate_external_paths(frozen, root / "run", frozen / "cache")
            resolved = validate_external_paths(
                frozen,
                root / "run",
                root / "run" / "cache",
            )
        self.assertEqual(resolved[0].name, "frozen")

    def test_input_symlink_is_rejected_before_resolve(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frozen = root / "frozen"
            alias = root / "frozen-alias"
            frozen.mkdir()
            alias.symlink_to(frozen, target_is_directory=True)
            with self.assertRaisesRegex(CCODRunnerError, "符号链接"):
                validate_external_paths(alias, root / "run", root / "cache")

    def test_query_watchdog_uses_short_real_deadline(self) -> None:
        started = time.monotonic()
        with self.assertRaises(QueryDeadlineExceeded):
            with query_deadline(0.02):
                time.sleep(0.20)
        self.assertLess(time.monotonic() - started, 0.15)

    def test_parent_state_watchdog_uses_short_timeout(self) -> None:
        process = _FakeProcess()
        outcome = monitor_worker(
            process,
            remaining_state_s=0.02,
            rss_limit_mib=6144.0,
            poll_interval_s=0.002,
            rss_reader=lambda _pid: 8.0,
        )
        self.assertEqual(outcome.violation, "state_timeout")
        self.assertTrue(process.terminated)

    def test_worker_command_starts_with_current_interpreter(self) -> None:
        import sys

        command = _worker_command(
            frozen_dir=Path("/tmp/frozen"),
            run_dir=Path("/tmp/run"),
            cache_dir=Path("/tmp/cache"),
            config_path=Path("/tmp/config.json"),
            state_ordinal=0,
            attempt=1,
            query_timeout_s=120.0,
            execution_id=EXECUTION_ID,
            runner_hash="sha256:" + "90" * 32,
            invocation=1,
        )
        self.assertEqual(command[0], sys.executable)
        parsed = build_parser().parse_args(command[2:])
        self.assertTrue(parsed.worker)
        self.assertEqual(parsed.invocation, 1)
        self.assertEqual(parsed.state_ordinal, 0)
        self.assertEqual(parsed.attempt, 1)
        self.assertEqual(parsed.query_timeout_s, 120.0)

    def test_parser_rejects_nonfinite_query_timeout(self) -> None:
        """内部 deadline 参数也不得接受 NaN/Inf。"""
        parser = build_parser()
        for raw in ("nan", "inf", "-inf"):
            with self.subTest(raw=raw), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parser.parse_args(
                        [
                            "--frozen-dir",
                            "/tmp/frozen",
                            "--run-dir",
                            "/tmp/run",
                            "--query-timeout-s",
                            raw,
                        ]
                    )

    def test_worker_final_report_requires_bounded_finite_rss(self) -> None:
        base = {
            "schema_version": "eosbench-ccod-worker-report-v1",
            "execution_id": EXECUTION_ID,
            "runner_implementation_hash": "sha256:" + "90" * 32,
            "frozen_snapshot_hash": "sha256:" + "ab" * 32,
            "invocation": 1,
            "state_ordinal": 0,
            "attempt": 1,
            "status": "success",
            "restored": False,
            "planned_queries": 1,
            "cache_hits": 1,
            "evaluated_queries": 0,
            "fresh_query_keys": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            for peak_rss in (None, 6144.1):
                payload = dict(base)
                payload["peak_rss_mib"] = peak_rss
                payload["report_hash"] = sha256_json(payload)
                path.write_text(
                    json.dumps(payload, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(CCODRunnerError, "peak_rss"):
                    _verify_worker_report(
                        path,
                        execution_id=EXECUTION_ID,
                        runner_hash="sha256:" + "90" * 32,
                        frozen_snapshot_hash="sha256:" + "ab" * 32,
                        invocation=1,
                        state_ordinal=0,
                        attempt=1,
                    )

    def test_hidden_worker_rejects_non_running_parent_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            frozen = root / "frozen"
            run = root / "run"
            frozen.mkdir()
            run.mkdir()
            parent_manifest = {
                "schema_version": "eosbench-ccod-runner-v1",
                "execution_id": EXECUTION_ID,
                "execution_identity": {
                    "runner_implementation_hash": "sha256:" + "90" * 32,
                },
                "frozen_snapshot_hash": "sha256:" + "ab" * 32,
                "invocation": 1,
                "status": "complete",
            }
            parent_manifest["manifest_hash"] = sha256_json(parent_manifest)
            (run / "execution_manifest.json").write_text(
                json.dumps(parent_manifest, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            args = SimpleNamespace(
                frozen_dir=frozen,
                run_dir=run,
                cache_dir=run / "cache",
                state_ordinal=0,
                attempt=1,
                query_timeout_s=120.0,
                execution_id=EXECUTION_ID,
                invocation=1,
                runner_hash="sha256:" + "90" * 32,
                frozen_snapshot_hash="sha256:" + "ab" * 32,
            )
            with redirect_stderr(io.StringIO()):
                returncode = _worker_main(args)
            self.assertEqual(returncode, 1)
            self.assertFalse((run / "workers").exists())

    def test_hidden_worker_rejects_wrong_invocation_without_writes(self) -> None:
        """已授权的运行清单也不能授权另一次 invocation。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            frozen = root / "frozen"
            run = root / "run"
            frozen.mkdir()
            run.mkdir()
            parent_manifest = {
                "schema_version": "eosbench-ccod-runner-v1",
                "execution_id": EXECUTION_ID,
                "execution_identity": {
                    "runner_implementation_hash": "sha256:" + "90" * 32,
                },
                "frozen_snapshot_hash": "sha256:" + "ab" * 32,
                "invocation": 2,
                "status": "running",
            }
            parent_manifest["manifest_hash"] = sha256_json(parent_manifest)
            (run / "execution_manifest.json").write_text(
                json.dumps(parent_manifest, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            args = SimpleNamespace(
                frozen_dir=frozen,
                run_dir=run,
                cache_dir=run / "cache",
                state_ordinal=0,
                attempt=1,
                query_timeout_s=120.0,
                execution_id=EXECUTION_ID,
                invocation=1,
                runner_hash="sha256:" + "90" * 32,
                frozen_snapshot_hash="sha256:" + "ab" * 32,
            )
            with redirect_stderr(io.StringIO()):
                returncode = _worker_main(args)
            self.assertEqual(returncode, 1)
            self.assertFalse((run / "workers").exists())

    def test_hidden_worker_binds_timeout_state_attempt_and_paths(self) -> None:
        """手工 worker 不得绕过冻结 deadline 或父进程授权坐标。"""
        cases = (
            ("timeout", {"query_timeout_s": 119.0}, "query timeout"),
            ("state", {"state_ordinal": 1}, "授权前缀"),
            ("attempt", {"attempt": 2}, "active"),
            ("paths", {"use_other_cache": True}, "路径"),
        )
        for name, overrides, expected_error in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                frozen = root / "frozen"
                run = root / "run"
                cache = run / "cache"
                frozen.mkdir()
                run.mkdir()
                parent_manifest = {
                    "schema_version": "eosbench-ccod-runner-v1",
                    "execution_id": EXECUTION_ID,
                    "execution_identity": {
                        "runner_implementation_hash": "sha256:" + "90" * 32,
                    },
                    "frozen_snapshot_hash": "sha256:" + "ab" * 32,
                    "paths": {
                        "frozen_dir": str(frozen),
                        "run_dir": str(run),
                        "cache_dir": str(cache),
                    },
                    "invocation": 1,
                    "state_ordinals_this_invocation": [0],
                    "state_progress": {
                        "0": {
                            "attempts_started": 1,
                            "active_attempt": 1,
                            "active_started_unix_ns": 1,
                        }
                    },
                    "status": "running",
                }
                parent_manifest["manifest_hash"] = sha256_json(parent_manifest)
                (run / "execution_manifest.json").write_text(
                    json.dumps(parent_manifest, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
                args = SimpleNamespace(
                    frozen_dir=frozen,
                    run_dir=run,
                    cache_dir=(
                        run / "other-cache"
                        if overrides.get("use_other_cache")
                        else cache
                    ),
                    state_ordinal=overrides.get("state_ordinal", 0),
                    attempt=overrides.get("attempt", 1),
                    query_timeout_s=overrides.get("query_timeout_s", 120.0),
                    execution_id=EXECUTION_ID,
                    invocation=1,
                    runner_hash="sha256:" + "90" * 32,
                    frozen_snapshot_hash="sha256:" + "ab" * 32,
                )
                stderr = io.StringIO()
                with redirect_stderr(stderr):
                    returncode = _worker_main(args)
                self.assertEqual(returncode, 1)
                self.assertIn(expected_error, stderr.getvalue())
                self.assertFalse((run / "workers").exists())

    def test_invocation_evidence_paths_coexist_and_refuse_overwrite(self) -> None:
        """跨 invocation 证据共存，同坐标 journal 不得覆盖。"""
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory).resolve()
            self.assertNotEqual(
                _worker_report_path(run, 1, 0, 1),
                _worker_report_path(run, 2, 0, 1),
            )
            for invocation in (1, 2):
                emit = _heartbeat_journal(
                    run,
                    execution_id=EXECUTION_ID,
                    invocation=invocation,
                    state_ordinal=0,
                    attempt=1,
                )
                emit("state_start", {})
            with self.assertRaisesRegex(CCODRunnerError, "拒绝覆盖"):
                _heartbeat_journal(
                    run,
                    execution_id=EXECUTION_ID,
                    invocation=1,
                    state_ordinal=0,
                    attempt=1,
                )
            journals = sorted((run / "workers").glob("*.journal.jsonl"))
            self.assertEqual(len(journals), 2)
            self.assertIn("invocation_001", journals[0].name)
            self.assertIn("invocation_002", journals[1].name)

    def test_broken_worker_symlink_cannot_escape_run_tree(self) -> None:
        """即使链接目标不存在，worker 目录也不得把 journal 引到树外。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            run = root / "run"
            run.mkdir()
            outside = root / "outside" / "missing-workers"
            (run / "workers").symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(CCODRunnerError, "符号链接"):
                _heartbeat_journal(
                    run,
                    execution_id=EXECUTION_ID,
                    invocation=1,
                    state_ordinal=0,
                    attempt=1,
                )
            self.assertFalse(outside.exists())

    def test_parent_interrupt_reaps_spawned_worker(self) -> None:
        """监控阶段中断必须终止并回收 worker，不能留给下一 invocation。"""
        process = mock.Mock()
        process.poll.return_value = None
        process.wait.return_value = 0
        process.communicate.return_value = ("", "")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            run = root / "run"
            run.mkdir()
            with mock.patch(
                "scripts.run_ccod_diagnostic.subprocess.Popen",
                return_value=process,
            ), mock.patch(
                "scripts.run_ccod_diagnostic.monitor_worker",
                side_effect=KeyboardInterrupt,
            ):
                with self.assertRaises(KeyboardInterrupt):
                    _launch_state_attempt(
                        frozen_dir=root / "frozen",
                        run_dir=run,
                        cache_dir=root / "cache",
                        config_path=root / "config.json",
                        invocation=1,
                        state_ordinal=0,
                        attempt=1,
                        remaining_state_s=1.0,
                        execution_id=EXECUTION_ID,
                        runner_hash="sha256:" + "90" * 32,
                        frozen_snapshot_hash="sha256:" + "ab" * 32,
                    )
        process.terminate.assert_called_once_with()
        process.wait.assert_called()
        process.communicate.assert_called_with(timeout=2.0)

    def test_worker_report_rejects_wrong_invocation(self) -> None:
        """final report 必须绑定当前 invocation。"""
        payload = {
            "schema_version": "eosbench-ccod-worker-report-v1",
            "execution_id": EXECUTION_ID,
            "runner_implementation_hash": "sha256:" + "90" * 32,
            "frozen_snapshot_hash": "sha256:" + "ab" * 32,
            "invocation": 2,
            "state_ordinal": 0,
            "attempt": 1,
            "status": "success",
            "restored": False,
            "planned_queries": 1,
            "cache_hits": 1,
            "evaluated_queries": 0,
            "fresh_query_keys": [],
            "peak_rss_mib": 64.0,
        }
        payload["report_hash"] = sha256_json(payload)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(
                json.dumps(payload, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(CCODRunnerError, "身份冲突"):
                _verify_worker_report(
                    path,
                    execution_id=EXECUTION_ID,
                    runner_hash="sha256:" + "90" * 32,
                    frozen_snapshot_hash="sha256:" + "ab" * 32,
                    invocation=1,
                    state_ordinal=0,
                    attempt=1,
                )

    def test_worker_report_requires_full_schema_and_provenance(self) -> None:
        """哈希正确也不能接受缺字段、额外字段或错误来源。"""
        base = {
            "schema_version": "eosbench-ccod-worker-report-v1",
            "execution_id": EXECUTION_ID,
            "runner_implementation_hash": "sha256:" + "90" * 32,
            "frozen_snapshot_hash": "sha256:" + "ab" * 32,
            "invocation": 1,
            "state_ordinal": 0,
            "attempt": 1,
            "status": "success",
            "restored": False,
            "planned_queries": 1,
            "cache_hits": 1,
            "evaluated_queries": 0,
            "fresh_query_keys": [],
            "peak_rss_mib": 64.0,
        }
        mutations = (
            ("missing_schema", lambda row: row.pop("schema_version")),
            (
                "wrong_runner",
                lambda row: row.__setitem__(
                    "runner_implementation_hash", "sha256:" + "91" * 32
                ),
            ),
            (
                "wrong_snapshot",
                lambda row: row.__setitem__(
                    "frozen_snapshot_hash", "sha256:" + "ac" * 32
                ),
            ),
            ("extra", lambda row: row.__setitem__("unexpected", True)),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            for name, mutate in mutations:
                with self.subTest(name=name):
                    payload = dict(base)
                    mutate(payload)
                    payload["report_hash"] = sha256_json(payload)
                    path.write_text(
                        json.dumps(payload, ensure_ascii=False) + "\n",
                        encoding="utf-8",
                    )
                    with self.assertRaises(CCODRunnerError):
                        _verify_worker_report(
                            path,
                            execution_id=EXECUTION_ID,
                            runner_hash="sha256:" + "90" * 32,
                            frozen_snapshot_hash="sha256:" + "ab" * 32,
                            invocation=1,
                            state_ordinal=0,
                            attempt=1,
                        )


class ParentOrchestrationTest(unittest.TestCase):
    """直接调用父进程入口，锁定默认限额、只读验收与跨恢复 attempt。"""

    def _patch_parent(self, plan, cache):
        """安装不触发真实 verifier/oracle 的父进程测试边界。"""
        stack = ExitStack()
        runner_hash = "sha256:" + "90" * 32
        identity = {
            "schema_version": "synthetic-execution-v1",
            "runner_implementation_hash": runner_hash,
        }

        def summarize(_plan, _identity, rows):
            rows = list(rows)
            successful = [row for row in rows if row.get("status") == "success"]
            failed = [row for row in rows if row.get("status") == "failed"]
            invalid = any(
                row.get("failure_kind") in INVALID_FAILURE_KINDS
                for row in failed
            )
            complete = len(successful) == len(plan.query_rows)
            return {
                "execution_status": (
                    "invalid" if invalid else ("complete" if complete else "incomplete")
                ),
                "completed_states": len(successful),
                "completed_queries": len(successful),
                "signal_gate": "fail" if complete else "not_evaluated",
                "method_decision": "no_go" if complete else None,
            }

        stack.enter_context(
            mock.patch(
                "scripts.prepare_ccod_diagnostic.verify_frozen_artifacts",
                return_value={"status": "verified"},
            )
        )
        stack.enter_context(
            mock.patch(
                "scripts.run_ccod_diagnostic.runner_implementation_hash",
                return_value=runner_hash,
            )
        )
        stack.enter_context(
            mock.patch(
                "scripts.run_ccod_diagnostic._frozen_snapshot",
                return_value={"snapshot_hash": "sha256:" + "ab" * 32},
            )
        )
        stack.enter_context(
            mock.patch(
                "algorithms.ccod.cache.CounterfactualLabelCache",
                return_value=cache,
            )
        )
        stack.enter_context(
            mock.patch(
                "algorithms.ccod.execution.load_frozen_diagnostic_plan",
                return_value=plan,
            )
        )
        stack.enter_context(
            mock.patch(
                "algorithms.ccod.execution.build_execution_identity",
                return_value=identity,
            )
        )
        stack.enter_context(
            mock.patch(
                "algorithms.ccod.execution.execution_identity_hash",
                return_value=EXECUTION_ID,
            )
        )
        stack.enter_context(
            mock.patch(
                "algorithms.ccod.execution.summarize_signal_gate",
                side_effect=summarize,
            )
        )
        launcher = stack.enter_context(
            mock.patch("scripts.run_ccod_diagnostic._launch_state_attempt")
        )
        return stack, launcher

    def test_parser_defaults_to_one_new_state_and_supports_verify_only(self) -> None:
        parser = build_parser()
        default = parser.parse_args(
            ["--frozen-dir", "/tmp/frozen", "--run-dir", "/tmp/run"]
        )
        self.assertEqual(default.max_new_states, 1)
        self.assertIsNone(default.max_states)
        self.assertFalse(default.verify_only)
        readonly = parser.parse_args(
            [
                "--frozen-dir",
                "/tmp/frozen",
                "--run-dir",
                "/tmp/run",
                "--verify-only",
            ]
        )
        self.assertTrue(readonly.verify_only)

    def test_verify_only_is_strictly_read_only(self) -> None:
        plan = _plan(query_count=1)
        cache = _FakeCache()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            frozen = root / "frozen"
            run = root / "run"
            frozen.mkdir()
            stack, launcher = self._patch_parent(plan, cache)
            with stack:
                result = run_diagnostic(
                    frozen_dir=frozen,
                    run_dir=run,
                    verify_only=True,
                )
            self.assertFalse(run.exists())
        launcher.assert_not_called()
        self.assertTrue(result["verify_only"])
        self.assertEqual(result["writes_performed"], 0)

    def test_verify_only_rejects_end_of_read_drift(self) -> None:
        """只读 cache 扫描结束也必须重新绑定同一冻结快照。"""
        plan = _plan(query_count=1)
        cache = _FakeCache()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            frozen = root / "frozen"
            run = root / "run"
            frozen.mkdir()
            stack, launcher = self._patch_parent(plan, cache)
            with stack, mock.patch(
                "scripts.run_ccod_diagnostic._frozen_snapshot",
                side_effect=[
                    {"snapshot_hash": "sha256:" + "ab" * 32},
                    {"snapshot_hash": "sha256:" + "ac" * 32},
                ],
            ):
                with self.assertRaisesRegex(CCODRunnerError, "冻结包"):
                    run_diagnostic(
                        frozen_dir=frozen,
                        run_dir=run,
                        verify_only=True,
                    )
            self.assertFalse(run.exists())
        launcher.assert_not_called()

    def test_all_cache_hits_skip_parent_spawn(self) -> None:
        plan = _plan(query_count=2)
        cache = _FakeCache(
            {
                query["query_key"]: _FakeResult(query["query_key"]).to_manifest()
                for query in plan.query_rows
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            frozen = root / "frozen"
            run = root / "run"
            frozen.mkdir()
            stack, launcher = self._patch_parent(plan, cache)
            with stack:
                result = run_diagnostic(frozen_dir=frozen, run_dir=run)
        launcher.assert_not_called()
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["newly_started_states"], 0)

    def test_post_run_drift_suppresses_complete_gate(self) -> None:
        plan = _plan(query_count=1)
        query = plan.query_rows[0]
        cache = _FakeCache(
            {
                query["query_key"]: _FakeResult(query["query_key"]).to_manifest()
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            frozen = root / "frozen"
            run = root / "run"
            frozen.mkdir()
            stack, launcher = self._patch_parent(plan, cache)
            with stack, mock.patch(
                "scripts.run_ccod_diagnostic._frozen_snapshot",
                side_effect=[
                    {"snapshot_hash": "sha256:" + "ab" * 32},
                    {"snapshot_hash": "sha256:" + "cd" * 32},
                ],
            ):
                result = run_diagnostic(frozen_dir=frozen, run_dir=run)
                summary = json.loads(
                    (run / "signal_summary.json").read_text(encoding="utf-8")
                )
        launcher.assert_not_called()
        self.assertEqual(result["status"], "invalid")
        self.assertEqual(result["signal_gate"], "not_evaluated")
        self.assertIsNone(result["method_decision"])
        self.assertEqual(summary["execution_status"], "invalid")

    def test_drift_during_gate_write_suppresses_complete_manifest(self) -> None:
        """初次复核通过后发生的漂移也不能留下 complete manifest。"""
        plan = _plan(query_count=1)
        query = plan.query_rows[0]
        cache = _FakeCache(
            {query["query_key"]: _FakeResult(query["query_key"]).to_manifest()}
        )
        same = {"snapshot_hash": "sha256:" + "ab" * 32}
        changed = {"snapshot_hash": "sha256:" + "ac" * 32}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            frozen = root / "frozen"
            run = root / "run"
            frozen.mkdir()
            stack, launcher = self._patch_parent(plan, cache)
            with stack, mock.patch(
                "scripts.run_ccod_diagnostic._frozen_snapshot",
                side_effect=[same, same, changed],
            ):
                result = run_diagnostic(frozen_dir=frozen, run_dir=run)
        launcher.assert_not_called()
        self.assertEqual(result["status"], "invalid")
        self.assertEqual(result["signal_gate"], "not_evaluated")

    def test_invalid_run_directory_cannot_be_resumed(self) -> None:
        """invalid 是粘性终态，修复后必须换 run-dir 保留原始证据。"""
        plan = _plan(query_count=1)
        cache = _FakeCache()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            frozen = root / "frozen"
            run = root / "run"
            frozen.mkdir()
            stack, launcher = self._patch_parent(plan, cache)
            launcher.side_effect = RunnerAttemptError(
                "identity_mismatch",
                "合成身份冲突",
            )
            with stack:
                first = run_diagnostic(frozen_dir=frozen, run_dir=run)
            self.assertEqual(first["status"], "invalid")

            stack, second_launcher = self._patch_parent(plan, cache)
            with stack, self.assertRaisesRegex(CCODRunnerError, "新的 run-dir"):
                run_diagnostic(frozen_dir=frozen, run_dir=run)
        self.assertEqual(launcher.call_count, 1)
        second_launcher.assert_not_called()

    def test_default_starts_only_first_incomplete_state(self) -> None:
        plan = _multi_plan(2)
        cache = _FakeCache()

        def finish_state(**kwargs):
            state_ordinal = kwargs["state_ordinal"]
            query = plan.queries_by_state[state_ordinal][0]
            cache.values[query["query_key"]] = _FakeResult(
                query["query_key"]
            ).to_manifest()
            return {
                "state_ordinal": state_ordinal,
                "peak_rss_mib": 1.0,
            }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            frozen = root / "frozen"
            run = root / "run"
            frozen.mkdir()
            stack, launcher = self._patch_parent(plan, cache)
            launcher.side_effect = finish_state
            with stack:
                result = run_diagnostic(frozen_dir=frozen, run_dir=run)
        self.assertEqual(launcher.call_count, 1)
        self.assertEqual(launcher.call_args.kwargs["state_ordinal"], 0)
        self.assertEqual(result["newly_started_states"], 1)
        self.assertEqual(result["completed_states"], 1)
        self.assertEqual(result["status"], "incomplete")

    def test_attempt_budget_exhaustion_persists_as_incomplete(self) -> None:
        plan = _plan(query_count=1)
        cache = _FakeCache()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            frozen = root / "frozen"
            run = root / "run"
            frozen.mkdir()
            stack, launcher = self._patch_parent(plan, cache)
            launcher.side_effect = RunnerAttemptError(
                "worker_error",
                "合成 worker 故障",
            )
            with stack:
                first = run_diagnostic(frozen_dir=frozen, run_dir=run)
            self.assertEqual(launcher.call_count, 2)
            self.assertEqual(
                first["state_progress"]["0"]["attempts_started"],
                2,
            )

            stack, second_launcher = self._patch_parent(plan, cache)
            with stack:
                second = run_diagnostic(frozen_dir=frozen, run_dir=run)
        second_launcher.assert_not_called()
        self.assertEqual(second["invocation"], 2)
        self.assertEqual(first["status"], "incomplete")
        self.assertEqual(second["status"], "incomplete")

    def test_absolute_prefix_allows_hot_cache_rerun_without_next_state(self) -> None:
        """同一单 state 前缀热跑不得顺带启动 state 1。"""
        plan = _multi_plan(2)
        cache = _FakeCache()

        def finish_state(**kwargs):
            state_ordinal = kwargs["state_ordinal"]
            query = plan.queries_by_state[state_ordinal][0]
            cache.values[query["query_key"]] = _FakeResult(
                query["query_key"]
            ).to_manifest()
            return {"state_ordinal": state_ordinal, "peak_rss_mib": 1.0}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            frozen = root / "frozen"
            run = root / "run"
            frozen.mkdir()
            stack, launcher = self._patch_parent(plan, cache)
            launcher.side_effect = finish_state
            with stack:
                first = run_diagnostic(
                    frozen_dir=frozen,
                    run_dir=run,
                    max_states=1,
                )
            self.assertEqual(launcher.call_count, 1)

            stack, hot_launcher = self._patch_parent(plan, cache)
            with stack:
                second = run_diagnostic(
                    frozen_dir=frozen,
                    run_dir=run,
                    max_states=1,
                )
        hot_launcher.assert_not_called()
        self.assertEqual(first["completed_states"], 1)
        self.assertEqual(second["invocation"], 2)
        self.assertEqual(second["newly_started_states"], 0)
        self.assertEqual(second["fresh_queries_this_invocation"], 0)
        self.assertEqual(second["state_ordinals_this_invocation"], [0])

    def test_interrupted_active_attempt_consumes_elapsed_budget(self) -> None:
        existing = {
            "state_progress": {
                "0": {
                    "attempts_started": 1,
                    "elapsed_s_hex": 1.0.hex(),
                    "active_attempt": 1,
                    "active_started_unix_ns": 1_000_000_000,
                    "last_failure_kind": None,
                }
            }
        }
        restored = _restore_state_progress(
            existing,
            1,
            now_unix_ns=3_000_000_000,
        )["0"]
        self.assertEqual(float.fromhex(restored["elapsed_s_hex"]), 3.0)
        self.assertEqual(restored["last_failure_kind"], "interrupted")
        self.assertIsNone(restored["active_attempt"])

    def test_run_lock_rejects_second_parent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory).resolve()
            first = _acquire_run_lock(run, EXECUTION_ID)
            try:
                with self.assertRaisesRegex(CCODRunnerError, "另一个"):
                    _acquire_run_lock(run, EXECUTION_ID)
            finally:
                first.close()
            second = _acquire_run_lock(run, EXECUTION_ID)
            second.close()


class FrozenPlanIntegrationTest(unittest.TestCase):
    """用正式无标签包锁定 prefix、identity 与 partial/invalid 语义。"""

    @classmethod
    def setUpClass(cls) -> None:
        from algorithms.ccod.execution import (
            build_execution_identity,
            execution_identity_hash,
            load_frozen_diagnostic_plan,
        )

        root = (
            Path(__file__).resolve().parents[1]
            / "output/ccod_diagnostic_v1_py310_final_9d74e46"
        )
        cls.plan = load_frozen_diagnostic_plan(root)
        cls.identity = build_execution_identity(
            cls.plan,
            runner_implementation_hash="sha256:" + "90" * 32,
        )
        cls.execution_id = execution_identity_hash(cls.identity)

    def test_first_twelve_states_have_179_queries_without_identity_change(self) -> None:
        from algorithms.ccod.execution import build_execution_identity

        ordinals, queries = execution_prefix(self.plan, 12)
        full_ordinals, full_queries = execution_prefix(self.plan, None)
        self.assertEqual(ordinals, tuple(range(12)))
        self.assertEqual(len(queries), 179)
        self.assertEqual(len(full_ordinals), 100)
        self.assertEqual(len(full_queries), 1570)
        after_prefix = build_execution_identity(
            self.plan,
            runner_implementation_hash="sha256:" + "90" * 32,
        )
        self.assertEqual(after_prefix, self.identity)
        self.plan.assert_pristine()

    def test_max_states_partial_remains_incomplete(self) -> None:
        from algorithms.ccod.execution import summarize_signal_gate

        _, queries = execution_prefix(self.plan, 12)
        self.assertEqual(len(queries), 179)
        summary = summarize_signal_gate(self.plan, self.identity, [])
        self.assertEqual(summary["execution_status"], "incomplete")
        self.assertEqual(summary["signal_gate"], "not_evaluated")
        self.assertEqual(summary["failed_queries"], 0)

    def test_attempt_exhaustion_marks_missing_rows_incomplete(self) -> None:
        from algorithms.ccod.execution import summarize_signal_gate

        cache = _FakeCache()
        failures = _terminal_missing_queries(
            self.plan,
            cache,
            0,
            "attempt_exhausted",
        )
        rows = _result_rows_from_cache(
            self.plan,
            cache,
            self.execution_id,
            set(),
            failures,
        )
        failed = [row for row in rows if row["status"] == "failed"]
        self.assertEqual(len(failed), len(self.plan.queries_by_state[0]))
        self.assertTrue(all(row["attempts_exhausted"] == 2 for row in failed))
        summary = summarize_signal_gate(self.plan, self.identity, rows)
        self.assertEqual(summary["execution_status"], "incomplete")
        self.assertEqual(summary["signal_gate"], "not_evaluated")
        self.assertEqual(summary["failed_queries"], len(failed))


if __name__ == "__main__":
    unittest.main()
