# -*- coding: utf-8 -*-
"""CCOD 单机 runner 的 cache-first、checkpoint 与护栏单元测试。"""

from __future__ import annotations

from contextlib import ExitStack, redirect_stderr
import hashlib
import io
import json
import os
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
    MAX_CACHE_RECORD_BYTES,
    QueryDeadlineExceeded,
    RunnerAttemptError,
    _acquire_run_lock,
    _assert_parent_lease,
    _heartbeat_journal,
    _open_label_cache,
    _real_runtime_factory,
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


def _synthetic_parent_lease() -> dict:
    """构造仅供命令组装测试使用的完整父锁租约。"""
    payload = {
        "schema_version": "eosbench-ccod-parent-lease-v1",
        "execution_id": EXECUTION_ID,
        "parent_pid": 12345,
        "lease_id": "ab" * 32,
        "lock_device": 1,
        "lock_inode": 2,
        "acquired_unix_ns": 3,
    }
    payload["lease_hash"] = sha256_json(payload)
    return payload


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

    def test_released_parent_lease_blocks_cache_publication(self) -> None:
        """父锁在 evaluate 后释放时，结果不得进入权威 cache。"""
        plan = _plan(query_count=1)
        cache = _FakeCache()
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory).resolve()
            lease = _acquire_run_lock(run, EXECUTION_ID)

            def assert_live() -> None:
                _assert_parent_lease(
                    run,
                    EXECUTION_ID,
                    lease.payload,
                    require_direct_parent=False,
                )

            # 先证明同一租约在锁仍存活时可以通过。
            assert_live()

            class LeaseDroppingOracle:
                def evaluate(self, _prepared, _action_key, *, continuation_config):
                    del _prepared, _action_key, continuation_config
                    lease.close()
                    return _FakeResult(plan.query_rows[0]["query_key"])

            with self.assertRaisesRegex(RunnerAttemptError, "释放运行锁"):
                execute_state_worker(
                    plan,
                    0,
                    cache,
                    lambda _event, _details: None,
                    query_timeout_s=0.5,
                    runtime_factory=lambda _plan, _ordinal: (
                        LeaseDroppingOracle(),
                        object(),
                    ),
                    authorization_check=assert_live,
                )
        self.assertEqual(cache.values, {})


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

    def test_cache_shard_symlink_cannot_write_frozen_tree(self) -> None:
        """cache 两位 shard 即使预置链接也不得逃逸到冻结目录。"""
        from algorithms.ccod.continuation import CounterfactualError

        identity = _query(0, "sha256:" + "56" * 32)["query_identity"]
        key = sha256_json(identity)
        shard = key.removeprefix("sha256:")[:2]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            frozen = root / "frozen"
            cache_root = root / "cache"
            frozen.mkdir()
            cache_root.mkdir()
            sentinel = frozen / "sentinel.txt"
            sentinel.write_text("冻结", encoding="utf-8")
            (cache_root / shard).symlink_to(frozen, target_is_directory=True)
            before = {
                path.relative_to(frozen).as_posix(): path.read_bytes()
                for path in frozen.rglob("*")
                if path.is_file()
            }
            cache = _open_label_cache(cache_root)
            with mock.patch(
                "algorithms.ccod.cache._validate_result_identity",
                return_value=None,
            ):
                with self.assertRaises(CounterfactualError):
                    cache.store(identity, _FakeResult(key))
            after = {
                path.relative_to(frozen).as_posix(): path.read_bytes()
                for path in frozen.rglob("*")
                if path.is_file()
            }
        self.assertEqual(after, before)

    def test_cache_load_rejects_fifo_and_oversized_record(self) -> None:
        """不可信 record 不能用 FIFO 阻塞父进程，也不能无界占用内存。"""
        from algorithms.ccod.continuation import CounterfactualError

        identity = _query(0, "sha256:" + "56" * 32)["query_identity"]
        key = sha256_json(identity).removeprefix("sha256:")
        shard, filename = key[:2], f"{key}.json"
        with tempfile.TemporaryDirectory() as directory:
            cache_root = Path(directory).resolve() / "cache"
            shard_dir = cache_root / shard
            shard_dir.mkdir(parents=True)
            record = shard_dir / filename
            os.mkfifo(record)
            cache = _open_label_cache(cache_root)
            with self.assertRaisesRegex(CounterfactualError, "普通文件"):
                cache.load(identity)
            record.unlink()
            with record.open("wb") as handle:
                handle.truncate(MAX_CACHE_RECORD_BYTES + 1)
            with self.assertRaisesRegex(CounterfactualError, "普通文件"):
                cache.load(identity)

    def test_secure_cache_roundtrip_uses_frozen_record_schema(self) -> None:
        """安全 cache 仍须兼容冻结记录格式并能完成原子读写。"""
        identity = _query(0, "sha256:" + "56" * 32)["query_identity"]
        key = sha256_json(identity)
        payload = {"query_key": key, "q_h_hex": 0.25.hex()}
        payload["result_hash"] = sha256_json(payload)
        result = SimpleNamespace(to_manifest=lambda: dict(payload))
        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "algorithms.ccod.cache._validate_result_identity",
            return_value=None,
        ):
            cache = _open_label_cache(Path(directory) / "cache")
            path = cache.store(identity, result)
            self.assertTrue(path.is_file())
            self.assertEqual(cache.load(identity), payload)
            self.assertEqual(cache.store(identity, result), path)

    def test_cache_root_symlink_inserted_after_open_is_rejected(self) -> None:
        """构造 cache 视图后替换根目录也不能把记录写入外部目录。"""
        from algorithms.ccod.continuation import CounterfactualError

        identity = _query(0, "sha256:" + "56" * 32)["query_identity"]
        key = sha256_json(identity)
        payload = {"query_key": key, "q_h_hex": 0.25.hex()}
        payload["result_hash"] = sha256_json(payload)
        result = SimpleNamespace(to_manifest=lambda: dict(payload))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            cache_root = root / "cache"
            outside = root / "outside"
            outside.mkdir()
            cache = _open_label_cache(cache_root)
            cache_root.symlink_to(outside, target_is_directory=True)
            with mock.patch(
                "algorithms.ccod.cache._validate_result_identity",
                return_value=None,
            ), self.assertRaises((CCODRunnerError, CounterfactualError)):
                cache.store(identity, result)
            self.assertEqual(list(outside.iterdir()), [])

    def test_cache_factory_rejects_preexisting_root_symlink(self) -> None:
        """直接调用 cache 工厂也不能让 resolve 吞掉已有根链接。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            outside = root / "outside"
            alias = root / "cache-alias"
            outside.mkdir()
            alias.symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(CCODRunnerError, "根目录不允许符号链接"):
                _open_label_cache(alias)

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
            parent_pid=12345,
            parent_lease_id="ab" * 32,
        )
        self.assertEqual(command[0], sys.executable)
        parsed = build_parser().parse_args(command[2:])
        self.assertTrue(parsed.worker)
        self.assertEqual(parsed.invocation, 1)
        self.assertEqual(parsed.state_ordinal, 0)
        self.assertEqual(parsed.attempt, 1)
        self.assertEqual(parsed.query_timeout_s, 120.0)
        self.assertEqual(parsed.parent_pid, 12345)
        self.assertEqual(parsed.parent_lease_id, "ab" * 32)

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

    def test_runtime_replay_identity_error_is_not_recoverable(self) -> None:
        """冻结 state/trace 缺损不得被降级成可重试 worker_error。"""
        plan = SimpleNamespace(
            selected_states=({},),
            frozen_dir=Path("/tmp/synthetic-frozen"),
        )
        with self.assertRaises(RunnerAttemptError) as raised:
            _real_runtime_factory(plan, 0)
        self.assertEqual(raised.exception.failure_kind, "identity_mismatch")

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

    def test_hidden_worker_requires_a_live_parent_lock(self) -> None:
        """遗留的 running manifest 与租约字节不能替代仍被持有的 flock。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            frozen = root / "frozen"
            run = root / "run"
            cache = run / "cache"
            frozen.mkdir()
            run.mkdir()
            probe = _acquire_run_lock(run, EXECUTION_ID)
            probe.close()
            lock_stat = (run / ".runner.lock").stat()
            lease = {
                "schema_version": "eosbench-ccod-parent-lease-v1",
                "execution_id": EXECUTION_ID,
                "parent_pid": os.getppid(),
                "lease_id": "cd" * 32,
                "lock_device": int(lock_stat.st_dev),
                "lock_inode": int(lock_stat.st_ino),
                "acquired_unix_ns": 1,
            }
            lease["lease_hash"] = sha256_json(lease)
            (run / ".runner.lock").write_text(
                json.dumps(lease, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
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
                "parent_lease": lease,
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
                cache_dir=cache,
                state_ordinal=0,
                attempt=1,
                query_timeout_s=120.0,
                execution_id=EXECUTION_ID,
                invocation=1,
                runner_hash="sha256:" + "90" * 32,
                frozen_snapshot_hash="sha256:" + "ab" * 32,
                parent_pid=lease["parent_pid"],
                parent_lease_id=lease["lease_id"],
            )
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                returncode = _worker_main(args)
            self.assertEqual(returncode, 1)
            self.assertIn("释放运行锁", stderr.getvalue())
            self.assertFalse((run / "workers").exists())
            self.assertFalse(cache.exists())

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
                        parent_lease=_synthetic_parent_lease(),
                        expected_query_keys=("sha256:" + "11" * 32,),
                    )
        process.terminate.assert_called_once_with()
        process.wait.assert_called()
        process.communicate.assert_called_with(timeout=2.0)

    def test_nonzero_exit_cannot_be_hidden_by_success_report(self) -> None:
        """进程异常退出时，即使 report 自哈希正确也不能判为成功。"""
        process = mock.Mock()
        process.pid = 12345
        lease = _synthetic_parent_lease()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            run = root / "run"
            run.mkdir()
            report_path = _worker_report_path(run, 1, 0, 1)

            def finish_worker(_process, **_kwargs):
                payload = {
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
                    "peak_rss_mib": 32.0,
                }
                payload["report_hash"] = sha256_json(payload)
                report_path.parent.mkdir(parents=True)
                report_path.write_text(
                    json.dumps(payload, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
                return SimpleNamespace(
                    violation=None,
                    returncode=7,
                    stderr="",
                )

            with mock.patch(
                "scripts.run_ccod_diagnostic.subprocess.Popen",
                return_value=process,
            ), mock.patch(
                "scripts.run_ccod_diagnostic.monitor_worker",
                side_effect=finish_worker,
            ), self.assertRaisesRegex(RunnerAttemptError, "退出码非零"):
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
                    parent_lease=lease,
                    expected_query_keys=("sha256:" + "11" * 32,),
                )

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
                "scripts.run_ccod_diagnostic._open_label_cache",
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

    def test_verify_only_cache_corruption_is_read_only(self) -> None:
        """只读验收遇到坏 cache 必须失败，且不能创建 run-dir。"""
        from algorithms.ccod.continuation import CounterfactualError

        plan = _plan(query_count=1)

        class CorruptCache(_FakeCache):
            def load(self, identity):
                del identity
                raise CounterfactualError("合成坏 cache record")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            frozen = root / "frozen"
            run = root / "run"
            frozen.mkdir()
            stack, launcher = self._patch_parent(plan, CorruptCache())
            with stack, self.assertRaisesRegex(RunnerAttemptError, "合成坏"):
                run_diagnostic(
                    frozen_dir=frozen,
                    run_dir=run,
                    verify_only=True,
                )
            self.assertFalse(run.exists())
        launcher.assert_not_called()

    def test_cache_corruption_becomes_sticky_invalid_after_lock(self) -> None:
        """普通运行的坏 cache 必须落盘为 invalid，修复后也不能洗白。"""
        from algorithms.ccod.continuation import CounterfactualError

        plan = _plan(query_count=1)

        class ToggleCache(_FakeCache):
            corrupt = True

            def load(self, identity):
                if self.corrupt:
                    raise CounterfactualError("合成坏 cache record")
                return super().load(identity)

        cache = ToggleCache()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            frozen = root / "frozen"
            run = root / "run"
            frozen.mkdir()
            stack, launcher = self._patch_parent(plan, cache)
            with stack:
                first = run_diagnostic(frozen_dir=frozen, run_dir=run)
            launcher.assert_not_called()
            self.assertEqual(first["status"], "invalid")
            self.assertIn("cache_corrupt", first["failure"])
            summary = json.loads(
                (run / "signal_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["execution_status"], "invalid")
            self.assertEqual(summary["failure_counts"], {"cache_corrupt": 1})

            cache.corrupt = False
            stack, second_launcher = self._patch_parent(plan, cache)
            with stack, self.assertRaisesRegex(CCODRunnerError, "新的 run-dir"):
                run_diagnostic(frozen_dir=frozen, run_dir=run)
        second_launcher.assert_not_called()

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
            evidence = result["artifact_evidence"]
            query_path = run / "query_results.jsonl"
            summary_path = run / "signal_summary.json"
            self.assertEqual(evidence["query_results"]["row_count"], 2)
            self.assertEqual(
                evidence["query_results"]["sha256"],
                "sha256:" + hashlib.sha256(query_path.read_bytes()).hexdigest(),
            )
            summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(
                evidence["signal_summary"]["sha256"],
                "sha256:" + hashlib.sha256(summary_path.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                evidence["signal_summary"]["summary_hash"],
                summary_payload["summary_hash"],
            )
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
                side_effect=[same, same, same, changed],
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
            evidence_roots = [
                run / "evidence" / f"invocation_{value:03d}"
                for value in (1, 2)
            ]
            for evidence_root in evidence_roots:
                self.assertTrue((evidence_root / "evidence.json").is_file())
                self.assertTrue((evidence_root / "query_results.jsonl").is_file())
                self.assertTrue((evidence_root / "signal_summary.json").is_file())
            self.assertNotEqual(
                (evidence_roots[0] / "query_results.jsonl").read_bytes(),
                (evidence_roots[1] / "query_results.jsonl").read_bytes(),
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

    def test_run_lock_releases_after_progress_restore_error(self) -> None:
        """持锁区早期异常即使保留 traceback，也必须立即释放 flock。"""
        plan = _plan(query_count=1)
        cache = _FakeCache()
        retained = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            frozen = root / "frozen"
            run = root / "run"
            frozen.mkdir()
            stack, launcher = self._patch_parent(plan, cache)
            with stack, mock.patch(
                "scripts.run_ccod_diagnostic._restore_state_progress",
                side_effect=RuntimeError("合成进度恢复故障"),
            ):
                try:
                    run_diagnostic(frozen_dir=frozen, run_dir=run)
                except RuntimeError as exc:
                    retained.append(exc)
            launcher.assert_not_called()
            self.assertIsNotNone(retained[0].__traceback__)
            probe = _acquire_run_lock(run, EXECUTION_ID)
            probe.close()

    def test_run_lock_releases_after_final_manifest_write_error(self) -> None:
        """收尾发布异常不能让 traceback 继续占有父进程锁。"""
        import importlib

        runner_module = importlib.import_module("scripts.run_ccod_diagnostic")
        real_write_json = runner_module._write_json
        plan = _plan(query_count=1)
        query = plan.query_rows[0]
        cache = _FakeCache(
            {query["query_key"]: _FakeResult(query["query_key"]).to_manifest()}
        )
        retained = []

        def fail_final_manifest(path, payload, hash_field):
            if (
                Path(path).name == "execution_manifest.json"
                and payload.get("status") != "running"
            ):
                raise RuntimeError("合成最终清单写入故障")
            return real_write_json(path, payload, hash_field)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            frozen = root / "frozen"
            run = root / "run"
            frozen.mkdir()
            stack, launcher = self._patch_parent(plan, cache)
            with stack, mock.patch(
                "scripts.run_ccod_diagnostic._write_json",
                side_effect=fail_final_manifest,
            ):
                try:
                    run_diagnostic(frozen_dir=frozen, run_dir=run)
                except RuntimeError as exc:
                    retained.append(exc)
            launcher.assert_not_called()
            self.assertIsNotNone(retained[0].__traceback__)
            probe = _acquire_run_lock(run, EXECUTION_ID)
            probe.close()


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
        self.assertTrue(all(row["attempts_started"] == 2 for row in failed))
        self.assertTrue(all(row["attempt_budget"] == 2 for row in failed))
        self.assertTrue(
            all(not row["state_budget_exhausted"] for row in failed)
        )
        summary = summarize_signal_gate(self.plan, self.identity, rows)
        self.assertEqual(summary["execution_status"], "incomplete")
        self.assertEqual(summary["signal_gate"], "not_evaluated")
        self.assertEqual(summary["failed_queries"], len(failed))

    def test_state_deadline_records_only_the_attempt_actually_started(self) -> None:
        """一次 attempt 耗尽 state wall-time 时不得伪造第二次进程启动。"""
        from algorithms.ccod.execution import summarize_signal_gate

        cache = _FakeCache()
        failures = _terminal_missing_queries(
            self.plan,
            cache,
            0,
            "state_timeout",
            attempts_started=1,
            state_budget_exhausted=True,
        )
        rows = _result_rows_from_cache(
            self.plan,
            cache,
            self.execution_id,
            set(),
            failures,
        )
        failed = [row for row in rows if row["status"] == "failed"]
        self.assertTrue(all(row["attempts_started"] == 1 for row in failed))
        self.assertTrue(all(row["state_budget_exhausted"] for row in failed))
        summary = summarize_signal_gate(self.plan, self.identity, rows)
        self.assertEqual(summary["execution_status"], "incomplete")


if __name__ == "__main__":
    unittest.main()
