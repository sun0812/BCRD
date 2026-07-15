#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""以单机、缓存优先方式执行冻结的 CCOD 诊断计划。

父进程只负责验收冻结包、串行启动一次性 state worker、监控硬时限/RSS，
以及从内容寻址缓存重建结果。worker 被中断后绝不复用其 Oracle；已经原子
发布的查询缓存则可由下一次尝试继续使用。
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import signal
import subprocess
import sys
import tempfile
import time
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


RUNNER_SCHEMA_VERSION = "eosbench-ccod-runner-v1"
QUERY_RESULT_SCHEMA_VERSION = "eosbench-ccod-query-result-v1"
CHECKPOINT_SCHEMA_VERSION = "eosbench-ccod-state-checkpoint-v1"
WORKER_REPORT_SCHEMA_VERSION = "eosbench-ccod-worker-report-v1"
RUNNER_IMPLEMENTATION_SCHEMA_VERSION = "eosbench-ccod-runner-implementation-v1"

# final report 是跨进程运行证据，必须用精确字段集阻止旧 worker 或额外
# 未受审计元数据混入 manifest。
_WORKER_SUCCESS_FIELDS = frozenset(
    {
        "schema_version",
        "execution_id",
        "runner_implementation_hash",
        "frozen_snapshot_hash",
        "invocation",
        "state_ordinal",
        "attempt",
        "status",
        "restored",
        "planned_queries",
        "cache_hits",
        "evaluated_queries",
        "fresh_query_keys",
        "peak_rss_mib",
        "report_hash",
    }
)
_WORKER_FAILURE_FIELDS = frozenset(
    {
        "schema_version",
        "execution_id",
        "runner_implementation_hash",
        "frozen_snapshot_hash",
        "invocation",
        "state_ordinal",
        "attempt",
        "status",
        "failure_kind",
        "error_type",
        "error_message",
        "peak_rss_mib",
        "report_hash",
    }
)

FROZEN_PYTHON = (3, 10, 20)
FROZEN_WORKERS = 1
FROZEN_MAX_ATTEMPTS = 2
FROZEN_QUERY_TIMEOUT_S = 120.0
FROZEN_STATE_TIMEOUT_S = 1200.0
FROZEN_RSS_LIMIT_MIB = 6144.0
RECOVERABLE_FAILURE_KINDS = frozenset(
    {
        "query_timeout",
        "state_timeout",
        "rss_exceeded",
        "worker_exit",
        "worker_error",
        "attempt_exhausted",
        "interrupted",
    }
)
INVALID_FAILURE_KINDS = frozenset(
    {
        "identity_mismatch",
        "cache_corrupt",
        "hash_mismatch",
        "frozen_drift",
        "runner_drift",
    }
)


class CCODRunnerError(RuntimeError):
    """runner 身份、输入、缓存或护栏不闭合时抛出的异常。"""


class QueryDeadlineExceeded(CCODRunnerError):
    """单查询超过 worker 内部真实时限。"""


class RunnerAttemptError(CCODRunnerError):
    """携带稳定 failure_kind 的单次 worker 失败。"""

    def __init__(self, failure_kind: str, message: str) -> None:
        allowed = RECOVERABLE_FAILURE_KINDS | INVALID_FAILURE_KINDS
        if failure_kind not in allowed:
            raise ValueError(f"未知 runner failure_kind: {failure_kind}")
        super().__init__(message)
        self.failure_kind = failure_kind

    @property
    def recoverable(self) -> bool:
        """返回该故障是否允许保留同一科学身份等待人工恢复。"""
        return self.failure_kind in RECOVERABLE_FAILURE_KINDS


@dataclass(frozen=True)
class WorkerOutcome:
    """父进程 watchdog 对一次性 worker 的最终判定。"""

    returncode: Optional[int]
    violation: Optional[str]
    stdout: str
    stderr: str
    peak_rss_mib: float


def _canonical_bytes(value: Any) -> bytes:
    """生成与仓库一致的规范 UTF-8 JSON。"""
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CCODRunnerError(f"对象无法规范 JSON 序列化: {exc}") from exc


def _sha256_json(value: Any) -> str:
    """返回带算法前缀的规范 JSON SHA-256。"""
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    """流式计算普通文件哈希。"""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise CCODRunnerError(f"无法哈希文件 {path}: {exc}") from exc
    return "sha256:" + digest.hexdigest()


def _mutable_json_copy(value: Any) -> Any:
    """把 execution 的递归只读视图转换为缓存/Oracle 可消费对象。"""
    if isinstance(value, Mapping):
        return {str(key): _mutable_json_copy(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_mutable_json_copy(item) for item in value]
    return value


def _with_hash(payload: Mapping[str, Any], hash_field: str) -> Dict[str, Any]:
    """复制记录并加入覆盖其余字段的内容哈希。"""
    result = dict(payload)
    result.pop(hash_field, None)
    result[hash_field] = _sha256_json(result)
    return result


def _reject_symlink_components(path: Path, *, label: str) -> None:
    """拒绝现有路径部分中的符号链接，仅放行 macOS 固有根级别名。"""
    cursor = path
    system_aliases = {Path("/tmp"), Path("/var"), Path("/etc")}
    while True:
        if (
            cursor.exists()
            and cursor.is_symlink()
            and cursor not in system_aliases
        ):
            raise CCODRunnerError(f"{label} 不允许符号链接组件: {cursor}")
        if cursor == cursor.parent:
            return
        cursor = cursor.parent


def _reject_input_symlink_components(path: Path, *, label: str) -> None:
    """在 resolve 前拒绝用户路径别名，仅放行 macOS 固有根级链接。"""
    absolute = path.expanduser()
    if not absolute.is_absolute():
        absolute = Path.cwd() / absolute
    cursor = Path(absolute.anchor)
    system_aliases = {Path("/tmp"), Path("/var"), Path("/etc")}
    for part in absolute.parts[1:]:
        cursor = cursor / part
        if not cursor.is_symlink():
            continue
        if cursor in system_aliases:
            continue
        raise CCODRunnerError(f"{label} 不允许符号链接组件: {cursor}")


def _atomic_write(path: Path, payload: bytes) -> None:
    """同目录 fsync 后原子替换派生文件。"""
    # macOS 的 ``/var`` 本身是系统级 ``/private/var`` 符号链接；先规范化，
    # 再检查实际写入树，避免把系统布局误判为用户注入的目录重定向。
    if path.is_symlink():
        raise CCODRunnerError(f"原子写入目标不得为符号链接: {path}")
    _reject_symlink_components(path.parent, label="写入目录")
    path = path.resolve(strict=False)
    _reject_symlink_components(path.parent, label="写入目录")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError as exc:
        raise CCODRunnerError(f"无法原子写入 {path}: {exc}") from exc
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def _write_json(
    path: Path, payload: Mapping[str, Any], hash_field: str
) -> Dict[str, Any]:
    """加入内容哈希并原子写入单个 JSON object。"""
    record = _with_hash(payload, hash_field)
    _atomic_write(path, _canonical_bytes(record) + b"\n")
    return record


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """按调用方给定顺序原子重建 JSONL。"""
    body = b"".join(_canonical_bytes(dict(row)) + b"\n" for row in rows)
    _atomic_write(path, body)


def _read_json(path: Path) -> Dict[str, Any]:
    """读取单个 JSON object 并拒绝重复键。"""

    def reject_duplicates(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise CCODRunnerError(f"{path} 含重复 JSON 键: {key}")
            result[key] = value
        return result

    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle, object_pairs_hook=reject_duplicates)
    except CCODRunnerError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise CCODRunnerError(f"无法读取 JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CCODRunnerError(f"{path} 必须包含 JSON object")
    return value


def _acquire_run_lock(run_dir: Path, execution_id: str) -> Any:
    """用内核 flock 保证同一 run dir 同时只有一个父进程。"""
    lock_path = run_dir / ".runner.lock"
    if lock_path.is_symlink():
        raise CCODRunnerError(f"运行锁不得为符号链接: {lock_path}")
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor: Optional[int] = None
    handle: Any = None
    try:
        descriptor = os.open(lock_path, flags, 0o600)
        handle = os.fdopen(descriptor, "r+b", buffering=0)
        descriptor = None
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        payload = _canonical_bytes(
            {
                "execution_id": execution_id,
                "pid": os.getpid(),
                "acquired_unix_ns": time.time_ns(),
            }
        ) + b"\n"
        handle.seek(0)
        handle.truncate(0)
        handle.write(payload)
        os.fsync(handle.fileno())
        return handle
    except BlockingIOError as exc:
        if handle is not None:
            handle.close()
        elif descriptor is not None:
            os.close(descriptor)
        raise CCODRunnerError("运行目录已被另一个 CCOD 父进程锁定") from exc
    except OSError as exc:
        if handle is not None:
            handle.close()
        elif descriptor is not None:
            os.close(descriptor)
        raise CCODRunnerError(f"无法获取运行目录锁: {exc}") from exc


def require_frozen_runtime() -> Dict[str, Any]:
    """在 verifier 或任何来源读取前锁死 CPython 3.10.20 final。"""
    info = sys.version_info
    version = (int(info[0]), int(info[1]), int(info[2]))
    implementation = platform.python_implementation().lower()
    if (
        implementation != "cpython"
        or version != FROZEN_PYTHON
        or info.releaselevel != "final"
        or info.serial != 0
    ):
        expected = ".".join(str(part) for part in FROZEN_PYTHON)
        actual = ".".join(str(part) for part in version)
        raise CCODRunnerError(
            f"仅允许 CPython {expected} final；当前为 "
            f"{implementation} {actual} {info.releaselevel}.{info.serial}"
        )
    # execution 核心也维护相同护栏；两层均在 verifier 之前执行。
    from algorithms.ccod.execution import require_frozen_execution_runtime

    require_frozen_execution_runtime()
    return {
        "python_implementation": "cpython",
        "python_version": ".".join(str(part) for part in version),
        "releaselevel": info.releaselevel,
        "serial": info.serial,
    }


def _is_within(path: Path, root: Path) -> bool:
    """判断解析后的路径是否位于另一目录内（含相等）。"""
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def validate_external_paths(
    frozen_dir: str | Path,
    run_dir: str | Path,
    cache_dir: str | Path,
) -> Tuple[Path, Path, Path]:
    """要求所有可写路径与冻结树完全分离。"""
    raw_paths = (
        ("冻结目录", Path(frozen_dir)),
        ("运行目录", Path(run_dir)),
        ("缓存目录", Path(cache_dir)),
    )
    for label, raw_path in raw_paths:
        _reject_input_symlink_components(raw_path, label=label)
    frozen = raw_paths[0][1].resolve()
    run = raw_paths[1][1].resolve()
    cache = raw_paths[2][1].resolve()
    if not frozen.is_dir():
        raise CCODRunnerError(f"冻结目录不存在: {frozen}")
    for label, path in (("冻结目录", frozen), ("运行目录", run), ("缓存目录", cache)):
        _reject_symlink_components(path, label=label)
    if (
        _is_within(run, frozen)
        or _is_within(cache, frozen)
        or _is_within(frozen, run)
        or _is_within(frozen, cache)
    ):
        raise CCODRunnerError("冻结目录与运行/缓存目录必须完全分离")
    return frozen, run, cache


def execution_prefix(
    plan: Any,
    max_states: Optional[int],
) -> Tuple[Tuple[int, ...], Tuple[Mapping[str, Any], ...]]:
    """选择冻结 state 前缀，不复制或重写 query identity。"""
    state_count = len(plan.selected_states)
    if max_states is None:
        limit = state_count
    elif (
        isinstance(max_states, bool)
        or not isinstance(max_states, int)
        or not (1 <= max_states <= state_count)
    ):
        raise CCODRunnerError(f"max_states 必须位于 [1, {state_count}]")
    else:
        limit = max_states
    ordinals = tuple(range(limit))
    queries = tuple(
        query
        for state_ordinal in ordinals
        for query in plan.queries_by_state[state_ordinal]
    )
    return ordinals, queries


def _peak_rss_mib() -> float:
    """把当前进程 ru_maxrss 统一换算为 MiB。"""
    raw = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return raw / (1024.0 * 1024.0)
    return raw / 1024.0


@contextmanager
def query_deadline(seconds: float) -> Iterator[None]:
    """用真实 SIGALRM 限制 worker 内单次查询。

    信号只在主线程可用；正式 worker 总在主线程执行。测试若从其他线程调用，
    会在退出时做事后时限校验，但不会伪装成硬中断。
    """
    if seconds <= 0.0:
        raise ValueError("query deadline 必须为正数")
    started = time.monotonic()
    enabled = hasattr(signal, "setitimer") and hasattr(signal, "ITIMER_REAL")
    old_handler: Any = None
    old_timer = (0.0, 0.0)

    def handle_timeout(signum: int, frame: Any) -> None:
        del signum, frame
        raise QueryDeadlineExceeded(f"query 超过 {seconds:.3f} 秒")

    if enabled:
        try:
            old_handler = signal.getsignal(signal.SIGALRM)
            old_timer = signal.getitimer(signal.ITIMER_REAL)
            requested = seconds
            if old_timer[0] > 0.0:
                requested = min(requested, old_timer[0])
            signal.signal(signal.SIGALRM, handle_timeout)
            signal.setitimer(signal.ITIMER_REAL, requested)
        except (AttributeError, OSError, RuntimeError, ValueError):
            enabled = False
    try:
        yield
    finally:
        elapsed = time.monotonic() - started
        if enabled:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, old_handler)
            if old_timer[0] > 0.0:
                remaining = max(0.000001, old_timer[0] - elapsed)
                signal.setitimer(signal.ITIMER_REAL, remaining, old_timer[1])
        if elapsed > seconds and sys.exc_info()[0] is None:
            raise QueryDeadlineExceeded(f"query 超过 {seconds:.3f} 秒")


def _load_cached_group(
    plan: Any, state_ordinal: int, cache: Any
) -> Dict[str, Mapping[str, Any]]:
    """完整加载一个 state 的现有缓存，坏记录由 cache.load 直接报错。"""
    cached: Dict[str, Mapping[str, Any]] = {}
    for query in plan.queries_by_state[state_ordinal]:
        value = cache.load(_mutable_json_copy(query["query_identity"]))
        if value is not None:
            key = str(query["query_key"])
            if value.get("query_key") != key:
                raise CCODRunnerError("缓存 query_key 与冻结计划不一致")
            cached[key] = value
    return cached


def _continuation_for_plan(plan: Any) -> Any:
    """由真实计划构造 H=5 continuation；轻量测试替身使用默认值。"""
    from algorithms.ccod.continuation import ContinuationConfig

    manifest = getattr(plan, "run_manifest", {})
    payload = (
        manifest.get("continuation_config") if isinstance(manifest, Mapping) else None
    )
    if payload is None:
        return ContinuationConfig(horizon=5)
    if not isinstance(payload, Mapping):
        raise CCODRunnerError("run manifest continuation_config 非 object")
    try:
        config = ContinuationConfig(**dict(payload))
    except (TypeError, ValueError) as exc:
        raise CCODRunnerError(f"continuation_config 非法: {exc}") from exc
    if config.to_payload() != dict(payload):
        raise CCODRunnerError("continuation_config 存在缺省或额外字段漂移")
    return config


def execute_state_worker(
    plan: Any,
    state_ordinal: int,
    cache: Any,
    heartbeat: Callable[[str, Mapping[str, Any]], None],
    *,
    query_timeout_s: float = FROZEN_QUERY_TIMEOUT_S,
    runtime_factory: Optional[Callable[[Any, int], Any]] = None,
    rss_limit_mib: float = FROZEN_RSS_LIMIT_MIB,
) -> Dict[str, Any]:
    """缓存优先执行一个完整状态，存在 miss 时只 restore/prepare 一次。"""
    queries = tuple(plan.queries_by_state[state_ordinal])
    cached_at_start = _load_cached_group(plan, state_ordinal, cache)
    missing = [
        query for query in queries if str(query["query_key"]) not in cached_at_start
    ]
    heartbeat(
        "state_start",
        {
            "state_ordinal": state_ordinal,
            "planned_queries": len(queries),
            "cache_hits": len(cached_at_start),
        },
    )
    if not missing:
        report = {
            "state_ordinal": state_ordinal,
            "restored": False,
            "planned_queries": len(queries),
            "cache_hits": len(queries),
            "evaluated_queries": 0,
            "fresh_query_keys": [],
            "peak_rss_mib": _peak_rss_mib(),
        }
        heartbeat("state_complete", report)
        return report

    actual_factory = (
        _real_runtime_factory if runtime_factory is None else runtime_factory
    )
    runtime = actual_factory(plan, state_ordinal)
    if not isinstance(runtime, tuple) or len(runtime) not in (2, 3):
        raise CCODRunnerError(
            "runtime_factory 必须返回 (oracle, prepared[, continuation])"
        )
    oracle, prepared = runtime[0], runtime[1]
    continuation = runtime[2] if len(runtime) == 3 else _continuation_for_plan(plan)
    evaluated: List[str] = []
    peak_rss = _peak_rss_mib()
    if peak_rss > rss_limit_mib:
        raise CCODRunnerError("worker 在查询前已超过 RSS 上限")

    for query in missing:
        identity = _mutable_json_copy(query["query_identity"])
        action_key = _mutable_json_copy(query["action_key"])
        # 与 evaluate 紧邻地再次检查，支持中断恢复或另一 writer 已发布同 key。
        if cache.load(identity) is not None:
            continue
        heartbeat(
            "query_start",
            {
                "state_ordinal": state_ordinal,
                "query_ordinal": query.get("query_ordinal"),
            },
        )
        with query_deadline(query_timeout_s):
            result = oracle.evaluate(
                prepared,
                action_key,
                continuation_config=continuation,
            )
            manifest = result.to_manifest()
            if manifest.get("query_key") != query.get("query_key"):
                raise CCODRunnerError("Oracle 结果 query_key 偏离冻结计划")
            if "state_hash" in manifest and manifest.get("state_hash") != query.get(
                "state_hash"
            ):
                raise CCODRunnerError("Oracle 结果 state_hash 偏离冻结计划")
            if (
                "forced_action_key" in manifest
                and manifest.get("forced_action_key") != action_key
            ):
                raise CCODRunnerError("Oracle 结果 forced_action_key 偏离冻结计划")
            cache.store(identity, result)
            reloaded = cache.load(identity)
            if reloaded is None or reloaded != manifest:
                raise CCODRunnerError("cache store/load 闭环校验失败")
        key = str(query["query_key"])
        evaluated.append(key)
        peak_rss = max(peak_rss, _peak_rss_mib())
        if peak_rss > rss_limit_mib:
            raise CCODRunnerError("worker 查询后超过 RSS 上限")
        heartbeat(
            "query_complete",
            {
                "state_ordinal": state_ordinal,
                "query_ordinal": query.get("query_ordinal"),
                "query_key": key,
            },
        )

    final_group = _load_cached_group(plan, state_ordinal, cache)
    if len(final_group) != len(queries):
        raise CCODRunnerError("worker 结束时 state cache 仍不完整")
    report = {
        "state_ordinal": state_ordinal,
        "restored": True,
        "planned_queries": len(queries),
        "cache_hits": len(queries) - len(evaluated),
        "evaluated_queries": len(evaluated),
        "fresh_query_keys": evaluated,
        "peak_rss_mib": peak_rss,
    }
    heartbeat("state_complete", report)
    return report


def publish_state_checkpoint_if_complete(
    plan: Any,
    execution_id: str,
    state_ordinal: int,
    cache: Any,
    checkpoint_dir: str | Path,
) -> bool:
    """父进程复核整组 cache 后才发布 advisory checkpoint。"""
    queries = tuple(plan.queries_by_state[state_ordinal])
    checkpoint_root = Path(checkpoint_dir)
    if checkpoint_root.is_symlink():
        raise CCODRunnerError(f"checkpoint 目录不允许符号链接: {checkpoint_root}")
    checkpoint_root = checkpoint_root.resolve(strict=False)
    checkpoint_path = checkpoint_root / f"state_{state_ordinal:03d}.json"
    try:
        cached = _load_cached_group(plan, state_ordinal, cache)
    except Exception:
        checkpoint_path.unlink(missing_ok=True)
        raise
    if len(cached) != len(queries):
        checkpoint_path.unlink(missing_ok=True)
        return False
    query_keys = [str(query["query_key"]) for query in queries]
    if set(query_keys) != set(cached):
        checkpoint_path.unlink(missing_ok=True)
        return False
    state_hash = str(plan.selected_states[state_ordinal]["state_hash"])
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "execution_id": execution_id,
        "run_id": str(plan.run_id),
        "state_ordinal": state_ordinal,
        "state_hash": state_hash,
        "query_count": len(query_keys),
        "query_keys": query_keys,
        "cache_is_source_of_truth": True,
    }
    _write_json(
        checkpoint_path,
        payload,
        "checkpoint_hash",
    )
    return True


def _terminate_worker(process: Any) -> None:
    """先温和终止，再在短等待后强杀一次性 worker。"""
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=2.0)
    except (subprocess.TimeoutExpired, TimeoutError):
        process.kill()
        process.wait(timeout=2.0)


def _default_rss_reader(pid: int) -> float:
    """通过系统 ps 读取子进程当前 RSS，并转换为 MiB。"""
    completed = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(pid)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        timeout=2.0,
        check=False,
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        raise CCODRunnerError("ps 未返回可用的 worker RSS")
    try:
        return float(completed.stdout.strip().splitlines()[0]) / 1024.0
    except ValueError as exc:
        raise CCODRunnerError("无法解析 worker RSS") from exc


def monitor_worker(
    process: Any,
    *,
    remaining_state_s: float,
    rss_limit_mib: float,
    poll_interval_s: float = 0.25,
    rss_reader: Callable[[int], float] = _default_rss_reader,
) -> WorkerOutcome:
    """父进程监控 state 累计 deadline 和 worker RSS。"""
    started = time.monotonic()
    peak_rss = 0.0
    violation: Optional[str] = None
    while process.poll() is None:
        elapsed = time.monotonic() - started
        if elapsed >= remaining_state_s:
            violation = "state_timeout"
            _terminate_worker(process)
            break
        try:
            peak_rss = max(peak_rss, float(rss_reader(int(process.pid))))
        except (OSError, subprocess.SubprocessError, CCODRunnerError):
            violation = "rss_unavailable"
            _terminate_worker(process)
            break
        if peak_rss > rss_limit_mib:
            violation = "rss_limit"
            _terminate_worker(process)
            break
        time.sleep(min(poll_interval_s, max(0.0, remaining_state_s - elapsed)))
    stdout, stderr = process.communicate()
    return WorkerOutcome(
        returncode=process.returncode,
        violation=violation,
        stdout=str(stdout or ""),
        stderr=str(stderr or ""),
        peak_rss_mib=peak_rss,
    )


def _worker_command(
    *,
    frozen_dir: Path,
    run_dir: Path,
    cache_dir: Path,
    config_path: Path,
    state_ordinal: int,
    attempt: int,
    query_timeout_s: float,
    execution_id: str,
    runner_hash: str,
    invocation: int,
    frozen_snapshot_hash: Optional[str] = None,
) -> List[str]:
    """构造一次性 worker 命令；首项固定为当前解释器。"""
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--frozen-dir",
        str(frozen_dir),
        "--run-dir",
        str(run_dir),
        "--cache-dir",
        str(cache_dir),
        "--config",
        str(config_path),
        "--state-ordinal",
        str(state_ordinal),
        "--attempt",
        str(attempt),
        "--invocation",
        str(invocation),
        "--query-timeout-s",
        repr(float(query_timeout_s)),
        "--execution-id",
        execution_id,
        "--runner-hash",
        runner_hash,
    ]
    if frozen_snapshot_hash is not None:
        command.extend(["--frozen-snapshot-hash", frozen_snapshot_hash])
    return command


def _safe_file(root: Path, raw_path: Any, *, label: str) -> Path:
    """解析根目录内的普通文件，拒绝绝对路径、逃逸和符号链接。"""
    if not isinstance(raw_path, str) or not raw_path:
        raise CCODRunnerError(f"{label} 缺少相对路径")
    relative = Path(raw_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise CCODRunnerError(f"{label} 不是安全相对路径")
    candidate = root / relative
    try:
        candidate.resolve(strict=True).relative_to(root.resolve())
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise CCODRunnerError(f"{label} 不存在或逃逸根目录") from exc
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise CCODRunnerError(f"{label} 不允许符号链接")
    if not candidate.is_file():
        raise CCODRunnerError(f"{label} 不是普通文件")
    return candidate


def _real_runtime_factory(plan: Any, state_ordinal: int) -> Tuple[Any, Any, Any]:
    """恢复一个真实状态并只 prepare 一次，供同组全部 query 复用。"""
    from algorithms.ccod.continuation import ContinuationOracle
    from schedulers.scenario_loader import load_scheduling_problem_from_json
    from schedulers.state_replay import (
        ConstraintConfig,
        EnumeratorConfig,
        ObjectiveConfig,
        restore_state,
    )

    state = plan.selected_states[state_ordinal]
    canonical = state["canonical_source"]
    scenario_path = _safe_file(
        REPO_ROOT,
        canonical["scenario_ref"]["relative_path"],
        label="scenario_ref",
    )
    trace_path = _safe_file(
        plan.frozen_dir,
        canonical["trace_ref"]["relative_path"],
        label="trace_ref",
    )
    trace = _read_json(trace_path)
    problem = load_scheduling_problem_from_json(scenario_path)
    constraint = ConstraintConfig.from_payload(trace["constraint_config"])
    enumerator = EnumeratorConfig.from_payload(trace["enumerator_config"])
    objective = ObjectiveConfig.from_payload(trace["objective_config"])
    for field_name, actual in (
        ("constraint_hash", constraint.hash),
        ("enumerator_hash", enumerator.hash),
        ("objective_hash", objective.hash),
    ):
        if state.get(field_name) != actual or trace.get(field_name) != actual:
            raise CCODRunnerError(f"state/trace 的 {field_name} 不一致")
    replayed = restore_state(
        problem,
        trace,
        _mutable_json_copy(canonical["state_manifest"]),
        scenario_path=scenario_path,
        verify=True,
    )
    if replayed.state_manifest.get("state_hash") != state.get("state_hash"):
        raise CCODRunnerError("restore 后 state_hash 偏离冻结计划")
    oracle = ContinuationOracle(
        problem,
        constraint_config=constraint,
        enumerator_config=enumerator,
        objective_config=objective,
    )
    return oracle, oracle.prepare(replayed), _continuation_for_plan(plan)


def runner_implementation_hash() -> str:
    """绑定 runner/执行/缓存源码和 continuation 全依赖哈希。"""
    from algorithms.ccod.continuation import continuation_implementation_hash

    # 每次复核都重新读取磁盘依赖，禁止 lru_cache 掩盖运行期源码漂移。
    continuation_implementation_hash.cache_clear()
    continuation_hash = continuation_implementation_hash()
    relative_paths = (
        "scripts/run_ccod_diagnostic.py",
        "scripts/prepare_ccod_diagnostic.py",
        "algorithms/ccod/execution.py",
        "algorithms/ccod/cache.py",
        "algorithms/ccod/metrics.py",
    )
    files = {path: _sha256_file(REPO_ROOT / path) for path in relative_paths}
    return _sha256_json(
        {
            "schema_version": RUNNER_IMPLEMENTATION_SCHEMA_VERSION,
            "files": files,
            "continuation_implementation_hash": continuation_hash,
        }
    )


def _frozen_snapshot(frozen_dir: Path) -> Dict[str, Any]:
    """记录冻结包普通文件，执行结束时用于不变性复核。"""
    files = []
    for path in sorted(frozen_dir.rglob("*")):
        if path.is_symlink():
            raise CCODRunnerError(f"冻结包含符号链接: {path}")
        if path.is_file():
            files.append(
                {
                    "relative_path": path.relative_to(frozen_dir).as_posix(),
                    "sha256": _sha256_file(path),
                }
            )
    payload = {"file_count": len(files), "files": files}
    payload["snapshot_hash"] = _sha256_json(payload)
    return payload


def _result_rows_from_cache(
    plan: Any,
    cache: Any,
    execution_id: str,
    initially_cached: Set[str],
    terminal_failures: Mapping[str, str],
) -> List[Dict[str, Any]]:
    """按完整冻结顺序从缓存重建已有成功标签。"""
    rows = []
    for query in plan.query_rows:
        query_key = str(query["query_key"])
        failure_kind = terminal_failures.get(query_key)
        if failure_kind is not None:
            failed = {
                "schema_version": QUERY_RESULT_SCHEMA_VERSION,
                "execution_id": execution_id,
                "run_id": plan.run_id,
                "query_ordinal": query["query_ordinal"],
                "state_ordinal": query["state_ordinal"],
                "action_ordinal": query["action_ordinal"],
                "state_hash": query["state_hash"],
                "query_key": query_key,
                "status": "failed",
                "failure_kind": failure_kind,
                "attempts_exhausted": FROZEN_MAX_ATTEMPTS,
            }
            rows.append(_with_hash(failed, "row_hash"))
            continue
        cached = cache.load(_mutable_json_copy(query["query_identity"]))
        if cached is None:
            continue
        if cached.get("query_key") != query.get("query_key"):
            raise CCODRunnerError("结果重建时 cache/query identity 冲突")
        payload = {
            "schema_version": QUERY_RESULT_SCHEMA_VERSION,
            "execution_id": execution_id,
            "run_id": plan.run_id,
            "query_ordinal": query["query_ordinal"],
            "state_ordinal": query["state_ordinal"],
            "action_ordinal": query["action_ordinal"],
            "state_hash": query["state_hash"],
            "query_key": query_key,
            "status": "success",
            "cache_hit_at_invocation_start": query_key in initially_cached,
            "result": dict(cached),
        }
        rows.append(_with_hash(payload, "row_hash"))
    return rows


def _worker_report_path(
    run_dir: Path,
    invocation: int,
    state_ordinal: int,
    attempt: int,
) -> Path:
    """返回父子进程约定的 final report 路径。"""
    return (
        run_dir
        / "workers"
        / (
            f"invocation_{invocation:03d}_state_{state_ordinal:03d}"
            f"_attempt_{attempt}.json"
        )
    )


def _heartbeat_journal(
    run_dir: Path,
    *,
    execution_id: str,
    invocation: int,
    state_ordinal: int,
    attempt: int,
) -> Callable[[str, Mapping[str, Any]], None]:
    """创建不可覆盖 journal，并同步发布原子 latest heartbeat。"""
    worker_dir = run_dir / "workers"
    _reject_symlink_components(worker_dir, label="worker journal 目录")
    path = (
        worker_dir
        / (
            f"invocation_{invocation:03d}_state_{state_ordinal:03d}"
            f"_attempt_{attempt}.journal.jsonl"
        )
    ).resolve(strict=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise CCODRunnerError(f"worker journal 已存在，拒绝覆盖: {path}")
    latest_path = worker_dir / (
        f"invocation_{invocation:03d}_state_{state_ordinal:03d}.heartbeat.json"
    )
    sequence = 0

    def emit(event: str, details: Mapping[str, Any]) -> None:
        nonlocal sequence
        record = {
            "schema_version": "eosbench-ccod-worker-journal-v1",
            "execution_id": execution_id,
            "invocation": invocation,
            "state_ordinal": state_ordinal,
            "attempt": attempt,
            "sequence": sequence,
            "event": str(event),
            "details": dict(details),
        }
        sequence += 1
        try:
            with path.open("ab") as handle:
                handle.write(_canonical_bytes(record) + b"\n")
                handle.flush()
                os.fsync(handle.fileno())
            _write_json(latest_path, record, "heartbeat_hash")
        except OSError as exc:
            raise CCODRunnerError(f"无法写入 worker journal: {exc}") from exc

    return emit


def _worker_main(args: argparse.Namespace) -> int:
    """执行一个 state 后退出；失败时不复用进程。"""
    run: Optional[Path] = None
    parent_authorized = False
    try:
        require_frozen_runtime()
        frozen, run, cache_dir = validate_external_paths(
            args.frozen_dir,
            args.run_dir,
            args.cache_dir,
        )
        manifest = _existing_manifest(run)
        if manifest is None:
            raise CCODRunnerError("worker 缺少父进程 execution manifest")
        manifest_identity = manifest.get("execution_identity")
        if not isinstance(manifest_identity, Mapping):
            raise CCODRunnerError("父进程 execution manifest 身份非法")
        if (
            isinstance(args.invocation, bool)
            or not isinstance(args.invocation, int)
            or args.invocation <= 0
        ):
            raise CCODRunnerError("worker invocation 非法")
        if (
            isinstance(args.state_ordinal, bool)
            or not isinstance(args.state_ordinal, int)
            or args.state_ordinal < 0
            or isinstance(args.attempt, bool)
            or not isinstance(args.attempt, int)
            or not (1 <= args.attempt <= FROZEN_MAX_ATTEMPTS)
        ):
            raise CCODRunnerError("worker state/attempt 坐标非法")
        timeout = args.query_timeout_s
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(float(timeout))
            or float(timeout).hex() != FROZEN_QUERY_TIMEOUT_S.hex()
        ):
            raise CCODRunnerError("worker query timeout 偏离冻结的 120 秒")
        manifest_invocation = manifest.get("invocation")
        if (
            manifest.get("status") != "running"
            or manifest.get("execution_id") != args.execution_id
            or manifest.get("frozen_snapshot_hash") != args.frozen_snapshot_hash
            or manifest_identity.get("runner_implementation_hash") != args.runner_hash
            or isinstance(manifest_invocation, bool)
            or not isinstance(manifest_invocation, int)
            or manifest_invocation != args.invocation
        ):
            raise CCODRunnerError("worker 无法绑定父进程已验收的运行身份")
        expected_paths = {
            "frozen_dir": str(frozen),
            "run_dir": str(run),
            "cache_dir": str(cache_dir),
        }
        if manifest.get("paths") != expected_paths:
            raise CCODRunnerError("worker 路径与父进程 execution manifest 不一致")
        allowed_states = manifest.get("state_ordinals_this_invocation")
        if (
            not isinstance(allowed_states, list)
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for value in allowed_states
            )
            or len(set(allowed_states)) != len(allowed_states)
            or args.state_ordinal not in allowed_states
        ):
            raise CCODRunnerError("worker state 不在本次 invocation 授权前缀内")
        state_progress = manifest.get("state_progress")
        active_progress = (
            state_progress.get(str(args.state_ordinal))
            if isinstance(state_progress, Mapping)
            else None
        )
        if (
            not isinstance(active_progress, Mapping)
            or active_progress.get("active_attempt") != args.attempt
            or active_progress.get("attempts_started") != args.attempt
            or isinstance(active_progress.get("active_started_unix_ns"), bool)
            or not isinstance(active_progress.get("active_started_unix_ns"), int)
            or int(active_progress["active_started_unix_ns"]) <= 0
        ):
            raise CCODRunnerError("worker attempt 未绑定父进程当前 active 状态")
        if _frozen_snapshot(frozen).get("snapshot_hash") != args.frozen_snapshot_hash:
            raise CCODRunnerError("worker 启动时冻结包 snapshot 漂移")
        from algorithms.ccod.cache import CounterfactualLabelCache
        from algorithms.ccod.execution import (
            build_execution_identity,
            execution_identity_hash,
            load_frozen_diagnostic_plan,
        )

        plan = load_frozen_diagnostic_plan(frozen)
        current_runner_hash = runner_implementation_hash()
        if current_runner_hash != args.runner_hash:
            raise CCODRunnerError("worker 启动时 runner 源码哈希漂移")
        identity = build_execution_identity(
            plan, runner_implementation_hash=current_runner_hash
        )
        if execution_identity_hash(identity) != args.execution_id:
            raise CCODRunnerError("worker execution_id 与父进程不一致")
        # 只有完整绑定父进程、冻结快照、invocation 与执行身份后，
        # worker 才获得在外部运行目录写入 journal/report 的权限。
        parent_authorized = True
        report = execute_state_worker(
            plan,
            args.state_ordinal,
            CounterfactualLabelCache(cache_dir),
            _heartbeat_journal(
                run,
                execution_id=args.execution_id,
                invocation=args.invocation,
                state_ordinal=args.state_ordinal,
                attempt=args.attempt,
            ),
            query_timeout_s=args.query_timeout_s,
            rss_limit_mib=FROZEN_RSS_LIMIT_MIB,
        )
        report.update(
            {
                "schema_version": WORKER_REPORT_SCHEMA_VERSION,
                "execution_id": args.execution_id,
                "runner_implementation_hash": args.runner_hash,
                "frozen_snapshot_hash": args.frozen_snapshot_hash,
                "invocation": args.invocation,
                "attempt": args.attempt,
                "status": "success",
            }
        )
        _write_json(
            _worker_report_path(
                run,
                args.invocation,
                args.state_ordinal,
                args.attempt,
            ),
            report,
            "report_hash",
        )
        return 0
    except Exception as exc:
        if parent_authorized and run is not None and all(
            value is not None
            for value in (
                args.execution_id,
                args.invocation,
                args.state_ordinal,
                args.attempt,
            )
        ):
            failure = {
                "schema_version": WORKER_REPORT_SCHEMA_VERSION,
                "execution_id": args.execution_id,
                "runner_implementation_hash": args.runner_hash,
                "frozen_snapshot_hash": args.frozen_snapshot_hash,
                "invocation": args.invocation,
                "state_ordinal": args.state_ordinal,
                "attempt": args.attempt,
                "status": "failed",
                "failure_kind": _failure_kind(exc),
                "error_type": type(exc).__name__,
                "error_message": str(exc)[-2000:],
                "peak_rss_mib": _peak_rss_mib(),
            }
            try:
                _write_json(
                    _worker_report_path(
                        run,
                        args.invocation,
                        args.state_ordinal,
                        args.attempt,
                    ),
                    failure,
                    "report_hash",
                )
            except Exception:
                # stderr 仍为父进程的最后恢复证据，不能掩盖原始异常。
                pass
        print(f"worker 失败: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


def _verify_worker_report(
    path: Path,
    *,
    execution_id: str,
    runner_hash: str,
    frozen_snapshot_hash: str,
    invocation: int,
    state_ordinal: int,
    attempt: int,
) -> Dict[str, Any]:
    """父进程验收 worker final report。"""
    if not path.is_file():
        raise RunnerAttemptError("worker_exit", "worker 未发布 final report")
    payload = _read_json(path)
    stored = payload.get("report_hash")
    unhashed = {key: value for key, value in payload.items() if key != "report_hash"}
    if stored != _sha256_json(unhashed):
        raise RunnerAttemptError("hash_mismatch", "worker final report 内容哈希不一致")
    if payload.get("schema_version") != WORKER_REPORT_SCHEMA_VERSION:
        raise RunnerAttemptError("identity_mismatch", "worker final report schema 不一致")
    status = payload.get("status")
    expected_fields = (
        _WORKER_SUCCESS_FIELDS
        if status == "success"
        else _WORKER_FAILURE_FIELDS
        if status == "failed"
        else None
    )
    if expected_fields is None or set(payload) != expected_fields:
        raise RunnerAttemptError(
            "identity_mismatch",
            "worker final report status 或字段集非法",
        )
    if (
        payload.get("execution_id") != execution_id
        or payload.get("runner_implementation_hash") != runner_hash
        or payload.get("frozen_snapshot_hash") != frozen_snapshot_hash
        or payload.get("invocation") != invocation
        or payload.get("state_ordinal") != state_ordinal
        or payload.get("attempt") != attempt
    ):
        raise RunnerAttemptError("identity_mismatch", "worker final report 身份冲突")
    peak_rss = payload.get("peak_rss_mib")
    if (
        isinstance(peak_rss, bool)
        or not isinstance(peak_rss, (int, float))
        or not math.isfinite(float(peak_rss))
        or float(peak_rss) < 0.0
        or float(peak_rss) > FROZEN_RSS_LIMIT_MIB
    ):
        raise RunnerAttemptError(
            "rss_exceeded"
            if isinstance(peak_rss, (int, float))
            and not isinstance(peak_rss, bool)
            and math.isfinite(float(peak_rss))
            and float(peak_rss) > FROZEN_RSS_LIMIT_MIB
            else "worker_error",
            "worker final report 缺少合规 peak_rss_mib",
        )
    if status == "failed":
        failure_kind = payload.get("failure_kind")
        if failure_kind not in RECOVERABLE_FAILURE_KINDS | INVALID_FAILURE_KINDS:
            raise RunnerAttemptError(
                "identity_mismatch",
                "worker failure report 含未知 failure_kind",
            )
        raise RunnerAttemptError(
            str(failure_kind),
            f"worker 报告失败: {payload.get('error_type')}: "
            f"{payload.get('error_message')}",
        )
    if status != "success":
        raise RunnerAttemptError(
            "identity_mismatch",
            "worker final report status 非法",
        )
    restored = payload.get("restored")
    planned = payload.get("planned_queries")
    cache_hits = payload.get("cache_hits")
    evaluated = payload.get("evaluated_queries")
    fresh_keys = payload.get("fresh_query_keys")
    if (
        not isinstance(restored, bool)
        or isinstance(planned, bool)
        or not isinstance(planned, int)
        or planned <= 0
        or isinstance(cache_hits, bool)
        or not isinstance(cache_hits, int)
        or cache_hits < 0
        or isinstance(evaluated, bool)
        or not isinstance(evaluated, int)
        or evaluated < 0
        or cache_hits + evaluated != planned
        or not isinstance(fresh_keys, list)
        or len(fresh_keys) != evaluated
        or any(not isinstance(key, str) or not key for key in fresh_keys)
        or len(set(fresh_keys)) != len(fresh_keys)
    ):
        raise RunnerAttemptError(
            "identity_mismatch",
            "worker success report 查询计数非法",
        )
    return payload


def _launch_state_attempt(
    *,
    frozen_dir: Path,
    run_dir: Path,
    cache_dir: Path,
    config_path: Path,
    invocation: int,
    state_ordinal: int,
    attempt: int,
    remaining_state_s: float,
    execution_id: str,
    runner_hash: str,
    frozen_snapshot_hash: str,
) -> Dict[str, Any]:
    """启动并监控一个 state attempt。"""
    _reject_symlink_components(run_dir / "workers", label="worker 报告目录")
    report_path = _worker_report_path(
        run_dir,
        invocation,
        state_ordinal,
        attempt,
    )
    if report_path.exists():
        raise RunnerAttemptError(
            "identity_mismatch",
            f"worker final report 已存在，拒绝覆盖: {report_path}",
        )
    command = _worker_command(
        frozen_dir=frozen_dir,
        run_dir=run_dir,
        cache_dir=cache_dir,
        config_path=config_path,
        state_ordinal=state_ordinal,
        attempt=attempt,
        query_timeout_s=FROZEN_QUERY_TIMEOUT_S,
        execution_id=execution_id,
        runner_hash=runner_hash,
        invocation=invocation,
        frozen_snapshot_hash=frozen_snapshot_hash,
    )
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    outcome = monitor_worker(
        process,
        remaining_state_s=remaining_state_s,
        rss_limit_mib=FROZEN_RSS_LIMIT_MIB,
    )
    if outcome.violation is not None:
        failure_kind = {
            "state_timeout": "state_timeout",
            "rss_limit": "rss_exceeded",
            "rss_unavailable": "worker_error",
        }.get(outcome.violation, "worker_error")
        raise RunnerAttemptError(
            failure_kind,
            f"worker 护栏触发: {outcome.violation}",
        )
    if report_path.is_file():
        return _verify_worker_report(
            report_path,
            execution_id=execution_id,
            runner_hash=runner_hash,
            frozen_snapshot_hash=frozen_snapshot_hash,
            invocation=invocation,
            state_ordinal=state_ordinal,
            attempt=attempt,
        )
    if outcome.returncode != 0:
        detail = outcome.stderr.strip()[-2000:]
        raise RunnerAttemptError(
            "worker_exit",
            "worker 异常退出" + (f": {detail}" if detail else ""),
        )
    raise RunnerAttemptError("worker_error", "worker 成功退出但缺少 final report")


def _existing_manifest(run_dir: Path) -> Optional[Dict[str, Any]]:
    """读取并验证可恢复运行清单。"""
    path = run_dir / "execution_manifest.json"
    if not path.is_file():
        return None
    payload = _read_json(path)
    stored = payload.get("manifest_hash")
    unhashed = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if stored != _sha256_json(unhashed):
        raise CCODRunnerError("已有 execution manifest 内容哈希不一致")
    return payload


def _failure_kind(exc: BaseException) -> str:
    """把已知异常类型映射到稳定词表；文本仅用于旧异常兼容。"""
    if isinstance(exc, RunnerAttemptError):
        return exc.failure_kind
    if isinstance(exc, QueryDeadlineExceeded):
        return "query_timeout"
    try:
        from algorithms.ccod.continuation import CounterfactualError
        from algorithms.ccod.execution import ExecutionIdentityError

        if isinstance(exc, ExecutionIdentityError):
            return "identity_mismatch"
        if isinstance(exc, CounterfactualError):
            return "cache_corrupt"
    except ImportError:
        pass
    text = f"{type(exc).__name__}: {exc}".lower()
    if "querydeadline" in text or "query 超过" in text:
        return "query_timeout"
    if "state_timeout" in text or "deadline" in text or "累计" in text:
        return "state_timeout"
    if "rss_limit" in text or "rss 上限" in text:
        return "rss_exceeded"
    if "cache" in text and ("corrupt" in text or "哈希" in text or "hash" in text):
        return "cache_corrupt"
    if "identity" in text or "身份" in text:
        return "identity_mismatch"
    if "冻结" in text or "frozen" in text:
        return "frozen_drift"
    if "runner" in text and ("漂移" in text or "hash" in text):
        return "runner_drift"
    if "异常退出" in text or "returncode" in text:
        return "worker_exit"
    return "worker_error"


def _terminal_missing_queries(
    plan: Any,
    cache: Any,
    state_ordinal: int,
    default_kind: str,
) -> Dict[str, str]:
    """两次尝试耗尽后，仅为该 state 仍缺失/损坏的 query 物化失败。"""
    failures: Dict[str, str] = {}
    for query in plan.queries_by_state[state_ordinal]:
        key = str(query["query_key"])
        try:
            cached = cache.load(_mutable_json_copy(query["query_identity"]))
        except Exception:
            failures[key] = "cache_corrupt"
            continue
        if cached is None:
            failures[key] = default_kind
    return failures


def _restore_state_progress(
    existing: Optional[Mapping[str, Any]],
    state_count: int,
    *,
    now_unix_ns: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    """校验并恢复跨 invocation 的 attempt 与累计 wall-time 台账。"""
    raw_progress = existing.get("state_progress", {}) if existing is not None else {}
    if not isinstance(raw_progress, Mapping):
        raise CCODRunnerError("execution manifest 的 state_progress 非 object")
    unexpected = set(raw_progress) - {str(index) for index in range(state_count)}
    if unexpected:
        raise CCODRunnerError(f"state_progress 含计划外状态: {sorted(unexpected)}")
    current_ns = time.time_ns() if now_unix_ns is None else int(now_unix_ns)
    progress: Dict[str, Dict[str, Any]] = {}
    for state_ordinal in range(state_count):
        key = str(state_ordinal)
        raw = raw_progress.get(key, {})
        if not isinstance(raw, Mapping):
            raise CCODRunnerError(f"state_progress[{key}] 非 object")
        attempts = raw.get("attempts_started", 0)
        if (
            isinstance(attempts, bool)
            or not isinstance(attempts, int)
            or not (0 <= attempts <= FROZEN_MAX_ATTEMPTS)
        ):
            raise CCODRunnerError(f"state_progress[{key}] attempts 非法")
        try:
            elapsed = float.fromhex(str(raw.get("elapsed_s_hex", 0.0.hex())))
        except ValueError as exc:
            raise CCODRunnerError(f"state_progress[{key}] elapsed 非法") from exc
        if not math.isfinite(elapsed) or elapsed < 0.0:
            raise CCODRunnerError(f"state_progress[{key}] elapsed 非法")
        active_attempt = raw.get("active_attempt")
        active_started = raw.get("active_started_unix_ns")
        last_failure = raw.get("last_failure_kind")
        if last_failure is not None and last_failure not in (
            RECOVERABLE_FAILURE_KINDS | INVALID_FAILURE_KINDS
        ):
            raise CCODRunnerError(f"state_progress[{key}] failure_kind 非法")
        if active_attempt is not None:
            if (
                isinstance(active_attempt, bool)
                or not isinstance(active_attempt, int)
                or active_attempt != attempts
                or isinstance(active_started, bool)
                or not isinstance(active_started, int)
                or active_started <= 0
                or current_ns < active_started
            ):
                raise CCODRunnerError(f"state_progress[{key}] active attempt 非法")
            elapsed += (current_ns - active_started) / 1_000_000_000.0
            last_failure = "interrupted"
            active_attempt = None
            active_started = None
        elif active_started is not None:
            raise CCODRunnerError(f"state_progress[{key}] 孤立 active 时间戳")
        progress[key] = {
            "attempts_started": attempts,
            "elapsed_s_hex": elapsed.hex(),
            "active_attempt": active_attempt,
            "active_started_unix_ns": active_started,
            "last_failure_kind": last_failure,
        }
    return progress


def _terminal_failures_from_progress(
    plan: Any,
    cache: Any,
    progress: Mapping[str, Mapping[str, Any]],
) -> Dict[str, str]:
    """为已耗尽两次尝试且仍缺失的 query 重建稳定失败行。"""
    failures: Dict[str, str] = {}
    for state_ordinal in range(len(plan.selected_states)):
        state_progress = progress[str(state_ordinal)]
        attempts = int(state_progress["attempts_started"])
        last_failure = state_progress.get("last_failure_kind")
        if attempts < FROZEN_MAX_ATTEMPTS and last_failure not in INVALID_FAILURE_KINDS:
            continue
        default_kind = str(
            last_failure or "attempt_exhausted"
        )
        failures.update(
            _terminal_missing_queries(
                plan,
                cache,
                state_ordinal,
                default_kind,
            )
        )
    return failures


def run_diagnostic(
    *,
    frozen_dir: Path,
    run_dir: Path,
    cache_dir: Optional[Path] = None,
    config_path: Optional[Path] = None,
    max_states: Optional[int] = None,
    max_new_states: int = 1,
    verify_only: bool = False,
) -> Dict[str, Any]:
    """执行或只读验收冻结计划。

    ``max_states`` 限制允许触及的冻结前缀，``max_new_states`` 则限制
    本次 invocation 最多启动的尚未完成 state。两者分离后，同一
    ``--max-states 1`` 命令可在第二次运行时只验证热缓存，不会误启动 state 1。
    """
    runtime = require_frozen_runtime()
    if (
        isinstance(max_new_states, bool)
        or not isinstance(max_new_states, int)
        or not (1 <= max_new_states <= 100)
    ):
        raise CCODRunnerError("max_new_states 必须位于 [1, 100]")
    raw_cache = cache_dir if cache_dir is not None else run_dir / "cache"
    frozen, run, cache_path = validate_external_paths(frozen_dir, run_dir, raw_cache)
    config = (
        config_path.resolve()
        if config_path is not None
        else REPO_ROOT / "algorithms/ccod/configs/diagnostic_v1.json"
    )

    # verifier 前后绑定同一 runner 快照，防止旧内存代码配上新磁盘哈希。
    runner_hash_before_verification = runner_implementation_hash()
    # 完整 verifier 必须先于任何 Q_H 标签或计划执行。
    from scripts.prepare_ccod_diagnostic import verify_frozen_artifacts

    verification = verify_frozen_artifacts(frozen, config_path=config)
    runner_hash = runner_implementation_hash()
    if runner_hash != runner_hash_before_verification:
        raise CCODRunnerError("verifier 期间 runner 源码发生变化")
    from algorithms.ccod.cache import CounterfactualLabelCache
    from algorithms.ccod.execution import (
        build_execution_identity,
        execution_identity_hash,
        load_frozen_diagnostic_plan,
        summarize_signal_gate,
    )

    plan = load_frozen_diagnostic_plan(frozen)
    state_count = len(plan.selected_states)
    state_ordinals, _ = execution_prefix(plan, max_states)
    identity = build_execution_identity(plan, runner_implementation_hash=runner_hash)
    execution_id = execution_identity_hash(identity)
    frozen_snapshot = _frozen_snapshot(frozen)
    existing = _existing_manifest(run) if run.is_dir() else None
    if existing is not None and (
        existing.get("execution_id") != execution_id
        or existing.get("execution_identity") != identity
    ):
        raise CCODRunnerError("运行目录属于不同 execution identity")
    cache = CounterfactualLabelCache(cache_path)
    initially_cached = {
        str(query["query_key"])
        for query in plan.query_rows
        if cache.load(_mutable_json_copy(query["query_identity"])) is not None
    }

    if verify_only:
        rows = _result_rows_from_cache(
            plan,
            cache,
            execution_id,
            initially_cached,
            {},
        )
        summary = summarize_signal_gate(plan, identity, rows)
        return {
            "schema_version": RUNNER_SCHEMA_VERSION,
            "execution_id": execution_id,
            "execution_identity": identity,
            "run_id": plan.run_id,
            "runtime": runtime,
            "verification": verification,
            "frozen_snapshot_hash": frozen_snapshot["snapshot_hash"],
            "status": summary["execution_status"],
            "completed_states": summary["completed_states"],
            "completed_queries": summary["completed_queries"],
            "signal_gate": summary["signal_gate"],
            "method_decision": summary["method_decision"],
            "verify_only": True,
            "writes_performed": 0,
        }

    run.mkdir(parents=True, exist_ok=True)
    run_lock = _acquire_run_lock(run, execution_id)
    # 上锁后重新读取，避免两个父进程在只读预检与写清单之间发生 TOCTOU。
    existing = _existing_manifest(run)
    if existing is not None and (
        existing.get("execution_id") != execution_id
        or existing.get("execution_identity") != identity
    ):
        run_lock.close()
        raise CCODRunnerError("运行目录属于不同 execution identity")
    if existing is None:
        unexpected = [
            path
            for path in run.iterdir()
            if path.name not in {".DS_Store", ".runner.lock"}
        ]
        if unexpected:
            run_lock.close()
            raise CCODRunnerError("非空运行目录缺少 execution manifest")
        invocation = 1
    else:
        previous_invocation = existing.get("invocation")
        if (
            isinstance(previous_invocation, bool)
            or not isinstance(previous_invocation, int)
            or previous_invocation <= 0
        ):
            run_lock.close()
            raise CCODRunnerError("execution manifest 的 invocation 非法")
        invocation = previous_invocation + 1
    progress = _restore_state_progress(existing, state_count)

    manifest: Dict[str, Any] = {
        "schema_version": RUNNER_SCHEMA_VERSION,
        "execution_id": execution_id,
        "execution_identity": identity,
        "run_id": plan.run_id,
        "runtime": runtime,
        "verification": verification,
        "frozen_snapshot_hash": frozen_snapshot["snapshot_hash"],
        "paths": {
            "frozen_dir": str(frozen),
            "run_dir": str(run),
            "cache_dir": str(cache_path),
        },
        "cache_is_source_of_truth": True,
        "invocation": invocation,
        "max_states_this_invocation": max_states,
        "max_new_states_this_invocation": max_new_states,
        "state_ordinals_this_invocation": list(state_ordinals),
        "verify_only": False,
        "status": "running",
        "completed_states": 0,
        "completed_queries": 0,
        "signal_gate": "not_evaluated",
        "failure": None,
        "state_progress": progress,
    }
    _write_json(run / "execution_manifest.json", manifest, "manifest_hash")
    reports: List[Dict[str, Any]] = []
    pending_error: Optional[RunnerAttemptError] = None
    newly_started_states = 0

    try:
        for state_ordinal in state_ordinals:
            if publish_state_checkpoint_if_complete(
                plan,
                execution_id,
                state_ordinal,
                cache,
                run / "checkpoints",
            ):
                continue
            state_progress = progress[str(state_ordinal)]
            attempts_started = int(state_progress["attempts_started"])
            if attempts_started >= FROZEN_MAX_ATTEMPTS:
                continue
            if newly_started_states >= max_new_states:
                break
            newly_started_states += 1
            final_error: Optional[RunnerAttemptError] = None
            while attempts_started < FROZEN_MAX_ATTEMPTS:
                elapsed = float.fromhex(str(state_progress["elapsed_s_hex"]))
                remaining = FROZEN_STATE_TIMEOUT_S - elapsed
                if remaining <= 0.0:
                    final_error = RunnerAttemptError(
                        "state_timeout",
                        "state 累计 deadline 已耗尽",
                    )
                    state_progress["attempts_started"] = FROZEN_MAX_ATTEMPTS
                    state_progress["last_failure_kind"] = final_error.failure_kind
                    _write_json(
                        run / "execution_manifest.json",
                        manifest,
                        "manifest_hash",
                    )
                    break
                attempt = attempts_started + 1
                state_progress.update(
                    {
                        "attempts_started": attempt,
                        "active_attempt": attempt,
                        "active_started_unix_ns": time.time_ns(),
                        "last_failure_kind": None,
                    }
                )
                _write_json(
                    run / "execution_manifest.json",
                    manifest,
                    "manifest_hash",
                )
                attempt_started = time.monotonic()
                try:
                    report = _launch_state_attempt(
                        frozen_dir=frozen,
                        run_dir=run,
                        cache_dir=cache_path,
                        config_path=config,
                        invocation=invocation,
                        state_ordinal=state_ordinal,
                        attempt=attempt,
                        remaining_state_s=remaining,
                        execution_id=execution_id,
                        runner_hash=runner_hash,
                        frozen_snapshot_hash=frozen_snapshot["snapshot_hash"],
                    )
                    if not publish_state_checkpoint_if_complete(
                        plan,
                        execution_id,
                        state_ordinal,
                        cache,
                        run / "checkpoints",
                    ):
                        raise CCODRunnerError("worker 成功但父进程复核 cache 不完整")
                    reports.append(report)
                    final_error = None
                except RunnerAttemptError as exc:
                    final_error = exc
                except KeyboardInterrupt:
                    final_error = RunnerAttemptError(
                        "interrupted",
                        "父进程收到键盘中断",
                    )
                except Exception as exc:
                    final_error = RunnerAttemptError("worker_error", str(exc))
                finally:
                    updated_elapsed = elapsed + (time.monotonic() - attempt_started)
                    state_progress.update(
                        {
                            "elapsed_s_hex": updated_elapsed.hex(),
                            "active_attempt": None,
                            "active_started_unix_ns": None,
                            "last_failure_kind": (
                                final_error.failure_kind
                                if final_error is not None
                                else None
                            ),
                        }
                    )
                    _write_json(
                        run / "execution_manifest.json",
                        manifest,
                        "manifest_hash",
                    )
                if final_error is None:
                    break
                if not final_error.recoverable:
                    pending_error = final_error
                    break
                if final_error.failure_kind == "interrupted":
                    break
                attempts_started = attempt
            if final_error is not None:
                if pending_error is None and not final_error.recoverable:
                    pending_error = final_error
                break
    finally:
        rows: List[Dict[str, Any]] = []
        summary: Dict[str, Any] = {}
        try:
            terminal_failures = _terminal_failures_from_progress(
                plan,
                cache,
                progress,
            )
            rows = _result_rows_from_cache(
                plan,
                cache,
                execution_id,
                initially_cached,
                terminal_failures,
            )
        except Exception as exc:
            if pending_error is None:
                pending_error = RunnerAttemptError("cache_corrupt", str(exc))
        if _frozen_snapshot(frozen) != frozen_snapshot:
            pending_error = RunnerAttemptError(
                "frozen_drift",
                "执行期间冻结包内容发生变化",
            )
        if runner_implementation_hash() != runner_hash:
            pending_error = RunnerAttemptError(
                "runner_drift",
                "执行期间 runner 源码发生变化",
            )
        successful_rows = [row for row in rows if row.get("status") == "success"]
        successful_keys = {str(row["query_key"]) for row in successful_rows}
        completed_states = sum(
            all(
                str(query["query_key"]) in successful_keys
                for query in plan.queries_by_state[index]
            )
            for index in range(state_count)
        )
        try:
            if pending_error is not None and not pending_error.recoverable:
                # invalid 运行禁止触发完整 gate；从空输入模板构造无结论摘要。
                summary = summarize_signal_gate(plan, identity, [])
                summary.update(
                    {
                        "execution_status": "invalid",
                        "status": "invalid",
                        "signal_gate": "not_evaluated",
                        "decision": "not_evaluated",
                        "method_decision": None,
                        "passed": None,
                        "completed_states": completed_states,
                        "completed_queries": len(successful_rows),
                        "failed_queries": max(1, summary.get("failed_queries", 0)),
                        "failure_counts": {pending_error.failure_kind: 1},
                        "scientific_results_hash": None,
                    }
                )
            else:
                summary = summarize_signal_gate(plan, identity, rows)
            _write_jsonl(run / "query_results.jsonl", rows)
            _write_json(run / "signal_summary.json", summary, "summary_hash")
        except Exception as exc:
            if pending_error is None:
                pending_error = RunnerAttemptError("cache_corrupt", str(exc))
            summary = {
                "execution_status": "invalid",
                "signal_gate": "not_evaluated",
                "method_decision": None,
            }
        status = str(summary.get("execution_status", "invalid"))
        if status not in {"complete", "incomplete", "invalid"}:
            status = "invalid"
        manifest.update(
            {
                "status": status,
                "completed_states": completed_states,
                "completed_queries": len(successful_rows),
                "signal_gate": summary.get("signal_gate", "not_evaluated"),
                "method_decision": summary.get("method_decision"),
                "failure": (
                    f"{type(pending_error).__name__}: {pending_error}"
                    if pending_error is not None
                    else None
                ),
                "state_reports": reports,
                "newly_started_states": newly_started_states,
                "fresh_queries_this_invocation": sum(
                    str(row["query_key"]) not in initially_cached
                    for row in successful_rows
                ),
                "peak_parent_rss_mib": _peak_rss_mib(),
            }
        )
        _write_json(run / "execution_manifest.json", manifest, "manifest_hash")

    final_manifest = _read_json(run / "execution_manifest.json")
    run_lock.close()
    return final_manifest


def _positive_int(text: str) -> int:
    """解析命令行严格正整数。"""
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("必须为整数") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError("必须为正整数")
    return value


def _positive_float(text: str) -> float:
    """解析命令行严格正浮点数。"""
    try:
        value = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("必须为数值") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError("必须为有限正数")
    return value


def build_parser() -> argparse.ArgumentParser:
    """构造公开 runner 与内部 worker 共用的命令行解析器。"""
    parser = argparse.ArgumentParser(description="执行冻结的 CCOD 单机诊断计划")
    parser.add_argument("--frozen-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "algorithms/ccod/configs/diagnostic_v1.json",
    )
    parser.add_argument(
        "--max-states",
        type=_positive_int,
        default=None,
        help="只允许执行冻结 state 的前 N 个（默认为全计划）",
    )
    parser.add_argument(
        "--max-new-states",
        type=_positive_int,
        default=1,
        help="本次最多启动多少个尚未完成的 state（默认 1）",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="只验收冻结包与已有 cache，不创建运行文件或启动 worker",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--state-ordinal", type=int, default=None, help=argparse.SUPPRESS
    )
    parser.add_argument("--attempt", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--invocation",
        type=_positive_int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--query-timeout-s",
        type=_positive_float,
        default=FROZEN_QUERY_TIMEOUT_S,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--execution-id", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--runner-hash", default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--frozen-snapshot-hash",
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """命令行入口。"""
    args = build_parser().parse_args(argv)
    if args.worker:
        required = (
            args.cache_dir,
            args.state_ordinal,
            args.attempt,
            args.invocation,
            args.execution_id,
            args.runner_hash,
            args.frozen_snapshot_hash,
        )
        if any(value is None for value in required):
            print("worker 缺少内部参数", file=sys.stderr)
            return 2
        return _worker_main(args)
    try:
        manifest = run_diagnostic(
            frozen_dir=args.frozen_dir,
            run_dir=args.run_dir,
            cache_dir=args.cache_dir,
            config_path=args.config,
            max_states=args.max_states,
            max_new_states=args.max_new_states,
            verify_only=args.verify_only,
        )
    except (CCODRunnerError, OSError, ValueError) as exc:
        print(f"CCOD runner 失败: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "execution_id": manifest["execution_id"],
                "status": manifest["status"],
                "completed_states": manifest["completed_states"],
                "completed_queries": manifest["completed_queries"],
                "signal_gate": manifest["signal_gate"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 1 if manifest["status"] == "invalid" else 0


if __name__ == "__main__":
    raise SystemExit(main())
