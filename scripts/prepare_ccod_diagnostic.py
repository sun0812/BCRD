#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""冻结 CCOD 100-state catalog 与无标签 query plan。

本脚本只读取场景 JSON 与六个 dev balanced final schedules，生成紧凑 replay
sidecars、去重 catalog 和最多 100×16 个 ActionKeys；不会读取或复制旧的 26 GiB
trajectory JSONL，也不会执行 continuation 或查看任何 Q_H 标签。
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, suppress
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from algorithms.ccod.cache import build_cache_identity, cache_key
from algorithms.ccod.continuation import (
    ContinuationConfig,
    continuation_implementation_hash,
)
from algorithms.ccod.diagnostic import (
    STATE_CATALOG_SCHEMA_VERSION,
    annotate_catalog_selection,
    catalog_prelabel_audit,
    merge_state_catalog,
    objective_name_from_weights,
    select_preregistered_states,
    selection_summary,
    validate_diagnostic_runtime,
)
from algorithms.ccod.instances import (
    build_recommended_split_manifest,
    discover_instance_identities,
)
from algorithms.ccod.selectors import diagnostic_query_prefix
from schedulers.scenario_loader import load_scheduling_problem_from_json
from schedulers.state_replay import (
    ConstraintConfig,
    EnumeratorConfig,
    ObjectiveConfig,
    candidate_action_key,
    canonical_json_bytes,
    restore_state,
    sha256_file,
    sha256_json,
)
from scripts.batch_export_trajectories import parse_schedule_filename
from scripts.build_replay_manifests import (
    REPLAY_IMPLEMENTATION_FILES,
    build_for_paths,
    collect_code_provenance,
    objective_config_from_schedule_path,
)


SPLIT_MANIFEST_SCHEMA_VERSION = "eosbench-ccod-split-manifest-v1"
CATALOG_HEADER_SCHEMA_VERSION = "eosbench-ccod-state-catalog-header-v1"
QUERY_PLAN_SCHEMA_VERSION = "eosbench-ccod-query-plan-v1"
RUN_MANIFEST_SCHEMA_VERSION = "eosbench-ccod-diagnostic-run-v1"
SOURCE_ID_SCHEMA_VERSION = "eosbench-ccod-source-id-v1"
VALIDATION_SUBSET_SCHEMA_VERSION = "eosbench-ccod-validation-subsets-v1"

PREPARATION_IMPLEMENTATION_FILES = (
    "algorithms/candidate_pool.py",
    "algorithms/objectives.py",
    "algorithms/random_utils.py",
    "algorithms/ccod/cache.py",
    "algorithms/ccod/continuation.py",
    "algorithms/ccod/diagnostic.py",
    "algorithms/ccod/instances.py",
    "algorithms/ccod/selectors.py",
    "schedulers/balance_utils.py",
    "schedulers/constraint_model.py",
    "schedulers/scenario_loader.py",
    "schedulers/state_replay.py",
    "schedulers/timeliness_utils.py",
    "schedulers/transition_utils.py",
    "scripts/batch_export_trajectories.py",
    "scripts/build_replay_manifests.py",
    "scripts/prepare_ccod_diagnostic.py",
)


class DiagnosticPreparationError(ValueError):
    """预注册输入或产物无法闭合验证时抛出的异常。"""


def _preparation_implementation_bundle() -> Dict[str, Any]:
    """冻结会影响 catalog、query plan 与缓存身份的本地实现。"""
    payload: Dict[str, Any] = {
        "schema_version": "eosbench-ccod-preparation-implementation-v1",
        "python_implementation": sys.implementation.name,
        "python_version": list(sys.version_info[:3]),
        "files": {
            relative_path: sha256_file(REPO_ROOT / relative_path)
            for relative_path in PREPARATION_IMPLEMENTATION_FILES
        },
    }
    payload["bundle_hash"] = sha256_json(payload)
    return payload


def _assert_preparation_implementation_unchanged(
    expected: Mapping[str, Any],
) -> None:
    actual = _preparation_implementation_bundle()
    if actual != dict(expected):
        expected_files = expected.get("files", {})
        actual_files = actual.get("files", {})
        changed_files = sorted(
            path
            for path in set(expected_files) | set(actual_files)
            if expected_files.get(path) != actual_files.get(path)
        )
        details = ", ".join(changed_files) if changed_files else "运行时身份"
        raise DiagnosticPreparationError(
            "预注册运行期间关键源码发生变化，整包结果作废: " + details
        )


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise DiagnosticPreparationError(f"无法读取 JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise DiagnosticPreparationError(f"{path} 必须包含 JSON object")
    return value


def _scientific_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    """对按稳定顺序排列的规范行计算内容哈希。"""
    return sha256_json([dict(row) for row in rows])


def _publish_bytes(path: Path, payload: bytes) -> None:
    """原子发布不可变产物；已存在时只接受逐字节相同内容。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        if path.read_bytes() != payload:
            raise DiagnosticPreparationError(f"已冻结产物内容冲突: {path}")
        return
    temporary: Path | None = None
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
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise DiagnosticPreparationError(f"并发发布产物冲突: {path}")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _publish_json(path: Path, payload: Mapping[str, Any]) -> None:
    body = json.dumps(
        dict(payload),
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    _publish_bytes(path, body)


def _publish_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    body = b"".join(canonical_json_bytes(dict(row)) + b"\n" for row in rows)
    _publish_bytes(path, body)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """读取 JSONL，并拒绝空行和非 object 记录。"""
    rows: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.endswith("\n") or not line.strip():
                    raise DiagnosticPreparationError(
                        f"{path}:{line_number} 不是完整 JSONL 记录"
                    )
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise DiagnosticPreparationError(
                        f"{path}:{line_number} 必须为 JSON object"
                    )
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise DiagnosticPreparationError(f"无法读取 JSONL {path}: {exc}") from exc
    if not rows:
        raise DiagnosticPreparationError(f"JSONL 不能为空: {path}")
    return rows


def _verify_embedded_hash(
    payload: Mapping[str, Any],
    hash_field: str,
    *,
    label: str,
) -> None:
    expected = payload.get(hash_field)
    actual = sha256_json(
        {key: value for key, value in payload.items() if key != hash_field}
    )
    if expected != actual:
        raise DiagnosticPreparationError(f"{label} 的 {hash_field} 不一致")


def _strict_relative_file(root: Path, raw_path: Any, *, label: str) -> Path:
    """解析冻结包内引用，拒绝绝对路径、逃逸、符号链接和缺失文件。"""
    if not isinstance(raw_path, str) or not raw_path:
        raise DiagnosticPreparationError(f"{label} 缺少相对路径")
    relative = Path(raw_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise DiagnosticPreparationError(f"{label} 不是安全相对路径: {raw_path}")
    candidate = root / relative
    resolved_root = root.resolve()
    try:
        candidate.resolve().relative_to(resolved_root)
    except ValueError as exc:
        raise DiagnosticPreparationError(f"{label} 逃逸冻结目录") from exc
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise DiagnosticPreparationError(f"{label} 不允许符号链接: {cursor}")
    if not candidate.is_file():
        raise DiagnosticPreparationError(f"{label} 文件不存在: {candidate}")
    return candidate


def _strict_repo_file(raw_path: Any, *, label: str) -> Path:
    """解析仓库相对输入引用，确保不会越过仓库根目录。"""
    return _strict_relative_file(REPO_ROOT, raw_path, label=label)


def _reject_existing_symlink_components(path: Path) -> None:
    """拒绝发布路径中已经存在的符号链接组件。"""
    for component in reversed((path, *path.parents)):
        if component.is_symlink():
            raise FileExistsError(f"发布路径不允许符号链接: {component}")


def _verify_trace_sidecars(
    out_dir: Path,
    catalog_manifest: Mapping[str, Any],
    catalog_rows: Sequence[Mapping[str, Any]],
    source_inventory: Mapping[str, Any],
) -> set[Path]:
    """校验六份 compact trace 与 600 条 state sidecar，不触碰旧轨迹。"""
    inventory = catalog_manifest.get("trace_inventory")
    if not isinstance(inventory, list) or len(inventory) != 6:
        raise DiagnosticPreparationError("trace inventory 必须恰有六个来源")
    expected_files: set[Path] = set()
    trace_by_hash: Dict[str, Dict[str, Any]] = {}
    state_by_trace: Dict[str, Dict[str, Dict[str, Any]]] = {}
    sidecar_pairs: set[Tuple[str, str]] = set()
    frozen_code: Dict[str, Any] | None = None
    for entry in inventory:
        if not isinstance(entry, Mapping):
            raise DiagnosticPreparationError("trace inventory row 必须为 object")
        trace_ref = entry.get("trace_ref")
        if not isinstance(trace_ref, Mapping):
            raise DiagnosticPreparationError("trace inventory 缺少 trace_ref")
        trace_path = _strict_relative_file(
            out_dir,
            trace_ref.get("relative_path"),
            label="trace_ref.relative_path",
        )
        states_path = _strict_relative_file(
            out_dir,
            trace_ref.get("states_relative_path"),
            label="trace_ref.states_relative_path",
        )
        expected_files.update({trace_path, states_path})
        if sha256_file(trace_path) != trace_ref.get("trace_file_hash"):
            raise DiagnosticPreparationError("trace sidecar 文件哈希不一致")
        if sha256_file(states_path) != trace_ref.get("states_file_hash"):
            raise DiagnosticPreparationError("states sidecar 文件哈希不一致")
        trace = _load_json(trace_path)
        _verify_embedded_hash(trace, "trace_hash", label="trace manifest")
        for field_name in ("trace_id", "trace_hash"):
            expected = entry.get(field_name)
            if trace.get(field_name) != expected or trace_ref.get(field_name) != expected:
                raise DiagnosticPreparationError(
                    f"trace inventory 的 {field_name} 引用不一致"
                )
        trace_hash = str(trace["trace_hash"])
        if trace_hash in trace_by_hash:
            raise DiagnosticPreparationError("trace inventory 含重复 trace_hash")

        scenario = trace.get("scenario")
        source_schedule = trace.get("source_schedule")
        if not isinstance(scenario, Mapping) or not isinstance(
            source_schedule, Mapping
        ):
            raise DiagnosticPreparationError("trace 缺少 scenario/source_schedule")
        scenario_path = _strict_repo_file(
            scenario.get("path"), label="trace scenario"
        )
        schedule_path = _strict_repo_file(
            source_schedule.get("path"), label="trace source schedule"
        )
        if sha256_file(scenario_path) != scenario.get("sha256"):
            raise DiagnosticPreparationError("trace scenario 文件哈希不一致")
        if sha256_file(schedule_path) != source_schedule.get("sha256"):
            raise DiagnosticPreparationError("trace schedule 文件哈希不一致")
        if entry.get("source_schedule_hash") != source_schedule.get("sha256"):
            raise DiagnosticPreparationError("trace inventory schedule 哈希不一致")
        instance_alias = str(entry.get("instance_alias", ""))
        source_family = str(entry.get("source_family", ""))
        instance_rows = source_inventory.get("instances")
        dev_sources = source_inventory.get("dev_sources")
        if not isinstance(instance_rows, list) or not isinstance(
            dev_sources, Mapping
        ):
            raise DiagnosticPreparationError("source inventory 结构非法")
        matching_instances = [
            row
            for row in instance_rows
            if isinstance(row, Mapping)
            and row.get("instance_alias") == instance_alias
        ]
        matching_schedules = [
            row
            for row in dev_sources.get(instance_alias, [])
            if isinstance(row, Mapping)
            and row.get("source_family") == source_family
        ]
        if len(matching_instances) != 1 or len(matching_schedules) != 1:
            raise DiagnosticPreparationError("trace 来源无法唯一映射到冻结库存")
        expected_instance = matching_instances[0]
        expected_schedule = matching_schedules[0]
        if (
            scenario.get("sha256") != expected_instance.get("raw_hash")
            or Path(str(scenario.get("path", ""))).name
            != expected_instance.get("filename")
        ):
            raise DiagnosticPreparationError("trace scenario 偏离冻结库存")
        if (
            source_schedule.get("sha256") != expected_schedule.get("sha256")
            or Path(str(source_schedule.get("path", ""))).name
            != expected_schedule.get("filename")
        ):
            raise DiagnosticPreparationError("trace schedule 偏离冻结库存")

        code = trace.get("code")
        if not isinstance(code, Mapping):
            raise DiagnosticPreparationError("trace 缺少 code provenance")
        normalized_code = dict(code)
        if frozen_code is None:
            frozen_code = normalized_code
        elif normalized_code != frozen_code:
            raise DiagnosticPreparationError("六份 trace 的 code provenance 不一致")

        states = _read_jsonl(states_path)
        if len(states) != entry.get("state_count"):
            raise DiagnosticPreparationError("trace state_count 与 sidecar 不一致")
        if (
            len(states) != len(trace.get("task_ids", []))
            or len(states) != len(trace.get("observed_action_keys", []))
        ):
            raise DiagnosticPreparationError("trace 任务/observed/state 数量不一致")
        by_manifest_hash: Dict[str, Dict[str, Any]] = {}
        for state in states:
            _verify_embedded_hash(
                state,
                "state_manifest_hash",
                label="state manifest",
            )
            if (
                state.get("trace_id") != trace.get("trace_id")
                or state.get("trace_hash") != trace_hash
            ):
                raise DiagnosticPreparationError("state manifest 引用了错误 trace")
            state_manifest_hash = str(state["state_manifest_hash"])
            if state_manifest_hash in by_manifest_hash:
                raise DiagnosticPreparationError("state sidecar 含重复 manifest hash")
            by_manifest_hash[state_manifest_hash] = state
            sidecar_pairs.add((trace_hash, state_manifest_hash))
        trace_by_hash[trace_hash] = {
            "entry": dict(entry),
            "trace": trace,
        }
        state_by_trace[trace_hash] = by_manifest_hash

    if frozen_code is None:
        raise DiagnosticPreparationError("trace code provenance 为空")
    implementation_files = frozen_code.get("implementation_files")
    preparation_files = catalog_manifest.get(
        "preparation_implementation", {}
    ).get("files")
    if not isinstance(implementation_files, Mapping) or not isinstance(
        preparation_files, Mapping
    ):
        raise DiagnosticPreparationError("实现文件 provenance 缺失")
    missing_replay_files = [
        path for path in REPLAY_IMPLEMENTATION_FILES if path not in preparation_files
    ]
    if missing_replay_files:
        raise DiagnosticPreparationError(
            f"preparation 缺少 replay 实现文件: {missing_replay_files}"
        )
    expected_replay_files = {
        path: preparation_files[path] for path in REPLAY_IMPLEMENTATION_FILES
    }
    if dict(implementation_files) != expected_replay_files:
        raise DiagnosticPreparationError("trace 与 preparation 实现哈希不一致")

    catalog_alias_pairs: List[Tuple[str, str]] = []
    for catalog_row in catalog_rows:
        aliases = catalog_row.get("source_aliases")
        canonical = catalog_row.get("canonical_source")
        if not isinstance(aliases, list) or not isinstance(canonical, Mapping):
            raise DiagnosticPreparationError("catalog 来源引用结构非法")
        catalog_alias_pairs.extend(
            (
                str(source.get("trace_hash")),
                str(source.get("state_manifest_hash")),
            )
            for source in aliases
            if isinstance(source, Mapping)
        )
        for source in [canonical, *aliases]:
            if not isinstance(source, Mapping):
                raise DiagnosticPreparationError("catalog source alias 不是 object")
            trace_hash = str(source.get("trace_hash"))
            trace_record = trace_by_hash.get(trace_hash)
            if trace_record is None:
                raise DiagnosticPreparationError("catalog 引用了 inventory 外 trace")
            if source.get("trace_ref") != trace_record["entry"].get("trace_ref"):
                raise DiagnosticPreparationError("catalog trace_ref 与 inventory 不一致")
            state = state_by_trace[trace_hash].get(
                str(source.get("state_manifest_hash"))
            )
            if state is None:
                raise DiagnosticPreparationError("catalog 引用了 sidecar 外状态")
            for field_name in (
                "state_hash",
                "physical_state_hash",
                "step",
                "task_id",
                "schedule_hash",
                "ordered_candidate_hash",
                "candidate_membership_hash",
                "candidate_count",
            ):
                if state.get(field_name) != catalog_row.get(field_name):
                    raise DiagnosticPreparationError(
                        f"catalog/source 的 {field_name} 不一致"
                    )
            if state.get("observed_action_key") != source.get(
                "observed_action_key"
            ):
                raise DiagnosticPreparationError("source observed action 不一致")
            scenario_ref = source.get("scenario_ref")
            trace_scenario = trace_record["trace"]["scenario"]
            if not isinstance(scenario_ref, Mapping) or (
                scenario_ref.get("relative_path") != trace_scenario.get("path")
                or scenario_ref.get("sha256") != trace_scenario.get("sha256")
            ):
                raise DiagnosticPreparationError("catalog scenario_ref 不一致")
        if canonical.get("state_manifest") != state_by_trace[
            str(canonical["trace_hash"])
        ].get(str(canonical["state_manifest_hash"])):
            raise DiagnosticPreparationError("canonical state manifest 不一致")
    if len(catalog_alias_pairs) != len(set(catalog_alias_pairs)):
        raise DiagnosticPreparationError("catalog source aliases 含重复 sidecar 引用")
    if set(catalog_alias_pairs) != sidecar_pairs:
        raise DiagnosticPreparationError("catalog 与全部 sidecar states 不是双向完备映射")
    return expected_files


def _verify_query_prefixes_by_replay(
    out_dir: Path,
    selected_by_ordinal: Mapping[int, Mapping[str, Any]],
    grouped_queries: Mapping[int, Sequence[Mapping[str, Any]]],
    config: Mapping[str, Any],
) -> None:
    """独立恢复 100 个状态并重算冻结 query prefix。"""
    problems: Dict[str, Any] = {}
    traces: Dict[str, Dict[str, Any]] = {}
    for state_ordinal, state in selected_by_ordinal.items():
        instance_alias = str(state["instance_alias"])
        canonical_source = state["canonical_source"]
        scenario_path = _strict_repo_file(
            canonical_source["scenario_ref"]["relative_path"],
            label="selected scenario_ref",
        )
        if instance_alias not in problems:
            problems[instance_alias] = load_scheduling_problem_from_json(
                scenario_path
            )
        trace_ref = canonical_source["trace_ref"]
        trace_hash = str(trace_ref["trace_hash"])
        if trace_hash not in traces:
            trace_path = _strict_relative_file(
                out_dir,
                trace_ref["relative_path"],
                label="selected trace_ref",
            )
            traces[trace_hash] = _load_json(trace_path)
        problem = problems[instance_alias]
        replayed = restore_state(
            problem,
            traces[trace_hash],
            canonical_source["state_manifest"],
            scenario_path=scenario_path,
            verify=True,
        )
        action_keys = [
            candidate_action_key(problem, replayed.task_id, action)
            for action in replayed.candidates
        ]
        if len(action_keys) != int(state["candidate_count"]):
            raise DiagnosticPreparationError("重放候选数与 selected state 不一致")
        expected_prefix = diagnostic_query_prefix(
            str(state["state_hash"]),
            action_keys,
            state["observed_action_key"],
            budget=int(config["query"]["budget_per_state"]),
            seed=int(config["seed"]),
        )
        actual_queries = list(grouped_queries[state_ordinal])
        if len(expected_prefix) != len(actual_queries):
            raise DiagnosticPreparationError("重算 query prefix 长度不一致")
        for action_ordinal, (expected, actual) in enumerate(
            zip(expected_prefix, actual_queries)
        ):
            if (
                actual.get("instance_alias") != instance_alias
                or actual.get("action_ordinal") != action_ordinal
                or actual.get("action_key") != expected["action_key"]
                or actual.get("roles") != expected["selection_sources"]
            ):
                raise DiagnosticPreparationError(
                    "query prefix 无法由恢复后的 canonical candidates 重算"
                )


def verify_frozen_artifacts(
    out_dir: Path,
    *,
    config_path: Path | None = None,
) -> Dict[str, Any]:
    """不执行 Q_H，闭合验收已发布完整包及当前代码/输入身份。"""
    out_dir = out_dir.resolve()
    config_path = (
        config_path.resolve()
        if config_path is not None
        else REPO_ROOT / "algorithms/ccod/configs/diagnostic_v1.json"
    )
    config = _load_json(config_path)
    validate_diagnostic_runtime(config)
    config_hash = sha256_json(config)
    inventory_path = _strict_repo_file(
        config["source_inventory"]["path"],
        label="source inventory",
    )
    if sha256_file(inventory_path) != config["source_inventory"]["file_hash"]:
        raise DiagnosticPreparationError("当前 source inventory 哈希不一致")
    source_inventory = _load_json(inventory_path)
    if (
        source_inventory.get("schema_version")
        != config["source_inventory"]["schema_version"]
    ):
        raise DiagnosticPreparationError("当前 source inventory schema 不一致")

    split_manifest = _load_json(out_dir / "split_manifest.json")
    catalog_manifest = _load_json(out_dir / "catalog_manifest.json")
    run_manifest = _load_json(out_dir / "run_manifest.json")
    validation_subsets = _load_json(out_dir / "validation_subsets.json")
    for payload, hash_field, label in (
        (split_manifest, "manifest_hash", "split_manifest"),
        (catalog_manifest, "manifest_hash", "catalog_manifest"),
        (run_manifest, "manifest_hash", "run_manifest"),
        (validation_subsets, "subsets_hash", "validation_subsets"),
    ):
        _verify_embedded_hash(payload, hash_field, label=label)

    catalog_records = _read_jsonl(out_dir / "state_catalog.jsonl")
    catalog_header, catalog_rows = catalog_records[0], catalog_records[1:]
    _verify_embedded_hash(catalog_header, "header_hash", label="catalog header")
    for row in catalog_rows:
        _verify_embedded_hash(row, "row_hash", label="catalog row")
    catalog_hash = _scientific_hash(catalog_rows)
    if catalog_hash != catalog_manifest.get("catalog_hash"):
        raise DiagnosticPreparationError("catalog_hash 与 state_catalog 不一致")
    if len(catalog_rows) != catalog_header.get("pool_states"):
        raise DiagnosticPreparationError("catalog pool_states 数量不一致")

    selected = _read_jsonl(out_dir / "selected_states.jsonl")
    for row in selected:
        _verify_embedded_hash(row, "row_hash", label="selected state row")
    selection_hash = _scientific_hash(selected)
    if selection_hash != catalog_manifest.get("selection_hash"):
        raise DiagnosticPreparationError("selection_hash 不一致")
    if len(selected) != catalog_header.get("selected_states"):
        raise DiagnosticPreparationError("selected_states 数量不一致")
    selected_hashes = [str(row["state_hash"]) for row in selected]
    if len(selected_hashes) != len(set(selected_hashes)):
        raise DiagnosticPreparationError("selected_states 含重复 state_hash")
    selected_ordinals = [
        int(row["selection"]["selected_ordinal"]) for row in selected
    ]
    if selected_ordinals != list(range(len(selected))):
        raise DiagnosticPreparationError("selected_ordinal 不连续或有重复")
    selected_by_hash = {
        str(row["state_hash"]): row for row in selected
    }
    catalog_selected = [
        str(row["state_hash"])
        for row in catalog_rows
        if row.get("selection", {}).get("selected") is True
    ]
    if set(catalog_selected) != set(selected_hashes):
        raise DiagnosticPreparationError("catalog 与 selected_states 选择不一致")
    expected_selected = select_preregistered_states(catalog_rows, config)
    if selected != expected_selected:
        raise DiagnosticPreparationError("selection 无法从完整 catalog 确定性重算")
    if catalog_manifest.get("summary") != selection_summary(selected):
        raise DiagnosticPreparationError("selection summary 无法从 selected 重算")
    prelabel_audit = catalog_manifest.get("prelabel_audit")
    if not isinstance(prelabel_audit, Mapping):
        raise DiagnosticPreparationError("catalog manifest 缺少 prelabel audit")
    _verify_embedded_hash(prelabel_audit, "audit_hash", label="prelabel audit")
    if dict(prelabel_audit) != catalog_prelabel_audit(catalog_rows, config):
        raise DiagnosticPreparationError("prelabel audit 无法从 catalog 重算")

    expected_preparation = _preparation_implementation_bundle()
    if catalog_manifest.get("preparation_implementation") != expected_preparation:
        raise DiagnosticPreparationError("当前 preparation 实现已偏离冻结包")
    continuation = ContinuationConfig(
        horizon=int(config["query"]["continuation_horizon"])
    )
    expected_continuation_hash = continuation_implementation_hash()
    shared = {
        "config_hash": config_hash,
        "split_manifest_hash": split_manifest["manifest_hash"],
        "source_inventory_hash": config["source_inventory"]["file_hash"],
        "catalog_hash": catalog_hash,
        "selection_hash": selection_hash,
        "validation_subsets_hash": validation_subsets["subsets_hash"],
        "preparation_implementation_hash": expected_preparation["bundle_hash"],
    }
    for field_name, expected in shared.items():
        for label, payload in (
            ("catalog header", catalog_header),
            ("catalog manifest", catalog_manifest),
            ("run manifest", run_manifest),
        ):
            if payload.get(field_name) != expected:
                raise DiagnosticPreparationError(
                    f"{label} 的 {field_name} 交叉引用不一致"
                )
    if split_manifest.get("source_inventory_hash") != shared[
        "source_inventory_hash"
    ]:
        raise DiagnosticPreparationError("split/source inventory 哈希不一致")
    if run_manifest.get("catalog_manifest_hash") != catalog_manifest.get(
        "manifest_hash"
    ):
        raise DiagnosticPreparationError("run 未绑定完整 catalog manifest")

    expected_validation = select_validation_subsets(selected, config)
    if validation_subsets != expected_validation:
        raise DiagnosticPreparationError("validation subsets 无法从 selection 重算")
    exhaustive_hashes = {
        str(row["state_hash"])
        for row in validation_subsets["exhaustive"]["states"]
    }
    strong_hashes = {
        str(row["state_hash"])
        for row in validation_subsets["strong"]["states"]
    }
    if len(exhaustive_hashes) != 20 or len(strong_hashes) != 10:
        raise DiagnosticPreparationError("validation subset 数量不一致")
    if not strong_hashes.issubset(exhaustive_hashes):
        raise DiagnosticPreparationError("strong 验证子集不是 exhaustive 的子集")
    if not exhaustive_hashes.issubset(selected_by_hash):
        raise DiagnosticPreparationError("验证子集引用了 selection 外状态")

    sidecar_files = _verify_trace_sidecars(
        out_dir,
        catalog_manifest,
        catalog_rows,
        source_inventory,
    )
    query_records = _read_jsonl(out_dir / "query_plan.jsonl")
    query_header, query_rows = query_records[0], query_records[1:]
    _verify_embedded_hash(query_header, "header_hash", label="query header")
    selected_by_ordinal = {
        int(row["selection"]["selected_ordinal"]): row for row in selected
    }
    grouped: Dict[int, List[Dict[str, Any]]] = {}
    query_keys: List[str] = []
    for query_ordinal, row in enumerate(query_rows):
        _verify_embedded_hash(row, "row_hash", label="query row")
        if row.get("query_ordinal") != query_ordinal:
            raise DiagnosticPreparationError("query_ordinal 不连续")
        if row.get("run_id") != run_manifest.get("run_id"):
            raise DiagnosticPreparationError("query row 的 run_id 不一致")
        state_ordinal = int(row["state_ordinal"])
        state = selected_by_ordinal.get(state_ordinal)
        if state is None or row.get("state_hash") != state.get("state_hash"):
            raise DiagnosticPreparationError("query state 引用不一致")
        action_key = row.get("action_key")
        identity = row.get("query_identity")
        if not isinstance(action_key, Mapping) or not isinstance(identity, Mapping):
            raise DiagnosticPreparationError("query action/identity 必须为 object")
        if sha256_json(action_key) != row.get("action_key_hash"):
            raise DiagnosticPreparationError("query action_key_hash 不一致")
        if identity.get("state_hash") != row.get("state_hash") or identity.get(
            "action_key"
        ) != action_key:
            raise DiagnosticPreparationError("query identity 内外状态/动作不一致")
        expected_identity_fields = {
            "constraint_hash": state["constraint_hash"],
            "enumerator_hash": state["enumerator_hash"],
            "objective_hash": state["objective_hash"],
            "horizon": continuation.horizon,
            "continuation_hash": continuation.hash,
            "continuation_implementation_hash": expected_continuation_hash,
        }
        if any(
            identity.get(field_name) != expected
            for field_name, expected in expected_identity_fields.items()
        ):
            raise DiagnosticPreparationError("query identity 配置哈希不一致")
        if cache_key(identity) != row.get("query_key"):
            raise DiagnosticPreparationError("query identity/key 不一致")
        if action_key.get("task_id") != state.get("task_id"):
            raise DiagnosticPreparationError("query action 指向错误 task")
        grouped.setdefault(state_ordinal, []).append(row)
        query_keys.append(str(row["query_key"]))
    if len(query_keys) != len(set(query_keys)):
        raise DiagnosticPreparationError("query plan 含重复 query_key")
    if set(grouped) != set(selected_by_ordinal):
        raise DiagnosticPreparationError("query plan 没有完整覆盖 selected states")
    budget = int(config["query"]["budget_per_state"])
    for state_ordinal, state_queries in grouped.items():
        state = selected_by_ordinal[state_ordinal]
        action_ordinals = [int(row["action_ordinal"]) for row in state_queries]
        if action_ordinals != list(range(len(state_queries))):
            raise DiagnosticPreparationError("action_ordinal 不连续或有重复")
        expected_queries = min(budget, int(state["candidate_count"]))
        if len(state_queries) != expected_queries:
            raise DiagnosticPreparationError("每状态 query 数不等于 min(B,C)")
        observed_rows = [row for row in state_queries if "observed" in row["roles"]]
        skip_rows = [row for row in state_queries if "skip" in row["roles"]]
        if len(observed_rows) != 1 or len(skip_rows) != 1:
            raise DiagnosticPreparationError("每状态必须恰有 observed/SKIP 锚点")
        if observed_rows[0]["action_key"] != state["observed_action_key"]:
            raise DiagnosticPreparationError("observed query 与冻结动作不一致")
        skip_key = skip_rows[0]["action_key"]
        if skip_key.get("kind") != "skip" or skip_key.get("task_id") != state[
            "task_id"
        ]:
            raise DiagnosticPreparationError("SKIP query 不是当前任务的 SKIP")
    _verify_query_prefixes_by_replay(
        out_dir,
        selected_by_ordinal,
        grouped,
        config,
    )

    query_plan_hash = _scientific_hash(query_rows)
    if query_plan_hash != run_manifest.get("query_plan_hash"):
        raise DiagnosticPreparationError("query_plan_hash 不一致")
    query_header_expected = {
        "run_id": run_manifest["run_id"],
        "query_plan_hash": query_plan_hash,
        "states": len(selected),
        "queries": len(query_rows),
    }
    if any(
        query_header.get(field_name) != expected
        for field_name, expected in query_header_expected.items()
    ):
        raise DiagnosticPreparationError("query header 与 run manifest 不一致")
    if run_manifest.get("continuation_config") != continuation.to_payload() or (
        run_manifest.get("continuation_implementation_hash")
        != expected_continuation_hash
    ):
        raise DiagnosticPreparationError("run continuation 身份不一致")
    run_identity_fields = (
        "schema_version",
        "config_hash",
        "split_manifest_hash",
        "source_inventory_hash",
        "catalog_hash",
        "selection_hash",
        "validation_subsets_hash",
        "preparation_implementation_hash",
        "catalog_manifest_hash",
        "continuation_config",
        "continuation_implementation_hash",
    )
    run_identity = {
        field_name: run_manifest[field_name]
        for field_name in run_identity_fields
    }
    if sha256_json(run_identity) != run_manifest.get("run_id"):
        raise DiagnosticPreparationError("run_id 与 run identity 不一致")
    if (
        run_manifest.get("planned_states") != len(selected)
        or run_manifest.get("planned_queries") != len(query_rows)
        or run_manifest.get("restore_count") != len(selected)
    ):
        raise DiagnosticPreparationError("run manifest 计划数量不一致")

    top_level_files = {
        out_dir / filename
        for filename in (
            "split_manifest.json",
            "catalog_manifest.json",
            "state_catalog.jsonl",
            "selected_states.jsonl",
            "validation_subsets.json",
            "query_plan.jsonl",
            "run_manifest.json",
        )
    }
    actual_files = {path for path in out_dir.rglob("*") if path.is_file()}
    if actual_files != top_level_files | sidecar_files:
        raise DiagnosticPreparationError("冻结包包含缺失或未登记文件")
    return {
        "run_id": run_manifest["run_id"],
        "catalog_hash": catalog_hash,
        "selection_hash": selection_hash,
        "query_plan_hash": query_plan_hash,
        "states": len(selected),
        "queries": len(query_rows),
        "trace_states": sum(
            int(row["state_count"])
            for row in catalog_manifest["trace_inventory"]
        ),
    }


@contextmanager
def _repository_working_directory() -> Iterator[None]:
    """确保 replay provenance 中只出现稳定的仓库相对路径。"""
    previous = Path.cwd()
    os.chdir(REPO_ROOT)
    try:
        yield
    finally:
        os.chdir(previous)


def _repo_relative(path: Path) -> Path:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise DiagnosticPreparationError(
            f"诊断输入必须位于仓库内: {path}"
        ) from exc


def _validate_source_inventory(
    config: Mapping[str, Any],
    inventory_path: Path,
    scenario_dir: Path,
    schedule_dir: Path,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """验证十个实例身份与六个 dev 来源，sealed-test schedule 不会被读取。"""
    expected_inventory = config["source_inventory"]
    if sha256_file(inventory_path) != expected_inventory["file_hash"]:
        raise DiagnosticPreparationError("source inventory 文件哈希不一致")
    inventory = _load_json(inventory_path)
    if inventory.get("schema_version") != expected_inventory["schema_version"]:
        raise DiagnosticPreparationError("source inventory schema 不一致")
    policy = inventory.get("split_policy")
    if not isinstance(policy, Mapping):
        raise DiagnosticPreparationError("source inventory 缺少 split_policy")
    for field_name in (
        "policy_version",
        "canonicalizer_version",
        "train",
        "dev",
        "test",
        "sealed_test",
    ):
        if policy.get(field_name) != config["split"].get(field_name):
            raise DiagnosticPreparationError(
                f"source inventory split_policy.{field_name} 不一致"
            )

    identities = discover_instance_identities(
        scenario_dir,
        pattern="Scenario_Sats1_M100_T7.0d_cities_*.json",
        collection_key="output_re_s1",
    )
    split_manifest = build_recommended_split_manifest(identities)
    inventory_instances = inventory.get("instances")
    if not isinstance(inventory_instances, list) or len(inventory_instances) != 10:
        raise DiagnosticPreparationError("source inventory 必须恰好记录十个实例")
    by_alias = {identity.instance_alias: identity for identity in identities}
    for raw_row in inventory_instances:
        if not isinstance(raw_row, Mapping):
            raise DiagnosticPreparationError("inventory instance row 必须为 object")
        alias = str(raw_row.get("instance_alias", ""))
        identity = by_alias.get(alias)
        if identity is None:
            raise DiagnosticPreparationError(f"inventory 包含未知实例: {alias}")
        if (
            raw_row.get("raw_hash") != identity.raw_hash
            or raw_row.get("semantic_hash") != identity.semantic_hash
        ):
            raise DiagnosticPreparationError(f"实例身份哈希不一致: {alias}")
        scenario_path = scenario_dir / str(raw_row.get("filename", ""))
        scenario = _load_json(scenario_path)
        if scenario.get("scenario_id") != raw_row.get("scenario_id"):
            raise DiagnosticPreparationError(f"scenario_id 不一致: {alias}")

    dev_sources = inventory.get("dev_sources")
    if not isinstance(dev_sources, Mapping) or set(dev_sources) != set(
        config["split"]["dev"]
    ):
        raise DiagnosticPreparationError("source inventory 只能包含两个 dev instances")
    expected_families = config["sources"]["solver_families"]
    for alias in config["split"]["dev"]:
        rows = dev_sources.get(alias)
        if not isinstance(rows, list) or [row.get("source_family") for row in rows] != expected_families:
            raise DiagnosticPreparationError(f"{alias} 的 solver family 清单不完整")
        expected_scenario_id = next(
            row["scenario_id"]
            for row in inventory_instances
            if row["instance_alias"] == alias
        )
        for row in rows:
            schedule_path = schedule_dir / str(row.get("filename", ""))
            if sha256_file(schedule_path) != row.get("sha256"):
                raise DiagnosticPreparationError(f"schedule 文件哈希不一致: {schedule_path}")
            parsed = parse_schedule_filename(schedule_path.stem)
            if (
                int(parsed["class_id"]) != 3
                or parsed["solver_name"] != row["source_family"]
            ):
                raise DiagnosticPreparationError(f"schedule 文件名语义不一致: {schedule_path}")
            objective = objective_config_from_schedule_path(schedule_path)
            if objective_name_from_weights(objective.normalized().weights) != "balanced":
                raise DiagnosticPreparationError(f"schedule 不是 balanced objective: {schedule_path}")
            schedule = _load_json(schedule_path)
            if schedule.get("scenario_id") != expected_scenario_id:
                raise DiagnosticPreparationError(f"schedule scenario_id 不一致: {schedule_path}")
    return inventory, split_manifest


def _trace_source_id(
    *,
    instance_alias: str,
    source_family: str,
    schedule_hash: str,
) -> str:
    return sha256_json(
        {
            "schema_version": SOURCE_ID_SCHEMA_VERSION,
            "instance_alias": instance_alias,
            "source_family": source_family,
            "source_schedule_hash": schedule_hash,
        }
    )


def _build_source_records(
    config: Mapping[str, Any],
    inventory: Mapping[str, Any],
    scenario_dir: Path,
    schedule_dir: Path,
    out_dir: Path,
    preparation_implementation: Mapping[str, Any],
    replay_code_provenance: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """构建六份 compact replay sidecars，并展开为可合并来源行。"""
    replay = config["replay"]
    inventory_instances = {
        row["instance_alias"]: row
        for row in inventory["instances"]
    }
    source_records: List[Dict[str, Any]] = []
    trace_inventory: List[Dict[str, Any]] = []
    for instance_alias in config["split"]["dev"]:
        instance_row = inventory_instances[instance_alias]
        scenario_path = scenario_dir / instance_row["filename"]
        scenario_relative = _repo_relative(scenario_path)
        for source_row in inventory["dev_sources"][instance_alias]:
            _assert_preparation_implementation_unchanged(
                preparation_implementation
            )
            family = str(source_row["source_family"])
            schedule_path = schedule_dir / source_row["filename"]
            schedule_relative = _repo_relative(schedule_path)
            replay_dir = out_dir / "sources" / instance_alias
            with _repository_working_directory():
                trace_path, states_path, trace, states = build_for_paths(
                    scenario_relative,
                    schedule_relative,
                    replay_dir,
                    max_candidates=int(replay["max_candidates"]),
                    placement_mode=str(replay["placement_mode"]),
                    downlink_duration_ratio=float(
                        replay["downlink_duration_ratio"]
                    ),
                    agility_profile=str(replay["agility_profile"]),
                    non_agile_transition_s=float(
                        replay["non_agile_transition_s"]
                    ),
                    code_provenance=replay_code_provenance,
                )
            scenario_hash = str(trace.get("scenario", {}).get("sha256", ""))
            if scenario_hash != instance_row["raw_hash"]:
                raise DiagnosticPreparationError(
                    f"replay 使用的场景哈希漂移: {instance_alias}"
                )
            objective = ObjectiveConfig.from_payload(trace["objective_config"])
            if objective_name_from_weights(objective.normalized().weights) != "balanced":
                raise DiagnosticPreparationError("replay trace objective 意外漂移")
            schedule_hash = str(trace["source_schedule"]["sha256"])
            if schedule_hash != source_row["sha256"]:
                raise DiagnosticPreparationError(
                    f"replay 使用的 schedule 哈希漂移: {schedule_path}"
                )
            if sha256_file(scenario_path) != instance_row["raw_hash"]:
                raise DiagnosticPreparationError(
                    f"replay 后场景文件发生变化: {scenario_path}"
                )
            if sha256_file(schedule_path) != source_row["sha256"]:
                raise DiagnosticPreparationError(
                    f"replay 后 schedule 文件发生变化: {schedule_path}"
                )
            source_id = _trace_source_id(
                instance_alias=instance_alias,
                source_family=family,
                schedule_hash=schedule_hash,
            )
            trace_ref = {
                "relative_path": trace_path.relative_to(out_dir).as_posix(),
                "states_relative_path": states_path.relative_to(out_dir).as_posix(),
                "trace_id": trace["trace_id"],
                "trace_hash": trace["trace_hash"],
                "trace_file_hash": sha256_file(trace_path),
                "states_file_hash": sha256_file(states_path),
            }
            scenario_ref = {
                "root_id": "eosbench_output",
                "relative_path": scenario_relative.as_posix(),
                "sha256": instance_row["raw_hash"],
            }
            trace_inventory.append(
                {
                    "instance_alias": instance_alias,
                    "source_family": family,
                    "source_id": source_id,
                    "source_schedule_hash": schedule_hash,
                    "trace_id": trace["trace_id"],
                    "trace_hash": trace["trace_hash"],
                    "trace_ref": trace_ref,
                    "state_count": len(states),
                }
            )
            for state in states:
                stats = dict(state["candidate_set_stats"])
                source_records.append(
                    {
                        "instance_alias": instance_alias,
                        "split": "dev",
                        "objective_name": "balanced",
                        "source_family": family,
                        "source_id": source_id,
                        "trace_id": trace["trace_id"],
                        "trace_hash": trace["trace_hash"],
                        "state_hash": state["state_hash"],
                        "physical_state_hash": state["physical_state_hash"],
                        "objective_hash": trace["objective_hash"],
                        "constraint_hash": trace["constraint_hash"],
                        "enumerator_hash": trace["enumerator_hash"],
                        "step": int(state["step"]),
                        "task_id": state["task_id"],
                        "schedule_hash": state["schedule_hash"],
                        "ordered_candidate_hash": state["ordered_candidate_hash"],
                        "candidate_membership_hash": state[
                            "candidate_membership_hash"
                        ],
                        "candidate_count": int(state["candidate_count"]),
                        "candidate_set_stats": stats,
                        "cap_reached": bool(stats["cap_reached"]),
                        "observed_action_key": state["observed_action_key"],
                        "state_manifest_hash": state["state_manifest_hash"],
                        "state_manifest": state,
                        "scenario_ref": scenario_ref,
                        "trace_ref": trace_ref,
                    }
                )
            _assert_preparation_implementation_unchanged(
                preparation_implementation
            )
    return source_records, trace_inventory


def _balanced_subset_counts(
    instances: Sequence[str],
    total: int,
) -> Dict[str, int]:
    if not instances:
        raise DiagnosticPreparationError("无法对空实例集合分配验证子集")
    base, remainder = divmod(total, len(instances))
    return {
        instance: base + (1 if index < remainder else 0)
        for index, instance in enumerate(instances)
    }


def _validation_subset_rank(
    row: Mapping[str, Any],
    *,
    seed: int,
    purpose: str,
) -> str:
    return sha256_json(
        {
            "schema_version": "eosbench-ccod-validation-subset-rank-v1",
            "seed": seed,
            "purpose": purpose,
            "instance_alias": row["instance_alias"],
            "state_hash": row["state_hash"],
        }
    )


def _choose_validation_subset(
    selected: Sequence[Mapping[str, Any]],
    *,
    instances: Sequence[str],
    count: int,
    seed: int,
    purpose: str,
    candidate_min: int,
    candidate_max: int | None,
    excluded_hashes: set[str],
) -> List[Dict[str, Any]]:
    """只使用预标签字段，按实例配额与稳定哈希选择状态。"""
    quotas = _balanced_subset_counts(instances, count)
    output: List[Dict[str, Any]] = []
    for instance_alias in instances:
        pool = [
            dict(row)
            for row in selected
            if row["instance_alias"] == instance_alias
            and row["state_hash"] not in excluded_hashes
            and int(row["candidate_count"]) >= candidate_min
            and (
                candidate_max is None
                or int(row["candidate_count"]) <= candidate_max
            )
        ]
        pool.sort(
            key=lambda row: (
                _validation_subset_rank(
                    row,
                    seed=seed,
                    purpose=purpose,
                ),
                str(row["state_hash"]),
            )
        )
        quota = quotas[instance_alias]
        if len(pool) < quota:
            raise DiagnosticPreparationError(
                f"{purpose}/{instance_alias} 候选不足: {len(pool)} < {quota}"
            )
        for row in pool[:quota]:
            output.append(
                {
                    "state_hash": row["state_hash"],
                    "instance_alias": instance_alias,
                    "selected_ordinal": row["selection"]["selected_ordinal"],
                    "candidate_count": int(row["candidate_count"]),
                    "subset_rank": _validation_subset_rank(
                        row,
                        seed=seed,
                        purpose=purpose,
                    ),
                }
            )
    output.sort(
        key=lambda row: (
            instances.index(row["instance_alias"]),
            row["subset_rank"],
            row["state_hash"],
        )
    )
    return output


def select_validation_subsets(
    selected: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    """在任何 Q_H 产生前冻结 exhaustive 与 beam-8 状态集合。"""
    validity = config["label_validity"]
    instances = list(config["split"]["dev"])
    seed = int(config["seed"])
    exhaustive = _choose_validation_subset(
        selected,
        instances=instances,
        count=int(validity["exhaustive_states"]),
        seed=seed,
        purpose="exhaustive",
        candidate_min=int(validity["exhaustive_candidate_count_min"]),
        candidate_max=int(validity["exhaustive_candidate_count_max"]),
        excluded_hashes=set(),
    )
    exhaustive_hashes = {row["state_hash"] for row in exhaustive}
    exhaustive_pool = [
        dict(row)
        for row in selected
        if row["state_hash"] in exhaustive_hashes
    ]
    strong = _choose_validation_subset(
        exhaustive_pool,
        instances=instances,
        count=int(validity["strong_states"]),
        seed=seed,
        purpose="strong_beam8",
        candidate_min=int(validity["strong_candidate_count_min"]),
        candidate_max=int(validity["exhaustive_candidate_count_max"]),
        excluded_hashes=set(),
    )
    payload: Dict[str, Any] = {
        "schema_version": VALIDATION_SUBSET_SCHEMA_VERSION,
        "exhaustive": {
            "policy": validity["exhaustive_selection"],
            "candidate_count_min": validity["exhaustive_candidate_count_min"],
            "candidate_count_max": validity["exhaustive_candidate_count_max"],
            "states": exhaustive,
        },
        "strong": {
            "policy": validity["strong_selection"],
            "candidate_count_min": validity["strong_candidate_count_min"],
            "beam_width": validity["beam_width"],
            "subset_of_exhaustive": validity[
                "strong_subset_of_exhaustive"
            ],
            "states": strong,
        },
    }
    payload["subsets_hash"] = sha256_json(payload)
    return payload


def _freeze_catalog(
    config: Mapping[str, Any],
    inventory: Mapping[str, Any],
    split_manifest: Mapping[str, Any],
    scenario_dir: Path,
    schedule_dir: Path,
    out_dir: Path,
    preparation_implementation: Mapping[str, Any],
    replay_code_provenance: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    source_records, trace_inventory = _build_source_records(
        config,
        inventory,
        scenario_dir,
        schedule_dir,
        out_dir,
        preparation_implementation,
        replay_code_provenance,
    )
    catalog = merge_state_catalog(source_records, config)
    reversed_catalog = merge_state_catalog(list(reversed(source_records)), config)
    if catalog != reversed_catalog:
        raise DiagnosticPreparationError("catalog 合并依赖来源遍历顺序")
    selected = select_preregistered_states(catalog, config)
    reversed_selected = select_preregistered_states(list(reversed(catalog)), config)
    if selected != reversed_selected:
        raise DiagnosticPreparationError("状态选样依赖 catalog 遍历顺序")
    annotated = annotate_catalog_selection(catalog, selected)
    validation_subsets = select_validation_subsets(selected, config)

    config_hash = sha256_json(dict(config))
    normalized_split = dict(split_manifest)
    normalized_split["schema_version"] = SPLIT_MANIFEST_SCHEMA_VERSION
    normalized_split["source_inventory_hash"] = config["source_inventory"][
        "file_hash"
    ]
    normalized_split["manifest_hash"] = sha256_json(normalized_split)
    _publish_json(out_dir / "split_manifest.json", normalized_split)

    catalog_hash = _scientific_hash(annotated)
    selection_hash = _scientific_hash(selected)
    catalog_header = {
        "record_type": "header",
        "schema_version": CATALOG_HEADER_SCHEMA_VERSION,
        "state_schema_version": STATE_CATALOG_SCHEMA_VERSION,
        "config_hash": config_hash,
        "split_manifest_hash": normalized_split["manifest_hash"],
        "source_inventory_hash": config["source_inventory"]["file_hash"],
        "catalog_hash": catalog_hash,
        "selection_hash": selection_hash,
        "validation_subsets_hash": validation_subsets["subsets_hash"],
        "preparation_implementation_hash": preparation_implementation[
            "bundle_hash"
        ],
        "pool_states": len(annotated),
        "selected_states": len(selected),
    }
    catalog_header["header_hash"] = sha256_json(catalog_header)
    _publish_jsonl(out_dir / "state_catalog.jsonl", [catalog_header, *annotated])
    _publish_jsonl(out_dir / "selected_states.jsonl", selected)
    _publish_json(out_dir / "validation_subsets.json", validation_subsets)

    catalog_manifest: Dict[str, Any] = {
        "schema_version": "eosbench-ccod-catalog-manifest-v1",
        "config_hash": config_hash,
        "split_manifest_hash": normalized_split["manifest_hash"],
        "source_inventory_hash": config["source_inventory"]["file_hash"],
        "trace_inventory": trace_inventory,
        "catalog_hash": catalog_hash,
        "selection_hash": selection_hash,
        "validation_subsets_hash": validation_subsets["subsets_hash"],
        "preparation_implementation_hash": preparation_implementation[
            "bundle_hash"
        ],
        "preparation_implementation": dict(preparation_implementation),
        "summary": selection_summary(selected),
        "prelabel_audit": catalog_prelabel_audit(annotated, config),
        "raw_trajectory_jsonl_read_bytes": 0,
        "raw_trajectory_jsonl_copied_bytes": 0,
    }
    catalog_manifest["manifest_hash"] = sha256_json(catalog_manifest)
    _publish_json(out_dir / "catalog_manifest.json", catalog_manifest)
    return selected, catalog_manifest


def _freeze_query_plan(
    config: Mapping[str, Any],
    selected: Sequence[Mapping[str, Any]],
    catalog_manifest: Mapping[str, Any],
    out_dir: Path,
) -> Dict[str, Any]:
    """恢复 100 个状态并冻结 observed/SKIP/uniform 查询，不执行 Q_H。"""
    continuation = ContinuationConfig(
        horizon=int(config["query"]["continuation_horizon"])
    )
    implementation_hash = continuation_implementation_hash()
    run_identity = {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "config_hash": catalog_manifest["config_hash"],
        "split_manifest_hash": catalog_manifest["split_manifest_hash"],
        "source_inventory_hash": catalog_manifest["source_inventory_hash"],
        "catalog_hash": catalog_manifest["catalog_hash"],
        "selection_hash": catalog_manifest["selection_hash"],
        "validation_subsets_hash": catalog_manifest[
            "validation_subsets_hash"
        ],
        "preparation_implementation_hash": catalog_manifest[
            "preparation_implementation"
        ]["bundle_hash"],
        "catalog_manifest_hash": catalog_manifest["manifest_hash"],
        "continuation_config": continuation.to_payload(),
        "continuation_implementation_hash": implementation_hash,
    }
    run_id = sha256_json(run_identity)
    query_rows: List[Dict[str, Any]] = []
    problems: Dict[str, Any] = {}
    traces: Dict[str, Dict[str, Any]] = {}

    for state_ordinal, selected_row in enumerate(selected):
        instance_alias = str(selected_row["instance_alias"])
        canonical_source = selected_row["canonical_source"]
        scenario_ref = canonical_source["scenario_ref"]
        scenario_path = REPO_ROOT / scenario_ref["relative_path"]
        if instance_alias not in problems:
            problems[instance_alias] = load_scheduling_problem_from_json(scenario_path)
        problem = problems[instance_alias]
        trace_ref = canonical_source["trace_ref"]
        trace_path = out_dir / trace_ref["relative_path"]
        trace_key = str(trace_ref["trace_hash"])
        if trace_key not in traces:
            traces[trace_key] = _load_json(trace_path)
        trace = traces[trace_key]
        replayed = restore_state(
            problem,
            trace,
            canonical_source["state_manifest"],
            scenario_path=scenario_path,
            verify=True,
        )
        action_keys = [
            candidate_action_key(problem, replayed.task_id, action)
            for action in replayed.candidates
        ]
        prefix = diagnostic_query_prefix(
            str(selected_row["state_hash"]),
            action_keys,
            selected_row["observed_action_key"],
            budget=int(config["query"]["budget_per_state"]),
            seed=int(config["seed"]),
        )
        if len(prefix) != min(
            int(config["query"]["budget_per_state"]),
            int(selected_row["candidate_count"]),
        ):
            raise DiagnosticPreparationError("query prefix 数量违反 min(B,C)")
        constraint = ConstraintConfig.from_payload(trace["constraint_config"])
        enumerator = EnumeratorConfig.from_payload(trace["enumerator_config"])
        objective = ObjectiveConfig.from_payload(trace["objective_config"])
        for action_ordinal, planned in enumerate(prefix):
            identity = build_cache_identity(
                state_hash=str(selected_row["state_hash"]),
                action_key=planned["action_key"],
                constraint_config=constraint,
                enumerator_config=enumerator,
                objective_config=objective,
                continuation_config=continuation,
            )
            query_key = cache_key(identity)
            row: Dict[str, Any] = {
                "schema_version": QUERY_PLAN_SCHEMA_VERSION,
                "run_id": run_id,
                "query_ordinal": len(query_rows),
                "state_ordinal": state_ordinal,
                "state_hash": selected_row["state_hash"],
                "instance_alias": instance_alias,
                "action_ordinal": action_ordinal,
                "action_key": planned["action_key"],
                "action_key_hash": sha256_json(planned["action_key"]),
                "roles": planned["selection_sources"],
                "query_identity": identity,
                "query_key": query_key,
            }
            row["row_hash"] = sha256_json(row)
            query_rows.append(row)

    if len({row["query_key"] for row in query_rows}) != len(query_rows):
        raise DiagnosticPreparationError("query plan 包含重复 query_key")
    query_plan_hash = _scientific_hash(query_rows)
    query_header = {
        "record_type": "header",
        "schema_version": "eosbench-ccod-query-plan-header-v1",
        "run_id": run_id,
        "query_plan_hash": query_plan_hash,
        "states": len(selected),
        "queries": len(query_rows),
    }
    query_header["header_hash"] = sha256_json(query_header)
    _publish_jsonl(out_dir / "query_plan.jsonl", [query_header, *query_rows])

    run_manifest: Dict[str, Any] = {
        **run_identity,
        "run_id": run_id,
        "query_plan_hash": query_plan_hash,
        "planned_states": len(selected),
        "planned_queries": len(query_rows),
        "restore_count": len(selected),
        "workers": 1,
        "status": "planned",
        "raw_trajectory_jsonl_read_bytes": 0,
        "raw_trajectory_jsonl_copied_bytes": 0,
    }
    run_manifest["manifest_hash"] = sha256_json(run_manifest)
    _publish_json(out_dir / "run_manifest.json", run_manifest)
    return run_manifest


def prepare_diagnostic(
    *,
    config_path: Path,
    scenario_dir: Path,
    schedule_dir: Path,
    out_dir: Path,
    catalog_only: bool,
) -> Dict[str, Any]:
    """执行输入验证、catalog 冻结，并按需冻结 query plan。"""
    preparation_implementation = _preparation_implementation_bundle()
    config_file_hash = sha256_file(config_path)
    # 只采一次并传给六个来源，保证 trace provenance 在整次运行内一致。
    replay_code_provenance = collect_code_provenance()
    config = _load_json(config_path)
    validate_diagnostic_runtime(config)
    inventory_path = REPO_ROOT / config["source_inventory"]["path"]
    inventory, split_manifest = _validate_source_inventory(
        config,
        inventory_path,
        scenario_dir,
        schedule_dir,
    )
    selected, catalog_manifest = _freeze_catalog(
        config,
        inventory,
        split_manifest,
        scenario_dir,
        schedule_dir,
        out_dir,
        preparation_implementation,
        replay_code_provenance,
    )
    run_manifest = None
    if not catalog_only:
        run_manifest = _freeze_query_plan(
            config,
            selected,
            catalog_manifest,
            out_dir,
        )
    _assert_preparation_implementation_unchanged(preparation_implementation)
    if sha256_file(config_path) != config_file_hash:
        raise DiagnosticPreparationError("预注册运行期间配置文件发生变化")
    final_inventory, final_split = _validate_source_inventory(
        config,
        inventory_path,
        scenario_dir,
        schedule_dir,
    )
    if final_inventory != inventory or final_split != split_manifest:
        raise DiagnosticPreparationError("预注册运行期间来源输入发生变化")
    if catalog_only:
        return {
            "status": "catalog_frozen",
            "catalog_manifest": catalog_manifest,
        }
    assert run_manifest is not None
    return {
        "status": "query_plan_frozen",
        "catalog_manifest": catalog_manifest,
        "run_manifest": run_manifest,
    }


def prepare_and_publish(
    config_path: Path,
    scenario_dir: Path,
    schedule_dir: Path,
    out_dir: Path,
    catalog_only: bool = False,
) -> Dict[str, Any]:
    """在唯一 staging 中构建整包，闭环验收后发布到新目录。"""
    requested_dir = out_dir.expanduser().absolute()
    if requested_dir.exists() or requested_dir.is_symlink():
        raise FileExistsError(f"冻结目录已存在，拒绝覆盖: {requested_dir}")
    # macOS 的 /var 是系统级符号链接；先规范化已有父目录，再检查最终目标。
    final_dir = requested_dir.resolve(strict=False)
    _reject_existing_symlink_components(final_dir)
    if final_dir.exists() or final_dir.is_symlink():
        raise FileExistsError(f"冻结目录已存在，拒绝覆盖: {final_dir}")
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = final_dir.parent / f".{final_dir.name}.publish.lock"
    try:
        lock_fd = os.open(
            lock_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
    except FileExistsError as exc:
        raise FileExistsError(f"冻结目录已有发布者: {lock_path}") from exc
    staging_dir: Path | None = None
    published = False
    try:
        staging_dir = Path(
            tempfile.mkdtemp(
                prefix=f".{final_dir.name}.staging-",
                dir=final_dir.parent,
            )
        )
        result = prepare_diagnostic(
            config_path=config_path.resolve(),
            scenario_dir=scenario_dir.resolve(),
            schedule_dir=schedule_dir.resolve(),
            out_dir=staging_dir,
            catalog_only=catalog_only,
        )
        if not catalog_only:
            verification = verify_frozen_artifacts(
                staging_dir,
                config_path=config_path,
            )
            if verification["run_id"] != result["run_manifest"]["run_id"]:
                raise DiagnosticPreparationError(
                    "发布前校验返回的 run_id 与本次冻结结果不一致"
                )
            result = {**result, "verification": verification}
        if final_dir.exists() or final_dir.is_symlink():
            raise FileExistsError(f"发布前目标目录被占用: {final_dir}")
        os.rename(staging_dir, final_dir)
        published = True
        directory_fsynced = False
        try:
            parent_fd = os.open(final_dir.parent, os.O_RDONLY)
            try:
                os.fsync(parent_fd)
                directory_fsynced = True
            finally:
                os.close(parent_fd)
        except OSError:
            # 某些文件系统不支持目录 open/fsync；rename 已经完成原子发布。
            pass
        result["publication"] = {
            "atomic_rename": True,
            "parent_directory_fsynced": directory_fsynced,
        }
        return result
    finally:
        if not published and staging_dir is not None:
            shutil.rmtree(staging_dir, ignore_errors=True)
        with suppress(OSError):
            os.close(lock_fd)
        with suppress(OSError):
            lock_path.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="冻结 CCOD 100-state catalog 与无标签 query plan。"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "algorithms/ccod/configs/diagnostic_v1.json",
    )
    parser.add_argument(
        "--scenario-dir",
        type=Path,
        default=REPO_ROOT / "output/output_re_s1",
    )
    parser.add_argument(
        "--schedule-dir",
        type=Path,
        default=REPO_ROOT / "output/schedules/output_re_s1",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--catalog-only",
        action="store_true",
        help="只冻结 split/catalog，不恢复 100 个状态生成 query plan。",
    )
    mode.add_argument(
        "--verify-only",
        action="store_true",
        help="只读验收已发布的完整冻结包，不生成或覆盖任何文件。",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    started = time.perf_counter()
    try:
        if args.verify_only:
            result = {
                "status": "verified",
                "verification": verify_frozen_artifacts(
                    args.out_dir,
                    config_path=args.config,
                ),
            }
        else:
            result = prepare_and_publish(
                args.config,
                args.scenario_dir,
                args.schedule_dir,
                args.out_dir,
                bool(args.catalog_only),
            )
    except (
        DiagnosticPreparationError,
        FileExistsError,
        FileNotFoundError,
        KeyError,
        OSError,
        ValueError,
    ) as exc:
        print(f"[fatal] CCOD diagnostic prepare 失败: {exc}", file=sys.stderr)
        return 1

    elapsed = time.perf_counter() - started
    print(f"[pass] status={result['status']}")
    if "verification" in result:
        verification = result["verification"]
        print(f"[pass] run_id={verification['run_id']}")
        print(f"[pass] catalog_hash={verification['catalog_hash']}")
        print(f"[pass] selection_hash={verification['selection_hash']}")
        print(f"[pass] query_plan_hash={verification['query_plan_hash']}")
        print(f"[pass] planned_states={verification['states']}")
        print(f"[pass] planned_queries={verification['queries']}")
    else:
        catalog = result["catalog_manifest"]
        print(f"[pass] catalog_hash={catalog['catalog_hash']}")
        print(f"[pass] selection_hash={catalog['selection_hash']}")
        print(
            "[summary] "
            + canonical_json_bytes(catalog["summary"]).decode("utf-8")
        )
    if "run_manifest" in result:
        run = result["run_manifest"]
        print(f"[pass] run_id={run['run_id']}")
        print(f"[pass] planned_queries={run['planned_queries']}")
    print(f"[timing] prepare_wall_s={elapsed:.3f}")
    if args.verify_only:
        print(f"[keep] verified artifacts unchanged -> {args.out_dir}")
    else:
        print(f"[keep] artifacts -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
