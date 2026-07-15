# -*- coding: utf-8 -*-
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from algorithms.ccod.diagnostic import (
    BALANCED_OBJECTIVE_NAME,
    DiagnosticCatalogError,
    DiagnosticConfigError,
    annotate_catalog_selection,
    catalog_prelabel_audit,
    merge_state_catalog,
    objective_name_from_weights,
    select_preregistered_states,
    selection_summary,
    validate_diagnostic_config,
)
from schedulers.state_replay import (
    ACTION_KEY_VERSION,
    MANIFEST_SCHEMA_VERSION,
    sha256_json,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "algorithms/ccod/configs/diagnostic_v1.json"


class CCODDiagnosticTest(unittest.TestCase):
    def _config(self):
        with CONFIG_PATH.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _hash(self, label: str) -> str:
        return sha256_json({"test_label": label})

    def _source(
        self,
        *,
        instance: str,
        index: int,
        family: str = "sa",
        trace_suffix: str = "a",
        observed_kind: str = "assign",
        candidate_count: int = 20,
        state_hash: str | None = None,
        cap_reached: bool = False,
    ):
        state_hash = state_hash or self._hash(f"state-{instance}-{index}")
        trace_id = self._hash(f"trace-{instance}-{family}-{trace_suffix}")
        trace_hash = self._hash(
            f"trace-hash-{instance}-{family}-{trace_suffix}"
        )
        observed = {
            "version": ACTION_KEY_VERSION,
            "kind": observed_kind,
            "task_id": f"task-{index}",
        }
        if observed_kind == "assign":
            observed["satellite_id"] = f"sat-{family}"
        physical_hash = self._hash(f"physical-{instance}-{index}")
        schedule_hash = self._hash(f"schedule-{instance}-{index}")
        ordered_hash = self._hash(f"ordered-{instance}-{index}")
        membership_hash = self._hash(f"membership-{instance}-{index}")
        state_manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "state_hash": state_hash,
            "physical_state_hash": physical_hash,
            "step": index,
            "task_id": f"task-{index}",
            "schedule_hash": schedule_hash,
            "ordered_candidate_hash": ordered_hash,
            "candidate_membership_hash": membership_hash,
            "candidate_count": candidate_count,
            "candidate_set_stats": {"cap_reached": cap_reached},
            "observed_action_key": observed,
            "trace_id": trace_id,
            "trace_hash": trace_hash,
        }
        state_manifest_hash = sha256_json(state_manifest)
        state_manifest["state_manifest_hash"] = state_manifest_hash
        return {
            "instance_alias": instance,
            "split": "dev",
            "objective_name": BALANCED_OBJECTIVE_NAME,
            "source_family": family,
            "source_id": self._hash(
                f"source-{instance}-{family}-{trace_suffix}"
            ),
            "trace_id": trace_id,
            "trace_hash": trace_hash,
            "state_hash": state_hash,
            "physical_state_hash": physical_hash,
            "objective_hash": self._hash("balanced"),
            "constraint_hash": self._hash("constraint"),
            "enumerator_hash": self._hash("enumerator"),
            "step": index,
            "task_id": f"task-{index}",
            "schedule_hash": schedule_hash,
            "ordered_candidate_hash": ordered_hash,
            "candidate_membership_hash": membership_hash,
            "candidate_count": candidate_count,
            "candidate_set_stats": {"cap_reached": cap_reached},
            "cap_reached": cap_reached,
            "observed_action_key": observed,
            "state_manifest_hash": state_manifest_hash,
            "state_manifest": state_manifest,
            "scenario_ref": {
                "root_id": "eosbench_output",
                "relative_path": f"output_re_s1/{instance}.json",
                "sha256": self._hash(f"scenario-{instance}"),
            },
            "trace_ref": {
                "relative_path": f"sources/{trace_id}.trace.json",
                "trace_id": trace_id,
                "trace_hash": trace_hash,
            },
        }

    def _pool(self):
        records = []
        families = ("sa", "ga", "aco")
        for instance in ("cities_08", "cities_04"):
            for index in range(60):
                records.append(
                    self._source(
                        instance=instance,
                        index=index,
                        family=families[index % len(families)],
                        observed_kind="skip" if index % 3 == 0 else "assign",
                        candidate_count=2 if index < 10 else 10 + index,
                    )
                )
        return records

    def test_repository_config_is_internally_consistent(self) -> None:
        with CONFIG_PATH.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        validate_diagnostic_config(config)
        self.assertEqual(config["split"]["dev"], ["cities_08", "cities_04"])
        self.assertEqual(config["state_selection"]["deduplicate_by"], "state_hash")

    def test_balanced_objective_mapping_is_strict(self) -> None:
        self.assertEqual(
            objective_name_from_weights((1.0, 1.0, 1.0, 1.0)),
            BALANCED_OBJECTIVE_NAME,
        )
        with self.assertRaisesRegex(DiagnosticConfigError, "只接受 balanced"):
            objective_name_from_weights((1.0, 0.0, 0.0, 0.0))

    def test_config_rejects_split_overlap_and_weak_dedup_identity(self) -> None:
        config = self._config()
        config["split"]["test"] = ["cities_08", "cities_10"]
        with self.assertRaisesRegex(DiagnosticConfigError, "互斥"):
            validate_diagnostic_config(config)

        config = self._config()
        config["state_selection"]["deduplicate_by"] = "physical_state_hash"
        with self.assertRaisesRegex(DiagnosticConfigError, "state_selection"):
            validate_diagnostic_config(config)

    def test_catalog_merge_is_order_invariant_and_freezes_one_observed(self) -> None:
        config = self._config()
        first = self._source(
            instance="cities_08",
            index=3,
            family="sa",
            trace_suffix="a",
            observed_kind="assign",
        )
        second = self._source(
            instance="cities_08",
            index=3,
            family="ga",
            trace_suffix="b",
            observed_kind="skip",
            state_hash=first["state_hash"],
        )
        catalog = merge_state_catalog([first, second], config)
        reversed_catalog = merge_state_catalog([second, first], config)
        self.assertEqual(catalog, reversed_catalog)
        self.assertEqual(len(catalog), 1)
        self.assertEqual(len(catalog[0]["source_aliases"]), 2)
        self.assertIn(
            catalog[0]["observed_action_key"]["kind"],
            {"assign", "skip"},
        )
        self.assertEqual(
            catalog[0]["observed_action_key"],
            catalog[0]["canonical_source"]["observed_action_key"],
        )

    def test_catalog_merge_rejects_same_hash_with_conflicting_candidates(self) -> None:
        config = self._config()
        first = self._source(instance="cities_08", index=4)
        second = self._source(
            instance="cities_08",
            index=4,
            family="ga",
            trace_suffix="b",
            state_hash=first["state_hash"],
            candidate_count=21,
        )
        with self.assertRaisesRegex(DiagnosticCatalogError, "语义冲突"):
            merge_state_catalog([first, second], config)

    def test_catalog_rejects_train_or_test_source(self) -> None:
        source = self._source(instance="cities_08", index=0)
        source["split"] = "test"
        with self.assertRaisesRegex(DiagnosticCatalogError, "dev split"):
            merge_state_catalog([source], self._config())

    def test_catalog_rejects_manifest_tampering_and_outer_mismatch(self) -> None:
        source = self._source(instance="cities_08", index=1)
        source["state_manifest"]["candidate_count"] += 1
        with self.assertRaisesRegex(DiagnosticCatalogError, "重算不一致"):
            merge_state_catalog([source], self._config())

        source = self._source(instance="cities_08", index=1)
        source["candidate_count"] += 1
        with self.assertRaisesRegex(DiagnosticCatalogError, "外层字段"):
            merge_state_catalog([source], self._config())

    def test_selection_is_deterministic_unique_and_meets_signal_floor(self) -> None:
        config = self._config()
        catalog = merge_state_catalog(self._pool(), config)
        first = select_preregistered_states(catalog, config)
        second = select_preregistered_states(list(reversed(catalog)), config)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 100)
        self.assertEqual(len({row["state_hash"] for row in first}), 100)
        self.assertGreaterEqual(sum(row["signal_eligible"] for row in first), 80)
        for instance in ("cities_08", "cities_04"):
            self.assertGreaterEqual(
                sum(
                    row["instance_alias"] == instance
                    and row["exhaustive_eligible"]
                    for row in first
                ),
                10,
            )
        self.assertEqual(
            {instance: sum(row["instance_alias"] == instance for row in first)
             for instance in ("cities_08", "cities_04")},
            {"cities_08": 50, "cities_04": 50},
        )
        self.assertTrue(all(row["candidate_count"] >= 2 for row in first))
        self.assertTrue(all(row["step"] <= 95 for row in first))

    def test_catalog_annotation_and_summary_are_compact_and_consistent(self) -> None:
        config = self._config()
        catalog = merge_state_catalog(self._pool(), config)
        selected = select_preregistered_states(catalog, config)
        annotated = annotate_catalog_selection(catalog, selected)
        self.assertEqual(
            sum(bool(row["selection"]["selected"]) for row in annotated),
            100,
        )
        summary = selection_summary(selected)
        self.assertEqual(summary["states"], 100)
        self.assertEqual(summary["unique_state_hashes"], 100)
        self.assertGreaterEqual(summary["signal_eligible_states"], 80)
        audit = catalog_prelabel_audit(annotated, config)
        self.assertEqual(audit["totals"]["catalog_states"], 120)
        self.assertEqual(
            audit["audit_hash"],
            sha256_json(
                {key: value for key, value in audit.items() if key != "audit_hash"}
            ),
        )
        self.assertNotIn("q_h", json.dumps(annotated, sort_keys=True))

    def test_selection_reports_exhaustive_inventory_shortage(self) -> None:
        config = self._config()
        pool = self._pool()
        replacement_pool = []
        for row in pool:
            if row["instance_alias"] == "cities_04":
                replacement_pool.append(
                    self._source(
                        instance="cities_04",
                        index=int(row["step"]),
                        family=str(row["source_family"]),
                        observed_kind=str(row["observed_action_key"]["kind"]),
                        candidate_count=10,
                    )
                )
            else:
                replacement_pool.append(row)
        catalog = merge_state_catalog(replacement_pool, config)
        with self.assertRaisesRegex(
            DiagnosticCatalogError,
            "cities_04 exhaustive-eligible 库存不足",
        ):
            select_preregistered_states(catalog, config)

    def test_selection_rejects_duplicate_catalog_hashes(self) -> None:
        config = self._config()
        catalog = merge_state_catalog(self._pool(), config)
        with self.assertRaisesRegex(DiagnosticCatalogError, "尚未按 state_hash 去重"):
            select_preregistered_states([*catalog, deepcopy(catalog[0])], config)

    def test_cap_hit_states_are_never_selected(self) -> None:
        config = self._config()
        pool = self._pool()
        for row in pool:
            if row["instance_alias"] == "cities_08" and int(row["step"]) < 5:
                replacement = self._source(
                    instance="cities_08",
                    index=int(row["step"]),
                    family=str(row["source_family"]),
                    observed_kind=str(row["observed_action_key"]["kind"]),
                    candidate_count=int(row["candidate_count"]),
                    cap_reached=True,
                )
                row.clear()
                row.update(replacement)
        selected = select_preregistered_states(
            merge_state_catalog(pool, config),
            config,
        )
        self.assertFalse(any(row["cap_reached"] for row in selected))

    def test_selection_rank_collision_keeps_input_order_invariance(self) -> None:
        config = self._config()
        catalog = merge_state_catalog(self._pool(), config)
        with patch(
            "algorithms.ccod.diagnostic._selection_rank",
            return_value="sha256:" + "0" * 64,
        ):
            first = select_preregistered_states(catalog, config)
            second = select_preregistered_states(list(reversed(catalog)), config)
        self.assertEqual(first, second)

    def test_annotation_resets_old_selection_and_rejects_unknown_hash(self) -> None:
        config = self._config()
        catalog = merge_state_catalog(self._pool(), config)
        selected = select_preregistered_states(catalog, config)
        annotated = annotate_catalog_selection(catalog, selected)
        reset = annotate_catalog_selection(annotated, [])
        self.assertFalse(any(row["selection"]["selected"] for row in reset))

        unknown = deepcopy(selected[0])
        unknown["state_hash"] = self._hash("unknown-selection")
        with self.assertRaisesRegex(DiagnosticCatalogError, "catalog 外"):
            annotate_catalog_selection(catalog, [unknown])


if __name__ == "__main__":
    unittest.main()
