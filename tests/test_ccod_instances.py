from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from algorithms.ccod.instances import (
    InstanceIdentity,
    InstanceIdentityError,
    RECOMMENDED_SPLIT,
    build_instance_identity,
    build_recommended_split_manifest,
    discover_instance_identities,
    semantic_scenario_hash,
    semantic_scenario_payload,
    split_by_semantic_hash,
)
from schedulers.state_replay import sha256_file


class CCODInstanceIdentityTest(unittest.TestCase):
    def _scenario(self):
        return {
            "scenario_id": "display-id",
            "scenario_type": "other",
            "metadata": {
                "name": "display-name",
                "description": "display-description",
                "extra": {"render": True},
                "creation_time": "2026-01-01T00:00:00",
                "duration": 3600.0,
                "time_step": 1.0,
            },
            "satellites": [
                {
                    "id": "SAT-B",
                    "observation_capability": {
                        "sensors": [
                            {"sensor_id": "SENSOR-2", "data_rate_Mbps": 2.0},
                            {"sensor_id": "SENSOR-1", "data_rate_Mbps": 1.0},
                        ]
                    },
                },
                {"id": "SAT-A"},
            ],
            "ground_stations": [{"id": "GS-B"}, {"id": "GS-A"}],
            "missions": [
                {"id": "TASK-B", "priority": 1.0},
                {"id": "TASK-A", "priority": 2.0},
            ],
            "observation_windows": [
                {
                    "satellite_id": "SAT-B",
                    "sensor_id": "SENSOR-2",
                    "mission_id": "TASK-B",
                    "time_windows": [
                        {
                            "start_time": "2026-01-01T00:02:00",
                            "end_time": "2026-01-01T00:03:00",
                            "orbit_number": 2,
                            "agile_data": {"pitch_angles": [2.0, 1.0]},
                        },
                        {
                            "start_time": "2026-01-01T00:01:00",
                            "end_time": "2026-01-01T00:02:00",
                            "orbit_number": 1,
                        },
                    ],
                },
                {
                    "satellite_id": "SAT-A",
                    "sensor_id": "SENSOR-1",
                    "mission_id": "TASK-A",
                    "time_windows": [],
                },
            ],
            "communication_windows": [
                {
                    "satellite_id": "SAT-B",
                    "ground_station_id": "GS-B",
                    "time_windows": [
                        {
                            "start_time": "2026-01-01T00:10:00",
                            "end_time": "2026-01-01T00:11:00",
                        }
                    ],
                },
                {
                    "satellite_id": "SAT-A",
                    "ground_station_id": "GS-A",
                    "time_windows": [],
                },
            ],
        }

    def test_display_fields_and_unordered_collection_order_do_not_change_hash(self):
        original = self._scenario()
        reordered = deepcopy(original)
        reordered["scenario_id"] = "another-display-id"
        reordered["metadata"]["name"] = "another-name"
        reordered["metadata"]["description"] = None
        reordered["metadata"]["extra"] = {"other": "value"}
        for field in ("satellites", "missions", "observation_windows"):
            reordered[field].reverse()
        for group in reordered["observation_windows"]:
            group["time_windows"].reverse()

        self.assertEqual(
            semantic_scenario_hash(original),
            semantic_scenario_hash(reordered),
        )
        payload = semantic_scenario_payload(original)
        self.assertNotIn("scenario_id", payload)
        self.assertEqual(
            set(payload["metadata"]),
            {"creation_time", "duration", "time_step"},
        )

    def test_nested_sensor_order_remains_part_of_frozen_v1_semantics(self):
        left = self._scenario()
        right = deepcopy(left)
        right["satellites"][0]["observation_capability"]["sensors"].reverse()
        self.assertNotEqual(
            semantic_scenario_hash(left),
            semantic_scenario_hash(right),
        )

    def test_ordered_attitude_series_remains_part_of_semantics(self):
        left = self._scenario()
        right = deepcopy(left)
        right["observation_windows"][0]["time_windows"][0]["agile_data"][
            "pitch_angles"
        ].reverse()
        self.assertNotEqual(
            semantic_scenario_hash(left),
            semantic_scenario_hash(right),
        )

    def test_identity_contains_only_logical_names_not_absolute_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "output_re_s1"
            root.mkdir()
            path = root / "Scenario_Sats1_M100_T7.0d_cities_01.json"
            path.write_text(
                json.dumps(self._scenario(), ensure_ascii=False),
                encoding="utf-8",
            )
            identity = build_instance_identity(path)

        payload = identity.to_payload()
        self.assertEqual(payload["instance_alias"], "cities_01")
        self.assertEqual(payload["instance_key"], "output_re_s1/cities_01")
        self.assertEqual(
            payload["source_stem"],
            "Scenario_Sats1_M100_T7.0d_cities_01",
        )
        self.assertNotIn(temporary, json.dumps(payload))
        self.assertRegex(payload["raw_hash"], r"^sha256:[0-9a-f]{64}$")
        self.assertRegex(payload["semantic_hash"], r"^sha256:[0-9a-f]{64}$")

    def test_semantic_hash_split_is_deterministic_and_disjoint(self):
        aliases = [f"cities_{index:02d}" for index in range(1, 11)]
        rows = [
            InstanceIdentity(
                raw_hash="sha256:" + f"{index + 20:064x}",
                semantic_hash="sha256:" + f"{10 - index:064x}",
                instance_alias=alias,
                instance_key=f"output_re_s1/{alias}",
                source_stem=f"Scenario_{alias}",
            )
            for index, alias in enumerate(aliases)
        ]
        first = split_by_semantic_hash(rows)
        second = split_by_semantic_hash(reversed(rows))
        self.assertEqual(first, second)
        self.assertEqual(first["train"], list(reversed(aliases[4:])))
        self.assertEqual(first["dev"], ["cities_04", "cities_03"])
        self.assertEqual(first["test"], ["cities_02", "cities_01"])
        folded = first["train"] + first["dev"] + first["test"]
        self.assertEqual(len(folded), len(set(folded)))

    def test_recommended_manifest_validates_inventory_and_freezes_622(self):
        rows = []
        recommended = (
            tuple(RECOMMENDED_SPLIT["train"])
            + tuple(RECOMMENDED_SPLIT["dev"])
            + tuple(RECOMMENDED_SPLIT["test"])
        )
        for index, alias in enumerate(reversed(recommended)):
            rows.append(
                InstanceIdentity(
                    raw_hash="sha256:" + f"{index + 100:064x}",
                    semantic_hash="sha256:"
                    + f"{recommended.index(alias) + 200:064x}",
                    instance_alias=alias,
                    instance_key=f"output_re_s1/{alias}",
                    source_stem=f"Scenario_{alias}",
                )
            )
        manifest = build_recommended_split_manifest(rows)
        for fold in ("train", "dev", "test"):
            self.assertEqual(manifest[fold], list(RECOMMENDED_SPLIT[fold]))
        self.assertTrue(manifest["sealed_test"])
        self.assertTrue(manifest["semantic_order_matches_recommendation"])
        folded = manifest["train"] + manifest["dev"] + manifest["test"]
        self.assertEqual(len(folded), 10)
        self.assertEqual(len(set(folded)), 10)

        with self.assertRaisesRegex(InstanceIdentityError, "库存"):
            build_recommended_split_manifest(rows[:-1])

    def test_repository_inventory_matches_preregistered_full_hashes(self):
        scenario_dir = (
            Path(__file__).resolve().parents[1] / "output/output_re_s1"
        )
        if not scenario_dir.is_dir():
            self.skipTest("本地未提供 Sats1/M100 场景库存")
        identities = discover_instance_identities(
            scenario_dir,
            pattern="Scenario_Sats1_M100_T7.0d_cities_*.json",
            collection_key="output_re_s1",
        )
        expected = {
            "cities_01": "sha256:40fac50bb648d09501d12fc0a64e48ad9db087a19db497c00e953df2423545be",
            "cities_02": "sha256:d39f8f947bcb13210d5eb6bd7ccebac5b9e232f6ef67714cb91760bdfee9a0fb",
            "cities_03": "sha256:24ea2194243b919d5023219eea47f4d75d47e7b28ffcdc36d8a101ae651ce5c5",
            "cities_04": "sha256:c9c9a911e75bd153f8a0bd3cee3595ec134e60817a31b186bb6a319c3c330929",
            "cities_05": "sha256:37876477bb7a9663c58c9a45535db9cd14fb5b435ee684d8eb0282be14fa932c",
            "cities_06": "sha256:5b12d361c3db753c2e3289871d51b5f6482d35f33591cc388d13b643f370708d",
            "cities_07": "sha256:5e0b812f8b5dd5a422edcce6ad7435e41826897af8ba4a167689f368e7cae510",
            "cities_08": "sha256:6d3e16fbc491a60085919d8d39e74a928a94ebe188adfb5ec83c39951e48299b",
            "cities_09": "sha256:4c467ce08ffd37e57f636a0566768c19fa1db437cdaa9e63064622980ab70a85",
            "cities_10": "sha256:d6157be8f7dba6d15f569788ca35043c28de4a219a7b68a9baae1fb0f551f244",
        }
        self.assertEqual(
            {row.instance_alias: row.semantic_hash for row in identities},
            expected,
        )
        manifest = build_recommended_split_manifest(identities)
        self.assertTrue(manifest["semantic_order_matches_recommendation"])

    def test_frozen_source_inventory_matches_local_dev_files(self):
        repo_root = Path(__file__).resolve().parents[1]
        config_path = repo_root / "algorithms/ccod/configs/diagnostic_v1.json"
        inventory_path = (
            repo_root / "algorithms/ccod/configs/diagnostic_v1_sources.json"
        )
        if not inventory_path.is_file():
            self.skipTest("本地未提供诊断来源库存")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        self.assertEqual(
            sha256_file(inventory_path),
            config["source_inventory"]["file_hash"],
        )
        self.assertEqual(
            inventory["schema_version"],
            config["source_inventory"]["schema_version"],
        )
        self.assertEqual(inventory["split_policy"]["dev"], config["split"]["dev"])
        self.assertEqual(
            inventory["split_policy"]["canonicalizer_version"],
            config["split"]["canonicalizer_version"],
        )

        scenario_dir = repo_root / "output/output_re_s1"
        schedule_dir = repo_root / "output/schedules/output_re_s1"
        for row in inventory["instances"]:
            scenario_path = scenario_dir / row["filename"]
            identity = build_instance_identity(
                scenario_path,
                instance_alias=row["instance_alias"],
                collection_key="output_re_s1",
            )
            self.assertEqual(identity.raw_hash, row["raw_hash"])
            self.assertEqual(identity.semantic_hash, row["semantic_hash"])
        for instance_alias in config["split"]["dev"]:
            source_rows = inventory["dev_sources"][instance_alias]
            self.assertEqual(
                [row["source_family"] for row in source_rows],
                config["sources"]["solver_families"],
            )
            for row in source_rows:
                self.assertEqual(
                    sha256_file(schedule_dir / row["filename"]),
                    row["sha256"],
                )


if __name__ == "__main__":
    unittest.main()
