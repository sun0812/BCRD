# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.prepare_ccod_diagnostic import (
    prepare_and_publish,
    select_validation_subsets,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "algorithms/ccod/configs/diagnostic_v1.json"


class CCODPrelabelPreparationTest(unittest.TestCase):
    def _config(self):
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    def _selected(self):
        rows = []
        ordinal = 0
        for instance_alias in ("cities_08", "cities_04"):
            for index in range(50):
                rows.append(
                    {
                        "instance_alias": instance_alias,
                        "state_hash": (
                            f"sha256:{instance_alias.removeprefix('cities_')}"
                            f"{index:062d}"
                        ),
                        "candidate_count": 17 + index,
                        "selection": {"selected_ordinal": ordinal},
                    }
                )
                ordinal += 1
        return rows

    def test_validation_subsets_are_stable_balanced_and_nested(self) -> None:
        selected = self._selected()
        first = select_validation_subsets(selected, self._config())
        second = select_validation_subsets(list(reversed(selected)), self._config())
        self.assertEqual(first, second)

        exhaustive = first["exhaustive"]["states"]
        strong = first["strong"]["states"]
        self.assertEqual(len(exhaustive), 20)
        self.assertEqual(len(strong), 10)
        self.assertEqual(
            {city: sum(row["instance_alias"] == city for row in exhaustive)
             for city in ("cities_08", "cities_04")},
            {"cities_08": 10, "cities_04": 10},
        )
        self.assertEqual(
            {city: sum(row["instance_alias"] == city for row in strong)
             for city in ("cities_08", "cities_04")},
            {"cities_08": 5, "cities_04": 5},
        )
        self.assertTrue(
            {row["state_hash"] for row in strong}.issubset(
                {row["state_hash"] for row in exhaustive}
            )
        )
        self.assertTrue(first["strong"]["subset_of_exhaustive"])
        self.assertTrue(all(17 <= row["candidate_count"] <= 128 for row in exhaustive))

    def test_validation_subset_fails_when_one_instance_is_insufficient(self) -> None:
        selected = [
            row for row in self._selected()
            if row["instance_alias"] != "cities_04" or row["candidate_count"] < 17
        ]
        with self.assertRaisesRegex(ValueError, "exhaustive/cities_04 候选不足"):
            select_validation_subsets(selected, self._config())

    def test_publish_refuses_to_touch_existing_frozen_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frozen = root / "frozen"
            frozen.mkdir()
            sentinel = frozen / "sentinel.txt"
            sentinel.write_text("保留", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "拒绝覆盖"):
                prepare_and_publish(
                    CONFIG_PATH,
                    root / "missing-scenarios",
                    root / "missing-schedules",
                    frozen,
                )
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "保留")

    def test_complete_package_is_verified_before_atomic_publish(self) -> None:
        """完整冻结包必须先通过闭合校验，才能获得最终目录名。"""
        run_id = "sha256:" + "ab" * 32
        prepared = {
            "status": "query_plan_frozen",
            "catalog_manifest": {},
            "run_manifest": {"run_id": run_id},
        }
        verification = {
            "run_id": run_id,
            "catalog_hash": "sha256:" + "01" * 32,
            "selection_hash": "sha256:" + "02" * 32,
            "query_plan_hash": "sha256:" + "03" * 32,
            "states": 100,
            "queries": 1600,
        }
        with tempfile.TemporaryDirectory() as directory:
            final_dir = Path(directory) / "frozen"
            with patch(
                "scripts.prepare_ccod_diagnostic.prepare_diagnostic",
                return_value=prepared,
            ), patch(
                "scripts.prepare_ccod_diagnostic.verify_frozen_artifacts",
                return_value=verification,
            ) as verifier:
                result = prepare_and_publish(
                    CONFIG_PATH,
                    Path(directory) / "scenarios",
                    Path(directory) / "schedules",
                    final_dir,
                )

            verifier.assert_called_once()
            self.assertTrue(final_dir.is_dir())
            self.assertEqual(result["verification"], verification)

    def test_failed_verification_cannot_publish_staging_directory(self) -> None:
        """闭合校验失败时最终目录必须保持不存在。"""
        prepared = {
            "status": "query_plan_frozen",
            "catalog_manifest": {},
            "run_manifest": {"run_id": "sha256:" + "ab" * 32},
        }
        with tempfile.TemporaryDirectory() as directory:
            final_dir = Path(directory) / "frozen"
            with patch(
                "scripts.prepare_ccod_diagnostic.prepare_diagnostic",
                return_value=prepared,
            ), patch(
                "scripts.prepare_ccod_diagnostic.verify_frozen_artifacts",
                side_effect=ValueError("故意破坏的冻结包"),
            ):
                with self.assertRaisesRegex(ValueError, "故意破坏"):
                    prepare_and_publish(
                        CONFIG_PATH,
                        Path(directory) / "scenarios",
                        Path(directory) / "schedules",
                        final_dir,
                    )

            self.assertFalse(final_dir.exists())
            self.assertFalse(
                any(Path(directory).glob(".frozen.staging-*"))
            )


if __name__ == "__main__":
    unittest.main()
