# -*- coding: utf-8 -*-
"""CCOD 诊断准备阶段的快速、无标签单元测试。"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from algorithms.ccod.instances import RECOMMENDED_SPLIT
from scripts import prepare_ccod_diagnostic as prepare


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "algorithms/ccod/configs/diagnostic_v1.json"
INVENTORY_PATH = (
    REPO_ROOT / "algorithms/ccod/configs/diagnostic_v1_sources.json"
)
SCENARIO_DIR = REPO_ROOT / "output/output_re_s1"
SCHEDULE_DIR = REPO_ROOT / "output/schedules/output_re_s1"


class CCODPreparePureFunctionTest(unittest.TestCase):
    """验证不可变发布、稳定身份与轻量清单哈希。"""

    def test_publish_bytes_accepts_identical_content_and_rejects_conflict(
        self,
    ) -> None:
        """相同内容应幂等，冲突内容不得覆盖已经冻结的字节。"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested/frozen.jsonl"
            original = b'{"state_hash":"sha256:one"}\n'
            prepare._publish_bytes(path, original)
            original_inode = path.stat().st_ino

            prepare._publish_bytes(path, original)
            self.assertEqual(path.stat().st_ino, original_inode)
            self.assertEqual(path.read_bytes(), original)

            with self.assertRaisesRegex(
                prepare.DiagnosticPreparationError,
                "已冻结产物内容冲突",
            ):
                prepare._publish_bytes(path, b'{"state_hash":"sha256:two"}\n')
            self.assertEqual(path.read_bytes(), original)

    def test_source_id_is_frozen_deterministic_and_input_sensitive(self) -> None:
        """来源 ID 必须只由冻结协议中的三个逻辑字段决定。"""
        arguments = {
            "instance_alias": "cities_08",
            "source_family": "sa",
            "schedule_hash": "sha256:" + "ab" * 32,
        }
        expected = (
            "sha256:6bbcfe50168822a6527ac427da2e0450"
            "eae93b854e6f19a23053d67dca13a838"
        )
        self.assertEqual(prepare._trace_source_id(**arguments), expected)
        self.assertEqual(prepare._trace_source_id(**arguments), expected)
        self.assertNotEqual(
            prepare._trace_source_id(
                **{**arguments, "source_family": "ga"},
            ),
            expected,
        )
        self.assertNotEqual(
            prepare._trace_source_id(
                **{**arguments, "instance_alias": "cities_04"},
            ),
            expected,
        )

    def test_scientific_hash_is_key_order_invariant_but_row_order_sensitive(
        self,
    ) -> None:
        """catalog/query 行内采用规范 JSON，行序仍属于科学身份。"""
        rows = [
            {"query_key": "q-1", "roles": ["observed", "uniform"]},
            {"query_key": "q-2", "roles": ["skip"]},
        ]
        same_rows = [
            {"roles": ["observed", "uniform"], "query_key": "q-1"},
            {"roles": ["skip"], "query_key": "q-2"},
        ]
        original = json.loads(json.dumps(rows))

        digest = prepare._scientific_hash(rows)
        self.assertRegex(digest, r"^sha256:[0-9a-f]{64}$")
        self.assertEqual(digest, prepare._scientific_hash(same_rows))
        self.assertNotEqual(digest, prepare._scientific_hash(list(reversed(rows))))
        self.assertNotEqual(
            digest,
            prepare._scientific_hash(
                [rows[0], {"query_key": "q-3", "roles": ["skip"]}],
            ),
        )
        self.assertEqual(rows, original)

    def test_repo_relative_rejects_path_escape(self) -> None:
        """可搬迁引用只允许解析后仍位于仓库根目录内的路径。"""
        inside = CONFIG_PATH
        self.assertEqual(
            prepare._repo_relative(inside),
            Path("algorithms/ccod/configs/diagnostic_v1.json"),
        )

        with tempfile.TemporaryDirectory() as directory:
            escaped = Path(directory) / "outside.json"
            escaped.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(
                prepare.DiagnosticPreparationError,
                "诊断输入必须位于仓库内",
            ):
                prepare._repo_relative(escaped)


class CCODFrozenInventoryTest(unittest.TestCase):
    """使用真实小型元数据验证预注册 split 与 sealed 边界。"""

    def test_real_inventory_returns_only_frozen_split_and_reads_dev_schedules(
        self,
    ) -> None:
        """库存必须闭合为 6/2/2，且 schedule JSON 只读取两个 dev 实例。"""
        required = (INVENTORY_PATH, SCENARIO_DIR, SCHEDULE_DIR)
        if not all(path.exists() for path in required):
            self.skipTest("本地未提供冻结的 CCOD 场景与 schedule 库存")

        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        inventory = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
        expected_schedule_names = {
            row["filename"]
            for instance_alias in config["split"]["dev"]
            for row in inventory["dev_sources"][instance_alias]
        }
        loaded_schedule_names: list[str] = []
        real_load_json = prepare._load_json

        def recording_load_json(path: Path):
            resolved = Path(path).resolve()
            try:
                resolved.relative_to(SCHEDULE_DIR.resolve())
            except ValueError:
                pass
            else:
                loaded_schedule_names.append(resolved.name)
            return real_load_json(Path(path))

        with patch.object(
            prepare,
            "_load_json",
            side_effect=recording_load_json,
        ):
            loaded_inventory, split_manifest = prepare._validate_source_inventory(
                config,
                INVENTORY_PATH,
                SCENARIO_DIR,
                SCHEDULE_DIR,
            )

        for fold in ("train", "dev", "test"):
            self.assertEqual(
                split_manifest[fold],
                list(RECOMMENDED_SPLIT[fold]),
            )
            self.assertEqual(split_manifest[fold], config["split"][fold])
        self.assertTrue(split_manifest["sealed_test"])
        folded = (
            split_manifest["train"]
            + split_manifest["dev"]
            + split_manifest["test"]
        )
        self.assertEqual(len(folded), 10)
        self.assertEqual(len(set(folded)), 10)
        self.assertEqual(set(loaded_inventory["dev_sources"]), {"cities_08", "cities_04"})
        self.assertEqual(set(loaded_schedule_names), expected_schedule_names)
        self.assertEqual(len(loaded_schedule_names), len(expected_schedule_names))
        self.assertTrue(
            all(
                "cities_02" not in name and "cities_10" not in name
                for name in loaded_schedule_names
            )
        )


if __name__ == "__main__":
    unittest.main()
