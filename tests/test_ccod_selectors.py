# -*- coding: utf-8 -*-
"""CCOD 确定性候选选择器的独立测试。"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from algorithms.ccod.selectors import (
    SelectorError,
    UNIFORM_RANK_SCHEMA_VERSION,
    diagnostic_query_prefix,
    stable_uniform_rank,
)
from schedulers.state_replay import canonical_json_bytes, sha256_json


def _action(kind: str, index: int = 0) -> dict:
    """构造足以覆盖排序契约的轻量 ActionKey。"""
    if kind == "skip":
        return {"version": "test-v1", "kind": "skip", "task_id": "T"}
    return {
        "version": "test-v1",
        "kind": "assign",
        "task_id": "T",
        "candidate_id": index,
    }


class CCODSelectorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.state_hash = "sha256:" + "1" * 64
        self.skip = _action("skip")
        self.assignments = tuple(_action("assign", index) for index in range(6))

    def test_uniform_rank_matches_frozen_hash_payload(self) -> None:
        candidates = (self.skip, *self.assignments)
        actual = stable_uniform_rank(self.state_hash, candidates, seed=19)
        expected = sorted(
            candidates,
            key=lambda action_key: (
                sha256_json(
                    {
                        "schema_version": UNIFORM_RANK_SCHEMA_VERSION,
                        "seed": 19,
                        "state_hash": self.state_hash,
                        "action_key": action_key,
                    }
                ),
                canonical_json_bytes(action_key),
            ),
        )
        self.assertEqual(
            [canonical_json_bytes(item) for item in actual],
            [canonical_json_bytes(item) for item in expected],
        )

    def test_uniform_rank_deduplicates_by_canonical_bytes(self) -> None:
        duplicate = {
            "candidate_id": 2,
            "task_id": "T",
            "kind": "assign",
            "version": "test-v1",
        }
        candidates = (self.assignments[2], self.skip, duplicate, self.assignments[0])
        first = stable_uniform_rank(self.state_hash, candidates, seed=7)
        second = stable_uniform_rank(
            self.state_hash, tuple(reversed(candidates)), seed=7
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)

    def test_hash_collision_uses_canonical_bytes_as_tie_breaker(self) -> None:
        candidates = (self.assignments[3], self.skip, self.assignments[1])
        with patch("algorithms.ccod.selectors.sha256_json", return_value="same"):
            ranked = stable_uniform_rank(self.state_hash, candidates, seed=3)
        self.assertEqual(
            [canonical_json_bytes(item) for item in ranked],
            sorted(canonical_json_bytes(item) for item in candidates),
        )

    def test_prefix_keeps_anchors_then_adds_only_uniform_fillers(self) -> None:
        candidates = (self.skip, *self.assignments)
        observed = self.assignments[3]
        first = diagnostic_query_prefix(
            self.state_hash,
            candidates,
            observed,
            budget=4,
            seed=29,
        )
        second = diagnostic_query_prefix(
            self.state_hash,
            tuple(reversed(candidates)),
            [dict(reversed(tuple(observed.items())))],
            budget=4,
            seed=29,
        )
        self.assertEqual(first, second)
        self.assertEqual(first[0]["action_key"], observed)
        self.assertEqual(first[0]["selection_sources"], ["observed"])
        self.assertEqual(first[1]["action_key"], self.skip)
        self.assertEqual(first[1]["selection_sources"], ["skip"])
        self.assertEqual(
            [row["selection_sources"] for row in first[2:]],
            [["stable_uniform"], ["stable_uniform"]],
        )
        self.assertEqual([row["query_rank"] for row in first], list(range(4)))

    def test_observed_skip_merges_roles_and_small_pool_queries_all(self) -> None:
        duplicate_skip = dict(reversed(tuple(self.skip.items())))
        candidates = (self.assignments[0], duplicate_skip, self.skip)
        result = diagnostic_query_prefix(
            self.state_hash,
            candidates,
            [self.skip, duplicate_skip],
            budget=16,
            seed=5,
        )
        self.assertEqual(len(result), 2)
        self.assertEqual(
            result[0]["selection_sources"], ["observed", "skip"]
        )
        self.assertEqual(result[1]["selection_sources"], ["stable_uniform"])
        self.assertEqual(
            {canonical_json_bytes(row["action_key"]) for row in result},
            {
                canonical_json_bytes(self.skip),
                canonical_json_bytes(self.assignments[0]),
            },
        )

    def test_distinct_anchors_must_fit_budget(self) -> None:
        with self.assertRaisesRegex(SelectorError, "budget 小于"):
            diagnostic_query_prefix(
                self.state_hash,
                (self.skip, self.assignments[0]),
                self.assignments[0],
                budget=1,
                seed=1,
            )

    def test_observed_is_exactly_one_canonical_action(self) -> None:
        with self.assertRaisesRegex(SelectorError, "只能提供一个"):
            diagnostic_query_prefix(
                self.state_hash,
                (self.skip, self.assignments[0]),
                (self.skip, self.assignments[0]),
                budget=2,
                seed=1,
            )
        with self.assertRaisesRegex(SelectorError, "不在当前候选集合"):
            diagnostic_query_prefix(
                self.state_hash,
                (self.skip, self.assignments[0]),
                self.assignments[1],
                budget=2,
                seed=1,
            )

    def test_invalid_public_inputs_are_rejected(self) -> None:
        with self.assertRaisesRegex(SelectorError, "state_hash"):
            stable_uniform_rank("", (self.skip,), seed=1)
        with self.assertRaisesRegex(SelectorError, "seed"):
            stable_uniform_rank(self.state_hash, (self.skip,), seed=True)
        with self.assertRaisesRegex(SelectorError, "budget"):
            diagnostic_query_prefix(
                self.state_hash,
                (self.skip,),
                self.skip,
                budget=True,
                seed=1,
            )
        with self.assertRaisesRegex(SelectorError, "恰好包含一个 SKIP"):
            diagnostic_query_prefix(
                self.state_hash,
                self.assignments,
                self.assignments[0],
                budget=2,
                seed=1,
            )

    def test_python_hash_seed_does_not_change_output(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        script = """
import json
from algorithms.ccod.selectors import diagnostic_query_prefix
skip = {"version": "test-v1", "kind": "skip", "task_id": "T"}
items = [skip]
for text in {"5", "1", "4", "2", "3", "0"}:
    value = int(text)
    items.append({
        "version": "test-v1",
        "kind": "assign",
        "task_id": "T",
        "candidate_id": value,
    })
observed = {"version": "test-v1", "kind": "assign", "task_id": "T", "candidate_id": 3}
result = diagnostic_query_prefix(
    "sha256:" + "a" * 64,
    items,
    observed,
    budget=5,
    seed=31,
)
print(json.dumps(result, sort_keys=True, separators=(",", ":")))
"""
        outputs = []
        for hash_seed in ("1", "987654"):
            environment = dict(os.environ)
            environment["PYTHONHASHSEED"] = hash_seed
            completed = subprocess.run(
                [sys.executable, "-c", script],
                cwd=repository,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )
            outputs.append(json.loads(completed.stdout))
        self.assertEqual(outputs[0], outputs[1])


if __name__ == "__main__":
    unittest.main()
