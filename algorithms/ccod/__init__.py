# -*- coding: utf-8 -*-
"""CCOD 的反事实标注与查询算法。"""

from algorithms.ccod.continuation import (
    CounterfactualError,
    CounterfactualResult,
    ContinuationOracle,
    ContinuationConfig,
    GreedyChoice,
    PreparedReplayedState,
    QUERY_IDENTITY_SCHEMA_VERSION,
    build_query_identity,
    choose_objective_greedy_action,
    continuation_implementation_hash,
    evaluate_replayed_state,
    force_action,
)
from algorithms.ccod.cache import (
    CounterfactualLabelCache,
    build_cache_identity,
    cache_key,
)

__all__ = [
    "CounterfactualError",
    "CounterfactualResult",
    "ContinuationOracle",
    "ContinuationConfig",
    "GreedyChoice",
    "PreparedReplayedState",
    "QUERY_IDENTITY_SCHEMA_VERSION",
    "build_query_identity",
    "choose_objective_greedy_action",
    "continuation_implementation_hash",
    "evaluate_replayed_state",
    "force_action",
    "CounterfactualLabelCache",
    "build_cache_identity",
    "cache_key",
]
