# Research intake

Assess and sharpen the current paper proposal in `/Users/sun/Documents/code/sat/EOS-Bench/方案.md` for the local EOS-Bench fork. The public benchmark anchor is **EOS-Bench: A Comprehensive Benchmark for Earth Observation Satellite Scheduling**.

Research problem: a fast learned scheduler must choose among large, dynamic feasible sets of `(task, satellite, observation-window)` actions. Existing solver schedules expose only the selected action. One-hot behavior cloning treats all feasible unselected actions as equally negative despite heterogeneous downstream opportunity costs.

Claimed method: **Budgeted Counterfactual Regret Distillation (BCRD)**. Warm-start a lightweight candidate-scoring policy from heterogeneous MIP/heuristic/metaheuristic/PPO trajectories. At selected offline states, force only a budgeted set of informative feasible alternatives, use deterministic cheap finite-horizon continuation to estimate candidate returns, form local normalized regret labels, train shared policy and nonnegative regret heads, and decode over the full feasible set with `policy_score - beta * predicted_regret`. No counterfactual rollout occurs at inference. Learner-state aggregation is optional and should remain secondary unless diagnostics support it.

Available local evidence: schedule-to-trajectory exporter, schema validator, streaming BC dataset, approximately 19K-parameter BC policy, and initial BC metrics. Counterfactual labeling, two-head model, learned-scheduler integration, and multi-instance end-to-end evaluation are not yet implemented.

Desired contribution type: method, with an empirical falsification plan. Intended venue and deadline were not specified. Compute should be assessed against the actual modest local setup described in `方案.md`, not an assumed large lab campaign.
