"""
Targeted re-run after fixing the predictive_prewarmer() reference-frame
bug in gateway/elastic_gateway.py (see CHANGELOG.md, "v3.1").

Scope: ONLY the cyclical scenario's four predictive policies
(serverless_predictive_{ewma,lastgap,movavg,fixed}) are re-run. Every
other slice of the primary results set is structurally unaffected by
this bug and is intentionally NOT re-run:
  - container            never touches elastic_gateway.py at all.
  - serverless_reactive  has PREDICTIVE=0 and never calls
                          predictive_prewarmer() (it returns
                          immediately -- see the function's first
                          line).
  - low / moderate       scenarios have no idle gaps >= GAP_MIN_S, so
                          the prewarmer's "empty" branch is reached
                          rarely/never regardless of the bug -- these
                          scenarios don't exercise it.
The learning-curve run IS re-run for both its policies (ewma, to fix
it; reactive, alongside it, purely so the learning-curve figure keeps
a clean paired reactive/ewma comparison from the same run rather than
splicing two different sessions' reactive curves together).

This script re-uses loadgen/orchestrate_and_run.py's own functions
unmodified (same port assignments, same trial_seed() function, so
trial N of a re-run predictive policy faces the EXACT SAME idle-jitter
and payload sequence as trial N of the container/reactive runs already
on disk -- the matched-seed design is preserved across this partial
re-run). It then merges the new per-trial raw CSVs (which land at the
same paths as before and simply overwrite the stale ones) into the
four aggregate result files, replacing only the rows that belong to
the re-run slice and leaving every other row untouched.

Usage: python3 loadgen/rerun_predictive_fix.py
"""
import asyncio
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import orchestrate_and_run as base

AFFECTED_CYCLICAL_POLICIES = [
    "serverless_predictive_ewma",
    "serverless_predictive_lastgap",
    "serverless_predictive_movavg",
    "serverless_predictive_fixed",
]
LEARNING_RERUN_POLICIES = ["serverless_reactive", "serverless_predictive_ewma"]

RESULTS_DIR = os.path.join(base.ROOT, "results")


def _merge_csv(path, fieldnames, new_rows, drop_mask):
    """Load existing CSV (if any), drop rows matching drop_mask(row),
    append new_rows, write back with the same fieldnames/formatting
    orchestrate_and_run.write_csv uses."""
    kept = []
    if os.path.exists(path):
        old = pd.read_csv(path, dtype=str, keep_default_na=False)
        for _, r in old.iterrows():
            row = r.to_dict()
            if not drop_mask(row):
                kept.append(row)
    kept.extend(new_rows)
    base.write_csv(path, kept, fieldnames)
    print(f"[*] merged -> {path} ({len(kept)} rows, {len(new_rows)} new/replaced)", flush=True)


async def main():
    resource_rows = []
    all_trials = []
    all_gw_stats = []
    learning_gw_stats = []

    async with base.aiohttp.ClientSession() as session:
        for policy in AFFECTED_CYCLICAL_POLICIES:
            await base.run_policy_scenario_independent(
                session, policy, "cyclical", base.CYCLICAL_SPEC, base.N_TRIALS_CYCLICAL,
                resource_rows, all_trials, all_gw_stats)

        base.LEARNING_POLICIES = LEARNING_RERUN_POLICIES
        await base.run_learning_curve_run(session, resource_rows, learning_gw_stats)

    # --- merge into results/trial_summary.csv ---
    _merge_csv(
        os.path.join(RESULTS_DIR, "trial_summary.csv"),
        ["policy", "scenario", "trial", "seed", "n_requests", "n_ok", "n_cold"],
        all_trials,
        drop_mask=lambda row: row["policy"] in AFFECTED_CYCLICAL_POLICIES and row["scenario"] == "cyclical",
    )

    # --- merge into results/gateway_stats.csv ---
    gw_fields = ["policy", "scenario", "trial", "cold_starts", "predictive_cold_starts", "reactive_cold_starts",
                 "warm_hits", "wasted_prewarms", "spawn_failures", "gap_observations", "pool_size",
                 "predicted_gap_s", "predictor", "predictive"]
    _merge_csv(
        os.path.join(RESULTS_DIR, "gateway_stats.csv"),
        gw_fields,
        all_gw_stats,
        drop_mask=lambda row: row["policy"] in AFFECTED_CYCLICAL_POLICIES and row["scenario"] == "cyclical",
    )

    # --- merge into results/learning_gateway_stats.csv ---
    learning_policy_tags = {f"learning:{p}" for p in LEARNING_RERUN_POLICIES}
    _merge_csv(
        os.path.join(RESULTS_DIR, "learning_gateway_stats.csv"),
        gw_fields,
        learning_gw_stats,
        drop_mask=lambda row: row["policy"] in learning_policy_tags,
    )

    # --- merge into results/resource_samples.csv ---
    affected_cyclical_prefixes = tuple(f"{p}:cyclical:" for p in AFFECTED_CYCLICAL_POLICIES)
    affected_learning_prefixes = tuple(f"learning:{p}" for p in LEARNING_RERUN_POLICIES)

    def _resource_drop(row):
        label = row["label"]
        return label.startswith(affected_cyclical_prefixes) or label.startswith(affected_learning_prefixes)

    _merge_csv(
        os.path.join(RESULTS_DIR, "resource_samples.csv"),
        ["t", "label", "cpu_pct_sum", "rss_mb_sum", "n_procs"],
        resource_rows,
        drop_mask=_resource_drop,
    )

    print("[*] targeted re-run + merge complete.", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
