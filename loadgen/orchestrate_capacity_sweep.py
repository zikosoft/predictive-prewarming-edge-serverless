"""
Supplementary capacity-sweep run: same cyclical scenario, same 6
policies, same 15-independent-trial / matched-seed protocol as
loadgen/orchestrate_and_run.py, but with MAX_INSTANCES raised so the
burst (15 concurrent requests) is no longer capacity-constrained.

Why: the corrected primary run (MAX_INSTANCES=3) found that predictive
pre-warming does not significantly reduce cyclical-scenario latency,
because with only 3 concurrent slots for a 15-request burst, ~80% of
requests queue for capacity (mean queue_ms ~480ms) far more than they
wait on any single cold start (mean cold_ms ~119ms) -- pre-warming one
instance ahead of the burst cannot fix a capacity shortfall. This sweep
asks: does the predictor show a benefit once capacity is no longer the
bottleneck? Output goes to a SEPARATE results directory
(results/raw_capacity{N}/) so it never overwrites or gets mixed into
the primary (MAX_INSTANCES=3) results already analyzed.

Usage: MAX_INSTANCES=8 python3 loadgen/orchestrate_capacity_sweep.py
(defaults to 8 if not set -- comfortably above the 15-request burst
size's likely bottleneck point while still leaving some queuing
headroom to observe, unlike e.g. MAX_INSTANCES=15 which would trivially
eliminate all queuing and be a less informative single data point).
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import orchestrate_and_run as base

CAPACITY = int(os.environ.get("MAX_INSTANCES", "8"))
base.MAX_INSTANCES = CAPACITY
base.RESULTS_DIR = os.path.join(base.ROOT, "results", f"raw_capacity{CAPACITY}")
os.makedirs(base.RESULTS_DIR, exist_ok=True)
# port bases shifted well clear of the primary run's range, in case both are ever run concurrently
base.GW_PORT_BY_POLICY = {
    "serverless_reactive": (9950, 9200),
    "serverless_predictive_ewma": (9951, 9250),
    "serverless_predictive_lastgap": (9952, 9260),
    "serverless_predictive_movavg": (9953, 9270),
    "serverless_predictive_fixed": (9954, 9280),
}
base.CONTAINER_GW_PORT = 9899


async def main():
    resource_rows = []
    all_trials = []
    all_gw_stats = []
    async with base.aiohttp.ClientSession() as session:
        for policy in base.CYCLICAL_POLICIES:
            await base.run_policy_scenario_independent(
                session, policy, "cyclical", base.CYCLICAL_SPEC, base.N_TRIALS_CYCLICAL,
                resource_rows, all_trials, all_gw_stats)

    out_dir = os.path.join(base.ROOT, "results")
    base.write_csv(os.path.join(out_dir, f"resource_samples_capacity{CAPACITY}.csv"), resource_rows,
                    ["t", "label", "cpu_pct_sum", "rss_mb_sum", "n_procs"])
    base.write_csv(os.path.join(out_dir, f"trial_summary_capacity{CAPACITY}.csv"), all_trials,
                    ["policy", "scenario", "trial", "seed", "n_requests", "n_ok", "n_cold"])
    gw_fields = ["policy", "scenario", "trial", "cold_starts", "predictive_cold_starts", "reactive_cold_starts",
                 "warm_hits", "wasted_prewarms", "spawn_failures", "gap_observations", "pool_size",
                 "predicted_gap_s", "predictor", "predictive"]
    base.write_csv(os.path.join(out_dir, f"gateway_stats_capacity{CAPACITY}.csv"), all_gw_stats, gw_fields)
    print(f"[*] capacity-sweep (MAX_INSTANCES={CAPACITY}) experiment complete.", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
