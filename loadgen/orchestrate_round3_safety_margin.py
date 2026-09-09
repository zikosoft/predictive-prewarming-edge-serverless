"""
Round-3 (FGCS finalization pass) — A10: pre-warm safety-margin
(STARTUP_LEAD_S) sensitivity sweep, across all four predictive policies,
so the main conclusions can be shown not to depend on one arbitrarily
generous 1.0s margin. N=6 independent trials per (policy, margin) pair --
explicitly sanctioned by the revision plan as a smaller sensitivity
study rather than a full N=15 primary experiment. The nominal margin
(1.0s) is reused from the already-collected N=15 primary run for each
predictor, not re-run.

Reuses the identical, unmodified measurement path
(orchestrate_and_run.py's run_policy_scenario_independent); only
STARTUP_LEAD_S is swept.

Output: results/raw_marginsweep/<margin>s/<policy>/raw_<policy>_cyclical_trial<N>.csv
        results/marginsweep_trial_summary.csv, _gateway_stats.csv, _resource_samples.csv
"""
import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import orchestrate_and_run as oar
import aiohttp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "results")
N_TRIALS = 6

NOMINAL_MARGIN_S = 1.0
MARGIN_SWEEP_VALUES = [0.1, 0.25, 0.5, 1.5]  # 1.0 (nominal) reused from primary N=15 run

PREDICTORS = ["serverless_predictive_ewma", "serverless_predictive_lastgap",
              "serverless_predictive_movavg", "serverless_predictive_fixed"]


def write_csv(path, rows, fieldnames):
    import csv
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


async def main():
    all_trials, all_gw_stats, resource_rows = [], [], []
    async with aiohttp.ClientSession() as session:
        for margin in MARGIN_SWEEP_VALUES:
            for policy in PREDICTORS:
                label = f"{margin}s/{policy}"
                print(f"\n=== [A10] SAFETY-MARGIN SWEEP: margin={margin}s policy={policy} ===", flush=True)
                oar.STARTUP_LEAD_S = margin
                oar.RESULTS_DIR = os.path.join(OUT_DIR, "raw_marginsweep", f"{margin}s", policy)
                os.makedirs(oar.RESULTS_DIR, exist_ok=True)
                t0 = time.time()
                trials_this, gw_this = [], []
                res_before = len(resource_rows)
                await oar.run_policy_scenario_independent(
                    session, policy, "cyclical", oar.CYCLICAL_SPEC,
                    N_TRIALS, resource_rows, trials_this, gw_this)
                for r in trials_this:
                    r["startup_lead_s"] = margin
                for r in gw_this:
                    r["startup_lead_s"] = margin
                for r in resource_rows[res_before:]:
                    r["startup_lead_s"] = margin
                all_trials.extend(trials_this)
                all_gw_stats.extend(gw_this)
                print(f"    done in {time.time()-t0:.1f}s ({N_TRIALS} trials)", flush=True)

                write_csv(os.path.join(OUT_DIR, "marginsweep_trial_summary.csv"), all_trials,
                          ["policy", "scenario", "trial", "seed", "n_requests", "n_ok", "n_cold", "startup_lead_s"])
                write_csv(os.path.join(OUT_DIR, "marginsweep_gateway_stats.csv"), all_gw_stats,
                          ["policy", "scenario", "trial", "cold_starts", "predictive_cold_starts", "reactive_cold_starts",
                           "warm_hits", "wasted_prewarms", "spawn_failures", "gap_observations", "pool_size",
                           "predicted_gap_s", "predictor", "predictive", "startup_lead_s"])
                write_csv(os.path.join(OUT_DIR, "marginsweep_resource_samples.csv"), resource_rows,
                          ["t", "label", "cpu_pct_sum", "rss_mb_sum", "n_procs", "startup_lead_s"])
    oar.STARTUP_LEAD_S = NOMINAL_MARGIN_S
    print("\n[*] [A10] SAFETY-MARGIN SWEEP COMPLETE.", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
