"""
Round-3 (FGCS finalization pass) — A6: one genuinely non-periodic
workload, run against all seven policies the revision plan asks for
(container, reactive, keep-alive-10s, EWMA, last-gap, moving-average,
fixed-schedule), N=15 independent seed-matched trials each, using the
identical measurement path as the primary run (only the workload
generator's idle-gap distribution changes -- see the new "bimodal"
branch added to loadgen/orchestrate_and_run.py's run_scenario()).

Design (bimodal / drifting idle-gap, no fixed period):
  each cycle independently draws "short" mode (idle ~ Uniform(2, 6) s)
  or "long" mode (idle ~ Uniform(10, 18) s) with equal probability, via
  the SAME per-trial RNG used for payload jitter (seed-matched across
  policies, exactly as in the primary cyclical scenario). Overall mean
  idle (~9.0 s) is deliberately close to the primary scenario's mean, so
  a ranking change is attributable to the SHAPE of the idle-gap
  distribution (bimodal / unbounded-relative-to-8-10s vs. narrow-jittered
  periodic), not to a change in average load. The short-mode range
  straddles the reactive/predictive pool's 6.0s scale-to-zero timeout
  (IDLE_TIMEOUT_S), so some "short" cycles never even fully cold-start;
  the long-mode range extends well past the primary scenario's 8-10s
  jitter bound, deliberately outside fixed-schedule's workload-informed
  9.0s constant. Burst size (15) and MAX_INSTANCES (3) are unchanged
  from the primary scenario -- only the idle-gap distribution differs,
  per the revision plan's "do not change multiple parameters at once."

The purpose is NOT to show universal superiority of any policy; it is to
test whether the primary ranking (fixed-schedule > adaptive > reactive)
and fixed-schedule's specific advantage survive when the workload is no
longer close to periodic.

Output: results/raw_nonperiodic/<policy_label>/raw_<policy>_bimodal_trial<N>.csv
        results/nonperiodic_trial_summary.csv, _gateway_stats.csv, _resource_samples.csv
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
N_TRIALS = 15

BIMODAL_SPEC = {
    "kind": "bimodal", "n_cycles": 5, "burst_n": 15,
    "short_min_s": 2.0, "short_max_s": 6.0,
    "long_min_s": 10.0, "long_max_s": 18.0,
}

# (label, policy, extra env overrides applied to oar module globals before the run)
RUNS = [
    ("container", "container", {}),
    ("serverless_reactive", "serverless_reactive", {}),
    ("serverless_keepalive10", "serverless_reactive", {"IDLE_TIMEOUT_S": 10.0}),
    ("serverless_predictive_ewma", "serverless_predictive_ewma", {}),
    ("serverless_predictive_lastgap", "serverless_predictive_lastgap", {}),
    ("serverless_predictive_movavg", "serverless_predictive_movavg", {}),
    ("serverless_predictive_fixed", "serverless_predictive_fixed", {}),
]


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
        for label, policy, overrides in RUNS:
            print(f"\n=== [A6 NON-PERIODIC] {label} ===", flush=True)
            # reset to defaults, then apply this run's overrides
            oar.IDLE_TIMEOUT_S = 6.0
            oar.FIXED_PREDICTED_GAP_S = 9.0
            for k, v in overrides.items():
                setattr(oar, k, v)
            oar.RESULTS_DIR = os.path.join(OUT_DIR, "raw_nonperiodic", label)
            os.makedirs(oar.RESULTS_DIR, exist_ok=True)
            # write bimodal-labeled per-trial CSVs (scenario name "bimodal"
            # so they never collide with / are mistaken for cyclical files)
            trials_this, gw_this = [], []
            res_before = len(resource_rows)
            t0 = time.time()

            # run_policy_scenario_independent writes files named
            # raw_<policy>_<scenario_name>_trial<N>.csv where scenario_name
            # is whatever we pass as the scenario label; use "bimodal" so
            # container's files (also used at label "serverless_reactive"
            # in the primary run) never collide.
            await oar.run_policy_scenario_independent(
                session, policy, "bimodal", BIMODAL_SPEC, N_TRIALS,
                resource_rows, trials_this, gw_this)
            for r in trials_this:
                r["run_label"] = label
            for r in gw_this:
                r["run_label"] = label
            for r in resource_rows[res_before:]:
                r["run_label"] = label
            all_trials.extend(trials_this)
            all_gw_stats.extend(gw_this)
            print(f"    done in {time.time()-t0:.1f}s ({N_TRIALS} trials)", flush=True)

            write_csv(os.path.join(OUT_DIR, "nonperiodic_trial_summary.csv"), all_trials,
                      ["policy", "scenario", "trial", "seed", "n_requests", "n_ok", "n_cold", "run_label"])
            write_csv(os.path.join(OUT_DIR, "nonperiodic_gateway_stats.csv"), all_gw_stats,
                      ["policy", "scenario", "trial", "cold_starts", "predictive_cold_starts", "reactive_cold_starts",
                       "warm_hits", "wasted_prewarms", "spawn_failures", "gap_observations", "pool_size",
                       "predicted_gap_s", "predictor", "predictive", "run_label"])
            write_csv(os.path.join(OUT_DIR, "nonperiodic_resource_samples.csv"), resource_rows,
                      ["t", "label", "cpu_pct_sum", "rss_mb_sum", "n_procs", "run_label"])

    oar.IDLE_TIMEOUT_S = 6.0
    oar.FIXED_PREDICTED_GAP_S = 9.0
    print("\n[*] [A6] NON-PERIODIC WORKLOAD RUN COMPLETE.", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
