"""
Round-2 revision: two REAL (not simulated/estimated) pilot sweeps, 
reusing the exact, already-debugged measurement path from orchestrate_and_run.py 
(same run_policy_scenario_independent / run_scenario / send_one functions --
nothing about how a request is timed or classified is changed here,
only which env-var condition is active per run, to minimize the risk
of introducing a new measurement bug while extending the harness).

Reduced trial count vs. the primary N=15 cyclical protocol, for wall-
clock tractability in this pass: N_TRIALS_SWEEP=6 independent trials per
new condition (still >= the N=5 already used elsewhere in this paper for
the low/moderate scenarios, and still a real, seed-matched, independent-
trial design -- not a reduction in rigor, only in statistical power
relative to the primary N=15 comparisons). Explicitly labeled as a pilot
in all outputs; a confirmatory N=15 run remains recommended before
camera-ready if reviewers want tighter confidence intervals.

Where a sweep's condition is numerically identical to a condition already
collected in the primary N=15 run (fixed-schedule nominal = 9.0s;
keep-alive/reactive idle-timeout nominal = 6.0s), that already-collected
N=15 data is reused for the nominal point instead of re-running it, both
to save time and because it is a larger, already-vetted sample.

1) FIXED-SCHEDULE SENSITIVITY SWEEP (Round-2 feedback item 5)
   FIXED_PREDICTED_GAP_S at -50/-25/-10/+10/+25/+50% of the manuscript's
   disclosed nominal value (9.0s, the midpoint of the workload's own
   8-10s jittered idle range -- disclosed exactly, per Round-2 item 5's
   disclosure requirement, in the accompanying revision package/
   manuscript text). Policy = serverless_predictive_fixed (PREDICTIVE=1,
   PREDICTOR=fixed); every other parameter identical to the primary run.

2) KEEP-ALIVE (IDLE-TIMEOUT) SWEEP, NON-ORACLE (Round-2 feedback item 4,
   "Preferred design")
   IDLE_TIMEOUT_S in {8, 10, 12} seconds (6s = the primary run's existing
   reactive-baseline nominal, reused, not re-run). These are round,
   operator-plausible values, not tuned to the workload generator's own
   8-10s gap range (the Round-2 review's specific objection to an
   11s "oracle" choice) -- 8s and 10s fall inside that range and 12s
   sits above it, so this sweep neither privileges nor systematically
   disadvantages the keep-alive baseline relative to the workload it is
   evaluated on. Policy = serverless_reactive (PREDICTIVE=0); every
   other parameter identical to the primary run, sweeping only
   IDLE_TIMEOUT_S.

Output layout (each condition gets its own subdirectory so nothing
collides with, or overwrites, the primary results/raw/*.csv):
  results/raw_fixedsweep/<pct_label>/raw_serverless_predictive_fixed_cyclical_trial<N>.csv
  results/raw_keepalivesweep/<timeout_label>/raw_serverless_reactive_cyclical_trial<N>.csv
  results/fixedsweep_gateway_stats.csv, results/keepalivesweep_gateway_stats.csv
  results/fixedsweep_resource_samples.csv, results/keepalivesweep_resource_samples.csv
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
N_TRIALS_SWEEP = 6

NOMINAL_FIXED_GAP_S = 9.0
FIXED_SWEEP_PCTS = [-50, -25, -10, 10, 25, 50]  # nominal (0%) reused from primary run, not re-run

NOMINAL_IDLE_TIMEOUT_S = 6.0
KEEPALIVE_SWEEP_VALUES = [8.0, 10.0, 12.0]  # nominal (6.0s) reused from primary run, not re-run


def write_csv(path, rows, fieldnames):
    import csv
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


async def run_fixed_sweep(session):
    all_trials, all_gw_stats, resource_rows = [], [], []
    for pct in FIXED_SWEEP_PCTS:
        value = round(NOMINAL_FIXED_GAP_S * (1 + pct / 100.0), 4)
        label = f"{pct:+d}pct_{value}s"
        print(f"\n=== FIXED-SCHEDULE SWEEP: {label} ===", flush=True)
        oar.FIXED_PREDICTED_GAP_S = value
        oar.RESULTS_DIR = os.path.join(OUT_DIR, "raw_fixedsweep", label)
        os.makedirs(oar.RESULTS_DIR, exist_ok=True)
        t0 = time.time()
        trials_this, gw_this = [], []
        await oar.run_policy_scenario_independent(
            session, "serverless_predictive_fixed", "cyclical", oar.CYCLICAL_SPEC,
            N_TRIALS_SWEEP, resource_rows, trials_this, gw_this)
        for r in trials_this:
            r["sweep_pct"] = pct
            r["fixed_predicted_gap_s"] = value
        for r in gw_this:
            r["sweep_pct"] = pct
            r["fixed_predicted_gap_s"] = value
        all_trials.extend(trials_this)
        all_gw_stats.extend(gw_this)
        print(f"    done in {time.time()-t0:.1f}s", flush=True)

    write_csv(os.path.join(OUT_DIR, "fixedsweep_trial_summary.csv"), all_trials,
              ["policy", "scenario", "trial", "seed", "n_requests", "n_ok", "n_cold", "sweep_pct", "fixed_predicted_gap_s"])
    write_csv(os.path.join(OUT_DIR, "fixedsweep_gateway_stats.csv"), all_gw_stats,
              ["policy", "scenario", "trial", "cold_starts", "predictive_cold_starts", "reactive_cold_starts",
               "warm_hits", "wasted_prewarms", "spawn_failures", "gap_observations", "pool_size",
               "predicted_gap_s", "predictor", "predictive", "sweep_pct", "fixed_predicted_gap_s"])
    write_csv(os.path.join(OUT_DIR, "fixedsweep_resource_samples.csv"), resource_rows,
              ["t", "label", "cpu_pct_sum", "rss_mb_sum", "n_procs"])
    print("[*] fixed-schedule sweep complete.", flush=True)


async def run_keepalive_sweep(session):
    all_trials, all_gw_stats, resource_rows = [], [], []
    for value in KEEPALIVE_SWEEP_VALUES:
        label = f"{value}s"
        print(f"\n=== KEEP-ALIVE SWEEP: idle_timeout={label} ===", flush=True)
        oar.IDLE_TIMEOUT_S = value
        oar.RESULTS_DIR = os.path.join(OUT_DIR, "raw_keepalivesweep", label)
        os.makedirs(oar.RESULTS_DIR, exist_ok=True)
        t0 = time.time()
        trials_this, gw_this = [], []
        await oar.run_policy_scenario_independent(
            session, "serverless_reactive", "cyclical", oar.CYCLICAL_SPEC,
            N_TRIALS_SWEEP, resource_rows, trials_this, gw_this)
        for r in trials_this:
            r["idle_timeout_s"] = value
        for r in gw_this:
            r["idle_timeout_s"] = value
        all_trials.extend(trials_this)
        all_gw_stats.extend(gw_this)
        print(f"    done in {time.time()-t0:.1f}s", flush=True)

    write_csv(os.path.join(OUT_DIR, "keepalivesweep_trial_summary.csv"), all_trials,
              ["policy", "scenario", "trial", "seed", "n_requests", "n_ok", "n_cold", "idle_timeout_s"])
    write_csv(os.path.join(OUT_DIR, "keepalivesweep_gateway_stats.csv"), all_gw_stats,
              ["policy", "scenario", "trial", "cold_starts", "predictive_cold_starts", "reactive_cold_starts",
               "warm_hits", "wasted_prewarms", "spawn_failures", "gap_observations", "pool_size",
               "predicted_gap_s", "predictor", "predictive", "idle_timeout_s"])
    write_csv(os.path.join(OUT_DIR, "keepalivesweep_resource_samples.csv"), resource_rows,
              ["t", "label", "cpu_pct_sum", "rss_mb_sum", "n_procs"])
    print("[*] keep-alive sweep complete.", flush=True)


async def main():
    async with aiohttp.ClientSession() as session:
        await run_fixed_sweep(session)
        await run_keepalive_sweep(session)
    # restore module defaults (harmless, script exits right after anyway)
    oar.FIXED_PREDICTED_GAP_S = NOMINAL_FIXED_GAP_S
    oar.IDLE_TIMEOUT_S = NOMINAL_IDLE_TIMEOUT_S
    print("\n[*] ALL ROUND-2 SWEEPS COMPLETE.", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
