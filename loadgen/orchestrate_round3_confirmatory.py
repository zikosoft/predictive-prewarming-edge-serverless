"""
Confirmatory N=15 reruns of
the two Round-2 N=6 pilot sweeps, using the identical, unmodified
measurement path (orchestrate_and_run.py's run_policy_scenario_independent
-- nothing about how a request is timed/classified changes here, only
which env-var condition is active and the trial count).

A4 -- keep-alive/idle-timeout sweep: IDLE_TIMEOUT_S in {8, 10, 12}
seconds, N=15 independent trials each (6.0s nominal reused from the
already-collected N=15 primary run, not re-run).

A5 -- fixed-schedule period sweep: FIXED_PREDICTED_GAP_S at
{-50,-25,-10,+10,+25,+50}% of the disclosed 9.0s nominal, N=15
independent trials each (0%/nominal reused from the already-collected
N=15 primary run, not re-run).

Fix vs. the Round-2 pilot script (orchestrate_round2_sweeps.py): every
resource-sample row is now tagged with its sweep condition (the pilot's
known limitation -- rows were labeled only by policy:scenario:trial-index,
identical across every swept condition, so CPU-s/GB-s could not be
attributed per condition). That fix is applied here so this confirmatory
run CAN report per-condition CPU-seconds/GB-seconds, per revision-plan
item A4.

Output layout (separate from both the primary run and the Round-2 pilot,
nothing is overwritten):
  results/raw_keepalivesweep_n15/<timeout>s/raw_serverless_reactive_cyclical_trial<N>.csv
  results/raw_fixedsweep_n15/<pct_label>/raw_serverless_predictive_fixed_cyclical_trial<N>.csv
  results/keepalivesweep_n15_trial_summary.csv, _gateway_stats.csv, _resource_samples.csv
  results/fixedsweep_n15_trial_summary.csv, _gateway_stats.csv, _resource_samples.csv

Run with: python3 loadgen/orchestrate_round3_confirmatory.py [keepalive|fixed|both]
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
N_TRIALS_CONFIRMATORY = 15

NOMINAL_FIXED_GAP_S = 9.0
FIXED_SWEEP_PCTS = [-50, -25, -10, 10, 25, 50]

NOMINAL_IDLE_TIMEOUT_S = 6.0
KEEPALIVE_SWEEP_VALUES = [8.0, 10.0, 12.0]


def write_csv(path, rows, fieldnames):
    import csv
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


async def run_keepalive_confirmatory(session):
    all_trials, all_gw_stats, resource_rows = [], [], []
    for value in KEEPALIVE_SWEEP_VALUES:
        label = f"{value}s"
        print(f"\n=== [A4] KEEP-ALIVE N=15 CONFIRMATORY: idle_timeout={label} ===", flush=True)
        oar.IDLE_TIMEOUT_S = value
        oar.RESULTS_DIR = os.path.join(OUT_DIR, "raw_keepalivesweep_n15", label)
        os.makedirs(oar.RESULTS_DIR, exist_ok=True)
        t0 = time.time()
        trials_this, gw_this = [], []
        res_before = len(resource_rows)
        await oar.run_policy_scenario_independent(
            session, "serverless_reactive", "cyclical", oar.CYCLICAL_SPEC,
            N_TRIALS_CONFIRMATORY, resource_rows, trials_this, gw_this)
        for r in trials_this:
            r["idle_timeout_s"] = value
        for r in gw_this:
            r["idle_timeout_s"] = value
        for r in resource_rows[res_before:]:
            r["idle_timeout_s"] = value  # per-condition tag fix
        all_trials.extend(trials_this)
        all_gw_stats.extend(gw_this)
        print(f"    done in {time.time()-t0:.1f}s ({N_TRIALS_CONFIRMATORY} trials)", flush=True)
        write_csv(os.path.join(OUT_DIR, "keepalivesweep_n15_trial_summary.csv"), all_trials,
                  ["policy", "scenario", "trial", "seed", "n_requests", "n_ok", "n_cold", "idle_timeout_s"])
        write_csv(os.path.join(OUT_DIR, "keepalivesweep_n15_gateway_stats.csv"), all_gw_stats,
                  ["policy", "scenario", "trial", "cold_starts", "predictive_cold_starts", "reactive_cold_starts",
                   "warm_hits", "wasted_prewarms", "spawn_failures", "gap_observations", "pool_size",
                   "predicted_gap_s", "predictor", "predictive", "idle_timeout_s"])
        write_csv(os.path.join(OUT_DIR, "keepalivesweep_n15_resource_samples.csv"), resource_rows,
                  ["t", "label", "cpu_pct_sum", "rss_mb_sum", "n_procs", "idle_timeout_s"])
    oar.IDLE_TIMEOUT_S = NOMINAL_IDLE_TIMEOUT_S
    print("[*] [A4] keep-alive N=15 confirmatory sweep complete.", flush=True)


async def run_fixed_confirmatory(session):
    all_trials, all_gw_stats, resource_rows = [], [], []
    for pct in FIXED_SWEEP_PCTS:
        value = round(NOMINAL_FIXED_GAP_S * (1 + pct / 100.0), 4)
        label = f"{pct:+d}pct_{value}s"
        print(f"\n=== [A5] FIXED-SCHEDULE N=15 CONFIRMATORY: {label} ===", flush=True)
        oar.FIXED_PREDICTED_GAP_S = value
        oar.RESULTS_DIR = os.path.join(OUT_DIR, "raw_fixedsweep_n15", label)
        os.makedirs(oar.RESULTS_DIR, exist_ok=True)
        t0 = time.time()
        trials_this, gw_this = [], []
        res_before = len(resource_rows)
        await oar.run_policy_scenario_independent(
            session, "serverless_predictive_fixed", "cyclical", oar.CYCLICAL_SPEC,
            N_TRIALS_CONFIRMATORY, resource_rows, trials_this, gw_this)
        for r in trials_this:
            r["sweep_pct"] = pct
            r["fixed_predicted_gap_s"] = value
        for r in gw_this:
            r["sweep_pct"] = pct
            r["fixed_predicted_gap_s"] = value
        for r in resource_rows[res_before:]:
            r["sweep_pct"] = pct
            r["fixed_predicted_gap_s"] = value
        all_trials.extend(trials_this)
        all_gw_stats.extend(gw_this)
        print(f"    done in {time.time()-t0:.1f}s ({N_TRIALS_CONFIRMATORY} trials)", flush=True)
        write_csv(os.path.join(OUT_DIR, "fixedsweep_n15_trial_summary.csv"), all_trials,
                  ["policy", "scenario", "trial", "seed", "n_requests", "n_ok", "n_cold", "sweep_pct", "fixed_predicted_gap_s"])
        write_csv(os.path.join(OUT_DIR, "fixedsweep_n15_gateway_stats.csv"), all_gw_stats,
                  ["policy", "scenario", "trial", "cold_starts", "predictive_cold_starts", "reactive_cold_starts",
                   "warm_hits", "wasted_prewarms", "spawn_failures", "gap_observations", "pool_size",
                   "predicted_gap_s", "predictor", "predictive", "sweep_pct", "fixed_predicted_gap_s"])
        write_csv(os.path.join(OUT_DIR, "fixedsweep_n15_resource_samples.csv"), resource_rows,
                  ["t", "label", "cpu_pct_sum", "rss_mb_sum", "n_procs", "sweep_pct", "fixed_predicted_gap_s"])
    oar.FIXED_PREDICTED_GAP_S = NOMINAL_FIXED_GAP_S
    print("[*] [A5] fixed-schedule N=15 confirmatory sweep complete.", flush=True)


async def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    async with aiohttp.ClientSession() as session:
        if which in ("keepalive", "both"):
            await run_keepalive_confirmatory(session)
        if which in ("fixed", "both"):
            await run_fixed_confirmatory(session)
    print("\n[*] ROUND-3 CONFIRMATORY SWEEPS COMPLETE.", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
