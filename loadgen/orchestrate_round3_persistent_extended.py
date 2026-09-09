"""
Round-3 (FGCS finalization pass) — A9: extend the persistent-state
(no-reset, 25-cycle) learning-curve run from EWMA-only to the three
remaining adaptive/non-adaptive predictors (last-gap, moving-average,
fixed-schedule), so the paper's central methodological claim --
independent trial resets structurally deny adaptive forecasters their
first-cycle opportunity, and a persistent-state run is needed to see
their true steady-state quality (Section 5.5/5.6) -- is not illustrated
by only one of the four predictors. Reactive and EWMA are unchanged
(already collected in the primary run's learning-curve data,
results/raw_learning/); not re-run here.

Reuses the identical measurement path as orchestrate_and_run.py's
run_learning_curve_run (single persistent gateway process, no resets
across cycles); descriptive only, never mixed into the significance
tests (same convention as the existing EWMA/reactive learning-curve run).

Output: results/raw_learning_extended/raw_learning_<policy>.csv
        results/learning_extended_gateway_stats.csv
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
LEARNING_DIR = os.path.join(OUT_DIR, "raw_learning_extended")
os.makedirs(LEARNING_DIR, exist_ok=True)

EXTRA_POLICIES = ["serverless_predictive_lastgap", "serverless_predictive_movavg",
                   "serverless_predictive_fixed"]


def write_csv(path, rows, fieldnames):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


async def main():
    resource_rows = []
    all_gw_stats = []
    spec = {"kind": "cyclical", "n_cycles": oar.N_LEARNING_CYCLES,
            "idle_min_s": 8.0, "idle_max_s": 10.0, "burst_n": 15}
    async with aiohttp.ClientSession() as session:
        for policy in EXTRA_POLICIES:
            print(f"\n=== [A9] persistent-state (no-reset, {oar.N_LEARNING_CYCLES} cycles) / {policy} ===", flush=True)
            t0 = time.time()
            gw, gw_port = await oar._start_elastic_gateway(policy)
            await oar.wait_ready(session, f"http://127.0.0.1:{gw_port}/stats")
            pids = lambda gw=gw: oar._pids_for(gw)
            target = lambda gw_port=gw_port: f"http://127.0.0.1:{gw_port}/invoke"
            rng = __import__("random").Random(oar.trial_seed("learning_extended", 0))
            rows = []
            sampler = oar.ResourceSampler(pids, resource_rows, f"learning_extended:{policy}")
            sampler.start()
            await oar.run_scenario(session, "learning_cyclical", spec, target, rows, rng)
            await sampler.stop()
            write_csv(os.path.join(LEARNING_DIR, f"raw_learning_{policy}.csv"),
                      rows, ["t", "request_id", "cycle_id", "latency_ms", "queue_ms", "cold_ms",
                             "service_ms", "status", "cold", "ok", "err"])
            try:
                async with session.get(f"http://127.0.0.1:{gw_port}/stats", timeout=aiohttp.ClientTimeout(total=1)) as r:
                    gw_stats = await r.json()
            except Exception:
                gw_stats = {}
            all_gw_stats.append({"policy": f"learning_extended:{policy}", "scenario": "learning_cyclical", "trial": 0, **gw_stats})
            gw.terminate()
            await asyncio.sleep(0.5)
            print(f"    done in {time.time()-t0:.1f}s", flush=True)

    gw_fields = ["policy", "scenario", "trial", "cold_starts", "predictive_cold_starts", "reactive_cold_starts",
                 "warm_hits", "wasted_prewarms", "spawn_failures", "gap_observations", "pool_size",
                 "predicted_gap_s", "predictor", "predictive"]
    write_csv(os.path.join(OUT_DIR, "learning_extended_gateway_stats.csv"), all_gw_stats, gw_fields)
    write_csv(os.path.join(OUT_DIR, "learning_extended_resource_samples.csv"), resource_rows,
              ["t", "label", "cpu_pct_sum", "rss_mb_sum", "n_procs"])
    print("\n[*] [A9] EXTENDED PERSISTENT-STATE RUN COMPLETE.", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
