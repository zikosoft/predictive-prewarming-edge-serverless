"""
Experiment orchestrator (v3, corrected after external review).

Architectures / policies:
  container                      -- N persistent replicas, fronted by
                                     gateway/container_gateway.py (same
                                     one-hop path as the on-demand
                                     architectures -- see that file's
                                     docstring for why this changed).
  serverless_reactive             -- gateway/elastic_gateway.py, PREDICTIVE=0
                                     (multi-instance, scale-to-zero, no forecasting)
  serverless_predictive_ewma      -- PREDICTIVE=1, PREDICTOR=ewma (the paper's
                                     main contribution)
  serverless_predictive_lastgap   -- PREDICTIVE=1, PREDICTOR=lastgap (naive
                                     baseline: forecast = most recent gap)
  serverless_predictive_movavg    -- PREDICTIVE=1, PREDICTOR=movavg (simple
                                     unweighted moving average baseline)
  serverless_predictive_fixed     -- PREDICTIVE=1, PREDICTOR=fixed (assume a
                                     constant, hand-set schedule -- the
                                     "you already knew the period" baseline)

Scenarios:
  low        -- constant 5 req/s, 12 s (no idle gaps: predictor has nothing to learn)
  moderate   -- ramp 5->25 req/s, 16 s (no idle gaps: predictor has nothing to learn)
  cyclical   -- N_CYCLES_PER_TRIAL repeated (idle 8-10 s jittered, then 15
                concurrent requests) cycles; the scenario the predictive
                policies target. Only run for the four predictor ablations plus
                container and reactive (6 policies total).

Independence / pseudoreplication fix (after external review): every
trial is now an INDEPENDENT REPLICATE -- the gateway process (and
therefore all predictor/pool state) is restarted fresh at the start of
EVERY trial, not just at the start of each scenario. N_INDEPENDENT_TRIALS
trials are run per (scenario, policy). Analysis aggregates each trial to
one summary value before any significance test, so the unit of
statistical analysis is the trial, not the individual request (see
analysis/analyze.py). Trial `trial` of every (scenario, policy) pair
shares an RNG seed derived from (scenario, trial) alone -- NOT the
policy/arch -- so all policies face the identical sequence of idle-gap
jitter and payload draws for a given trial index: a matched/paired
design, not just an independent one.

Separately, a dedicated LEARNING-CURVE run (no resets across cycles)
reproduces the "does the predictor improve with repeated exposure"
figure for the reactive and ewma policies specifically -- see
run_learning_curve_run(). Its output is analyzed descriptively only and
is never mixed into the significance tests.

Output: results/raw/raw_<policy>_<scenario>_trial<N>.csv (per-request,
including the queue/cold/service latency decomposition read from the
gateway's response headers), results/resource_samples.csv,
results/trial_summary.csv, results/gateway_stats.csv,
results/raw_learning/raw_learning_<policy>_cycle<C>.csv.
"""
import asyncio
import csv
import hashlib
import json
import os
import random
import subprocess
import sys
import time

import aiohttp
import psutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_DIR = os.path.join(ROOT, "app")
GATEWAY_DIR = os.path.join(ROOT, "gateway")
RESULTS_DIR = os.path.join(ROOT, "results", "raw")
LEARNING_DIR = os.path.join(ROOT, "results", "raw_learning")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LEARNING_DIR, exist_ok=True)

N_CONTAINER_REPLICAS = 3
CONTAINER_GW_PORT = 9800
MAX_INSTANCES = 3
IDLE_TIMEOUT_S = 6.0          # instance-level scale-to-zero timeout
STARTUP_LEAD_S = 1.0          # predictor's safety margin ahead of predicted arrival
EWMA_ALPHA = 0.4
MOVAVG_WINDOW = 5
FIXED_PREDICTED_GAP_S = 9.0   # midpoint of the 8-10s jittered idle range
N_TRIALS_SHORT = 5            # low / moderate: short scenarios, still independent trials
N_TRIALS_CYCLICAL = 15        # cyclical: independent replicates for the significance test
N_CYCLES_PER_TRIAL = 5
N_LEARNING_CYCLES = 25        # dedicated persistent-state learning-curve run

SCENARIOS_SHORT = {
    "low": {"kind": "constant", "rate_rps": 5, "duration_s": 12},
    "moderate": {"kind": "ramp", "start_rps": 5, "end_rps": 25, "duration_s": 16},
}
CYCLICAL_SPEC = {"kind": "cyclical", "n_cycles": N_CYCLES_PER_TRIAL, "idle_min_s": 8.0, "idle_max_s": 10.0, "burst_n": 15}

SHORT_POLICIES = ["container", "serverless_reactive", "serverless_predictive_ewma"]
CYCLICAL_POLICIES = ["container", "serverless_reactive", "serverless_predictive_ewma",
                      "serverless_predictive_lastgap", "serverless_predictive_movavg",
                      "serverless_predictive_fixed"]
LEARNING_POLICIES = ["serverless_reactive", "serverless_predictive_ewma"]

GW_PORT_BY_POLICY = {
    "serverless_reactive": (9750, 9600),
    "serverless_predictive_ewma": (9760, 9650),
    "serverless_predictive_lastgap": (9761, 9660),
    "serverless_predictive_movavg": (9762, 9670),
    "serverless_predictive_fixed": (9763, 9680),
}


def trial_seed(scenario_name, trial):
    """Deterministic seed shared across ALL policies for a given
    (scenario, trial) pair, so every policy is compared against the
    identical idle-jitter / payload sequence for that trial index."""
    h = hashlib.sha256(f"{scenario_name}:{trial}".encode()).hexdigest()
    return int(h[:8], 16)


def make_payload(rng, i):
    return json.dumps({"event_id": i, "value": rng.uniform(0, 100), "type": rng.choice(["metric", "log", "trace"])})


class ResourceSampler:
    def __init__(self, pid_provider, out_rows, label):
        self.pid_provider = pid_provider
        self.out_rows = out_rows
        self.label = label
        self._task = None
        self._stop = False

    async def _loop(self):
        procs = {}
        while not self._stop:
            for pid in self.pid_provider():
                if pid not in procs:
                    try:
                        procs[pid] = psutil.Process(pid)
                        procs[pid].cpu_percent(None)
                    except Exception:
                        continue
            total_cpu, total_rss, alive = 0.0, 0, 0
            for pid in list(procs.keys()):
                try:
                    p = procs[pid]
                    total_cpu += p.cpu_percent(None)
                    total_rss += p.memory_info().rss
                    alive += 1
                except Exception:
                    procs.pop(pid, None)
            self.out_rows.append({"t": time.time(), "label": self.label,
                                   "cpu_pct_sum": round(total_cpu, 2),
                                   "rss_mb_sum": round(total_rss / (1024 * 1024), 2),
                                   "n_procs": alive})
            await asyncio.sleep(0.2)

    def start(self):
        self._task = asyncio.ensure_future(self._loop())

    async def stop(self):
        self._stop = True
        if self._task:
            await asyncio.sleep(0)
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass


def _hdr_f(headers, key, default=0.0):
    try:
        return float(headers.get(key, default))
    except (TypeError, ValueError):
        return default


async def send_one(session, url, payload, request_id, cycle_id):
    t0 = time.perf_counter()
    try:
        async with session.post(url, data=payload, headers={"Content-Type": "application/json"},
                                 timeout=aiohttp.ClientTimeout(total=8)) as r:
            await r.read()
            dt = (time.perf_counter() - t0) * 1000.0
            return {"t": time.time(), "request_id": request_id, "cycle_id": cycle_id,
                    "latency_ms": dt, "queue_ms": _hdr_f(r.headers, "X-Queue-Ms"),
                    "cold_ms": _hdr_f(r.headers, "X-Cold-Ms"), "service_ms": _hdr_f(r.headers, "X-Service-Ms"),
                    "status": r.status, "cold": r.headers.get("X-Cold-Start", ""), "ok": r.status == 200}
    except Exception as e:
        dt = (time.perf_counter() - t0) * 1000.0
        return {"t": time.time(), "request_id": request_id, "cycle_id": cycle_id,
                "latency_ms": dt, "queue_ms": 0.0, "cold_ms": 0.0, "service_ms": 0.0,
                "status": -1, "cold": "", "ok": False, "err": str(e)}


async def run_scenario(session, scenario_name, spec, target_urls_fn, rows_out, rng):
    tasks = []
    i = 0
    if spec["kind"] == "constant":
        end = time.perf_counter() + spec["duration_s"]
        while time.perf_counter() < end:
            payload = make_payload(rng, i); i += 1
            tasks.append(asyncio.ensure_future(send_one(session, target_urls_fn(), payload, i, -1)))
            await asyncio.sleep(1.0 / spec["rate_rps"])
    elif spec["kind"] == "ramp":
        end = time.perf_counter() + spec["duration_s"]
        t0 = time.perf_counter()
        while time.perf_counter() < end:
            frac = (time.perf_counter() - t0) / spec["duration_s"]
            rate = max(spec["start_rps"] + frac * (spec["end_rps"] - spec["start_rps"]), 1.0)
            payload = make_payload(rng, i); i += 1
            tasks.append(asyncio.ensure_future(send_one(session, target_urls_fn(), payload, i, -1)))
            await asyncio.sleep(1.0 / rate)
    elif spec["kind"] == "cyclical":
        for c in range(spec["n_cycles"]):
            idle = rng.uniform(spec["idle_min_s"], spec["idle_max_s"])
            await asyncio.sleep(idle)
            burst_tasks = []
            for _ in range(spec["burst_n"]):
                payload = make_payload(rng, i); i += 1
                burst_tasks.append(asyncio.ensure_future(send_one(session, target_urls_fn(), payload, i, c)))
            tasks.extend(burst_tasks)
            await asyncio.gather(*burst_tasks)  # let this cycle's burst fully settle before starting the next idle timer
    elif spec["kind"] == "bimodal":
        for c in range(spec["n_cycles"]):
            mode = rng.choice(["short", "long"])
            if mode == "short":
                idle = rng.uniform(spec["short_min_s"], spec["short_max_s"])
            else:
                idle = rng.uniform(spec["long_min_s"], spec["long_max_s"])
            await asyncio.sleep(idle)
            burst_tasks = []
            for _ in range(spec["burst_n"]):
                payload = make_payload(rng, i); i += 1
                burst_tasks.append(asyncio.ensure_future(send_one(session, target_urls_fn(), payload, i, c)))
            tasks.extend(burst_tasks)
            await asyncio.gather(*burst_tasks)
    results = await asyncio.gather(*tasks)
    rows_out.extend(results)
    return results


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


async def wait_ready(session, url, timeout_s=15):
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=0.4)) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        await asyncio.sleep(0.1)
    return False


async def _start_container_gateway(session):
    procs = []
    for k in range(N_CONTAINER_REPLICAS):
        port = 9001 + k
        p = subprocess.Popen([sys.executable, os.path.join(GATEWAY_DIR, "container_backend.py")],
                              env={**os.environ, "PORT": str(port)},
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        procs.append((p, port))
    for _p, port in procs:
        await wait_ready(session, f"http://127.0.0.1:{port}/health")
    gw = subprocess.Popen([sys.executable, os.path.join(GATEWAY_DIR, "container_gateway.py")],
                           env={**os.environ, "GATEWAY_PORT": str(CONTAINER_GW_PORT),
                                "PORT_BASE": "9001", "N_CONTAINER_REPLICAS": str(N_CONTAINER_REPLICAS)},
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    await wait_ready(session, f"http://127.0.0.1:{CONTAINER_GW_PORT}/stats")
    return procs, gw


def _stop_procs(procs, gw=None):
    for p, _ in procs:
        p.terminate()
    if gw is not None:
        gw.terminate()


async def _start_elastic_gateway(policy):
    predictor = policy.replace("serverless_predictive_", "") if policy.startswith("serverless_predictive_") else "ewma"
    predictive = "1" if policy.startswith("serverless_predictive_") else "0"
    gw_port, port_base = GW_PORT_BY_POLICY[policy]
    env = os.environ.copy()
    env.update({
        "GATEWAY_PORT": str(gw_port), "MAX_INSTANCES": str(MAX_INSTANCES),
        "IDLE_TIMEOUT_S": str(IDLE_TIMEOUT_S), "PREDICTIVE": predictive, "PREDICTOR": predictor,
        "EWMA_ALPHA": str(EWMA_ALPHA), "MOVAVG_WINDOW": str(MOVAVG_WINDOW),
        "FIXED_PREDICTED_GAP_S": str(FIXED_PREDICTED_GAP_S),
        "STARTUP_LEAD_S": str(STARTUP_LEAD_S), "PORT_BASE": str(port_base),
    })
    gw = subprocess.Popen([sys.executable, os.path.join(GATEWAY_DIR, "elastic_gateway.py")],
                           env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return gw, gw_port


def _pids_for(gw):
    pids_ = [gw.pid]
    try:
        pids_ += [c.pid for c in psutil.Process(gw.pid).children(recursive=True)]
    except Exception:
        pass
    return pids_


async def run_policy_scenario_independent(session, policy, scenario_name, spec, n_trials,
                                           resource_rows, all_trials, all_gw_stats):
    """Runs n_trials INDEPENDENT trials of (policy, scenario): the
    backend (container replicas + gateway, or elastic gateway) is
    torn down and restarted fresh before every trial, so predictor /
    pool state never carries across trials."""
    for trial in range(n_trials):
        seed = trial_seed(scenario_name, trial)
        rng = random.Random(seed)
        print(f"[*] {policy} / {scenario_name} / trial {trial+1}/{n_trials} (seed={seed})", flush=True)

        if policy == "container":
            procs, gw = await _start_container_gateway(session)
            pids = lambda: _pids_for(gw) + [p.pid for p, _ in procs]
            target = lambda: f"http://127.0.0.1:{CONTAINER_GW_PORT}/invoke"
        else:
            gw, gw_port = await _start_elastic_gateway(policy)
            await wait_ready(session, f"http://127.0.0.1:{gw_port}/stats")
            pids = lambda gw=gw: _pids_for(gw)
            target = lambda gw_port=gw_port: f"http://127.0.0.1:{gw_port}/invoke"

        sampler = ResourceSampler(pids, resource_rows, f"{policy}:{scenario_name}:{trial}")
        sampler.start()
        rows = []
        await run_scenario(session, scenario_name, spec, target, rows, rng)
        await sampler.stop()

        write_csv(os.path.join(RESULTS_DIR, f"raw_{policy}_{scenario_name}_trial{trial}.csv"),
                  rows, ["t", "request_id", "cycle_id", "latency_ms", "queue_ms", "cold_ms",
                         "service_ms", "status", "cold", "ok", "err"])
        n_ok = sum(1 for r in rows if r.get("ok"))
        n_cold = sum(1 for r in rows if r.get("cold") == "1")
        gw_stats = {}
        if policy != "container":
            try:
                async with session.get(f"http://127.0.0.1:{gw_port}/stats", timeout=aiohttp.ClientTimeout(total=1)) as r:
                    gw_stats = await r.json()
            except Exception:
                pass
        all_trials.append({"policy": policy, "scenario": scenario_name, "trial": trial, "seed": seed,
                            "n_requests": len(rows), "n_ok": n_ok, "n_cold": n_cold})
        all_gw_stats.append({"policy": policy, "scenario": scenario_name, "trial": trial, **gw_stats})

        if policy == "container":
            _stop_procs(procs, gw)
        else:
            gw.terminate()
        await asyncio.sleep(0.5)


async def run_learning_curve_run(session, resource_rows, all_gw_stats):
    """Dedicated persistent-state run: the gateway is started ONCE and
    kept alive across N_LEARNING_CYCLES consecutive cycles, so the
    predictor's improvement over repeated exposure can be observed
    directly. Analyzed descriptively only -- never mixed into the
    significance tests (see module docstring)."""
    spec = {"kind": "cyclical", "n_cycles": N_LEARNING_CYCLES, "idle_min_s": 8.0, "idle_max_s": 10.0, "burst_n": 15}
    for policy in LEARNING_POLICIES:
        print(f"[*] learning-curve run / {policy} ({N_LEARNING_CYCLES} cycles, persistent state)...", flush=True)
        gw, gw_port = await _start_elastic_gateway(policy)
        await wait_ready(session, f"http://127.0.0.1:{gw_port}/stats")
        pids = lambda gw=gw: _pids_for(gw)
        target = lambda gw_port=gw_port: f"http://127.0.0.1:{gw_port}/invoke"
        rng = random.Random(trial_seed("learning", 0))
        rows = []
        sampler = ResourceSampler(pids, resource_rows, f"learning:{policy}")
        sampler.start()
        await run_scenario(session, "learning_cyclical", spec, target, rows, rng)
        await sampler.stop()
        write_csv(os.path.join(LEARNING_DIR, f"raw_learning_{policy}.csv"),
                  rows, ["t", "request_id", "cycle_id", "latency_ms", "queue_ms", "cold_ms",
                         "service_ms", "status", "cold", "ok", "err"])
        try:
            async with session.get(f"http://127.0.0.1:{gw_port}/stats", timeout=aiohttp.ClientTimeout(total=1)) as r:
                gw_stats = await r.json()
        except Exception:
            gw_stats = {}
        all_gw_stats.append({"policy": f"learning:{policy}", "scenario": "learning_cyclical", "trial": 0, **gw_stats})
        gw.terminate()
        await asyncio.sleep(0.5)


async def main():
    resource_rows = []
    all_trials = []
    all_gw_stats = []
    learning_gw_stats = []
    async with aiohttp.ClientSession() as session:
        for scenario_name, spec in SCENARIOS_SHORT.items():
            for policy in SHORT_POLICIES:
                await run_policy_scenario_independent(session, policy, scenario_name, spec, N_TRIALS_SHORT,
                                                        resource_rows, all_trials, all_gw_stats)
        for policy in CYCLICAL_POLICIES:
            await run_policy_scenario_independent(session, policy, "cyclical", CYCLICAL_SPEC, N_TRIALS_CYCLICAL,
                                                    resource_rows, all_trials, all_gw_stats)
        await run_learning_curve_run(session, resource_rows, learning_gw_stats)

    write_csv(os.path.join(ROOT, "results", "resource_samples.csv"), resource_rows,
              ["t", "label", "cpu_pct_sum", "rss_mb_sum", "n_procs"])
    write_csv(os.path.join(ROOT, "results", "trial_summary.csv"), all_trials,
              ["policy", "scenario", "trial", "seed", "n_requests", "n_ok", "n_cold"])
    gw_fields = ["policy", "scenario", "trial", "cold_starts", "predictive_cold_starts", "reactive_cold_starts",
                 "warm_hits", "wasted_prewarms", "spawn_failures", "gap_observations", "pool_size",
                 "predicted_gap_s", "predictor", "predictive"]
    write_csv(os.path.join(ROOT, "results", "gateway_stats.csv"), all_gw_stats, gw_fields)
    write_csv(os.path.join(ROOT, "results", "learning_gateway_stats.csv"), learning_gw_stats, gw_fields)
    print("[*] experiment complete.", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
