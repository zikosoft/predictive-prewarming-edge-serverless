"""
Real-infrastructure load generator and orchestrator -- runs the SAME
three scenarios, the SAME seeding scheme (matched across policies for
a given (scenario, trial) index), and the SAME independent-trial
protocol as loadgen/orchestrate_and_run.py, but against genuine
Kubernetes + Knative endpoints instead of the process-level emulation.

Run real-infra/scripts/01_install_knative.sh and
real-infra/scripts/02_build_and_deploy.sh first.

Policies:
  container            -- hits the K8s container Service directly
                           (http://localhost:8080/ingest); 3 always-on
                           pods, Kubernetes' own Service load-balances.
  serverless_reactive  -- hits the Knative Service URL directly; no
                           custom code at all -- Knative's own
                           activator/autoscaler does cold start and
                           scale-to-zero natively. This is the
                           "stock" reactive baseline.
  serverless_predictive_ewma
                        -- hits the SAME Knative Service URL for real
                           traffic, but a background asyncio task in
                           THIS process also observes real send
                           timestamps (never the other way around --
                           see PredictiveController, structurally
                           identical in spirit to
                           gateway/elastic_gateway.py's contamination
                           fix: it never treats its own priming
                           requests as real arrivals) and fires a
                           lightweight priming GET to /health ahead of
                           a forecast burst, so Knative cold-starts a
                           pod before the real burst lands on it.

Cold-start detection on real infra: app/telemetry_app.py already
returns `server_uptime_s` (seconds since the pod's Python process
started) in every response. A response with server_uptime_s below
COLD_THRESHOLD_S is classified cold -- a direct, self-reported signal
from the same application code used throughout this project, not a
latency-threshold guess.

Output: results/real/raw_<policy>_<scenario>_trial<N>.csv, same core
columns as the emulation (t, request_id, cycle_id, latency_ms, status,
cold, ok, err) so real-infra/analyze_real.py can compare the two
tracks side by side. (queue_ms/cold_ms/service_ms are not available
from stock Knative without deeper platform instrumentation, so those
columns are written as 0 here -- the decomposition figure stays a
controlled-emulation-only artifact.)
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

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "real")
os.makedirs(RESULTS_DIR, exist_ok=True)

CONTAINER_URL = os.environ.get("CONTAINER_URL", "http://localhost:8080")
COLD_THRESHOLD_S = float(os.environ.get("COLD_THRESHOLD_S", "3.0"))
EWMA_ALPHA = float(os.environ.get("EWMA_ALPHA", "0.4"))
STARTUP_LEAD_S = float(os.environ.get("STARTUP_LEAD_S", "1.5"))  # real cold starts are slower than the emulation's
GAP_MIN_S = float(os.environ.get("GAP_MIN_S", "1.0"))

N_TRIALS_SHORT = int(os.environ.get("N_TRIALS_SHORT", "5"))
N_TRIALS_CYCLICAL = int(os.environ.get("N_TRIALS_CYCLICAL", "8"))  # fewer than the emulation's 15: real pod scheduling is slower
N_CYCLES_PER_TRIAL = int(os.environ.get("N_CYCLES_PER_TRIAL", "5"))

SCENARIOS_SHORT = {
    "low": {"kind": "constant", "rate_rps": 5, "duration_s": 12},
    "moderate": {"kind": "ramp", "start_rps": 5, "end_rps": 25, "duration_s": 16},
}
CYCLICAL_SPEC = {"kind": "cyclical", "n_cycles": N_CYCLES_PER_TRIAL, "idle_min_s": 8.0, "idle_max_s": 10.0, "burst_n": 15}
POLICIES = ["container", "serverless_reactive", "serverless_predictive_ewma"]


def trial_seed(scenario_name, trial):
    """Identical scheme to loadgen/orchestrate_and_run.py: shared
    across policies for a given (scenario, trial), so the real-infra
    run faces the same matched idle-jitter/payload sequence as the
    emulation's runs at the same trial index (comparable, though not
    expected to be numerically identical given real scheduling
    variance)."""
    h = hashlib.sha256(f"{scenario_name}:{trial}".encode()).hexdigest()
    return int(h[:8], 16)


def make_payload(rng, i):
    return json.dumps({"event_id": i, "value": rng.uniform(0, 100), "type": rng.choice(["metric", "log", "trace"])})


def get_knative_url():
    out = subprocess.run(["kubectl", "get", "ksvc", "telemetry-serverless", "-o", "jsonpath={.status.url}"],
                          capture_output=True, text=True, check=True)
    url = out.stdout.strip()
    if not url:
        raise RuntimeError("Knative Service URL not found -- did scripts/02_build_and_deploy.sh succeed?")
    return url


class PredictiveController:
    """Observes real request send-timestamps (fed by send_one, NEVER
    by its own priming action) and fires a priming GET ahead of a
    forecast burst. Structurally the same non-contamination guarantee
    as gateway/elastic_gateway.py's _note_real_arrival: this class has
    no way to observe its own priming request as a real arrival,
    because priming requests are sent via `_prime()` below, which
    never calls `note_arrival()`."""

    def __init__(self, session, target_url):
        self.session = session
        self.target_url = target_url
        self.gap_s = None
        self._last_arrival = None
        self._empty_since = None
        self._task = None
        self._stop = False
        self.primes_fired = 0

    def note_arrival(self):
        now = time.time()
        prev = self._last_arrival
        self._last_arrival = now
        self._empty_since = None  # a real request just landed; no longer "idle since prewarm decision"
        if prev is not None:
            gap = now - prev
            if gap >= GAP_MIN_S:
                if self.gap_s is None:
                    self.gap_s = gap
                else:
                    self.gap_s = EWMA_ALPHA * gap + (1 - EWMA_ALPHA) * self.gap_s

    async def _prime(self):
        try:
            async with self.session.get(f"{self.target_url}/health", timeout=aiohttp.ClientTimeout(total=10)):
                pass
            self.primes_fired += 1
        except Exception:
            pass

    async def _loop(self):
        while not self._stop:
            await asyncio.sleep(0.2)
            if self._last_arrival is None:
                continue
            now = time.time()
            idle_elapsed = now - self._last_arrival
            if self._empty_since is not None:
                continue  # already primed for this idle period
            if self.gap_s is None:
                continue
            if idle_elapsed < max(self.gap_s - STARTUP_LEAD_S, 0.2):
                continue
            self._empty_since = now
            await self._prime()

    def start(self):
        self._task = asyncio.ensure_future(self._loop())

    async def stop(self):
        self._stop = True
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass


async def send_one(session, url, payload, request_id, cycle_id, controller=None):
    if controller is not None:
        controller.note_arrival()
    t0 = time.perf_counter()
    try:
        async with session.post(f"{url}/ingest", data=payload, headers={"Content-Type": "application/json"},
                                 timeout=aiohttp.ClientTimeout(total=15)) as r:
            body = await r.read()
            dt = (time.perf_counter() - t0) * 1000.0
            try:
                parsed = json.loads(body)
                uptime = float(parsed.get("server_uptime_s", 999))
            except Exception:
                uptime = 999.0
            cold = "1" if uptime < COLD_THRESHOLD_S else "0"
            return {"t": time.time(), "request_id": request_id, "cycle_id": cycle_id,
                    "latency_ms": dt, "queue_ms": 0.0, "cold_ms": 0.0, "service_ms": 0.0,
                    "status": r.status, "cold": cold, "ok": r.status == 200, "server_uptime_s": uptime}
    except Exception as e:
        dt = (time.perf_counter() - t0) * 1000.0
        return {"t": time.time(), "request_id": request_id, "cycle_id": cycle_id,
                "latency_ms": dt, "queue_ms": 0.0, "cold_ms": 0.0, "service_ms": 0.0,
                "status": -1, "cold": "", "ok": False, "err": str(e)}


async def run_scenario(session, spec, url, rows_out, rng, controller=None):
    tasks = []
    i = 0
    if spec["kind"] == "constant":
        end = time.perf_counter() + spec["duration_s"]
        while time.perf_counter() < end:
            payload = make_payload(rng, i); i += 1
            tasks.append(asyncio.ensure_future(send_one(session, url, payload, i, -1, controller)))
            await asyncio.sleep(1.0 / spec["rate_rps"])
    elif spec["kind"] == "ramp":
        end = time.perf_counter() + spec["duration_s"]
        t0 = time.perf_counter()
        while time.perf_counter() < end:
            frac = (time.perf_counter() - t0) / spec["duration_s"]
            rate = max(spec["start_rps"] + frac * (spec["end_rps"] - spec["start_rps"]), 1.0)
            payload = make_payload(rng, i); i += 1
            tasks.append(asyncio.ensure_future(send_one(session, url, payload, i, -1, controller)))
            await asyncio.sleep(1.0 / rate)
    elif spec["kind"] == "cyclical":
        for c in range(spec["n_cycles"]):
            idle = rng.uniform(spec["idle_min_s"], spec["idle_max_s"])
            await asyncio.sleep(idle)
            burst_tasks = []
            for _ in range(spec["burst_n"]):
                payload = make_payload(rng, i); i += 1
                burst_tasks.append(asyncio.ensure_future(send_one(session, url, payload, i, c, controller)))
            tasks.extend(burst_tasks)
            await asyncio.gather(*burst_tasks)
    results = await asyncio.gather(*tasks)
    rows_out.extend(results)
    return results


def write_csv(path, rows):
    fields = ["t", "request_id", "cycle_id", "latency_ms", "queue_ms", "cold_ms", "service_ms",
              "status", "cold", "ok", "err", "server_uptime_s"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


async def run_policy_scenario(session, policy, scenario_name, spec, n_trials, knative_url, all_trials):
    url = CONTAINER_URL if policy == "container" else knative_url
    for trial in range(n_trials):
        seed = trial_seed(scenario_name, trial)
        rng = random.Random(seed)
        print(f"[*] {policy} / {scenario_name} / trial {trial+1}/{n_trials} (seed={seed})", flush=True)
        controller = None
        if policy == "serverless_predictive_ewma":
            controller = PredictiveController(session, url)
            controller.start()
        rows = []
        await run_scenario(session, spec, url, rows, rng, controller)
        if controller is not None:
            await controller.stop()
        write_csv(os.path.join(RESULTS_DIR, f"raw_{policy}_{scenario_name}_trial{trial}.csv"), rows)
        n_ok = sum(1 for r in rows if r.get("ok"))
        n_cold = sum(1 for r in rows if r.get("cold") == "1")
        primes = controller.primes_fired if controller is not None else 0
        all_trials.append({"policy": policy, "scenario": scenario_name, "trial": trial, "seed": seed,
                            "n_requests": len(rows), "n_ok": n_ok, "n_cold": n_cold, "primes_fired": primes})
        await asyncio.sleep(1.0)


async def main():
    knative_url = get_knative_url()
    print(f"[*] Knative Service URL: {knative_url}", flush=True)
    print(f"[*] Container Service URL: {CONTAINER_URL}", flush=True)
    all_trials = []
    async with aiohttp.ClientSession() as session:
        for scenario_name, spec in SCENARIOS_SHORT.items():
            for policy in POLICIES:
                await run_policy_scenario(session, policy, scenario_name, spec, N_TRIALS_SHORT, knative_url, all_trials)
        for policy in POLICIES:
            await run_policy_scenario(session, policy, "cyclical", CYCLICAL_SPEC, N_TRIALS_CYCLICAL, knative_url, all_trials)

    with open(os.path.join(RESULTS_DIR, "trial_summary_real.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["policy", "scenario", "trial", "seed", "n_requests", "n_ok", "n_cold", "primes_fired"])
        w.writeheader()
        for r in all_trials:
            w.writerow(r)
    print("[*] real-infra experiment complete.", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
