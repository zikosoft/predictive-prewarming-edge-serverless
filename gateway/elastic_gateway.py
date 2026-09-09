"""
Elastic on-demand gateway with a reactive baseline and a family of
pluggable prewarming predictors, selected by PREDICTIVE / PREDICTOR:

  PREDICTIVE=0            "reactive" baseline: a multi-instance pool that
                          scales out only in response to observed
                          concurrent demand (a request finding no free
                          instance -> a new one is cold-started) and
                          scales an idle instance down after
                          IDLE_TIMEOUT_S seconds without traffic. This is
                          the standard behavior of commercial FaaS
                          concurrency-based autoscaling, realized here
                          without a commercial platform.

  PREDICTIVE=1            identical elastic pool, plus a forecaster
  PREDICTOR=ewma|lastgap| (see GapPredictor below) that learns the
           movavg|fixed   typical silence duration preceding a burst
                          from past cycles and proactively cold-starts
                          an instance *before* the next burst is
                          expected. When the forecast is wrong, the
                          pre-warmed instance simply idles out normally
                          (wasted_prewarms counter, exposed at /stats).

Predictor families (deliberately kept as a pluggable ablation, not just
EWMA in isolation -- see manuscript Section 4.2 for the full design
rationale and Section 5 for the comparison):
  ewma     exponentially weighted moving average of observed gaps
           (EWMA_ALPHA controls adaptation speed).
  lastgap  forecast = the single most recently observed gap (no
           smoothing at all -- the naive baseline EWMA must beat).
  movavg   simple unweighted moving average of the last MOVAVG_WINDOW
           observed gaps.
  fixed    forecast = a constant, operator-configured value
           (FIXED_PREDICTED_GAP_S) that never adapts -- the "assume you
           already know the schedule" baseline.

Gap-observation integrity (bug fixed after external review, see repo
history): gaps are now measured EXCLUSIVELY from genuine external
arrivals at /invoke (see `_note_real_arrival`), never from the pool's
internal empty->nonempty transitions. The previous implementation
inferred "a burst started" from `len(pool) > 0`, which is also true the
instant the predictive prewarmer reserves a placeholder slot for its
own speculative spawn -- so a successful anticipatory prewarm was
mis-recorded as a fresh, very short observed gap, corrupting the EWMA
into a self-reinforcing collapse toward its own trigger threshold
rather than tracking the true ~8-10s inter-burst silence. Measuring
gaps strictly from real request timestamps at the gateway's external
edge makes this contamination structurally impossible: the prewarmer
never calls `_note_real_arrival`.

Concurrency design note: pool capacity is reserved (a placeholder slot
inserted) under `pool_lock`, but the slow part -- spawning the
subprocess and waiting for it to become healthy -- happens OUTSIDE the
lock, so concurrent requests are never serialized behind each other's
cold starts. A per-request watchdog (asyncio.wait_for) also guarantees
no single request can hang the gateway indefinitely.

Latency decomposition: every response carries X-Queue-Ms (time spent
waiting for pool capacity, i.e. blocked because MAX_INSTANCES busy
instances were all occupied -- excludes any cold-start spawn time),
X-Cold-Ms (spawn + health-check time, 0 on a warm hit), and
X-Service-Ms (time spent in the downstream application call itself),
in addition to the existing X-Cold-Start flag and X-Gateway-Total-Ms,
so the queuing-cascade mechanism can be measured directly rather than
inferred.
"""
import asyncio
import itertools
import os
import subprocess
import sys
import time

from aiohttp import web, ClientSession, ClientTimeout

APP_DIR = os.path.join(os.path.dirname(__file__), "..", "app")
PORT_BASE = int(os.environ.get("PORT_BASE", "9600"))
MAX_INSTANCES = int(os.environ.get("MAX_INSTANCES", "3"))
IDLE_TIMEOUT_S = float(os.environ.get("IDLE_TIMEOUT_S", "8"))
PREDICTIVE = os.environ.get("PREDICTIVE", "0") == "1"
PREDICTOR_KIND = os.environ.get("PREDICTOR", "ewma")  # ewma | lastgap | movavg | fixed
EWMA_ALPHA = float(os.environ.get("EWMA_ALPHA", "0.3"))
MOVAVG_WINDOW = int(os.environ.get("MOVAVG_WINDOW", "5"))
FIXED_PREDICTED_GAP_S = float(os.environ.get("FIXED_PREDICTED_GAP_S", "9.0"))
STARTUP_LEAD_S = float(os.environ.get("STARTUP_LEAD_S", "0.9"))
REQUEST_WATCHDOG_S = float(os.environ.get("REQUEST_WATCHDOG_S", "8.0"))
SPAWN_HEALTH_TIMEOUT_S = float(os.environ.get("SPAWN_HEALTH_TIMEOUT_S", "5.0"))
GAP_MIN_S = float(os.environ.get("GAP_MIN_S", "1.0"))  # min silence to count as an inter-burst gap
QUEUE_POLL_S = 0.015

_port_counter = itertools.count()
pool_lock = asyncio.Lock()
pool = {}  # port -> {"proc","busy","last_used","started_at","predictive_spawn","served","pending"}
stats = {"cold_starts": 0, "predictive_cold_starts": 0, "reactive_cold_starts": 0,
         "warm_hits": 0, "wasted_prewarms": 0, "spawn_failures": 0, "gap_observations": 0}


class GapPredictor:
    """Pluggable forecaster of inter-burst silence duration. `update()`
    is called ONLY from genuine external request arrivals (see
    `_note_real_arrival`) -- never from the prewarmer's own actions."""

    def __init__(self, kind):
        self.kind = kind
        self.gap_s = FIXED_PREDICTED_GAP_S if kind == "fixed" else None
        self._window = []

    def update(self, gap_s):
        if self.kind == "fixed":
            return
        elif self.kind == "lastgap":
            self.gap_s = gap_s
        elif self.kind == "movavg":
            self._window.append(gap_s)
            if len(self._window) > MOVAVG_WINDOW:
                self._window.pop(0)
            self.gap_s = sum(self._window) / len(self._window)
        else:  # ewma
            if self.gap_s is None:
                self.gap_s = gap_s
            else:
                self.gap_s = EWMA_ALPHA * gap_s + (1 - EWMA_ALPHA) * self.gap_s

    def predict(self):
        return self.gap_s


predictor = GapPredictor(PREDICTOR_KIND)
_last_real_arrival = {"t": None}


def _next_port():
    return PORT_BASE + next(_port_counter)


async def _wait_healthy(session, port, timeout_s):
    deadline = time.perf_counter() + timeout_s
    url = f"http://127.0.0.1:{port}/health"
    while time.perf_counter() < deadline:
        try:
            async with session.get(url, timeout=ClientTimeout(total=0.4)) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        await asyncio.sleep(0.02)
    return False


def _note_real_arrival():
    """Record a genuine external /invoke arrival and, if it follows a
    silence of at least GAP_MIN_S, feed that silence to the predictor
    as an observed inter-burst gap. This is the ONLY place gaps are
    recorded -- the prewarmer never touches this, so a speculative
    prewarm can never be mistaken for a real arrival."""
    now = time.time()
    prev = _last_real_arrival["t"]
    _last_real_arrival["t"] = now
    if prev is not None:
        gap = now - prev
        if gap >= GAP_MIN_S:
            predictor.update(gap)
            stats["gap_observations"] += 1


async def _reserve_slot():
    """Atomically either claim a free existing instance, or reserve a
    new placeholder slot (if under capacity), or report the pool is
    full. Never blocks on subprocess I/O while holding the lock."""
    async with pool_lock:
        for port, st in pool.items():
            if not st["busy"] and not st.get("pending"):
                st["busy"] = True
                return "existing", port
        if len(pool) < MAX_INSTANCES:
            port = _next_port()
            pool[port] = {"proc": None, "busy": True, "pending": True,
                          "last_used": time.time(), "started_at": time.time(),
                          "predictive_spawn": False, "served": 0}
            return "new", port
        return "full", None


async def _finalize_new_slot(session, port, predictive_spawn=False):
    """Spawn the subprocess for a reserved slot, outside the lock.
    On failure, removes the reservation and returns None."""
    env = os.environ.copy()
    env["INSTANCE_ID"] = f"elastic-{port}"
    t0 = time.perf_counter()
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "telemetry_app:app", "--host", "127.0.0.1",
         "--port", str(port), "--log-level", "warning"],
        cwd=APP_DIR, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    ok = await _wait_healthy(session, port, SPAWN_HEALTH_TIMEOUT_S)
    cold_ms = (time.perf_counter() - t0) * 1000.0
    async with pool_lock:
        if not ok:
            proc.kill()
            pool.pop(port, None)
            stats["spawn_failures"] += 1
            return None
        if port in pool:
            pool[port]["proc"] = proc
            pool[port]["pending"] = False
            pool[port]["predictive_spawn"] = predictive_spawn
    return cold_ms


async def _acquire_instance(session):
    """Returns (port, was_cold, cold_ms, queue_ms): queue_ms accumulates
    time spent retrying because the pool was at capacity (all instances
    busy), strictly excluding any cold-start spawn time."""
    queue_ms = 0.0
    while True:
        kind, port = await _reserve_slot()
        if kind == "existing":
            return port, False, 0.0, queue_ms
        if kind == "new":
            cold_ms = await _finalize_new_slot(session, port, predictive_spawn=False)
            if cold_ms is None:
                continue  # spawn failed; try again (maybe pool has room now)
            async with pool_lock:
                stats["cold_starts"] += 1
                stats["reactive_cold_starts"] += 1
            return port, True, cold_ms, queue_ms
        # kind == "full": wait briefly for a free instance, then retry reservation
        t_wait0 = time.perf_counter()
        await asyncio.sleep(QUEUE_POLL_S)
        queue_ms += (time.perf_counter() - t_wait0) * 1000.0


async def handle_invoke(request: web.Request):
    _note_real_arrival()
    session = request.app["session"]
    t0 = time.perf_counter()
    body = await request.read()
    try:
        port, was_cold, cold_ms, queue_ms = await asyncio.wait_for(
            _acquire_instance(session), timeout=REQUEST_WATCHDOG_S)
    except (asyncio.TimeoutError, Exception) as e:
        return web.json_response({"error": f"acquire_failed: {e}"}, status=503)
    t_svc0 = time.perf_counter()
    try:
        async with session.post(f"http://127.0.0.1:{port}/ingest", data=body,
                                 headers={"Content-Type": "application/json"},
                                 timeout=ClientTimeout(total=5)) as r:
            payload = await r.read()
            status = r.status
    except Exception as e:
        payload = str(e).encode()
        status = 502
    service_ms = (time.perf_counter() - t_svc0) * 1000.0
    async with pool_lock:
        if port in pool:
            pool[port]["busy"] = False
            pool[port]["last_used"] = time.time()
            pool[port]["served"] += 1
            pool[port]["predictive_spawn"] = False
        if not was_cold:
            stats["warm_hits"] += 1
    total_ms = (time.perf_counter() - t0) * 1000.0
    resp = web.Response(body=payload, status=status, content_type="application/json")
    resp.headers["X-Cold-Start"] = "1" if was_cold else "0"
    resp.headers["X-Queue-Ms"] = f"{queue_ms:.3f}"
    resp.headers["X-Cold-Ms"] = f"{cold_ms:.3f}"
    resp.headers["X-Service-Ms"] = f"{service_ms:.3f}"
    resp.headers["X-Gateway-Total-Ms"] = f"{total_ms:.3f}"
    return resp


async def handle_stats(request):
    async with pool_lock:
        return web.json_response({**stats, "pool_size": len(pool),
                                   "predicted_gap_s": predictor.predict(),
                                   "predictor": PREDICTOR_KIND, "predictive": PREDICTIVE})


async def reaper(app):
    while True:
        await asyncio.sleep(0.25)
        now = time.time()
        to_kill = []
        async with pool_lock:
            for p, st in list(pool.items()):
                if not st["busy"] and not st.get("pending") and now - st["last_used"] > IDLE_TIMEOUT_S:
                    to_kill.append((p, pool.pop(p)))
        for p, st in to_kill:
            if st["predictive_spawn"] and st["served"] == 0:
                stats["wasted_prewarms"] += 1
            if st["proc"] is not None:
                st["proc"].terminate()
                await asyncio.get_event_loop().run_in_executor(None, _safe_wait, st["proc"])


def _safe_wait(proc):
    try:
        proc.wait(timeout=2)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


async def predictive_prewarmer(app):
    if not PREDICTIVE:
        return
    session = app["session"]
    already_prewarmed_for_this_idle_period = False
    while True:
        await asyncio.sleep(0.1)
        async with pool_lock:
            empty = len(pool) == 0
        if not empty:
            already_prewarmed_for_this_idle_period = False
            continue
        if already_prewarmed_for_this_idle_period:
            continue
        predicted = predictor.predict()
        if predicted is None:
            continue
        # Reference-frame fix (bug found after the contamination-bug fix,
        # see CHANGELOG): idle time must be measured from the same origin
        # the forecast itself is calibrated against -- the last genuine
        # external arrival at /invoke (`_note_real_arrival`) -- NOT from
        # when the pool happened to become empty. The pool can sit empty
        # for up to IDLE_TIMEOUT_S seconds *after* the reaper retires the
        # last idle instance, which lags the true end of the previous
        # burst; measuring from `empty_since` therefore starts the clock
        # late and made the trigger (predicted - STARTUP_LEAD_S) almost
        # unreachable for any predictor whose forecast tracked the true
        # ~8-10s inter-burst gap (only the "fixed" predictor's flat,
        # always-available threshold ever fired). Measuring from the last
        # real arrival puts the prewarmer's clock in the same reference
        # frame as the predictor it is driven by.
        last_arrival = _last_real_arrival["t"]
        if last_arrival is None:
            continue
        idle_elapsed = time.time() - last_arrival
        if idle_elapsed < max(predicted - STARTUP_LEAD_S, 0.05):
            continue
        kind, port = await _reserve_slot()
        if kind != "new":
            continue
        cold_ms = await _finalize_new_slot(session, port, predictive_spawn=True)
        if cold_ms is not None:
            async with pool_lock:
                stats["cold_starts"] += 1
                stats["predictive_cold_starts"] += 1
                # this slot was reserved (busy=True) purely to perform the
                # spawn; it isn't serving a request, so release it back to
                # the free pool so a subsequent real request can claim it.
                # NOTE: this pool mutation is deliberately NOT observed by
                # any gap-recording logic (see _note_real_arrival) -- it
                # must never feed back into the predictor it was produced by.
                if port in pool:
                    pool[port]["busy"] = False
                    pool[port]["last_used"] = time.time()
        # Don't re-fire every 0.1s poll tick while still idle and still
        # empty (a real arrival, handled above via the `not empty` branch,
        # is what re-arms this) -- one prewarm attempt per idle period.
        already_prewarmed_for_this_idle_period = True


async def on_startup(app):
    app["session"] = ClientSession()
    app["tasks"] = [
        asyncio.create_task(reaper(app)),
        asyncio.create_task(predictive_prewarmer(app)),
    ]


async def on_cleanup(app):
    for t in app["tasks"]:
        t.cancel()
    await app["session"].close()
    async with pool_lock:
        procs = [st["proc"] for st in pool.values() if st["proc"] is not None]
        pool.clear()
    for p in procs:
        p.terminate()


def build_app():
    app = web.Application()
    app.router.add_post("/invoke", handle_invoke)
    app.router.add_get("/stats", handle_stats)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    return app


if __name__ == "__main__":
    port = int(os.environ.get("GATEWAY_PORT", "9700"))
    web.run_app(build_app(), host="127.0.0.1", port=port, print=None)
