"""
Gateway for the "container" (always-on, persistent) architecture.

Added after external review flagged a measurement asymmetry: in the
previous version, the container arch's load was sent directly from the
load generator to each replica's port (zero gateway hops), while both
serverless arches were sent through elastic_gateway.py (one extra local
HTTP hop, plus asyncio-lock bookkeeping). That extra hop is itself a
few hundred microseconds to low milliseconds of overhead on loopback --
small, but systematic, and present in every scenario (not just the
cyclical one), so it inflated the apparent gap between "always-on" and
"on-demand" deployments independently of the actual deployment-model
difference the paper is about.

This gateway removes that confound: it fronts N persistent replicas
(started once at process launch, never scaled, never cold-started)
behind the exact same one-hop proxy shape as elastic_gateway.py,
round-robining across them, and emits the identical response header
set (X-Cold-Start, X-Queue-Ms, X-Cold-Ms, X-Service-Ms,
X-Gateway-Total-Ms) so the same analysis code applies uniformly. Queue
time and cold-start time are structurally always zero here -- that is
the point: the "always-on" architecture pays neither by construction,
and now that's demonstrated through the same instrumentation path
rather than asserted by a different (shorter) code path.
"""
import itertools
import os
import sys

from aiohttp import web, ClientSession, ClientTimeout
import time

N_REPLICAS = int(os.environ.get("N_CONTAINER_REPLICAS", "3"))
REPLICA_PORTS = [int(os.environ.get("PORT_BASE", "9001")) + i for i in range(N_REPLICAS)]
_rr = itertools.cycle(REPLICA_PORTS)


async def handle_invoke(request: web.Request):
    session = request.app["session"]
    t0 = time.perf_counter()
    body = await request.read()
    port = next(_rr)
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
    total_ms = (time.perf_counter() - t0) * 1000.0
    resp = web.Response(body=payload, status=status, content_type="application/json")
    resp.headers["X-Cold-Start"] = "0"
    resp.headers["X-Queue-Ms"] = "0.000"
    resp.headers["X-Cold-Ms"] = "0.000"
    resp.headers["X-Service-Ms"] = f"{service_ms:.3f}"
    resp.headers["X-Gateway-Total-Ms"] = f"{total_ms:.3f}"
    return resp


async def handle_stats(request):
    return web.json_response({"cold_starts": 0, "predictive_cold_starts": 0,
                               "reactive_cold_starts": 0, "warm_hits": 0,
                               "wasted_prewarms": 0, "spawn_failures": 0,
                               "gap_observations": 0, "pool_size": N_REPLICAS,
                               "predicted_gap_s": None, "predictor": "n/a",
                               "predictive": False})


async def on_startup(app):
    app["session"] = ClientSession()


async def on_cleanup(app):
    await app["session"].close()


def build_app():
    app = web.Application()
    app.router.add_post("/invoke", handle_invoke)
    app.router.add_get("/stats", handle_stats)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    return app


if __name__ == "__main__":
    port = int(os.environ.get("GATEWAY_PORT", "9800"))
    web.run_app(build_app(), host="127.0.0.1", port=port, print=None)
