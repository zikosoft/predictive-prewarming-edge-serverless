"""
Minimal real-time telemetry ingestion service.

Represents the application under test (AUT) used identically in both
deployment architectures (container-based and serverless-emulated),
so that the deployment model is the ONLY variable being measured.

Endpoint: POST /ingest  -> accepts a small JSON telemetry event,
performs light validation + a bounded CPU-bound aggregation step
(rolling stats), and returns an ack. This mirrors the "telemetry
ingestion" workload described in the manuscript (metrics/logs/traces
event processing).

Also exposes GET /health for readiness / warm-pool checks.
"""
import hashlib
import os
import time
from collections import deque

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

START_TIME = time.time()
INSTANCE_ID = os.environ.get("INSTANCE_ID", "unknown")
WINDOW = deque(maxlen=256)


def _cpu_bound_aggregate(payload: dict) -> dict:
    """Bounded CPU work to emulate real telemetry processing
    (parsing + hashing + rolling aggregation), independent of
    architecture so the comparison isolates deployment overhead."""
    blob = str(payload).encode("utf-8")
    digest = hashlib.sha256(blob).hexdigest()
    value = payload.get("value", 0)
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = 0.0
    WINDOW.append(value)
    rolling_avg = sum(WINDOW) / len(WINDOW) if WINDOW else 0.0
    # small deterministic extra workload to avoid trivial sub-ms responses
    acc = 0
    for i in range(2000):
        acc += (i * 2654435761) & 0xFFFFFFFF
    return {"digest": digest[:16], "rolling_avg": rolling_avg, "acc_check": acc % 997}


@app.get("/health")
async def health():
    return {"status": "ok", "instance": INSTANCE_ID, "uptime_s": round(time.time() - START_TIME, 3)}


@app.post("/ingest")
async def ingest(request: Request):
    t0 = time.perf_counter()
    payload = await request.json()
    result = _cpu_bound_aggregate(payload)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    return JSONResponse(
        {
            "status": "accepted",
            "instance": INSTANCE_ID,
            "processing_ms": round(dt_ms, 3),
            "server_uptime_s": round(time.time() - START_TIME, 3),
            **result,
        }
    )
