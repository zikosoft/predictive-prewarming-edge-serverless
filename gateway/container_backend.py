"""
Launcher for a single "container-based" (persistent) backend instance.
Used by orchestrate.py to start N long-lived replicas that stay warm
for the entire experiment (mirrors a Kubernetes Deployment replica /
Docker container that is never scaled to zero).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "9001"))
    os.environ["INSTANCE_ID"] = f"container-{port}"
    uvicorn.run("telemetry_app:app", host="127.0.0.1", port=port, log_level="warning")
