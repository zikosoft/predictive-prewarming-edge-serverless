# Real-infrastructure validation track

The main harness (`loadgen/orchestrate_and_run.py`) realizes all architectures as OS-level processes on one host, disclosed throughout the manuscript as a deliberate substitution for real containers/FaaS (the sandboxed environment it was built in has no container-registry access). This directory is the answer to the natural follow-up question an external review raised: **does the finding still hold on a real container platform and a real serverless platform?**

It deploys the exact same application code (`app/telemetry_app.py`) to:
- a real Kubernetes `Deployment` (3 always-on pods) for the **container** architecture, and
- a real **Knative Serving** `Service` for the **serverless (reactive)** architecture — Knative's own activator and autoscaler handle cold start and scale-to-zero natively, no custom code involved.

For the **predictive** policy, since Knative has no built-in pluggable forecaster, `loadgen_real.py` runs a small `PredictiveController` in the same process as the load generator: it observes real request timestamps (and *only* real ones — see the contamination bug this project already hit once, documented in `../CHANGELOG.md`) and fires a lightweight priming `GET /health` at the Knative Service ahead of a forecast burst, so Knative cold-starts a pod before the real traffic arrives. This is a legitimate, commonly used pattern for operator-side Knative pre-warming (no cluster RBAC or control-plane access needed — just an HTTP client).

## What you need

- Windows with **Docker Desktop**, Kubernetes enabled (Settings → Kubernetes → "Enable Kubernetes"). Confirmed target: Ryzen 9 5900X / 32GB RAM is comfortably enough for this workload (3 low-resource pods + Knative's own control plane).
- A shell to run bash scripts in: **WSL2** (recommended — open your WSL2 Ubuntu terminal) or **Git Bash**. All commands below assume one of these.
- `kubectl` (ships with Docker Desktop — check with `kubectl version --client`).
- Python 3.10+ with `pip install aiohttp` (nothing else — no Kubernetes client library is needed, everything talks plain HTTP).
- Internet access for the one-time Knative install (pulls YAML from GitHub and images from `gcr.io`/`ghcr.io` — Knative's own images, not the benchmark's) and for `*.sslip.io` DNS resolution (a public wildcard-DNS service Knative uses for automatic local URLs — no account, no config, just needs outbound DNS to work normally).

## Setup (one time, ~10-15 minutes)

```bash
cd edge-faas-container-benchmark   # this repo, wherever you unzipped/cloned it

# 1. Point kubectl at Docker Desktop's cluster
kubectl config use-context docker-desktop
kubectl get nodes                              # sanity check: should show one Ready node

# 2. Install Knative Serving + Kourier + magic DNS (asks for confirmation)
bash real-infra/scripts/01_install_knative.sh

# 3. Build the application image and deploy both architectures
bash real-infra/scripts/02_build_and_deploy.sh
```

The last script prints two URLs — verify both respond before continuing:

```bash
curl -s http://localhost:8080/health                                    # container
curl -s http://telemetry-serverless.default.127.0.0.1.sslip.io/health   # Knative (URL printed by the script; copy it exactly)
```

Both should return a small JSON blob like `{"status":"ok","instance":"...","uptime_s":...}`.

## Running the experiment

```bash
pip install aiohttp
python3 real-infra/loadgen_real.py
```

Defaults: 5 independent trials for `low`/`moderate`, 8 independent trials of 5 cycles each for `cyclical` (fewer than the process-level emulation's 15 — real pod scheduling is slower than spawning a local subprocess, so this keeps total runtime reasonable; raise `N_TRIALS_CYCLICAL` as an environment variable if you have time, e.g. `N_TRIALS_CYCLICAL=15 python3 real-infra/loadgen_real.py`). Expect on the order of **45-90 minutes** total, mostly the cyclical scenario's repeated 8-10s idle waits — this is normal, not a hang; it prints a progress line before every trial.

Then analyze:

```bash
python3 real-infra/analyze_real.py
```

This writes `results/real/summary_table_real.csv`, `trial_level_table_real.csv`, `omnibus_posthoc_real.csv`, and — if you've also run the process-level emulation (`analysis/analyze.py`) in this same repo — `emulation_vs_real_comparison.csv`, a direct side-by-side table of whether the two tracks agree.

## Sending results back

Everything needed to fold real-infra numbers into the manuscript is in `results/real/*.csv` — send that directory back (zip it) so it can be merged into the analysis and the manuscript's Results/Threats-to-Validity sections rewritten from genuine measurements instead of (or alongside) the process-level emulation.

## Troubleshooting

- **`kubectl wait` times out on Knative deployments**: `kubectl get pods -n knative-serving` and `kubectl describe pod <name> -n knative-serving` for the reason — usually a slow image pull on first install; re-run the wait command, no need to redo the whole script.
- **Knative curl hangs / DNS doesn't resolve `*.sslip.io`**: some corporate/VPN networks block wildcard public DNS. Check with `nslookup telemetry-serverless.default.127.0.0.1.sslip.io` — it should resolve to `127.0.0.1`. If it's blocked, add a manual hosts-file entry instead: `127.0.0.1  telemetry-serverless.default.127.0.0.1.sslip.io` in `C:\Windows\System32\drivers\etc\hosts` (needs an admin editor), or ask and a `config-domain` alternative (e.g. `example.com` + manual `/etc/hosts`) can be worked out.
- **Container Service never gets a LoadBalancer IP**: Docker Desktop usually assigns `localhost` within seconds; if `kubectl get svc telemetry-container-svc` shows `<pending>` after a minute, check Docker Desktop's Kubernetes status in its own UI (Settings → Kubernetes) — a restart of Docker Desktop's Kubernetes cluster usually fixes this.
- **`docker build` succeeds but pods show `ErrImageNeverPull` or similar**: confirm you're on the `docker-desktop` kubectl context (not some other cluster) — Docker Desktop's Kubernetes shares the Docker daemon's image store only with itself.
- **Something else entirely**: copy the exact error and the output of `kubectl get pods -A`, `kubectl get ksvc`, `kubectl get svc` and send it back — this whole track was written without access to a real Kubernetes cluster to test against, so a first-run hiccup here is expected and worth debugging together rather than working around blindly.
