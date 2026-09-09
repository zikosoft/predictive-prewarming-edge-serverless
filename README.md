# predictive-prewarming-edge-serverless

Open, fully reproducible benchmarking harness comparing an **always-on container** deployment against **on-demand (scale-to-zero) serverless-style** deployments — a purely reactive baseline plus four prewarming predictor policies (EWMA, last-gap, moving-average, fixed-schedule) — of an identical real-time telemetry-ingestion service. Companion code and data for a manuscript in preparation (not yet submitted).

Every number, table, and figure in the paper is generated from the raw CSVs in `results/raw/` and `results/raw_learning/` by `analysis/analyze.py` and `analysis/roi_model.py` — nothing is hand-typed. You can re-run the whole experiment and regenerate every result from scratch.

## What changed after external review

An external methodological review of an earlier draft identified five issues, all fixed in this version — see `CHANGELOG.md` for the full account of each:

1. The predictive pre-warming policy's own speculative pre-warms were being mis-recorded as real traffic arrivals, contaminating its own forecast (`gateway/elastic_gateway.py`, `_note_real_arrival`).
2. The container architecture bypassed the gateway entirely (direct-to-worker), while the serverless architectures were measured through an extra network hop — an asymmetric measurement path (`gateway/container_gateway.py` now unifies this).
3. Statistical tests were run on individual, correlated requests rather than independent trials (`analysis/analyze.py` now uses the trial as the unit of analysis, with matched-seed paired testing).
4. Only the EWMA predictor was evaluated, with no baseline ablation (`analysis/analyze.py` now compares EWMA against last-gap, moving-average, and fixed-schedule policies on identical, seed-matched traffic).
5. The repository and manuscript had drifted out of sync (this README, in particular, described a since-deleted file and an old two-architecture design).
6. **Found only after fixing #1**, and not part of the original external review: with the forecast no longer contaminated, `results/gateway_stats.csv` showed the prewarm trigger essentially never firing for the EWMA/last-gap/moving-average policies, despite an accurate forecast. Root cause: the trigger measured idle time from the wrong reference point (pool-became-empty, not the last genuine request) — fixed in `gateway/elastic_gateway.py`'s `predictive_prewarmer`. See `CHANGELOG.md`, "v3.1", for the full account, including why this bug was masked by bug #1 in the original (v2) draft.

## Why process-level emulation instead of real Docker/Knative?

The results in `results/` were produced in a sandboxed environment with no outbound access to container registries. To keep the deployment-model comparison meaningful without that access, all architectures are realized as **OS-level process instances** running the *exact same application code* (`app/telemetry_app.py`):

- **Container**: `N_CONTAINER_REPLICAS` persistent worker processes, started once, never torn down, fronted by `gateway/container_gateway.py`.
- **Serverless (reactive / predictive)**: `gateway/elastic_gateway.py`, an elastic pool that spawns a fresh worker on demand, health-checks it, proxies the request, and kills it after an idle timeout (reactive), optionally pre-warming ahead of a forecast burst (predictive).

This is disclosed in full in the manuscript's Methods / Threats to Validity sections. **A parallel, real-infrastructure validation track lives in `real-infra/`** — Kubernetes + Knative manifests and a matching orchestration script designed to run on a machine with Docker Desktop (tested target: Windows, Docker Desktop with Kubernetes enabled), producing the identical CSV schema so its results plug directly into the same `analyze.py`. See `real-infra/README.md`.

## Repository layout

```
app/            the application under test (identical across all policies)
gateway/        container_gateway.py, elastic_gateway.py (reactive + 4 predictor policies)
loadgen/        orchestrate_and_run.py (primary run), orchestrate_capacity_sweep.py (supplementary
                MAX_INSTANCES=8 sweep), rerun_predictive_fix.py (targeted-rerun template)
analysis/       analyze.py (statistics + figures), roi_model.py (cost model),
                analyze_capacity_sweep.py (supplementary sweep analysis)
results/        raw per-request CSVs, resource samples, summary tables, figures (all regenerable);
                raw_capacity8/ holds the supplementary MAX_INSTANCES=8 sweep's raw data
real-infra/     Kubernetes/Knative manifests + scripts for real-container/real-FaaS validation
paper/          the manuscript (build_docx.js regenerates it from the results/ artifacts)
k8s/            Kubernetes / Knative manifests (also used by real-infra/)
docker-compose.yml, app/Dockerfile   real-container reproduction of the persistent architecture
```

## Reproducing the process-level-emulation results

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Run the full experiment (container + reactive + 4 predictor policies,
#    3 scenarios, independent trials per policy, writes results/raw/*.csv
#    and results/raw_learning/*.csv)
python3 loadgen/orchestrate_and_run.py

# 2. Compute statistics and regenerate every table/figure
python3 analysis/analyze.py
python3 analysis/roi_model.py
```

Total run time: on the order of 1.5-2 hours on 2+ vCPUs (dominated by the cyclical scenario's independent-trial protocol — see below). No external network access is required — everything runs on localhost.

## Experimental design (summary)

- **low** / **moderate**: no idle gaps by design (control scenarios), 5 independent trials per policy, 3 policies (container, reactive, predictive-EWMA).
- **cyclical**: 5 repeated (idle 8-10s jittered, then 15 concurrent requests) cycles per trial, **15 independent trials per policy**, 6 policies (container, reactive, and 4 predictor ablations). Every trial's backend is restarted fresh (no state carried over), and trial `N` uses an identical RNG seed across all 6 policies — a matched/paired design, not just an independent one.
- **learning-curve run** (`results/raw_learning/`): a separate, dedicated run where the gateway is kept alive across 25 consecutive cycles (state persists, no resets) to show how the predictor's hit rate evolves with repeated exposure. Analyzed descriptively only — never mixed into the significance tests above.

Metrics: end-to-end latency (P50/P95/P99, per-request, client-timed) decomposed into queue / cold-start / service time (from gateway response headers), cold-start rate, error rate, aggregate CPU% and RSS (5 Hz `psutil` sampling), integrated into CPU-seconds / GB-seconds for the cost model. Statistical comparison: Kruskal-Wallis omnibus + pairwise Mann-Whitney with Holm-Bonferroni correction, plus a paired Wilcoxon signed-rank test exploiting the matched-seed design — all computed on trial-level medians, not individual requests.

## Supplementary: capacity-sensitivity sweep

`loadgen/orchestrate_capacity_sweep.py` re-runs the cyclical scenario's 6 policies (15 trials each) with `MAX_INSTANCES=8` instead of the primary run's 3, to test whether predictive pre-warming's benefit depends on capacity being the bottleneck (see the paper's Section 9). Output goes to `results/raw_capacity8/`, analyzed by `analysis/analyze_capacity_sweep.py`. **Headline finding, reported in full in the paper rather than omitted:** on this benchmark's 2-vCPU host, raising `MAX_INSTANCES` did not relieve the bottleneck — it moved it from software-level queueing to host-level CPU contention among more simultaneous cold-start spawns, so this specific experiment is confounded by hardware and should be re-run on genuine multi-core infrastructure (`real-infra/`) before drawing a capacity-sensitivity conclusion.

```bash
MAX_INSTANCES=8 python3 loadgen/orchestrate_capacity_sweep.py
python3 analysis/analyze_capacity_sweep.py
```

If a predictive policy's earlier data ever needs re-generating in isolation (e.g. after a gateway fix that only affects the predictive policies), `loadgen/rerun_predictive_fix.py` is a template for a targeted, scope-limited re-run that merges into the existing `results/*.csv` rather than re-running everything — see its docstring for exactly which slice of data it touches and why.

## Data availability & citation

Raw data and code: this repository. See `CITATION.cff` for citation metadata. An archived, versioned release with a citable DOI will be minted via Zenodo once the corrected protocol above (and, if available, the `real-infra/` validation) is finalized — **deliberately not yet archived**, since a DOI is permanent and the previous protocol had a confirmed measurement bug (see "What changed after external review").

## License

MIT — see `LICENSE`.

## Release checklist (for the maintainer)

1. Run `real-infra/` on genuine Kubernetes/Knative infrastructure (or explicitly decide to submit on process-level-emulation results alone, clearly labeled as such, which is the current state of the manuscript).
2. Manuscript rebuilt from the corrected results (`paper/build_docx.js`) — done; re-run this step if `results/` changes again.
3. `git tag v1.0.0 && git push --tags`.
4. Connect the repo to [Zenodo](https://zenodo.org/account/settings/github/) and create a DOI for the tagged release (`.zenodo.json` in this repo pre-fills the archive's metadata).
5. Add the DOI badge to the top of this README and to the manuscript's Data Availability statement.
6. Optionally submit a short companion "software" paper (e.g. to JOSS) describing this harness itself as a citable, reusable artifact.
