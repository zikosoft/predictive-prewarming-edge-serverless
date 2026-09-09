"""
Cost/latency break-even model (v2, corrected after external review).

Two fixes relative to the previous version:
  1. The "low" scenario is NOT idle -- it is 5 req/s of CONTINUOUS
     traffic (see loadgen/orchestrate_and_run.py: SCENARIOS_SHORT).
     Calling it "idle/baseline operation" was a mislabeling flagged in
     review. There is no genuinely idle scenario in this workload suite
     by design (the point of "low" is a no-idle-gaps control). The
     resource comparison below is now honestly labeled as "steady
     light load" (container vs. reactive), not "idle".
  2. Costs are now integrated as CPU-seconds and GB-seconds (Riemann
     sum over the 5 Hz resource samples) rather than averaged
     instantaneous percentages, which is what an actual cloud billing
     model charges for. Wasted prewarms (an instance spawned
     speculatively that never serves a request before idling out) are
     counted and costed explicitly.

Outputs: results/roi_table.csv (SLA violation rates), results/
cost_seconds_table.csv (CPU-seconds / GB-seconds per policy),
results/roi_breakeven.json (decision rule + a real cost-latency Pareto
frontier across the cyclical policies, still expressed in measured
units with no invented dollar figures -- the reader plugs in their own
cost-per-unit).
"""
import glob
import json
import os
import re

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "results", "raw")
OUT_DIR = os.path.join(ROOT, "results")

CYCLICAL_POLICIES = ["container", "serverless_reactive", "serverless_predictive_ewma",
                      "serverless_predictive_lastgap", "serverless_predictive_movavg",
                      "serverless_predictive_fixed"]
LABEL = {
    "container": "Container", "serverless_reactive": "Serverless (reactive)",
    "serverless_predictive_ewma": "Serverless (predictive, EWMA)",
    "serverless_predictive_lastgap": "Serverless (predictive, last-gap)",
    "serverless_predictive_movavg": "Serverless (predictive, moving avg.)",
    "serverless_predictive_fixed": "Serverless (predictive, fixed-schedule)",
}
_POLICY_ALT = "|".join(sorted(LABEL.keys(), key=len, reverse=True))
RAW_RE = re.compile(rf"raw_({_POLICY_ALT})_(low|moderate|cyclical)_trial(\d+)\.csv")
THRESHOLDS_MS = [50, 100, 200]


def load_all():
    frames = []
    for path in glob.glob(os.path.join(RAW_DIR, "raw_*.csv")):
        m = RAW_RE.match(os.path.basename(path))
        if not m:
            continue
        policy, scenario, trial = m.group(1), m.group(2), int(m.group(3))
        df = pd.read_csv(path)
        df["policy"], df["scenario"], df["trial"] = policy, scenario, trial
        frames.append(df)
    full = pd.concat(frames, ignore_index=True)
    full["ok"] = full["ok"].astype(str).str.lower().isin(["true", "1"])
    return full


def integrate_seconds(res_samples, policy, scenario):
    """Riemann-sum the 5 Hz resource samples into CPU-seconds and
    GB-seconds over the wall-clock duration actually observed, per
    trial, then average across trials -- this is what a metered biller
    would charge, unlike a plain mean of instantaneous percentages."""
    parts = res_samples["label"].str.split(":", n=2, expand=True)
    res_samples = res_samples.assign(policy=parts[0], scenario=parts[1], trial=parts[2])
    sub = res_samples[(res_samples.policy == policy) & (res_samples.scenario == scenario)]
    if sub.empty:
        return None, None
    cpu_s_per_trial, gb_s_per_trial = [], []
    for _trial, g in sub.groupby("trial"):
        g = g.sort_values("t")
        if len(g) < 2:
            continue
        dt = np.diff(g["t"].values)
        cpu_s = float(np.sum(dt * g["cpu_pct_sum"].values[:-1] / 100.0))
        gb_s = float(np.sum(dt * (g["rss_mb_sum"].values[:-1] / 1024.0)))
        cpu_s_per_trial.append(cpu_s)
        gb_s_per_trial.append(gb_s)
    if not cpu_s_per_trial:
        return None, None
    return float(np.mean(cpu_s_per_trial)), float(np.mean(gb_s_per_trial))


def main():
    df = load_all()
    res_samples = pd.read_csv(os.path.join(OUT_DIR, "resource_samples.csv"))
    gw = pd.read_csv(os.path.join(OUT_DIR, "gateway_stats.csv"))

    # 1. SLA violation rates (cyclical, all 6 policies)
    rows = []
    for policy in CYCLICAL_POLICIES:
        sub = df[(df.scenario == "cyclical") & (df.policy == policy) & (df.ok)]
        n = len(sub)
        row = {"policy": LABEL[policy], "n_requests_cyclical": n}
        for th in THRESHOLDS_MS:
            viol = (sub["latency_ms"] > th).sum()
            row[f"violation_rate_pct_gt_{th}ms"] = round(100 * viol / n, 2) if n else float("nan")
        rows.append(row)
    viol_table = pd.DataFrame(rows)
    viol_table.to_csv(os.path.join(OUT_DIR, "roi_table.csv"), index=False)

    # 2. Cost in CPU-seconds / GB-seconds per policy (cyclical), plus
    #    wasted-prewarm counts (spawned speculatively, never served a request)
    cost_rows = []
    for policy in CYCLICAL_POLICIES:
        cpu_s, gb_s = integrate_seconds(res_samples, policy, "cyclical")
        if cpu_s is None:
            continue
        gwp = gw[(gw.policy == policy) & (gw.scenario == "cyclical")]
        mean_wasted = float(gwp["wasted_prewarms"].mean()) if not gwp.empty and gwp["wasted_prewarms"].notna().any() else 0.0
        mean_predictive_cold = float(gwp["predictive_cold_starts"].mean()) if not gwp.empty and gwp["predictive_cold_starts"].notna().any() else 0.0
        cost_rows.append({"policy": LABEL[policy], "mean_cpu_seconds_per_trial": round(cpu_s, 3),
                          "mean_gb_seconds_per_trial": round(gb_s, 3),
                          "mean_wasted_prewarms_per_trial": round(mean_wasted, 2),
                          "mean_predictive_cold_starts_per_trial": round(mean_predictive_cold, 2)})
    cost_table = pd.DataFrame(cost_rows)
    cost_table.to_csv(os.path.join(OUT_DIR, "cost_seconds_table.csv"), index=False)

    # 3. Steady light-load resource comparison (container vs. reactive) --
    #    explicitly NOT labeled "idle": "low" is 5 req/s continuous traffic.
    steady_container_cpu, steady_container_gb = integrate_seconds(res_samples, "container", "low")
    steady_reactive_cpu, steady_reactive_gb = integrate_seconds(res_samples, "serverless_reactive", "low")

    # 4. Break-even decision rule + Pareto frontier (measured units only)
    v100 = {row["policy"]: row["violation_rate_pct_gt_100ms"] for row in rows}
    reactive_v100 = v100[LABEL["serverless_reactive"]]
    pareto = []
    for policy in CYCLICAL_POLICIES:
        c = next((r for r in cost_rows if r["policy"] == LABEL[policy]), None)
        if c is None:
            continue
        pareto.append({
            "policy": LABEL[policy],
            "mean_cpu_seconds_per_trial": c["mean_cpu_seconds_per_trial"],
            "violation_rate_pct_gt_100ms": v100[LABEL[policy]],
            "violation_rate_reduction_vs_reactive_pp": round(reactive_v100 - v100[LABEL[policy]], 2),
        })

    breakeven = {
        "steady_light_load_note": "The 'low' scenario is 5 req/s of continuous traffic, NOT an idle baseline -- there is no genuinely idle scenario in this workload suite by design.",
        "steady_light_load_container_cpu_seconds_per_trial": round(steady_container_cpu, 3) if steady_container_cpu else None,
        "steady_light_load_reactive_cpu_seconds_per_trial": round(steady_reactive_cpu, 3) if steady_reactive_cpu else None,
        "steady_light_load_container_gb_seconds_per_trial": round(steady_container_gb, 3) if steady_container_gb else None,
        "steady_light_load_reactive_gb_seconds_per_trial": round(steady_reactive_gb, 3) if steady_reactive_gb else None,
        "cyclical_pareto_frontier_cpu_seconds_vs_sla_violation": pareto,
        "decision_rule": (
            "Adopt a predictive pre-warming policy over the purely reactive one whenever: "
            "(cost_per_SLA_violation / cost_per_CPU_second) > "
            "(policy_extra_CPU_seconds_per_trial) / (violation_rate_reduction_pp/100 * requests_per_trial). "
            "Plug in your own cost-per-violation and cost-per-CPU-second; the measured numerator/denominator "
            "for each policy are in the Pareto frontier above and in cost_seconds_table.csv / roi_table.csv."
        ),
    }
    with open(os.path.join(OUT_DIR, "roi_breakeven.json"), "w") as f:
        json.dump(breakeven, f, indent=2)

    print(viol_table.to_string(index=False))
    print(cost_table.to_string(index=False))
    print(json.dumps(breakeven, indent=2))


if __name__ == "__main__":
    main()
