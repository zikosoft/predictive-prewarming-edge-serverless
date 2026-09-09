"""
Analyzes the two Round-2 pilot sweeps (loadgen/orchestrate_round2_sweeps.py
output) once they finish:

1) Fixed-schedule sensitivity sweep (Round-2 feedback item 5): FIXED_PREDICTED_GAP_S
   at -50/-25/-10/+10/+25/+50% of the disclosed nominal 9.0s, N=6 independent
   trials per condition (pilot scale), plus the existing N=15 nominal (0%)
   run reused from the primary results as the center point.

2) Keep-alive (idle-timeout) sweep, non-oracle (Round-2 feedback item 4):
   IDLE_TIMEOUT_S in {8, 10, 12}s, N=6 per condition, plus the existing N=15
   reactive-baseline run (IDLE_TIMEOUT_S=6.0s, the primary run's own reactive
   baseline) reused as the 4th point on this sweep.

For every condition, per Round-2 item 5's exact requested metrics: median
latency, P95, P99, cold-start rate, missed/wasted pre-warms (fixed-schedule
sweep only -- meaningless for plain reactive), CPU-s/GB-s per trial.

Everything here is computed from genuinely-collected raw per-request CSVs
(results/raw_fixedsweep/*/, results/raw_keepalivesweep/*/, plus the primary
results/raw/ for the reused nominal points) -- no simulated or estimated
numbers.
"""
import glob
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")


def load_raw_trials(pattern):
    """Load all raw per-request CSVs matching a glob pattern into one DataFrame,
    tagging each row with its full source path and parent-directory condition
    label (basenames collide across condition subdirectories -- trial0.csv
    exists once per condition -- so the full path, not just the basename,
    must be used to recover which condition/trial a row belongs to)."""
    frames = []
    for path in sorted(glob.glob(pattern)):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        df["__source_file"] = path
        df["condition"] = os.path.basename(os.path.dirname(path))
        df["__trial_file"] = os.path.basename(path)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def per_trial_summary(df, group_cols):
    """Collapse per-request rows to one row per (condition, trial): median/P95/P99
    total latency, cold-start rate, CPU-s/GB-s if resource columns are present."""
    rows = []
    for keys, g in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        lat_col = "latency_ms" if "latency_ms" in g.columns else (
            "total_ms" if "total_ms" in g.columns else None)
        cold_col = "cold" if "cold" in g.columns else (
            "cold_start" if "cold_start" in g.columns else None)
        row = dict(zip(group_cols, keys))
        if lat_col:
            lat = pd.to_numeric(g[lat_col], errors="coerce").dropna()
            row["median_ms"] = float(lat.median())
            row["p95_ms"] = float(lat.quantile(0.95))
            row["p99_ms"] = float(lat.quantile(0.99))
        if cold_col:
            cold = pd.to_numeric(g[cold_col], errors="coerce").dropna()
            row["cold_start_rate"] = float(cold.mean())
        row["n_requests"] = len(g)
        rows.append(row)
    return pd.DataFrame(rows)


def integrate_cost(resource_csv_path, label_filter=None):
    """Riemann-integrate cpu_pct_sum/rss_mb_sum over time per (label) trial,
    same method as analysis/roi_model.py's integrate_seconds()."""
    if not os.path.exists(resource_csv_path):
        return pd.DataFrame()
    df = pd.read_csv(resource_csv_path)
    if df.empty:
        return pd.DataFrame()
    rows = []
    for label, g in df.groupby("label"):
        g = g.sort_values("t")
        t = g["t"].values
        cpu = g["cpu_pct_sum"].values / 100.0  # percent -> cores
        rss_gb = g["rss_mb_sum"].values / 1024.0
        if len(t) < 2:
            continue
        dt = np.diff(t)
        cpu_s = float(np.sum(dt * (cpu[:-1] + cpu[1:]) / 2.0))
        gb_s = float(np.sum(dt * (rss_gb[:-1] + rss_gb[1:]) / 2.0))
        rows.append({"label": label, "cpu_s": cpu_s, "gb_s": gb_s})
    return pd.DataFrame(rows)


def main():
    print("=" * 70)
    print("FIXED-SCHEDULE SENSITIVITY SWEEP")
    print("=" * 70)
    fixed_raw = load_raw_trials(os.path.join(RESULTS, "raw_fixedsweep", "*", "raw_serverless_predictive_fixed_cyclical_trial*.csv"))
    if fixed_raw.empty:
        print("[no raw_fixedsweep data found yet]")
    else:
        group_cols = ["condition", "__source_file"]
        summ = per_trial_summary(fixed_raw, group_cols)
        agg = summ.groupby("condition").agg(
            n_trials=("median_ms", "count"),
            mean_of_median_ms=("median_ms", "mean"),
            mean_p95_ms=("p95_ms", "mean"),
            mean_p99_ms=("p99_ms", "mean"),
            mean_cold_start_rate=("cold_start_rate", "mean"),
        ).reset_index()
        print(agg.to_string(index=False))
        agg.to_csv(os.path.join(RESULTS, "v4d_fixedsweep_summary.csv"), index=False)

        gw = os.path.join(RESULTS, "fixedsweep_gateway_stats.csv")
        if os.path.exists(gw):
            gws = pd.read_csv(gw)
            gws["label"] = gws["sweep_pct"].astype(str) + "pct"
            wp = gws.groupby("sweep_pct").agg(
                mean_wasted_prewarms=("wasted_prewarms", "mean"),
                mean_predictive_cold_starts=("predictive_cold_starts", "mean"),
                mean_reactive_cold_starts=("reactive_cold_starts", "mean"),
                fixed_predicted_gap_s=("fixed_predicted_gap_s", "first"),
            ).reset_index()
            print("\n-- wasted/missed prewarms by condition --")
            print(wp.to_string(index=False))
            wp.to_csv(os.path.join(RESULTS, "v4d_fixedsweep_prewarm_stats.csv"), index=False)

        res = os.path.join(RESULTS, "fixedsweep_resource_samples.csv")
        cost = integrate_cost(res)
        if not cost.empty:
            print("\n-- per-trial CPU-s/GB-s (raw labels) --")
            print(cost.to_string(index=False))
            cost.to_csv(os.path.join(RESULTS, "v4d_fixedsweep_cost.csv"), index=False)

    print()
    print("=" * 70)
    print("KEEP-ALIVE (IDLE-TIMEOUT) SWEEP")
    print("=" * 70)
    ka_raw = load_raw_trials(os.path.join(RESULTS, "raw_keepalivesweep", "*", "raw_serverless_reactive_cyclical_trial*.csv"))
    if ka_raw.empty:
        print("[no raw_keepalivesweep data found yet]")
    else:
        group_cols = ["condition", "__source_file"]
        summ = per_trial_summary(ka_raw, group_cols)
        agg = summ.groupby("condition").agg(
            n_trials=("median_ms", "count"),
            mean_of_median_ms=("median_ms", "mean"),
            mean_p95_ms=("p95_ms", "mean"),
            mean_p99_ms=("p99_ms", "mean"),
            mean_cold_start_rate=("cold_start_rate", "mean"),
        ).reset_index()
        print(agg.to_string(index=False))
        agg.to_csv(os.path.join(RESULTS, "v4d_keepalivesweep_summary.csv"), index=False)

        res = os.path.join(RESULTS, "keepalivesweep_resource_samples.csv")
        cost = integrate_cost(res)
        if not cost.empty:
            print("\n-- per-trial CPU-s/GB-s (raw labels) --")
            print(cost.to_string(index=False))
            cost.to_csv(os.path.join(RESULTS, "v4d_keepalivesweep_cost.csv"), index=False)

    print()
    print("=" * 70)
    print("NOMINAL/REUSED POINTS FROM PRIMARY N=15 RUN (for reference)")
    print("=" * 70)
    primary = load_raw_trials(os.path.join(RESULTS, "raw", "raw_serverless_predictive_fixed_cyclical_trial*.csv"))
    if not primary.empty:
        summ = per_trial_summary(primary, ["__source_file"])
        print("fixed-schedule nominal (9.0s), N=%d trials: median(mean)=%.2f ms, P95(mean)=%.2f, P99(mean)=%.2f, cold_rate(mean)=%.4f" % (
            len(summ), summ["median_ms"].mean(), summ["p95_ms"].mean(), summ["p99_ms"].mean(), summ["cold_start_rate"].mean()))
    primary_r = load_raw_trials(os.path.join(RESULTS, "raw", "raw_serverless_reactive_cyclical_trial*.csv"))
    if not primary_r.empty:
        summ = per_trial_summary(primary_r, ["__source_file"])
        print("reactive/keep-alive nominal (6.0s), N=%d trials: median(mean)=%.2f ms, P95(mean)=%.2f, P99(mean)=%.2f, cold_rate(mean)=%.4f" % (
            len(summ), summ["median_ms"].mean(), summ["p95_ms"].mean(), summ["p99_ms"].mean(), summ["cold_start_rate"].mean()))


if __name__ == "__main__":
    main()
