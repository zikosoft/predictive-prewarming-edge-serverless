"""
Analysis for the supplementary MAX_INSTANCES=8 capacity-sweep
(loadgen/orchestrate_capacity_sweep.py). Answers: does predictive
pre-warming's benefit persist, shrink, or grow once burst concurrency
(15) is no longer far above instance capacity (raised from 3 to 8)?

Mirrors analysis/analyze.py's trial-as-unit-of-analysis approach on
this smaller (cyclical-only, one capacity setting) run, and additionally
reports a side-by-side comparison against the primary MAX_INSTANCES=3
results already analyzed in results/*.csv.

Run after loadgen/orchestrate_capacity_sweep.py has produced
results/raw_capacity8/*.csv and results/gateway_stats_capacity8.csv.

Outputs (results/):
  summary_table_capacity8.csv
  trial_level_table_capacity8.csv
  omnibus_posthoc_capacity8.csv
  decomposition_table_capacity8.csv
  capacity3_vs_capacity8_comparison.csv
  fig_decomposition_capacity8.png/.pdf
"""
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
RAW_DIR = os.path.join(RESULTS, "raw_capacity8")
CAPACITY = 8

POLICIES = ["container", "serverless_reactive", "serverless_predictive_ewma",
            "serverless_predictive_lastgap", "serverless_predictive_movavg",
            "serverless_predictive_fixed"]
LABEL = {"container": "Container", "serverless_reactive": "Serverless (reactive)",
         "serverless_predictive_ewma": "Serverless (predictive, EWMA)",
         "serverless_predictive_lastgap": "Serverless (predictive, last-gap)",
         "serverless_predictive_movavg": "Serverless (predictive, moving avg.)",
         "serverless_predictive_fixed": "Serverless (predictive, fixed-schedule)"}
RAW_RE = re.compile(r"raw_(container|serverless_reactive|serverless_predictive_ewma|"
                     r"serverless_predictive_lastgap|serverless_predictive_movavg|"
                     r"serverless_predictive_fixed)_cyclical_trial(\d+)\.csv")


def _hdr_or_col(df, col, default=0.0):
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series([default] * len(df))


def load_all():
    frames = []
    for path_ in glob.glob(os.path.join(RAW_DIR, "raw_*.csv")):
        m = RAW_RE.match(os.path.basename(path_))
        if not m:
            continue
        policy, trial = m.group(1), int(m.group(2))
        df = pd.read_csv(path_)
        df["policy"], df["trial"] = policy, trial
        frames.append(df)
    if not frames:
        raise SystemExit(f"No raw CSVs found in {RAW_DIR} -- run loadgen/orchestrate_capacity_sweep.py first.")
    full = pd.concat(frames, ignore_index=True)
    full["ok"] = full["ok"].astype(str).str.lower().isin(["true", "1"])
    full["cold"] = pd.to_numeric(full["cold"], errors="coerce").fillna(0).astype(int) if "cold" in full.columns \
        else (full["cold"].astype(str) == "1").astype(int)
    return full


def main():
    df = load_all()
    summary_rows, trial_rows, decomp_rows = [], [], []
    for policy in POLICIES:
        sub = df[df.policy == policy]
        if sub.empty:
            continue
        lat = sub.loc[sub.ok, "latency_ms"]
        n_cold = int((sub["cold"] == 1).sum())
        summary_rows.append({
            "policy": LABEL[policy], "n_trials": sub["trial"].nunique(), "n_requests": len(sub),
            "n_ok": int(sub.ok.sum()), "n_cold_starts": n_cold,
            "cold_start_rate_pct": round(100 * n_cold / len(sub), 2) if len(sub) else float("nan"),
            "p50_ms": round(np.percentile(lat, 50), 2) if len(lat) else float("nan"),
            "p95_ms": round(np.percentile(lat, 95), 2) if len(lat) else float("nan"),
            "mean_ms": round(lat.mean(), 2) if len(lat) else float("nan"),
        })
        for trial, g in sub.groupby("trial"):
            glat = g.loc[g.ok, "latency_ms"]
            if len(glat) == 0:
                continue
            trial_rows.append({"policy": policy, "trial": int(trial), "median_latency_ms": float(np.median(glat))})
        queue = _hdr_or_col(sub, "queue_ms")
        cold_ms = _hdr_or_col(sub, "cold_ms")
        service = _hdr_or_col(sub, "service_ms")
        decomp_rows.append({
            "policy": LABEL[policy], "mean_queue_ms": round(queue.mean(), 2),
            "mean_cold_ms": round(cold_ms.mean(), 2), "mean_service_ms": round(service.mean(), 2),
            "mean_total_ms": round(lat.mean(), 2) if len(lat) else float("nan"),
            "pct_requests_with_nonzero_queue": round(100 * (queue > 1.0).mean(), 2),
        })

    summary = pd.DataFrame(summary_rows)
    trial_tbl = pd.DataFrame(trial_rows)
    decomp = pd.DataFrame(decomp_rows)
    summary.to_csv(os.path.join(RESULTS, "summary_table_capacity8.csv"), index=False)
    trial_tbl.to_csv(os.path.join(RESULTS, "trial_level_table_capacity8.csv"), index=False)
    decomp.to_csv(os.path.join(RESULTS, "decomposition_table_capacity8.csv"), index=False)

    stat_rows = []
    groups = {p: trial_tbl[trial_tbl.policy == p].sort_values("trial")["median_latency_ms"].values for p in POLICIES}
    present = {k: v for k, v in groups.items() if len(v) >= 3}
    if len(present) >= 2:
        h, p_omni = stats.kruskal(*present.values())
        stat_rows.append({"test": "kruskal_wallis_omnibus", "statistic": round(float(h), 3), "p_value": p_omni})
    ga = groups.get("serverless_reactive")
    if ga is not None and len(ga) >= 3:
        for policy in POLICIES:
            if policy in ("container", "serverless_reactive"):
                continue
            gb = groups.get(policy)
            if gb is None or len(gb) < 3:
                continue
            u, p_mw = stats.mannwhitneyu(ga, gb, alternative="two-sided")
            row = {"pair": f"reactive_vs_{policy}", "test": "mannwhitney", "statistic": round(float(u), 3), "p_value": p_mw,
                   "median_reactive": float(np.median(ga)), "median_other": float(np.median(gb)),
                   "pct_change_vs_reactive": round(100 * (np.median(gb) - np.median(ga)) / np.median(ga), 1)}
            if len(ga) == len(gb):
                try:
                    w, p_w = stats.wilcoxon(ga, gb)
                    row["wilcoxon_paired_p"] = p_w
                except ValueError:
                    row["wilcoxon_paired_p"] = float("nan")
            stat_rows.append(row)
    pd.DataFrame(stat_rows).to_csv(os.path.join(RESULTS, "omnibus_posthoc_capacity8.csv"), index=False)

    # Figure: decomposition, same style as the primary fig_decomposition_cyclical.png
    order = [LABEL[p] for p in POLICIES if LABEL[p] in decomp["policy"].values]
    dsub = decomp.set_index("policy").loc[order]
    short_labels = [lbl.replace("Serverless (predictive, ", "Pred.\n(").replace("Serverless (", "Serv.\n(").replace(")", ")")
                    for lbl in order]
    fig, ax = plt.subplots(figsize=(9, 5.4))
    x = np.arange(len(order))
    ax.bar(x, dsub["mean_queue_ms"], label="Queue (waiting for capacity)")
    ax.bar(x, dsub["mean_cold_ms"], bottom=dsub["mean_queue_ms"], label="Cold-start (spawn + health-check)")
    ax.bar(x, dsub["mean_service_ms"], bottom=dsub["mean_queue_ms"] + dsub["mean_cold_ms"], label="Service (application)")
    ax.set_xticks(x); ax.set_xticklabels(short_labels, rotation=0, fontsize=9)
    ax.set_ylabel("Mean latency contribution (ms)")
    ax.set_title(f"Cyclical scenario, MAX_INSTANCES={CAPACITY}: measured latency decomposition")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "fig_decomposition_capacity8.png"), dpi=600)
    fig.savefig(os.path.join(RESULTS, "fig_decomposition_capacity8.pdf"))
    plt.close(fig)

    # capacity=3 (primary) vs capacity=8 (supplementary) comparison
    cmp_rows = []
    primary_summary_path = os.path.join(RESULTS, "summary_table.csv")
    primary_decomp_path = os.path.join(RESULTS, "decomposition_table.csv")
    if os.path.exists(primary_summary_path) and os.path.exists(primary_decomp_path):
        psum = pd.read_csv(primary_summary_path)
        psum = psum[psum.scenario == "cyclical"]
        pdec = pd.read_csv(primary_decomp_path)
        for policy in POLICIES:
            lbl = LABEL[policy]
            p3 = psum[psum.policy == lbl]
            p8 = summary[summary.policy == lbl]
            d3 = pdec[pdec.policy == lbl]
            d8 = decomp[decomp.policy == lbl]
            if p3.empty or p8.empty:
                continue
            cmp_rows.append({
                "policy": lbl,
                "median_ms_capacity3": p3["p50_ms"].values[0],
                "median_ms_capacity8": p8["p50_ms"].values[0],
                "cold_rate_pct_capacity3": p3["cold_start_rate_pct"].values[0],
                "cold_rate_pct_capacity8": p8["cold_start_rate_pct"].values[0],
                "mean_queue_ms_capacity3": d3["mean_queue_ms"].values[0] if not d3.empty else float("nan"),
                "mean_queue_ms_capacity8": d8["mean_queue_ms"].values[0] if not d8.empty else float("nan"),
            })
        pd.DataFrame(cmp_rows).to_csv(os.path.join(RESULTS, "capacity3_vs_capacity8_comparison.csv"), index=False)

    print("=== CAPACITY=8 SUMMARY ==="); print(summary.to_string(index=False))
    print("\n=== CAPACITY=8 DECOMPOSITION ==="); print(decomp.to_string(index=False))
    print("\n=== CAPACITY=8 STATS ==="); print(pd.DataFrame(stat_rows).to_string(index=False))
    if cmp_rows:
        print("\n=== CAPACITY 3 vs 8 ==="); print(pd.DataFrame(cmp_rows).to_string(index=False))


if __name__ == "__main__":
    main()
