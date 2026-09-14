"""
Statistical analysis (v3, corrected after external review) for the
container / reactive / predictive(x4 predictor ablations) comparison
across three scenarios (low / moderate / cyclical).

Key methodological changes from v2, made in direct response to an
external review (see repo history / CHANGELOG):

  - Unit of analysis for significance testing is now the TRIAL, not the
    individual request. Every trial is an independent replicate (the
    backend is restarted fresh per trial by the orchestrator), so a
    trial-level summary statistic (median latency) is computed first,
    and Kruskal-Wallis / Mann-Whitney / Wilcoxon run on those N=5 or
    N=15 trial-level values -- not on the underlying hundreds of
    correlated individual requests. Request-level percentiles are still
    reported descriptively (summary_table.csv) but are no longer the
    input to any p-value.
  - Because every policy in a given (scenario, trial) shares the same
    RNG seed, trials are MATCHED across policies: a paired Wilcoxon
    signed-rank test is reported alongside the independent-groups tests.
  - Four prewarming policies are now compared on the cyclical scenario
    (ewma, lastgap, movavg, fixed) against the reactive baseline and the
    always-on container, not just EWMA in isolation.
  - Latency is decomposed into queue/cold-start/service time per
    request (from the gateway's response headers), enabling a direct,
    measured demonstration of the queuing-cascade mechanism instead of
    an inferred one.
  - The learning-curve figure is now built from a dedicated persistent-
    state run (results/raw_learning/) with per-cycle granularity in the
    raw data itself, and is analyzed purely descriptively -- it is
    never mixed into the significance tests above.

Produces (results/):
  summary_table.csv            per (policy, scenario): n, p50/p95/p99, mean/std, cold-start rate (descriptive)
  trial_level_table.csv        per (policy, scenario, trial): median/p95 latency, n_cold -- the unit of analysis
  omnibus_table.csv            Kruskal-Wallis across policies per scenario, computed on trial-level medians
  posthoc_table.csv            pairwise Mann-Whitney + Holm-Bonferroni, PLUS paired Wilcoxon (matched by seed)
  baseline_comparison_table.csv cyclical-only: all 4 predictor ablations vs reactive vs container
  decomposition_table.csv      mean queue/cold/service ms per policy, cyclical scenario
  resource_summary.csv         mean/max CPU% and RSS per (policy, scenario)
  learning_curve_table.csv     cold starts per cycle index, from the dedicated persistent-state run
  fig_latency_boxplot_{low,moderate}.png / .pdf
  fig_latency_boxplot_cyclical.png / .pdf
  fig_latency_ecdf_cyclical.png / .pdf
  fig_decomposition_cyclical.png / .pdf
  fig_learning_curve.png / .pdf
  fig_resource.png / .pdf
Every number is computed directly from results/raw/*.csv,
results/raw_learning/*.csv and results/gateway_stats.csv -- nothing is
hand-typed. Figures are saved at 600 DPI PNG plus a matching vector PDF.
"""
import glob
import os
import re

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "results", "raw")
LEARNING_DIR = os.path.join(ROOT, "results", "raw_learning")
OUT_DIR = os.path.join(ROOT, "results")
FIG_DPI = 600

SCENARIOS = ["low", "moderate", "cyclical"]
SHORT_POLICIES = ["container", "serverless_reactive", "serverless_predictive_ewma"]
CYCLICAL_POLICIES = ["container", "serverless_reactive", "serverless_predictive_ewma",
                      "serverless_predictive_lastgap", "serverless_predictive_movavg",
                      "serverless_predictive_fixed"]
POLICIES_BY_SCENARIO = {"low": SHORT_POLICIES, "moderate": SHORT_POLICIES, "cyclical": CYCLICAL_POLICIES}
LABEL = {
    "container": "Container", "serverless_reactive": "Serverless (reactive)",
    "serverless_predictive_ewma": "Serverless (predictive, EWMA)",
    "serverless_predictive_lastgap": "Serverless (predictive, last-gap)",
    "serverless_predictive_movavg": "Serverless (predictive, moving avg.)",
    "serverless_predictive_fixed": "Serverless (predictive, fixed-schedule)",
}
ALL_POLICIES = list(LABEL.keys())
_POLICY_ALT = "|".join(sorted(ALL_POLICIES, key=len, reverse=True))
RAW_RE = re.compile(rf"raw_({_POLICY_ALT})_(low|moderate|cyclical)_trial(\d+)\.csv")


def _savefig(fig, name):
    fig.savefig(os.path.join(OUT_DIR, f"{name}.png"), dpi=FIG_DPI)
    fig.savefig(os.path.join(OUT_DIR, f"{name}.pdf"))
    plt.close(fig)


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
    full["cold"] = pd.to_numeric(full["cold"], errors="coerce").fillna(0).astype(int)
    for col in ("queue_ms", "cold_ms", "service_ms"):
        full[col] = pd.to_numeric(full[col], errors="coerce").fillna(0.0)
    return full


def summary_table(df):
    rows = []
    for scenario in SCENARIOS:
        for policy in POLICIES_BY_SCENARIO[scenario]:
            sub = df[(df.scenario == scenario) & (df.policy == policy)]
            lat = sub.loc[sub.ok, "latency_ms"]
            n, n_ok = len(sub), int(sub.ok.sum())
            n_cold = int((sub["cold"] == 1).sum())
            n_trials = sub["trial"].nunique()
            rows.append({
                "scenario": scenario, "policy": LABEL[policy], "n_trials": n_trials,
                "n_requests": n, "n_ok": n_ok,
                "error_rate_pct": round(100 * (n - n_ok) / n, 2) if n else float("nan"),
                "n_cold_starts": n_cold, "cold_start_rate_pct": round(100 * n_cold / n, 2) if n else float("nan"),
                "p50_ms": round(np.percentile(lat, 50), 2) if len(lat) else float("nan"),
                "p95_ms": round(np.percentile(lat, 95), 2) if len(lat) else float("nan"),
                "p99_ms": round(np.percentile(lat, 99), 2) if len(lat) else float("nan"),
                "mean_ms": round(lat.mean(), 2) if len(lat) else float("nan"),
                "std_ms": round(lat.std(), 2) if len(lat) else float("nan"),
            })
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_DIR, "summary_table.csv"), index=False)
    return out


def trial_level_table(df):
    """The unit of statistical analysis: one row per (policy, scenario,
    trial), summarizing that independent replicate."""
    rows = []
    for scenario in SCENARIOS:
        for policy in POLICIES_BY_SCENARIO[scenario]:
            sub = df[(df.scenario == scenario) & (df.policy == policy)]
            for trial, g in sub.groupby("trial"):
                lat = g.loc[g.ok, "latency_ms"]
                if len(lat) == 0:
                    continue
                rows.append({
                    "scenario": scenario, "policy": policy, "trial": int(trial),
                    "n_requests": len(g), "n_cold": int((g["cold"] == 1).sum()),
                    "median_latency_ms": float(np.median(lat)),
                    "p95_latency_ms": float(np.percentile(lat, 95)),
                    "mean_queue_ms": float(g["queue_ms"].mean()),
                    "mean_cold_ms": float(g["cold_ms"].mean()),
                })
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_DIR, "trial_level_table.csv"), index=False)
    return out


def holm_bonferroni(pvals):
    idx = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running_max = 0.0
    for rank, i in enumerate(idx):
        val = (m - rank) * pvals[i]
        running_max = max(running_max, val)
        adj[i] = min(running_max, 1.0)
    return adj


def omnibus_and_posthoc(trial_tbl):
    omni_rows, posthoc_rows = [], []
    for scenario in SCENARIOS:
        policies = POLICIES_BY_SCENARIO[scenario]
        groups = {p: trial_tbl[(trial_tbl.scenario == scenario) & (trial_tbl.policy == p)]
                  .sort_values("trial")["median_latency_ms"].values for p in policies}
        if any(len(g) < 3 for g in groups.values()):
            continue
        h_stat, p_omni = stats.kruskal(*groups.values())
        omni_rows.append({"scenario": scenario, "n_trials_per_policy": len(next(iter(groups.values()))),
                           "H_statistic": round(float(h_stat), 2), "p_value": p_omni,
                           "significant_at_0.05": bool(p_omni < 0.05)})

        if scenario == "cyclical":
            pairs = [("container", p) for p in policies if p != "container"] + \
                    [("serverless_reactive", p) for p in policies if p.startswith("serverless_predictive")]
        else:
            pairs = [("container", "serverless_reactive"), ("container", "serverless_predictive_ewma"),
                     ("serverless_reactive", "serverless_predictive_ewma")]
        raw_p, tmp = [], []
        for a, b in pairs:
            ga, gb = groups[a], groups[b]
            u, p = stats.mannwhitneyu(ga, gb, alternative="two-sided")
            effect = 1 - (2 * u) / (len(ga) * len(gb))
            # paired test: trials share a seed across policies (matched design)
            try:
                w_stat, p_paired = stats.wilcoxon(ga, gb)
            except ValueError:
                w_stat, p_paired = float("nan"), float("nan")
            tmp.append({"scenario": scenario, "pair": f"{LABEL[a]} vs {LABEL[b]}",
                        "n_a": len(ga), "n_b": len(gb),
                        "median_a_ms": round(float(np.median(ga)), 2), "median_b_ms": round(float(np.median(gb)), 2),
                        "U": round(float(u), 1), "p_mannwhitney_raw": p, "effect_size_rank_biserial": round(float(effect), 3),
                        "wilcoxon_paired_p": p_paired})
            raw_p.append(p)
        adj = holm_bonferroni(np.array(raw_p))
        for row, p_adj in zip(tmp, adj):
            row["p_mannwhitney_holm_adjusted"] = p_adj
            row["significant_at_0.05"] = bool(p_adj < 0.05)
            posthoc_rows.append(row)
    omni_df, posthoc_df = pd.DataFrame(omni_rows), pd.DataFrame(posthoc_rows)
    omni_df.to_csv(os.path.join(OUT_DIR, "omnibus_table.csv"), index=False)
    posthoc_df.to_csv(os.path.join(OUT_DIR, "posthoc_table.csv"), index=False)
    return omni_df, posthoc_df


def baseline_comparison_table(trial_tbl):
    """Cyclical-only: how much does EWMA actually buy over the three
    simpler baselines the external review asked for?"""
    rows = []
    reactive_med = trial_tbl[(trial_tbl.scenario == "cyclical") & (trial_tbl.policy == "serverless_reactive")]["median_latency_ms"].values
    for policy in CYCLICAL_POLICIES:
        sub = trial_tbl[(trial_tbl.scenario == "cyclical") & (trial_tbl.policy == policy)].sort_values("trial")
        med = sub["median_latency_ms"].values
        n_cold_mean = sub["n_cold"].mean()
        row = {"policy": LABEL[policy], "n_trials": len(med),
               "median_of_trial_medians_ms": round(float(np.median(med)), 2),
               "mean_cold_starts_per_trial": round(float(n_cold_mean), 2)}
        if policy != "serverless_reactive" and len(med) == len(reactive_med):
            try:
                _, p_paired = stats.wilcoxon(reactive_med, med)
            except ValueError:
                p_paired = float("nan")
            row["wilcoxon_vs_reactive_p"] = p_paired
            row["pct_change_vs_reactive"] = round(100 * (np.median(med) - np.median(reactive_med)) / np.median(reactive_med), 1)
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_DIR, "baseline_comparison_table.csv"), index=False)
    return out


def decomposition_table(df):
    rows = []
    for policy in CYCLICAL_POLICIES:
        sub = df[(df.scenario == "cyclical") & (df.policy == policy) & (df.ok)]
        if sub.empty:
            continue
        rows.append({"policy": LABEL[policy],
                     "mean_queue_ms": round(sub["queue_ms"].mean(), 2),
                     "mean_cold_ms": round(sub["cold_ms"].mean(), 2),
                     "mean_service_ms": round(sub["service_ms"].mean(), 2),
                     "mean_total_ms": round(sub["latency_ms"].mean(), 2),
                     "pct_requests_with_nonzero_queue": round(100 * (sub["queue_ms"] > 1.0).mean(), 1)})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_DIR, "decomposition_table.csv"), index=False)
    return out


def resource_summary():
    df = pd.read_csv(os.path.join(OUT_DIR, "resource_samples.csv"))
    parts = df["label"].str.split(":", n=2, expand=True)
    df["policy"], df["scenario"] = parts[0], parts[1]
    rows = []
    for scenario in SCENARIOS:
        for policy in POLICIES_BY_SCENARIO[scenario]:
            sub = df[(df.scenario == scenario) & (df.policy == policy)]
            if sub.empty:
                continue
            rows.append({"scenario": scenario, "policy": LABEL[policy],
                         "mean_cpu_pct_sum": round(sub.cpu_pct_sum.mean(), 2),
                         "max_cpu_pct_sum": round(sub.cpu_pct_sum.max(), 2),
                         "mean_rss_mb_sum": round(sub.rss_mb_sum.mean(), 2),
                         "max_rss_mb_sum": round(sub.rss_mb_sum.max(), 2),
                         "mean_n_procs": round(sub.n_procs.mean(), 2)})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_DIR, "resource_summary.csv"), index=False)
    return out


def learning_curve_table():
    """Built directly from per-cycle granularity in the dedicated
    persistent-state run (results/raw_learning/) -- descriptive only,
    never fed into the significance tests above."""
    rows = []
    for policy in ["serverless_reactive", "serverless_predictive_ewma"]:
        path = os.path.join(LEARNING_DIR, f"raw_learning_{policy}.csv")
        if not os.path.exists(path):
            continue
        g = pd.read_csv(path)
        g["cold"] = pd.to_numeric(g["cold"], errors="coerce").fillna(0).astype(int)
        for cycle_id, cg in g.groupby("cycle_id"):
            if cycle_id < 0:
                continue
            rows.append({"policy": LABEL[policy], "cycle_index": int(cycle_id) + 1,
                         "n_cold_starts": int((cg["cold"] == 1).sum()), "n_requests": len(cg)})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_DIR, "learning_curve_table.csv"), index=False)
    return out


def make_figures(df, learning_tbl):
    for scenario, policies, figname in [("low", SHORT_POLICIES, "fig_latency_boxplot_low"),
                                         ("moderate", SHORT_POLICIES, "fig_latency_boxplot_moderate"),
                                         ("cyclical", CYCLICAL_POLICIES, "fig_latency_boxplot_cyclical")]:
        fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(policies)), 4.6))
        data, labels = [], []
        for policy in policies:
            lat = df[(df.scenario == scenario) & (df.policy == policy) & (df.ok)]["latency_ms"]
            data.append(lat.values)
            labels.append(LABEL[policy].replace(" (", "\n(").replace(", ", ",\n"))
        ax.boxplot(data, tick_labels=labels, showfliers=False)
        ax.set_title(f"{scenario.capitalize()} scenario: end-to-end latency")
        ax.set_ylabel("Latency (ms)")
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis='x', labelsize=7.5)
        fig.tight_layout()
        _savefig(fig, figname)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for policy in CYCLICAL_POLICIES:
        lat = np.sort(df[(df.scenario == "cyclical") & (df.policy == policy) & (df.ok)]["latency_ms"].values)
        if len(lat) == 0:
            continue
        y = np.arange(1, len(lat) + 1) / len(lat)
        ax.plot(lat, y, label=LABEL[policy])
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("ECDF")
    ax.set_title("Cyclical scenario: latency ECDF (all independent trials pooled)")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _savefig(fig, "fig_latency_ecdf_cyclical")

    dec = pd.read_csv(os.path.join(OUT_DIR, "decomposition_table.csv"))
    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = np.arange(len(dec))
    ax.bar(x, dec["mean_queue_ms"], label="Queue (waiting for capacity)")
    ax.bar(x, dec["mean_cold_ms"], bottom=dec["mean_queue_ms"], label="Cold-start (spawn + health-check)")
    ax.bar(x, dec["mean_service_ms"], bottom=dec["mean_queue_ms"] + dec["mean_cold_ms"], label="Service (application)")
    ax.set_xticks(x)
    ax.set_xticklabels([p.replace(" (", "\n(").replace(", ", ",\n") for p in dec["policy"]], fontsize=7.5)
    ax.set_ylabel("Mean latency contribution (ms)")
    ax.set_title("Cyclical scenario: measured latency decomposition")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _savefig(fig, "fig_decomposition_cyclical")

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for policy_label in learning_tbl["policy"].unique():
        sub = learning_tbl[learning_tbl.policy == policy_label].sort_values("cycle_index")
        ax.plot(sub["cycle_index"], sub["n_cold_starts"], marker="o", label=policy_label)
    ax.set_xlabel("Cycle index (persistent-state run, no resets)")
    ax.set_ylabel("Cold starts (out of 15 requests in the cycle's burst)")
    ax.set_title("Predictor learning curve: dedicated persistent-state run")
    ax.set_ylim(-0.5, 16)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _savefig(fig, "fig_learning_curve")

    res = pd.read_csv(os.path.join(OUT_DIR, "resource_summary.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, col, title in zip(axes, ["mean_cpu_pct_sum", "mean_rss_mb_sum"], ["Mean aggregate CPU (%)", "Mean aggregate RSS (MB)"]):
        width = 0.8 / len(SHORT_POLICIES)
        xs = np.arange(len(SCENARIOS))
        for i, policy in enumerate(SHORT_POLICIES):
            vals = [res[(res.scenario == s) & (res.policy == LABEL[policy])][col].values[0]
                    if not res[(res.scenario == s) & (res.policy == LABEL[policy])].empty else 0 for s in SCENARIOS]
            ax.bar(xs + (i - 1) * width, vals, width, label=LABEL[policy])
        ax.set_xticks(xs)
        ax.set_xticklabels([s.capitalize() for s in SCENARIOS])
        ax.set_title(title)
        ax.legend(fontsize=7.5)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _savefig(fig, "fig_resource")


def main():
    df = load_all()
    summary = summary_table(df)
    trial_tbl = trial_level_table(df)
    omni, posthoc = omnibus_and_posthoc(trial_tbl)
    baseline_cmp = baseline_comparison_table(trial_tbl)
    dec = decomposition_table(df)
    res = resource_summary()
    learning_tbl = learning_curve_table()
    make_figures(df, learning_tbl)
    print("=== SUMMARY (descriptive) ==="); print(summary.to_string(index=False))
    print("\n=== TRIAL-LEVEL (unit of analysis) n rows ==="); print(len(trial_tbl))
    print("\n=== KRUSKAL-WALLIS OMNIBUS (on trial-level medians) ==="); print(omni.to_string(index=False))
    print("\n=== POST-HOC (Mann-Whitney/Holm + paired Wilcoxon) ==="); print(posthoc.to_string(index=False))
    print("\n=== BASELINE PREDICTOR COMPARISON (cyclical) ==="); print(baseline_cmp.to_string(index=False))
    print("\n=== LATENCY DECOMPOSITION (cyclical) ==="); print(dec.to_string(index=False))
    print("\n=== RESOURCE ==="); print(res.to_string(index=False))
    print("\n=== LEARNING CURVE (descriptive, dedicated run) ==="); print(learning_tbl.to_string(index=False))


if __name__ == "__main__":
    main()
