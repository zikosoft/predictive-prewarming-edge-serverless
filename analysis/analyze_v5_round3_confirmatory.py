"""
Analysis of the A4 (keep-alive N=15
confirmatory) and A5 (fixed-schedule N=15 confirmatory sensitivity sweep)
experiments produced by loadgen/orchestrate_round3_confirmatory.py.

Reuses, rather than reimplements, the exact methods already used elsewhere
in this study for methodological consistency:
  - per_trial_summary / load_raw_trials from analyze_v4d_sweeps.py (the
    original N=6 pilot's own analysis script) for median/P95/P99/cold-start
    descriptive aggregation -- so the N=15 confirmatory numbers are computed
    exactly the same way as the pilot numbers they replace, and are directly
    comparable.
  - friedman_omnibus / paired_stats / holm_bonferroni / hodges_lehmann_walsh
    / hl_confidence_interval / matched_pairs_rank_biserial from
    analyze_v4_corrected.py (the primary comparison's own analysis script)
    for the new paired inferential test this confirmatory N=15 scale now
    supports (trials are seed-matched across every swept condition and the
    reused nominal point, exactly as in the primary Family A/B design).

A4 adds an inferential result the N=6 pilot could not support: a Friedman
omnibus across the 4 keep-alive conditions (6.0s nominal + 8/10/12s), N=15
paired trials, followed by 3 paired Wilcoxon contrasts (each swept value vs.
the 6.0s nominal), Holm-3 corrected.

A5 adds SLA violation rates (matching Table 7's 50/100/200ms thresholds) and
a pooled opportunity-success rate (successes / (N_trials * 4), matching
Section 5.5's own pooled-count convention for a "structural opportunity"
statistic) per swept condition, N=15.

Outputs:
  results/v5_keepalive_confirmatory_summary.csv
  results/v5_keepalive_confirmatory_cost.csv
  results/v5_keepalive_confirmatory_friedman.csv
  results/v5_keepalive_confirmatory_pairwise_vs_nominal.csv
  results/v5_fixedsweep_confirmatory_summary.csv
  results/v5_fixedsweep_confirmatory_cost.csv
"""
import os
import sys

import numpy as np
import pandas as pd

from scipy import stats as scipy_stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_v4d_sweeps import load_raw_trials, per_trial_summary  # noqa: E402
# NOTE: analyze_v4_corrected.py's own `friedman_omnibus` is NOT reused here --
# it is hardcoded to the primary trial_tbl's specific scenario/policy
# structure (low/moderate/cyclical) AND has a hardcoded side-effect write to
# results/v4_friedman_omnibus.csv, the primary run's own Friedman result file
# quoted directly in the manuscript (Section 5.2). Calling it with this
# script's differently-shaped 4-condition keep-alive matrix would both fail
# (wrong DataFrame schema) and, if it didn't fail, risk silently overwriting
# that file -- the exact near-miss already caught and fixed once for A8
# (analyze_v5_capacity8_paired.py). scipy.stats.friedmanchisquare is called
# directly instead, which is what that function itself wraps.
from analyze_v4_corrected import (  # noqa: E402
    holm_bonferroni, hodges_lehmann_walsh,
    hl_confidence_interval, matched_pairs_rank_biserial, paired_stats,
)


def friedman_omnibus_generic(groups):
    """groups: list of 1-D arrays, one per condition, all matched/paired by
    trial index (same length, same trial order). Thin, side-effect-free
    wrapper around scipy.stats.friedmanchisquare -- the same test
    analyze_v4_corrected.py's own friedman_omnibus() calls internally."""
    chi2, p = scipy_stats.friedmanchisquare(*groups)
    return float(chi2), len(groups) - 1, float(p)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

THRESHOLDS_MS = [50, 100, 200]


def sla_violation_rates(df):
    """df: per-request rows (ok==True) for one condition. Returns dict of
    violation-rate percentages at each threshold, pooled across all trials
    of that condition -- same convention as roi_model.py's Table 7."""
    ok = df[df["ok"].astype(str).str.lower().isin(["true", "1"])]
    n = len(ok)
    out = {}
    for th in THRESHOLDS_MS:
        viol = (pd.to_numeric(ok["latency_ms"], errors="coerce") > th).sum()
        out[f"violation_rate_pct_gt_{th}ms"] = round(100 * viol / n, 2) if n else float("nan")
    return out


def integrate_cost_by_tag(resource_csv_path, tag_col):
    """Per-trial Riemann-integration of CPU-s/GB-s, grouped by (tag_col,
    label) -- label alone is ambiguous across swept conditions (same
    policy:scenario:trial string repeats at every condition), but combined
    with the new per-condition tag column it correctly isolates each
    condition's own trials. Averages across trials within a condition."""
    if not os.path.exists(resource_csv_path):
        return pd.DataFrame()
    df = pd.read_csv(resource_csv_path)
    if df.empty or tag_col not in df.columns:
        return pd.DataFrame()
    rows = []
    for (tag, label), g in df.groupby([tag_col, "label"]):
        g = g.sort_values("t")
        t = g["t"].values
        if len(t) < 2:
            continue
        cpu = g["cpu_pct_sum"].values / 100.0
        rss_gb = g["rss_mb_sum"].values / 1024.0
        dt = np.diff(t)
        cpu_s = float(np.sum(dt * (cpu[:-1] + cpu[1:]) / 2.0))
        gb_s = float(np.sum(dt * (rss_gb[:-1] + rss_gb[1:]) / 2.0))
        rows.append({tag_col: tag, "label": label, "cpu_s": cpu_s, "gb_s": gb_s})
    per_trial = pd.DataFrame(rows)
    if per_trial.empty:
        return per_trial
    agg = per_trial.groupby(tag_col).agg(
        n_trials=("cpu_s", "count"),
        mean_cpu_s_per_trial=("cpu_s", "mean"),
        mean_gb_s_per_trial=("gb_s", "mean"),
    ).reset_index()
    return agg


# ---------------------------------------------------------------------
# A4 -- keep-alive N=15 confirmatory
# ---------------------------------------------------------------------
def analyze_keepalive():
    print("=" * 70)
    print("A4 -- KEEP-ALIVE N=15 CONFIRMATORY")
    print("=" * 70)

    ka_raw = load_raw_trials(os.path.join(
        RESULTS, "raw_keepalivesweep_n15", "*", "raw_serverless_reactive_cyclical_trial*.csv"))
    nominal_raw = load_raw_trials(os.path.join(
        RESULTS, "raw", "raw_serverless_reactive_cyclical_trial*.csv"))
    nominal_raw = nominal_raw[nominal_raw["cycle_id"].notna()] if not nominal_raw.empty else nominal_raw
    nominal_raw = nominal_raw.copy()
    nominal_raw["condition"] = "6.0s"

    all_raw = pd.concat([ka_raw, nominal_raw], ignore_index=True)
    group_cols = ["condition", "__source_file"]
    summ = per_trial_summary(all_raw, group_cols)
    agg = summ.groupby("condition").agg(
        n_trials=("median_ms", "count"),
        mean_of_trial_median_ms=("median_ms", "mean"),
        mean_p95_ms=("p95_ms", "mean"),
        mean_p99_ms=("p99_ms", "mean"),
        mean_cold_start_rate=("cold_start_rate", "mean"),
    ).reset_index()

    # pooled SLA violation rates per condition (Table-7 convention)
    sla_rows = []
    for cond, g in all_raw.groupby("condition"):
        row = {"condition": cond}
        row.update(sla_violation_rates(g))
        sla_rows.append(row)
    sla_df = pd.DataFrame(sla_rows)
    agg = agg.merge(sla_df, on="condition", how="left")

    order = {"6.0s": 0, "8.0s": 1, "10.0s": 2, "12.0s": 3}
    agg["_ord"] = agg["condition"].map(order)
    agg = agg.sort_values("_ord").drop(columns="_ord")
    print(agg.to_string(index=False))
    agg.to_csv(os.path.join(RESULTS, "v5_keepalive_confirmatory_summary.csv"), index=False)

    # CPU-s / GB-s per condition, using the fixed per-condition resource tag
    cost8 = integrate_cost_by_tag(os.path.join(RESULTS, "keepalivesweep_n15_resource_samples.csv"), "idle_timeout_s")
    cost8["condition"] = cost8["idle_timeout_s"].astype(str) + "s"
    # nominal 6.0s cost from the primary run's own (untagged, single-condition) resource_samples.csv
    primary_res = pd.read_csv(os.path.join(RESULTS, "resource_samples.csv"))
    primary_res = primary_res[primary_res["label"].str.startswith("serverless_reactive:cyclical:")]
    rows = []
    for label, g in primary_res.groupby("label"):
        g = g.sort_values("t")
        t = g["t"].values
        if len(t) < 2:
            continue
        cpu = g["cpu_pct_sum"].values / 100.0
        rss_gb = g["rss_mb_sum"].values / 1024.0
        dt = np.diff(t)
        cpu_s = float(np.sum(dt * (cpu[:-1] + cpu[1:]) / 2.0))
        gb_s = float(np.sum(dt * (rss_gb[:-1] + rss_gb[1:]) / 2.0))
        rows.append({"cpu_s": cpu_s, "gb_s": gb_s})
    nom_df = pd.DataFrame(rows)
    nominal_cost = pd.DataFrame([{
        "condition": "6.0s", "idle_timeout_s": 6.0,
        "n_trials": len(nom_df),
        "mean_cpu_s_per_trial": nom_df["cpu_s"].mean(),
        "mean_gb_s_per_trial": nom_df["gb_s"].mean(),
    }])
    cost_all = pd.concat([nominal_cost, cost8[["condition", "idle_timeout_s", "n_trials",
                                                "mean_cpu_s_per_trial", "mean_gb_s_per_trial"]]],
                          ignore_index=True)
    cost_all["_ord"] = cost_all["condition"].map(order)
    cost_all = cost_all.sort_values("_ord").drop(columns="_ord")
    print("\n-- CPU-s / GB-s per trial by condition --")
    print(cost_all.to_string(index=False))
    cost_all.to_csv(os.path.join(RESULTS, "v5_keepalive_confirmatory_cost.csv"), index=False)

    # ---- paired inferential test: nominal (6.0s) vs. each swept value ----
    # Build a trial-indexed median-latency matrix: rows = trial (0..14,
    # seed-matched across all 4 conditions), columns = condition.
    piv = summ.copy()
    piv["trial_idx"] = piv["__source_file"].str.extract(r"trial(\d+)\.csv$").astype(int)
    wide = piv.pivot_table(index="trial_idx", columns="condition", values="median_ms")
    wide = wide[["6.0s", "8.0s", "10.0s", "12.0s"]].dropna()
    print(f"\n-- paired matrix: {len(wide)} matched trials across all 4 conditions --")

    groups = [wide[c].values for c in ["6.0s", "8.0s", "10.0s", "12.0s"]]
    chi2, df_, p_omni = friedman_omnibus_generic(groups)
    friedman_row = pd.DataFrame([{"chi_squared": chi2, "df": df_, "p_value": p_omni,
                                   "n_trials": len(wide), "n_conditions": 4}])
    friedman_row.to_csv(os.path.join(RESULTS, "v5_keepalive_confirmatory_friedman.csv"), index=False)
    print(f"Friedman omnibus (4 conditions, N={len(wide)}): chi2={chi2:.2f}, df={df_}, p={p_omni:.3g}")

    pair_rows = []
    raw_ps = []
    labels = []
    for cond in ["8.0s", "10.0s", "12.0s"]:
        a = wide["6.0s"].values
        b = wide[cond].values
        stats = paired_stats(a, b)
        pair_rows.append({"comparison": f"6.0s (nominal) vs {cond}", **stats})
        raw_ps.append(stats["p_raw"])
        labels.append(cond)
    adj = holm_bonferroni(raw_ps)
    for row, p_adj in zip(pair_rows, adj):
        row["p_holm_adjusted"] = p_adj
        row["significant_at_0.05_after_holm3"] = p_adj < 0.05
    pair_df = pd.DataFrame(pair_rows)
    print(pair_df.to_string(index=False))
    pair_df.to_csv(os.path.join(RESULTS, "v5_keepalive_confirmatory_pairwise_vs_nominal.csv"), index=False)


# ---------------------------------------------------------------------
# A5 -- fixed-schedule N=15 confirmatory sensitivity sweep
# ---------------------------------------------------------------------
def analyze_fixedsweep():
    print()
    print("=" * 70)
    print("A5 -- FIXED-SCHEDULE N=15 CONFIRMATORY SENSITIVITY SWEEP")
    print("=" * 70)

    fx_raw = load_raw_trials(os.path.join(
        RESULTS, "raw_fixedsweep_n15", "*", "raw_serverless_predictive_fixed_cyclical_trial*.csv"))
    nominal_raw = load_raw_trials(os.path.join(
        RESULTS, "raw", "raw_serverless_predictive_fixed_cyclical_trial*.csv"))
    nominal_raw = nominal_raw.copy()
    nominal_raw["condition"] = "+0pct_9.0s"

    all_raw = pd.concat([fx_raw, nominal_raw], ignore_index=True)
    group_cols = ["condition", "__source_file"]
    summ = per_trial_summary(all_raw, group_cols)
    agg = summ.groupby("condition").agg(
        n_trials=("median_ms", "count"),
        mean_of_trial_median_ms=("median_ms", "mean"),
        mean_p95_ms=("p95_ms", "mean"),
        mean_p99_ms=("p99_ms", "mean"),
        mean_cold_start_rate=("cold_start_rate", "mean"),
    ).reset_index()

    # Opportunity-success ("conversion") rate: SAME definition and method as
    # the already-published Section 5.5 quantity (analyze_v4b_opportunity.py):
    # per-trial conversion_rate = predictive_cold_starts / opportunities,
    # opportunities = 4 for fixed-schedule (cycles 2-5; a structural protocol
    # constant, not measured per trial), then averaged across trials.
    # predictive_cold_starts counts a PROACTIVE spawn fired by the prewarmer
    # ahead of the burst (elastic_gateway.py line ~350) -- i.e. successes,
    # not misses. An earlier version of this script had the sign backwards
    # (treated predictive_cold_starts as a miss count); caught by a sanity
    # check against the nominal point's known Section-5.5 value (80%) before
    # this analysis was used for anything, and fixed here.
    FIXED_OPPORTUNITIES = 4
    gw_n15 = pd.read_csv(os.path.join(RESULTS, "fixedsweep_n15_gateway_stats.csv"))
    gw_primary = pd.read_csv(os.path.join(RESULTS, "gateway_stats.csv"))
    gw_primary = gw_primary[(gw_primary.policy == "serverless_predictive_fixed") & (gw_primary.scenario == "cyclical")]

    def conversion_stats(g):
        rate = g["predictive_cold_starts"] / FIXED_OPPORTUNITIES
        wasted_rate = g["wasted_prewarms"] / g["predictive_cold_starts"].replace(0, np.nan)
        return {
            "opportunity_success_rate_pct": round(100 * float(rate.mean()), 2),
            "mean_wasted_prewarms_per_trial": round(float(g["wasted_prewarms"].mean()), 3),
            "mean_wasted_prewarm_rate_pct": round(100 * float(wasted_rate.mean()), 2) if wasted_rate.notna().any() else 0.0,
        }

    opp_rows = [{"condition": "+0pct_9.0s", **conversion_stats(gw_primary)}]
    for pct, g in gw_n15.groupby("sweep_pct"):
        value = g["fixed_predicted_gap_s"].iloc[0]
        label = f"{int(pct):+d}pct_{value}s"
        opp_rows.append({"condition": label, **conversion_stats(g)})
    opp_df = pd.DataFrame(opp_rows)
    agg = agg.merge(opp_df, on="condition", how="left")

    # Sanity check against the already-published Section 8.2 pilot's own
    # nominal-point value for THIS SAME "opportunity-success rate" column
    # (Table 9: "9.0 (N=15) nominal ... 100%"). Note this is a different
    # quantity from Table 4/Section 5.5's "4.0 of a possible 5" (=80%)
    # figure, which uses 5 raw cycles as its denominator rather than the
    # 4-opportunity structural denominator used here and in
    # analyze_v4b_opportunity.py -- two related but distinct metrics that
    # happen to sound similar; do not conflate them.
    nominal_check = opp_df.loc[opp_df["condition"] == "+0pct_9.0s", "opportunity_success_rate_pct"].iloc[0]
    assert abs(nominal_check - 100.0) < 0.5, (
        f"nominal fixed-schedule opportunity-success rate = {nominal_check}%, "
        f"expected 100% per Table 9's own nominal-point value -- formula/data mismatch, do not proceed")
    print(f"[sanity check OK] nominal opportunity-success rate = {nominal_check}% (Table 9 pilot expects 100%)")

    sla_rows = []
    for cond, g in all_raw.groupby("condition"):
        row = {"condition": cond}
        row.update(sla_violation_rates(g))
        sla_rows.append(row)
    sla_df = pd.DataFrame(sla_rows)
    agg = agg.merge(sla_df, on="condition", how="left")

    order = {"-50pct_4.5s": 0, "-25pct_6.75s": 1, "-10pct_8.1s": 2, "+0pct_9.0s": 3,
             "+10pct_9.9s": 4, "+25pct_11.25s": 5, "+50pct_13.5s": 6}
    agg["_ord"] = agg["condition"].map(order)
    agg = agg.sort_values("_ord").drop(columns="_ord", errors="ignore")
    print(agg.to_string(index=False))
    agg.to_csv(os.path.join(RESULTS, "v5_fixedsweep_confirmatory_summary.csv"), index=False)

    cost8 = integrate_cost_by_tag(os.path.join(RESULTS, "fixedsweep_n15_resource_samples.csv"), "sweep_pct")
    print("\n-- CPU-s / GB-s per trial by swept condition (nominal excluded; see cost_seconds_table.csv) --")
    print(cost8.to_string(index=False))
    cost8.to_csv(os.path.join(RESULTS, "v5_fixedsweep_confirmatory_cost.csv"), index=False)


if __name__ == "__main__":
    analyze_keepalive()
    analyze_fixedsweep()
