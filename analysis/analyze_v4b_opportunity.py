"""
Extends the opportunity-normalized analysis (manuscript Section 5.5) to
all four predictors using data already logged by the gateway
(results/gateway_stats.csv: predictive_cold_starts, reactive_cold_starts,
wasted_prewarms per trial per policy) -- no new experiment or
instrumentation needed, unlike per-observation forecast MAE/bias (which
would require logging each predicted-vs-actual gap pair and is not
present in this data; that remains [NEW EXPERIMENT REQUIRED]).

"Opportunity conversion rate" per trial = predictive_cold_starts /
(predictive_cold_starts + reactive_cold_starts), i.e. of the ~15 cold
starts a trial needed across its 5 cycles, the fraction the predictor
converted into a proactive (pre-request) spawn rather than a reactive
(at-request) one. This is the same quantity the manuscript's Section
5.5/5.6 already discusses descriptively; here it is computed for all
four predictors and tested statistically (paired Wilcoxon + Holm-6,
same RQ3 family logic) rather than asserted as "comparable" on
inspection alone -- directly answering Round-2 feedback item 7.

Also reports the wasted-prewarm rate per policy (wasted_prewarms /
predictive_cold_starts), the closest already-logged proxy to a
"false-positive prewarm rate."
"""
import itertools
import os

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "results")

LABEL = {
    "serverless_predictive_ewma": "EWMA",
    "serverless_predictive_lastgap": "Last-gap",
    "serverless_predictive_movavg": "Moving-average",
    "serverless_predictive_fixed": "Fixed-schedule",
}
PREDICTORS = list(LABEL.keys())


def holm_bonferroni(pvals):
    pvals = np.asarray(pvals, dtype=float)
    idx = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running_max = 0.0
    for rank, i in enumerate(idx):
        val = (m - rank) * pvals[i]
        running_max = max(running_max, val)
        adj[i] = min(running_max, 1.0)
    return adj


def main():
    gs = pd.read_csv(os.path.join(OUT_DIR, "gateway_stats.csv"))
    gs = gs[(gs.scenario == "cyclical") & (gs.policy.isin(PREDICTORS))].copy()
    # Opportunity denominator per manuscript Section 5.5's own definition:
    # 5 cycles/trial, but a gap is only recorded at the START of the burst
    # that follows it, so cycle 1 never offers an opportunity to any
    # forecaster. gap_observations == 4.0 for every trial/policy (verified
    # exactly, std=0), i.e. cycles 2-5 each produce one observed gap.
    # fixed needs no prior observation (only a reference timestamp), so its
    # opportunity window is cycles 2-5 = 4 opportunities/trial. The three
    # adaptive forecasters additionally need >=1 prior observed gap before
    # they can emit a non-null forecast, so their first usable cycle is 3,
    # giving cycles 3-5 = 3 opportunities/trial. This is a structural
    # property of the protocol (constant per policy family), not a
    # per-trial measured quantity, so the denominator below is a constant,
    # not (predictive_cold_starts + reactive_cold_starts) -- that sum
    # conflates genuine opportunity-deficit cycles (structurally unusable)
    # with true misses within a real opportunity, which is exactly what
    # Section 5.5 exists to separate.
    OPPORTUNITIES = {"serverless_predictive_fixed": 4, "serverless_predictive_ewma": 3,
                      "serverless_predictive_lastgap": 3, "serverless_predictive_movavg": 3}
    gs["opportunities"] = gs["policy"].map(OPPORTUNITIES)
    gs["conversion_rate"] = gs["predictive_cold_starts"] / gs["opportunities"]
    gs["wasted_prewarm_rate"] = gs["wasted_prewarms"] / gs["predictive_cold_starts"].replace(0, np.nan)

    # descriptive per-policy summary
    desc = gs.groupby("policy").agg(
        n_trials=("trial", "count"),
        mean_conversion_rate=("conversion_rate", "mean"),
        median_conversion_rate=("conversion_rate", "median"),
        min_conversion_rate=("conversion_rate", "min"),
        max_conversion_rate=("conversion_rate", "max"),
        mean_predictive_cold_starts=("predictive_cold_starts", "mean"),
        mean_reactive_cold_starts=("reactive_cold_starts", "mean"),
        mean_wasted_prewarms=("wasted_prewarms", "mean"),
    ).reset_index()
    desc["policy_label"] = desc["policy"].map(LABEL)
    desc.to_csv(os.path.join(OUT_DIR, "v4c_opportunity_conversion_descriptive.csv"), index=False)

    # inferential: is "comparable" actually supported? paired Wilcoxon + Holm-6
    # across the same 6 predictor-vs-predictor pairs as RQ3 Family B.
    series = {}
    for pred in PREDICTORS:
        series[pred] = gs[gs.policy == pred].sort_values("trial").set_index("trial")["conversion_rate"]
    rows = []
    for p1, p2 in itertools.combinations(PREDICTORS, 2):
        common = series[p1].index.intersection(series[p2].index)
        a, b = series[p1].loc[common].values, series[p2].loc[common].values
        diffs = a - b
        try:
            w, p = stats.wilcoxon(a, b, zero_method="wilcox", correction=False, mode="auto")
        except ValueError:
            w, p = float("nan"), float("nan")
        rows.append({"comparison": f"{LABEL[p1]} vs {LABEL[p2]}", "n_pairs": len(a),
                     "median_a": float(np.median(a)), "median_b": float(np.median(b)),
                     "median_diff": float(np.median(diffs)), "wilcoxon_W": float(w), "p_raw": float(p)})
    raw_p = [r["p_raw"] for r in rows]
    adj = holm_bonferroni(raw_p)
    for row, p_adj in zip(rows, adj):
        row["p_holm_adjusted"] = float(p_adj)
        row["significant_at_0.05_after_holm6"] = bool(p_adj < 0.05)
    inf = pd.DataFrame(rows)
    inf.to_csv(os.path.join(OUT_DIR, "v4c_opportunity_conversion_pairwise.csv"), index=False)

    pd.set_option("display.width", 200)
    print("=== OPPORTUNITY-CONVERSION RATE, descriptive, all 4 predictors (cyclical) ===")
    print(desc[["policy_label", "n_trials", "mean_conversion_rate", "median_conversion_rate",
                "min_conversion_rate", "max_conversion_rate", "mean_wasted_prewarms"]].to_string(index=False))
    print("\n=== Is 'comparable' supported? paired Wilcoxon + Holm-6 on conversion rate ===")
    print(inf.to_string(index=False))


if __name__ == "__main__":
    main()
