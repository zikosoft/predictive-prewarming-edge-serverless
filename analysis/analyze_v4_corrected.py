"""
Statistical reanalysis (v4), written in direct response to the FGCS
Round-2 revision feedback (external methodological critique of the
revision package, not of the raw data itself). This script does NOT
change how the experiment was run or how latency was measured -- it
re-analyzes the same trial-level medians already produced by
`analyze.py` (`results/trial_level_table.csv`) with a corrected
inferential design:

  1. H4 and RQ3 are now two SEPARATE correction families, each with its
     own Holm-Bonferroni budget, instead of one merged 4-test family
     that only ever answered H4:
       Family A (H4, 4 tests): each predictor vs. Reactive
       Family B (RQ3, 6 tests): every predictor vs. every other predictor
  2. All primary tests are the PAIRED Wilcoxon signed-rank test (trials
     are matched by seed across all six policies -- see README /
     CHANGELOG), not the unpaired Mann-Whitney test the original
     analyze.py reported as primary.
  3. A Friedman test (repeated-measures omnibus, appropriate for the
     matched-trial design) is reported per scenario, in addition to
     (not instead of) the existing between-groups Kruskal-Wallis.
  4. Effect size is the matched-pairs rank-biserial correlation
     (Kerby 2014: r = (sum_pos - sum_neg) / (sum_pos + sum_neg) of the
     signed-rank sums), not the independent-samples rank-biserial the
     original script computed from the Mann-Whitney U statistic.
  5. Hodges-Lehmann is now computed CORRECTLY as the median of all
     pairwise Walsh averages (d_i + d_j)/2 for i <= j of the n paired
     differences -- not simply the median of the n differences, which
     the Round-2 review correctly flagged as a mislabeling in the
     revision package (median of differences remains reported
     separately, correctly labeled).
  6. A distribution-free confidence interval for the paired median
     difference is reported using the Wilcoxon-signed-rank-based
     Hodges-Lehmann CI construction (via the Walsh-average order
     statistics and the exact/normal-approximation critical rank).

Nothing here is fabricated: every number below is computed directly
from results/trial_level_table.csv, which is itself produced by
analyze.py directly from results/raw/*.csv. This script produces
CSVs only -- it does not touch the manuscript text.

Outputs (results/):
  v4_family_A_H4_vs_reactive.csv     paired Wilcoxon, HL (Walsh), rank-biserial, Holm-4
  v4_family_B_RQ3_pairwise.csv       paired Wilcoxon, HL (Walsh), rank-biserial, Holm-6
  v4_friedman_omnibus.csv            Friedman chi-square per scenario
  v4_container_vs_others.csv         paired Wilcoxon, container vs each on-demand policy (descriptive family, not corrected -- container is not part of H4/RQ3)
"""
import itertools
import os

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "results")

LABEL = {
    "container": "Container",
    "serverless_reactive": "Reactive",
    "serverless_predictive_ewma": "EWMA",
    "serverless_predictive_lastgap": "Last-gap",
    "serverless_predictive_movavg": "Moving-average",
    "serverless_predictive_fixed": "Fixed-schedule",
}
PREDICTORS = ["serverless_predictive_ewma", "serverless_predictive_lastgap",
              "serverless_predictive_movavg", "serverless_predictive_fixed"]


def holm_bonferroni(pvals):
    """Standard Holm step-down. pvals: 1-D array-like. Returns adjusted
    p-values aligned to the input order (not sorted)."""
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


def hodges_lehmann_walsh(diffs):
    """Correct paired-sample Hodges-Lehmann estimator: median of all
    pairwise Walsh averages (d_i + d_j)/2 for i <= j (n*(n+1)/2 of
    them, including each d_i averaged with itself)."""
    d = np.asarray(diffs, dtype=float)
    n = len(d)
    walsh = []
    for i in range(n):
        for j in range(i, n):
            walsh.append((d[i] + d[j]) / 2.0)
    walsh = np.array(walsh)
    return float(np.median(walsh)), walsh


def hl_confidence_interval(walsh, alpha=0.05):
    """Distribution-free CI for the HL estimator from the order
    statistics of the Walsh averages, using the normal approximation
    to the Wilcoxon signed-rank null distribution to pick the rank
    (standard construction; adequate for n=15, exact tables agree
    closely with the normal approximation at n>=10)."""
    n_walsh = len(walsh)
    w_sorted = np.sort(walsh)
    n = int((-1 + (1 + 8 * n_walsh) ** 0.5) / 2)  # invert n*(n+1)/2
    mu = n * (n + 1) / 4.0
    sigma = (n * (n + 1) * (2 * n + 1) / 24.0) ** 0.5
    z = stats.norm.ppf(1 - alpha / 2)
    lo_rank = int(np.floor(mu - z * sigma))
    hi_rank = n_walsh - lo_rank + 1
    lo_rank = max(lo_rank, 1)
    hi_rank = min(hi_rank, n_walsh)
    return float(w_sorted[lo_rank - 1]), float(w_sorted[hi_rank - 1])


def matched_pairs_rank_biserial(diffs):
    """Kerby (2014) matched-pairs rank-biserial correlation, computed
    directly from the signed ranks (equivalent to the Wilcoxon
    signed-rank test's own internal statistic, so it is exactly
    consistent with the reported paired-Wilcoxon p-value, unlike a
    rank-biserial derived from the unpaired Mann-Whitney U on the same
    data, which the Round-2 review correctly flagged as a mismatch)."""
    d = np.asarray(diffs, dtype=float)
    d_nz = d[d != 0]
    if len(d_nz) == 0:
        return 0.0, 0, 0
    ranks = stats.rankdata(np.abs(d_nz))
    sum_pos = ranks[d_nz > 0].sum()
    sum_neg = ranks[d_nz < 0].sum()
    r = (sum_pos - sum_neg) / (sum_pos + sum_neg)
    return float(r), float(sum_pos), float(sum_neg)


def paired_stats(a, b):
    """a, b: matched (same trial index / seed) arrays of trial-level
    medians. Returns a dict of every quantity Round-2 asked for."""
    diffs = a - b  # a minus b, i.e. policy_a - policy_b
    median_diff = float(np.median(diffs))
    hl, walsh = hodges_lehmann_walsh(diffs)
    ci_lo, ci_hi = hl_confidence_interval(walsh)
    r_rb, sum_pos, sum_neg = matched_pairs_rank_biserial(diffs)
    try:
        w_stat, p_raw = stats.wilcoxon(a, b, zero_method="wilcox", correction=False, mode="auto")
    except ValueError:
        w_stat, p_raw = float("nan"), float("nan")
    return {
        "n_pairs": len(a),
        "median_a": float(np.median(a)), "median_b": float(np.median(b)),
        "median_paired_diff_ms": median_diff,
        "hodges_lehmann_ms": hl,
        "hl_ci95_lo_ms": ci_lo, "hl_ci95_hi_ms": ci_hi,
        "wilcoxon_W": float(w_stat), "p_raw": float(p_raw),
        "rank_biserial_matched": r_rb,
        "signed_rank_sum_pos": sum_pos, "signed_rank_sum_neg": sum_neg,
    }


def family_A_h4(trial_tbl, scenario="cyclical"):
    """H4: each of the four predictors vs. Reactive, paired by trial
    index (matched seed). One Holm-4 correction family."""
    rows = []
    reactive = trial_tbl[(trial_tbl.scenario == scenario) & (trial_tbl.policy == "serverless_reactive")] \
        .sort_values("trial").set_index("trial")["median_latency_ms"]
    for pred in PREDICTORS:
        pred_series = trial_tbl[(trial_tbl.scenario == scenario) & (trial_tbl.policy == pred)] \
            .sort_values("trial").set_index("trial")["median_latency_ms"]
        common = reactive.index.intersection(pred_series.index)
        a, b = pred_series.loc[common].values, reactive.loc[common].values
        row = {"family": "A (H4: predictor vs Reactive)", "comparison": f"{LABEL[pred]} vs Reactive"}
        row.update(paired_stats(a, b))
        rows.append(row)
    raw_p = [r["p_raw"] for r in rows]
    adj = holm_bonferroni(raw_p)
    for row, p_adj in zip(rows, adj):
        row["p_holm_adjusted"] = float(p_adj)
        row["significant_at_0.05_after_holm4"] = bool(p_adj < 0.05)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_DIR, "v4_family_A_H4_vs_reactive.csv"), index=False)
    return out


def family_B_rq3(trial_tbl, scenario="cyclical"):
    """RQ3: all 6 pairwise comparisons among the four predictors,
    paired by trial index. Separate Holm-6 correction family."""
    rows = []
    series = {}
    for pred in PREDICTORS:
        series[pred] = trial_tbl[(trial_tbl.scenario == scenario) & (trial_tbl.policy == pred)] \
            .sort_values("trial").set_index("trial")["median_latency_ms"]
    for p1, p2 in itertools.combinations(PREDICTORS, 2):
        common = series[p1].index.intersection(series[p2].index)
        a, b = series[p1].loc[common].values, series[p2].loc[common].values
        row = {"family": "B (RQ3: predictor vs predictor)", "comparison": f"{LABEL[p1]} vs {LABEL[p2]}"}
        row.update(paired_stats(a, b))
        rows.append(row)
    raw_p = [r["p_raw"] for r in rows]
    adj = holm_bonferroni(raw_p)
    for row, p_adj in zip(rows, adj):
        row["p_holm_adjusted"] = float(p_adj)
        row["significant_at_0.05_after_holm6"] = bool(p_adj < 0.05)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_DIR, "v4_family_B_RQ3_pairwise.csv"), index=False)
    return out


def container_vs_others(trial_tbl, scenario="cyclical"):
    """Descriptive/exploratory only -- container is architecturally
    different (always-on) and was never part of H4 or RQ3's planned
    contrasts; reported separately and NOT pooled into either Holm
    family, to avoid inflating either budget with an unplanned
    comparison."""
    rows = []
    container = trial_tbl[(trial_tbl.scenario == scenario) & (trial_tbl.policy == "container")] \
        .sort_values("trial").set_index("trial")["median_latency_ms"]
    others = ["serverless_reactive"] + PREDICTORS
    for pol in others:
        s = trial_tbl[(trial_tbl.scenario == scenario) & (trial_tbl.policy == pol)] \
            .sort_values("trial").set_index("trial")["median_latency_ms"]
        common = container.index.intersection(s.index)
        a, b = container.loc[common].values, s.loc[common].values
        row = {"comparison": f"Container vs {LABEL[pol]}"}
        row.update(paired_stats(a, b))
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_DIR, "v4_container_vs_others.csv"), index=False)
    return out


def friedman_omnibus(trial_tbl):
    rows = []
    for scenario, policies in [("low", ["container", "serverless_reactive", "serverless_predictive_ewma"]),
                                ("moderate", ["container", "serverless_reactive", "serverless_predictive_ewma"]),
                                ("cyclical", ["container", "serverless_reactive"] + PREDICTORS)]:
        mats = []
        ok = True
        for pol in policies:
            s = trial_tbl[(trial_tbl.scenario == scenario) & (trial_tbl.policy == pol)] \
                .sort_values("trial")["median_latency_ms"].values
            if len(s) < 3:
                ok = False
                break
            mats.append(s)
        if not ok or len({len(m) for m in mats}) != 1:
            continue
        chi2, p = stats.friedmanchisquare(*mats)
        rows.append({"scenario": scenario, "k_policies": len(policies), "n_trials": len(mats[0]),
                     "friedman_chi2": float(chi2), "df": len(policies) - 1, "p_value": float(p),
                     "significant_at_0.05": bool(p < 0.05)})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_DIR, "v4_friedman_omnibus.csv"), index=False)
    return out


def main():
    trial_tbl = pd.read_csv(os.path.join(OUT_DIR, "trial_level_table.csv"))
    friedman = friedman_omnibus(trial_tbl)
    famA = family_A_h4(trial_tbl)
    famB = family_B_rq3(trial_tbl)
    cvo = container_vs_others(trial_tbl)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print("=== FRIEDMAN OMNIBUS (repeated-measures, matched trials) ===")
    print(friedman.to_string(index=False))
    print("\n=== FAMILY A (H4): predictor vs Reactive, paired Wilcoxon + Holm-4 ===")
    print(famA[["comparison", "n_pairs", "median_paired_diff_ms", "hodges_lehmann_ms",
                "hl_ci95_lo_ms", "hl_ci95_hi_ms", "p_raw", "p_holm_adjusted",
                "significant_at_0.05_after_holm4", "rank_biserial_matched"]].to_string(index=False))
    print("\n=== FAMILY B (RQ3): predictor vs predictor, paired Wilcoxon + Holm-6 ===")
    print(famB[["comparison", "n_pairs", "median_paired_diff_ms", "hodges_lehmann_ms",
                "hl_ci95_lo_ms", "hl_ci95_hi_ms", "p_raw", "p_holm_adjusted",
                "significant_at_0.05_after_holm6", "rank_biserial_matched"]].to_string(index=False))
    print("\n=== CONTAINER vs OTHERS (descriptive, not Holm-corrected, not part of H4/RQ3) ===")
    print(cvo[["comparison", "n_pairs", "median_paired_diff_ms", "hodges_lehmann_ms", "p_raw"]].to_string(index=False))


if __name__ == "__main__":
    main()
