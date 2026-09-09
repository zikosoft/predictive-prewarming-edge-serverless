"""
A6 -- non-periodic (bimodal) workload validation. Tests whether the primary
ranking (fixed-schedule > adaptive predictors > reactive) and fixed-schedule's
specific advantage survive when the workload's idle-gap distribution is no
longer close to periodic (see the "bimodal" scenario docstring in
loadgen/orchestrate_round3_nonperiodic.py for the exact design).

All 7 policies share the same seed sequence (verified: trial 0's seed is
identical across every policy's raw CSV), so this is a fully paired,
seed-matched 7-way design -- the same paired-Wilcoxon/Holm-correction
machinery used for the primary comparison applies directly. Reuses
per_trial_summary/load_raw_trials (analyze_v4d_sweeps.py) and
paired_stats/holm_bonferroni/hodges_lehmann_walsh/hl_confidence_interval/
matched_pairs_rank_biserial (analyze_v4_corrected.py) -- NOT that module's
own friedman_omnibus/family_A_h4/family_B_rq3, which are hardcoded to the
primary trial_tbl's scenario/policy structure and have hardcoded output-file
side effects (the same reuse pattern, and the same reason, as
analyze_v5_capacity8_paired.py and analyze_v5_round3_confirmatory.py).

Outputs:
  results/v5_nonperiodic_descriptive.csv
  results/v5_nonperiodic_friedman.csv
  results/v5_nonperiodic_family_A_vs_reactive.csv
  results/v5_nonperiodic_family_B_pairwise.csv
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_v4d_sweeps import load_raw_trials, per_trial_summary  # noqa: E402
from analyze_v4_corrected import (  # noqa: E402
    holm_bonferroni, paired_stats,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

LABEL = {
    "container": "Container", "serverless_reactive": "Serverless (reactive)",
    "serverless_keepalive10": "Serverless (keep-alive 10s)",
    "serverless_predictive_ewma": "Serverless (predictive, EWMA)",
    "serverless_predictive_lastgap": "Serverless (predictive, last-gap)",
    "serverless_predictive_movavg": "Serverless (predictive, moving avg.)",
    "serverless_predictive_fixed": "Serverless (predictive, fixed-schedule)",
}
POLICIES = list(LABEL.keys())
PREDICTORS = ["serverless_predictive_ewma", "serverless_predictive_lastgap",
              "serverless_predictive_movavg", "serverless_predictive_fixed"]


def load_policy(policy):
    # each policy/label has its own directory (results/raw_nonperiodic/<label>/),
    # but the files inside are named after the underlying harness policy
    # argument, not the label -- e.g. serverless_keepalive10/ contains
    # raw_serverless_reactive_bimodal_trial*.csv (reactive policy, 10s
    # keep-alive override). Glob by directory + wildcard filename rather
    # than assuming the label appears in the filename.
    pattern = os.path.join(RESULTS, "raw_nonperiodic", policy, "raw_*_bimodal_trial*.csv")
    df = load_raw_trials(pattern)
    df["policy"] = policy
    return df


def main():
    all_raw = pd.concat([load_policy(p) for p in POLICIES], ignore_index=True)

    # ---- descriptive summary ----
    summ = per_trial_summary(all_raw, ["policy", "__source_file"])
    summ["trial_idx"] = summ["__source_file"].str.extract(r"trial(\d+)\.csv$").astype(int)
    desc = summ.groupby("policy").agg(
        n_trials=("median_ms", "count"),
        mean_of_trial_median_ms=("median_ms", "mean"),
        mean_p95_ms=("p95_ms", "mean"),
        mean_p99_ms=("p99_ms", "mean"),
        mean_cold_start_rate=("cold_start_rate", "mean"),
    ).reset_index()
    desc["policy_label"] = desc["policy"].map(LABEL)
    desc["_ord"] = desc["policy"].map({p: i for i, p in enumerate(POLICIES)})
    desc = desc.sort_values("_ord").drop(columns="_ord")
    print(desc[["policy_label", "n_trials", "mean_of_trial_median_ms", "mean_p95_ms",
                "mean_p99_ms", "mean_cold_start_rate"]].to_string(index=False))
    desc.to_csv(os.path.join(RESULTS, "v5_nonperiodic_descriptive.csv"), index=False)

    # ---- paired matrix: trial_idx x policy, median_ms ----
    wide = summ.pivot_table(index="trial_idx", columns="policy", values="median_ms")
    wide = wide[POLICIES].dropna()
    assert len(wide) == 15, f"expected 15 fully-matched trials, got {len(wide)}"

    # ---- Friedman omnibus across all 7 policies ----
    groups = [wide[p].values for p in POLICIES]
    chi2, p_omni = scipy_stats.friedmanchisquare(*groups)
    friedman_row = pd.DataFrame([{"chi_squared": float(chi2), "df": len(POLICIES) - 1,
                                   "p_value": float(p_omni), "n_trials": len(wide),
                                   "n_policies": len(POLICIES)}])
    friedman_row.to_csv(os.path.join(RESULTS, "v5_nonperiodic_friedman.csv"), index=False)
    print(f"\nFriedman omnibus (7 policies, N={len(wide)}): chi2={chi2:.2f}, "
          f"df={len(POLICIES)-1}, p={p_omni:.3g}")

    # ---- Family A analog: each predictor vs reactive, Holm-4 ----
    rows, raw_ps = [], []
    for pred in PREDICTORS:
        a, b = wide[pred].values, wide["serverless_reactive"].values
        s = paired_stats(a, b)
        rows.append({"predictor": LABEL[pred], **s})
        raw_ps.append(s["p_raw"])
    adj = holm_bonferroni(raw_ps)
    for row, p_adj in zip(rows, adj):
        row["p_holm_adjusted"] = p_adj
        row["significant_at_0.05_after_holm4"] = p_adj < 0.05
    famA = pd.DataFrame(rows)
    print("\n-- Family A analog (predictor vs reactive, Holm-4) --")
    print(famA[["predictor", "n_pairs", "median_a", "median_b", "median_paired_diff_ms",
                "hodges_lehmann_ms", "p_raw", "p_holm_adjusted", "significant_at_0.05_after_holm4"]].to_string(index=False))
    famA.to_csv(os.path.join(RESULTS, "v5_nonperiodic_family_A_vs_reactive.csv"), index=False)

    # ---- Family B analog: all 6 predictor-predictor pairs, Holm-6 ----
    rows, raw_ps = [], []
    for p1, p2 in itertools.combinations(PREDICTORS, 2):
        a, b = wide[p1].values, wide[p2].values
        s = paired_stats(a, b)
        rows.append({"comparison": f"{LABEL[p1]} vs {LABEL[p2]}", **s})
        raw_ps.append(s["p_raw"])
    adj = holm_bonferroni(raw_ps)
    for row, p_adj in zip(rows, adj):
        row["p_holm_adjusted"] = p_adj
        row["significant_at_0.05_after_holm6"] = p_adj < 0.05
    famB = pd.DataFrame(rows)
    print("\n-- Family B analog (predictor vs predictor, Holm-6) --")
    print(famB[["comparison", "n_pairs", "median_a", "median_b", "median_paired_diff_ms",
                "p_raw", "p_holm_adjusted", "significant_at_0.05_after_holm6"]].to_string(index=False))
    famB.to_csv(os.path.join(RESULTS, "v5_nonperiodic_family_B_pairwise.csv"), index=False)


if __name__ == "__main__":
    main()
