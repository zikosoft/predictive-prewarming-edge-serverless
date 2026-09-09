"""
Round-3 (FGCS finalization pass) — A8: re-analyze the existing
MAX_INSTANCES=8 supplementary-capacity data with the SAME corrected
paired method used for the primary MAX_INSTANCES=3 comparison
(analysis/analyze_v4_corrected.py), replacing the old unpaired
Mann-Whitney / "pending paired revision" language.

No new experiment is run here: results/trial_level_table_capacity8.csv
already holds the trial-level medians from the MAX_INSTANCES=8 run
(loadgen/orchestrate_capacity_sweep.py, already executed in an earlier
pass). This script reuses analyze_v4_corrected.py's exact paired-stats
implementation (Hodges-Lehmann via Walsh averages, matched-pairs
rank-biserial from signed-rank sums, Holm correction) unchanged, so the
capacity=8 numbers are directly comparable in method to Table 3/3b.

Outputs (results/):
  v5_capacity8_family_A_H4_vs_reactive.csv
  v5_capacity8_friedman_omnibus.csv
"""
import os

import pandas as pd
from scipy import stats

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_v4_corrected import family_A_h4, PREDICTORS, LABEL

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "results")


def main():
    tbl = pd.read_csv(os.path.join(OUT_DIR, "trial_level_table_capacity8.csv"))
    tbl = tbl.copy()
    tbl["scenario"] = "cyclical"  # single-scenario capacity-8 run; family_A_h4 filters on this

    # family_A_h4() has a side effect: it unconditionally overwrites
    # results/v4_family_A_H4_vs_reactive.csv (the PRIMARY MAX_INSTANCES=3
    # run's Table 3 source file). Back that file up and restore it right
    # after the call so this capacity=8 reanalysis never clobbers it.
    orig_path = os.path.join(OUT_DIR, "v4_family_A_H4_vs_reactive.csv")
    new_path = os.path.join(OUT_DIR, "v5_capacity8_family_A_H4_vs_reactive.csv")
    with open(orig_path) as f:
        backup = f.read()

    fam_a = family_A_h4(tbl, scenario="cyclical")
    fam_a.to_csv(new_path, index=False)

    with open(orig_path, "w") as f:
        f.write(backup)

    # Friedman omnibus, capacity=8, six policies
    policies = ["container", "serverless_reactive"] + PREDICTORS
    mats = []
    for pol in policies:
        s = tbl[tbl.policy == pol].sort_values("trial")["median_latency_ms"].values
        mats.append(s)
    assert len({len(m) for m in mats}) == 1, "unequal trial counts across policies"
    chi2, p = stats.friedmanchisquare(*mats)
    fried = pd.DataFrame([{
        "scenario": "cyclical_capacity8", "k_policies": len(policies), "n_trials": len(mats[0]),
        "friedman_chi2": float(chi2), "df": len(policies) - 1, "p_value": float(p),
        "significant_at_0.05": bool(p < 0.05),
    }])
    fried.to_csv(os.path.join(OUT_DIR, "v5_capacity8_friedman_omnibus.csv"), index=False)

    print("Friedman omnibus (capacity=8, 6 policies):")
    print(fried.to_string(index=False))
    print()
    print("Family A (H4, capacity=8), Holm-4:")
    print(fam_a[["comparison", "n_pairs", "median_a", "median_b", "median_paired_diff_ms",
                 "hodges_lehmann_ms", "hl_ci95_lo_ms", "hl_ci95_hi_ms", "wilcoxon_W",
                 "p_raw", "p_holm_adjusted", "significant_at_0.05_after_holm4",
                 "rank_biserial_matched"]].to_string(index=False))

    # ranking check: does the ordering among the four predictors change
    # at capacity=8 vs capacity=3? (medians only, descriptive)
    print()
    print("Predictor medians at capacity=8 (ascending = fastest first):")
    meds = {pred: tbl[tbl.policy == pred]["median_latency_ms"].median() for pred in PREDICTORS}
    for pred, m in sorted(meds.items(), key=lambda kv: kv[1]):
        print(f"  {LABEL[pred]:16s} {m:8.2f} ms")


if __name__ == "__main__":
    main()
