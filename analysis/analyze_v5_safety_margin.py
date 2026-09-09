"""
A10 -- pre-warm safety-margin (STARTUP_LEAD_S) sensitivity sweep. Explicitly
sanctioned by the revision plan as a smaller sensitivity study (N=6/condition)
rather than a full N=15 primary experiment, so this analysis is descriptive
only, matching the treatment the original Section 8.2 pilot sweeps received
before A4/A5 upgraded them -- no new inferential family is introduced here.

Reuses per_trial_summary/load_raw_trials (analyze_v4d_sweeps.py). The
nominal (1.0s) margin point is reused from the already-collected N=15
primary run for each predictor (not re-run), matching the same
reused-nominal-point convention as every other sweep in this study.

Outputs:
  results/v5_marginsweep_descriptive.csv
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_v4d_sweeps import load_raw_trials, per_trial_summary  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

LABEL = {
    "serverless_predictive_ewma": "EWMA", "serverless_predictive_lastgap": "Last-gap",
    "serverless_predictive_movavg": "Moving-average", "serverless_predictive_fixed": "Fixed-schedule",
}
PREDICTORS = list(LABEL.keys())
MARGINS = [0.1, 0.25, 0.5, 1.0, 1.5]


def main():
    frames = []
    for pred in PREDICTORS:
        pattern = os.path.join(RESULTS, "raw_marginsweep", "*", pred, f"raw_{pred}_cyclical_trial*.csv")
        df = load_raw_trials(pattern)
        if not df.empty:
            # path shape: results/raw_marginsweep/<margin>s/<policy>/raw_..._trial<N>.csv
            # load_raw_trials' own "condition" column takes the immediate
            # parent dir (the policy name here, since two levels are
            # nested), so extract the margin from the full source path
            # instead of relying on "condition".
            df["margin"] = df["__source_file"].str.extract(r"raw_marginsweep/([\d.]+)s/")[0].astype(float)
            df["policy"] = pred
            frames.append(df)
        nominal = load_raw_trials(os.path.join(RESULTS, "raw", f"raw_{pred}_cyclical_trial*.csv"))
        nominal["margin"] = 1.0
        nominal["policy"] = pred
        frames.append(nominal)
    all_raw = pd.concat(frames, ignore_index=True)

    summ = per_trial_summary(all_raw, ["policy", "margin", "__source_file"])
    desc = summ.groupby(["policy", "margin"]).agg(
        n_trials=("median_ms", "count"),
        mean_of_trial_median_ms=("median_ms", "mean"),
        mean_p95_ms=("p95_ms", "mean"),
        mean_p99_ms=("p99_ms", "mean"),
        mean_cold_start_rate=("cold_start_rate", "mean"),
    ).reset_index()
    desc["policy_label"] = desc["policy"].map(LABEL)
    desc = desc.sort_values(["policy", "margin"])
    pd.set_option("display.width", 200)
    print(desc[["policy_label", "margin", "n_trials", "mean_of_trial_median_ms", "mean_p95_ms",
                "mean_p99_ms", "mean_cold_start_rate"]].to_string(index=False))
    desc.to_csv(os.path.join(RESULTS, "v5_marginsweep_descriptive.csv"), index=False)


if __name__ == "__main__":
    main()
