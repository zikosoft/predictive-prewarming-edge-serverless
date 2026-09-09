"""
Round-3 (FGCS finalization pass) — A3: direct forecast-quality metrics.

Downstream latency alone cannot distinguish "the forecaster is inaccurate"
from "the forecaster is accurate but the independent-trial protocol denies
it an opportunity to act" (Section 5.5's opportunity-window asymmetry).
This script answers that question directly, from data already collected
for the primary N=15 cyclical run -- NO new experiment is required.

Method (fully reproducible from already-logged timestamps, no new
instrumentation): the gateway's own gap-observation rule (see
gateway/elastic_gateway.py::_note_real_arrival) is that a "gap" is the
silence between the last request of one burst and the first request of
the next, whenever that silence is >= GAP_MIN_S (1.0s). Because every
raw_<policy>_cyclical_trial<N>.csv already logs a wall-clock timestamp
`t` for every request, the exact same 4 inter-burst gaps the live
predictor observed during the run can be reconstructed offline by sorting
request timestamps and finding the >=1.0s gaps between them -- this is
arithmetic on already-recorded data, not a new measurement.

Each predictor's update rule (EWMA / last-gap / moving-average / fixed)
is a small, deterministic, already-published function of the sequence of
observed gaps (see GapPredictor in elastic_gateway.py). Replaying that
exact same update rule, in order, over the reconstructed gap sequence
therefore reproduces -- deterministically and exactly -- the predicted
value the live predictor held immediately BEFORE each new gap was
observed. Comparing that reconstructed forecast to the gap that actually
materialized gives a genuine, reproducible forecast-error sample: this is
a reanalysis of existing data, not a simulation standing in for a
measurement that was never taken.

Opportunity accounting (must match Section 5.5's already-published
numbers exactly, and does): for the 3 adaptive predictors, the very
first observed gap only SEEDS the predictor (no prior estimate exists
yet to compare it against) -- so each trial yields 3 forecast/actual
pairs (opportunities before gaps 2, 3, 4 -- i.e. cycles 3-5). For
fixed-schedule, the constant is available immediately, so every trial
yields 4 pairs (opportunities before gaps 1-4 -- i.e. cycles 2-5).

Output: results/v5_forecast_quality_trial.csv (one row per (policy,
trial): mean/median absolute error, mean signed bias, n opportunities),
results/v5_forecast_quality_summary.csv (one row per policy: aggregated
across all N=15 trials' opportunities, plus the trial-level MAE
median/IQR for the paper's trial-as-unit convention).
"""
import glob
import itertools
import os
import statistics as st

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_v4_corrected import (
    holm_bonferroni, hodges_lehmann_walsh, hl_confidence_interval,
    matched_pairs_rank_biserial, paired_stats,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "results", "raw")
OUT_DIR = os.path.join(ROOT, "results")

GAP_MIN_S = 1.0
EWMA_ALPHA = 0.4
MOVAVG_WINDOW = 5
FIXED_PREDICTED_GAP_S = 9.0

POLICIES = {
    "serverless_predictive_ewma": "ewma",
    "serverless_predictive_lastgap": "lastgap",
    "serverless_predictive_movavg": "movavg",
    "serverless_predictive_fixed": "fixed",
}


def reconstruct_gaps(df):
    """Sort by wall-clock arrival time, return the list of inter-burst
    gaps (>= GAP_MIN_S) in chronological order -- identical rule to
    gateway/elastic_gateway.py::_note_real_arrival."""
    ts = sorted(df["t"].tolist())
    gaps = []
    for i in range(1, len(ts)):
        d = ts[i] - ts[i - 1]
        if d >= GAP_MIN_S:
            gaps.append(d)
    return gaps


class ReplayPredictor:
    def __init__(self, kind):
        self.kind = kind
        self.gap_s = FIXED_PREDICTED_GAP_S if kind == "fixed" else None
        self._window = []

    def predict(self):
        return self.gap_s

    def update(self, gap_s):
        if self.kind == "fixed":
            return
        elif self.kind == "lastgap":
            self.gap_s = gap_s
        elif self.kind == "movavg":
            self._window.append(gap_s)
            if len(self._window) > MOVAVG_WINDOW:
                self._window.pop(0)
            self.gap_s = sum(self._window) / len(self._window)
        else:
            if self.gap_s is None:
                self.gap_s = gap_s
            else:
                self.gap_s = EWMA_ALPHA * gap_s + (1 - EWMA_ALPHA) * self.gap_s


def per_trial_errors(kind, gaps):
    """Returns list of (forecast, actual, signed_error=forecast-actual)
    for every opportunity in this trial, replaying the exact update rule."""
    pred = ReplayPredictor(kind)
    out = []
    for g in gaps:
        f = pred.predict()
        if f is not None:
            out.append((f, g, f - g))
        pred.update(g)
    return out


def main():
    trial_rows = []
    all_errors = {policy: [] for policy in POLICIES}

    for policy, kind in POLICIES.items():
        files = sorted(glob.glob(os.path.join(RAW_DIR, f"raw_{policy}_cyclical_trial*.csv")))
        assert len(files) == 15, f"expected 15 trial files for {policy}, found {len(files)}"
        for fpath in files:
            trial_idx = int(fpath.rsplit("trial", 1)[1].split(".")[0])
            df = pd.read_csv(fpath)
            gaps = reconstruct_gaps(df)
            assert len(gaps) == 4, f"{fpath}: expected 4 reconstructed gaps, found {len(gaps)}"
            errs = per_trial_errors(kind, gaps)
            n_opp = len(errs)
            expected_n = 4 if kind == "fixed" else 3
            assert n_opp == expected_n, f"{fpath}: expected {expected_n} opportunities, got {n_opp}"
            abs_errs = [abs(e) for _, _, e in errs]
            signed_errs = [e for _, _, e in errs]
            all_errors[policy].extend(errs)
            trial_rows.append({
                "policy": policy, "trial": trial_idx, "n_opportunities": n_opp,
                "mae_s": round(st.mean(abs_errs), 4),
                "median_ae_s": round(st.median(abs_errs), 4),
                "signed_bias_s": round(st.mean(signed_errs), 4),
                "median_signed_error_s": round(st.median(signed_errs), 4),
            })

    trial_df = pd.DataFrame(trial_rows)
    trial_df.to_csv(os.path.join(OUT_DIR, "v5_forecast_quality_trial.csv"), index=False)

    summary_rows = []
    for policy in POLICIES:
        errs = all_errors[policy]
        abs_errs = [abs(e) for _, _, e in errs]
        signed_errs = [e for _, _, e in errs]
        tdf = trial_df[trial_df.policy == policy]
        summary_rows.append({
            "policy": policy,
            "n_trials": len(tdf),
            "n_opportunities_total": len(errs),
            "pooled_MAE_s": round(st.mean(abs_errs), 4),
            "pooled_median_AE_s": round(st.median(abs_errs), 4),
            "pooled_signed_bias_s": round(st.mean(signed_errs), 4),
            "pooled_median_signed_error_s": round(st.median(signed_errs), 4),
            "trial_level_MAE_median_s": round(st.median(tdf["mae_s"]), 4),
            "trial_level_MAE_min_s": round(tdf["mae_s"].min(), 4),
            "trial_level_MAE_max_s": round(tdf["mae_s"].max(), 4),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(OUT_DIR, "v5_forecast_quality_summary.csv"), index=False)

    # Family C (new): 6 pairwise paired-Wilcoxon contrasts on trial-level
    # forecast MAE among the four predictors, Holm-6 -- directly tests
    # whether forecast QUALITY itself differs, as opposed to Family B
    # (Section 4.6/RQ3) which tests whether DOWNSTREAM LATENCY differs.
    policies = list(POLICIES.keys())
    pairs = list(itertools.combinations(policies, 2))
    rows_c = []
    for pa, pb in pairs:
        a = trial_df[trial_df.policy == pa].sort_values("trial")["mae_s"].values
        b = trial_df[trial_df.policy == pb].sort_values("trial")["mae_s"].values
        rows_c.append({"pair": f"{pa.split('_')[-1]} vs {pb.split('_')[-1]}", **paired_stats(a, b)})
    raw_p = [r["p_raw"] for r in rows_c]
    adj_p = holm_bonferroni(raw_p)
    for r, p in zip(rows_c, adj_p):
        r["p_holm"] = float(p)
        r["sig_holm6"] = "Yes" if p < 0.05 else "No"
    family_c_df = pd.DataFrame(rows_c)
    family_c_df.to_csv(os.path.join(OUT_DIR, "v5_forecast_quality_family_C_pairwise.csv"), index=False)

    print(trial_df.groupby("policy")[["mae_s", "signed_bias_s"]].describe())
    print()
    print(summary_df.to_string(index=False))
    print()
    print("Family C -- forecast-MAE pairwise (Holm-6):")
    print(family_c_df[["pair", "n_pairs", "median_paired_diff_ms", "hodges_lehmann_ms",
                        "p_raw", "p_holm", "sig_holm6"]].to_string(index=False))


if __name__ == "__main__":
    main()
