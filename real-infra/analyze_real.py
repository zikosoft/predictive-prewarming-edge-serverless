"""
Analysis for the real-infrastructure (Kubernetes + Knative) validation
track. Mirrors analysis/analyze.py's trial-as-unit-of-analysis
approach on this smaller run, and additionally produces a side-by-side
table against the process-level emulation's results (if available) to
report whether the corrected finding replicates on genuine
infrastructure.

Run after real-infra/loadgen_real.py has produced results/real/*.csv.

Outputs (results/real/):
  summary_table_real.csv
  trial_level_table_real.csv
  omnibus_posthoc_real.csv
  emulation_vs_real_comparison.csv   (only if results/summary_table.csv exists)
"""
import glob
import os
import re

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL_DIR = os.path.join(ROOT, "results", "real")
EMULATION_SUMMARY = os.path.join(ROOT, "results", "summary_table.csv")

POLICIES = ["container", "serverless_reactive", "serverless_predictive_ewma"]
LABEL = {"container": "Container (real k8s)", "serverless_reactive": "Serverless reactive (real Knative)",
         "serverless_predictive_ewma": "Serverless predictive EWMA (real Knative)"}
SCENARIOS = ["low", "moderate", "cyclical"]
RAW_RE = re.compile(r"raw_(container|serverless_reactive|serverless_predictive_ewma)_(low|moderate|cyclical)_trial(\d+)\.csv")


def load_all():
    frames = []
    for path in glob.glob(os.path.join(REAL_DIR, "raw_*.csv")):
        m = RAW_RE.match(os.path.basename(path))
        if not m:
            continue
        policy, scenario, trial = m.group(1), m.group(2), int(m.group(3))
        df = pd.read_csv(path)
        df["policy"], df["scenario"], df["trial"] = policy, scenario, trial
        frames.append(df)
    if not frames:
        raise SystemExit(f"No raw CSVs found in {REAL_DIR} -- run loadgen_real.py first.")
    full = pd.concat(frames, ignore_index=True)
    full["ok"] = full["ok"].astype(str).str.lower().isin(["true", "1"])
    full["cold"] = pd.to_numeric(full["cold"], errors="coerce").fillna(0).astype(int)
    return full


def main():
    df = load_all()
    summary_rows, trial_rows = [], []
    for scenario in SCENARIOS:
        for policy in POLICIES:
            sub = df[(df.scenario == scenario) & (df.policy == policy)]
            if sub.empty:
                continue
            lat = sub.loc[sub.ok, "latency_ms"]
            n_cold = int((sub["cold"] == 1).sum())
            summary_rows.append({"scenario": scenario, "policy": LABEL[policy], "n_requests": len(sub),
                                 "n_ok": int(sub.ok.sum()), "n_cold_starts": n_cold,
                                 "cold_start_rate_pct": round(100 * n_cold / len(sub), 2) if len(sub) else float("nan"),
                                 "p50_ms": round(np.percentile(lat, 50), 2) if len(lat) else float("nan"),
                                 "p95_ms": round(np.percentile(lat, 95), 2) if len(lat) else float("nan"),
                                 "mean_ms": round(lat.mean(), 2) if len(lat) else float("nan")})
            for trial, g in sub.groupby("trial"):
                glat = g.loc[g.ok, "latency_ms"]
                if len(glat) == 0:
                    continue
                trial_rows.append({"scenario": scenario, "policy": policy, "trial": int(trial),
                                   "median_latency_ms": float(np.median(glat)),
                                   "n_cold": int((g["cold"] == 1).sum())})
    summary = pd.DataFrame(summary_rows)
    trial_tbl = pd.DataFrame(trial_rows)
    summary.to_csv(os.path.join(REAL_DIR, "summary_table_real.csv"), index=False)
    trial_tbl.to_csv(os.path.join(REAL_DIR, "trial_level_table_real.csv"), index=False)

    stat_rows = []
    for scenario in SCENARIOS:
        groups = {p: trial_tbl[(trial_tbl.scenario == scenario) & (trial_tbl.policy == p)]
                  .sort_values("trial")["median_latency_ms"].values for p in POLICIES}
        if any(len(g) < 3 for g in groups.values()):
            continue
        h, p_omni = stats.kruskal(*groups.values())
        stat_rows.append({"scenario": scenario, "test": "kruskal_wallis_omnibus",
                          "statistic": round(float(h), 3), "p_value": p_omni})
        ga, gb = groups["serverless_reactive"], groups["serverless_predictive_ewma"]
        u, p_mw = stats.mannwhitneyu(ga, gb, alternative="two-sided")
        stat_rows.append({"scenario": scenario, "test": "reactive_vs_predictive_mannwhitney",
                          "statistic": round(float(u), 3), "p_value": p_mw})
        if len(ga) == len(gb):
            try:
                w, p_w = stats.wilcoxon(ga, gb)
                stat_rows.append({"scenario": scenario, "test": "reactive_vs_predictive_wilcoxon_paired",
                                  "statistic": round(float(w), 3), "p_value": p_w})
            except ValueError:
                pass
    pd.DataFrame(stat_rows).to_csv(os.path.join(REAL_DIR, "omnibus_posthoc_real.csv"), index=False)

    print("=== REAL-INFRA SUMMARY ==="); print(summary.to_string(index=False))
    print("\n=== REAL-INFRA STATS ==="); print(pd.DataFrame(stat_rows).to_string(index=False))

    if os.path.exists(EMULATION_SUMMARY):
        emu = pd.read_csv(EMULATION_SUMMARY)
        emu_map = {"container": "Container", "serverless_reactive": "Serverless (reactive)",
                   "serverless_predictive_ewma": "Serverless (predictive, EWMA)"}
        rows = []
        for scenario in SCENARIOS:
            for policy in POLICIES:
                real_row = summary[(summary.scenario == scenario) & (summary.policy == LABEL[policy])]
                emu_row = emu[(emu.scenario == scenario) & (emu.policy == emu_map[policy])]
                if real_row.empty or emu_row.empty:
                    continue
                rows.append({
                    "scenario": scenario, "policy": policy,
                    "emulation_p50_ms": emu_row["p50_ms"].values[0],
                    "real_infra_p50_ms": real_row["p50_ms"].values[0],
                    "emulation_cold_rate_pct": emu_row["cold_start_rate_pct"].values[0],
                    "real_infra_cold_rate_pct": real_row["cold_start_rate_pct"].values[0],
                })
        cmp_df = pd.DataFrame(rows)
        cmp_df.to_csv(os.path.join(REAL_DIR, "emulation_vs_real_comparison.csv"), index=False)
        print("\n=== EMULATION vs REAL-INFRA ==="); print(cmp_df.to_string(index=False))
    else:
        print(f"\n(No {EMULATION_SUMMARY} found -- run analysis/analyze.py on the emulation track first "
              "to get a side-by-side comparison table.)")


if __name__ == "__main__":
    main()
