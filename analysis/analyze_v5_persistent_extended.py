"""
A9 -- extends the persistent-state (no-reset, 25-cycle) learning-curve
analysis from EWMA-only to all three remaining predictors (last-gap,
moving-average, fixed-schedule). Reuses the exact per-cycle method already
used for the published EWMA/reactive curve (analyze.py's
learning_curve_table(): per cycle, count of user-facing cold starts,
descriptive only, never fed into the significance tests).

Outputs:
  results/v5_learning_extended_table.csv  (all 5 policies, per-cycle)
  results/v5_learning_extended_summary.csv  (steady-state summary per policy)
  results/fig_learning_curve_v2.png  (5-policy version of Figure 4, same style)
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

LABEL = {
    "serverless_reactive": "Reactive",
    "serverless_predictive_ewma": "EWMA",
    "serverless_predictive_lastgap": "Last-gap",
    "serverless_predictive_movavg": "Moving-average",
    "serverless_predictive_fixed": "Fixed-schedule",
}
SOURCE_DIR = {
    "serverless_reactive": os.path.join(RESULTS, "raw_learning"),
    "serverless_predictive_ewma": os.path.join(RESULTS, "raw_learning"),
    "serverless_predictive_lastgap": os.path.join(RESULTS, "raw_learning_extended"),
    "serverless_predictive_movavg": os.path.join(RESULTS, "raw_learning_extended"),
    "serverless_predictive_fixed": os.path.join(RESULTS, "raw_learning_extended"),
}


def build_table():
    rows = []
    for policy, label in LABEL.items():
        path = os.path.join(SOURCE_DIR[policy], f"raw_learning_{policy}.csv")
        if not os.path.exists(path):
            print(f"[skip] {path} not found")
            continue
        g = pd.read_csv(path)
        g["cold"] = pd.to_numeric(g["cold"], errors="coerce").fillna(0).astype(int)
        for cycle_id, cg in g.groupby("cycle_id"):
            if cycle_id < 0:
                continue
            rows.append({"policy": label, "cycle_index": int(cycle_id) + 1,
                         "n_cold_starts": int((cg["cold"] == 1).sum()), "n_requests": len(cg)})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RESULTS, "v5_learning_extended_table.csv"), index=False)
    return out


def summarize(tbl):
    """Steady-state summary: mean cold starts/cycle over cycles 1-3 (early,
    still-learning window) vs. cycles 4-25 (steady-state window), matching
    the EWMA narrative's own cycle-3 convergence point (Section 5.6)."""
    rows = []
    for label in LABEL.values():
        sub = tbl[tbl.policy == label].sort_values("cycle_index")
        if sub.empty:
            continue
        early = sub[sub.cycle_index <= 3]["n_cold_starts"]
        steady = sub[sub.cycle_index >= 4]["n_cold_starts"]
        rows.append({
            "policy": label,
            "mean_cold_starts_cycles_1_3": round(float(early.mean()), 2) if len(early) else float("nan"),
            "mean_cold_starts_cycles_4_25": round(float(steady.mean()), 2) if len(steady) else float("nan"),
            "min_cold_starts_steady": int(steady.min()) if len(steady) else None,
            "max_cold_starts_steady": int(steady.max()) if len(steady) else None,
            "n_cycles_at_min_steady": int((steady == steady.min()).sum()) if len(steady) else None,
            "n_cycles_total": len(sub),
        })
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RESULTS, "v5_learning_extended_summary.csv"), index=False)
    return out


def plot(tbl):
    fig, ax = plt.subplots(figsize=(9, 5.76))  # matches original Figure 4's embedded aspect ratio (4572000x2926080 EMU)
    for label, marker in zip(LABEL.values(), ["o", "s", "^", "D", "v"]):
        sub = tbl[tbl.policy == label].sort_values("cycle_index")
        if sub.empty:
            continue
        ax.plot(sub["cycle_index"], sub["n_cold_starts"], marker=marker, markersize=5,
                linewidth=1.5, label=label)
    ax.set_xlabel("Cycle index (25-cycle persistent-state run, no trial resets)")
    ax.set_ylabel("User-facing cold starts per cycle")
    ax.set_title("Persistent-state learning curve, all five policies")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_path = os.path.join(RESULTS, "fig_learning_curve_v2.png")
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    tbl = build_table()
    summ = summarize(tbl)
    print(summ.to_string(index=False))
    plot(tbl)
