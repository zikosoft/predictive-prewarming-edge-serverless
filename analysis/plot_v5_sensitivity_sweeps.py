"""
Regenerates Figure 7 (fixed-schedule period sensitivity + keep-alive
timeout sensitivity) from the N=15 confirmatory data (A4/A5), replacing the
N=6-pilot-scale version. Same two-panel layout, same visual style, same
axes/legend conventions as the original figure -- only the underlying data
(and the subtitle's N annotation) changes, per the manuscript's figure
policy: this figure's underlying dataset materially changed (pilot -> N=15
confirmatory), so keeping the old image would be factually inconsistent
with the confirmatory numbers now reported in Table 9/10 and the
surrounding text.

Reads: results/v5_fixedsweep_confirmatory_summary.csv,
       results/v5_keepalive_confirmatory_summary.csv
Writes: results/fig_sensitivity_sweeps_v2.png (new file; the original
        results/fig_sensitivity_sweeps.png is left untouched for the record)
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

fx = pd.read_csv(os.path.join(RESULTS, "v5_fixedsweep_confirmatory_summary.csv"))
ka = pd.read_csv(os.path.join(RESULTS, "v5_keepalive_confirmatory_summary.csv"))

pct_map = {"-50pct_4.5s": -50, "-25pct_6.75s": -25, "-10pct_8.1s": -10, "+0pct_9.0s": 0,
           "+10pct_9.9s": 10, "+25pct_11.25s": 25, "+50pct_13.5s": 50}
fx["pct"] = fx["condition"].map(pct_map)
fx = fx.sort_values("pct")

idle_map = {"6.0s": 6.0, "8.0s": 8.0, "10.0s": 10.0, "12.0s": 12.0}
ka["idle_s"] = ka["condition"].map(idle_map)
ka = ka.sort_values("idle_s")
reactive_nominal_ms = ka.loc[ka["idle_s"] == 6.0, "mean_of_trial_median_ms"].iloc[0]

fig, axes = plt.subplots(1, 2, figsize=(13, 5.16))  # matches original figure's embedded aspect ratio (5760720x2286000 EMU)

ax = axes[0]
ax.plot(fx["pct"], fx["mean_of_trial_median_ms"], "o-", color="tab:blue",
        label="Latency (mean-of-trial-median)", linewidth=2, markersize=8)
ax.axhline(reactive_nominal_ms, color="gray", linestyle="--", linewidth=1.3,
           label="Reactive baseline (6.0s, N=15)")
ax.axvline(0, color="lightgray", linestyle=":", linewidth=1)
ax.set_xlabel("Fixed-schedule period, % deviation from disclosed nominal (9.0s)")
ax.set_ylabel("Mean-of-trial-median latency (ms)")
ax.set_title("Fixed-schedule period sensitivity\n(N=15 confirmatory, all conditions)")

ax2 = ax.twinx()
ax2.plot(fx["pct"], fx["opportunity_success_rate_pct"], "s--", color="tab:orange",
         label="Opportunity-success rate", linewidth=2, markersize=8)
ax2.set_ylabel("Opportunity-success rate (%)", color="tab:orange")
ax2.tick_params(axis="y", labelcolor="tab:orange")
ax2.set_ylim(-5, 105)

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc="center left", fontsize=9)

ax = axes[1]
ax.plot(ka["idle_s"], ka["mean_of_trial_median_ms"], "o-", color="tab:blue", linewidth=2, markersize=8)
ax.axvspan(8.0, 10.0, color="navajowhite", alpha=0.4, label="Workload's disclosed idle-gap range (8-10s)")
ax.set_xlabel("Reactive idle (keep-alive) timeout (s)")
ax.set_ylabel("Mean-of-trial-median latency (ms)")
ax.set_title("Keep-alive timeout sensitivity\n(N=15 confirmatory, all conditions)")
ax.legend(loc="upper right", fontsize=9)

plt.tight_layout()
out_path = os.path.join(RESULTS, "fig_sensitivity_sweeps_v2.png")
plt.savefig(out_path, dpi=150)
print(f"Wrote {out_path}")
