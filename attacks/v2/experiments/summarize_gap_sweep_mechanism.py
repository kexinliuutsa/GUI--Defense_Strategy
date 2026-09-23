#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import numpy as np

ITEMS = [
    ("UI-TARS", "B30", "results/mechanism_analysis/gap_sweep_UITARS_B30/gap_sweep_summary.csv"),
    ("UI-TARS", "B100", "results/mechanism_analysis/gap_sweep_UITARS_B100/gap_sweep_summary.csv"),
    ("AgentCPM", "B30", "results/mechanism_analysis/gap_sweep_AgentCPM_B30/gap_sweep_summary.csv"),
    ("AutoGLM", "B30", "results/mechanism_analysis/gap_sweep_AutoGLM_B30/gap_sweep_summary.csv"),
]

out_dir = Path("results/mechanism_analysis")
out_dir.mkdir(parents=True, exist_ok=True)

all_rows = []
summary_rows = []

for agent, budget, path in ITEMS:
    p = Path(path)
    if not p.exists():
        print("missing:", p)
        continue

    df = pd.read_csv(p)
    df.insert(0, "agent", agent)
    df.insert(1, "budget", budget)
    all_rows.append(df)

    low = df[df["sweep_z_gap"] <= 0.20]
    mid = df[(df["sweep_z_gap"] >= 0.45) & (df["sweep_z_gap"] <= 0.55)]
    high = df[df["sweep_z_gap"] >= 0.80]

    low_escape = float(low["escape_rate_pct"].mean()) if len(low) else np.nan
    mid_escape = float(mid["escape_rate_pct"].mean()) if len(mid) else np.nan
    high_escape = float(high["escape_rate_pct"].mean()) if len(high) else np.nan

    first = float(df.sort_values("sweep_z_gap")["escape_rate_pct"].iloc[0])
    last = float(df.sort_values("sweep_z_gap")["escape_rate_pct"].iloc[-1])

    # Spearman trend without requiring scipy.
    corr = df["sweep_z_gap"].rank().corr(df["escape_rate_pct"].rank())

    summary_rows.append({
        "agent": agent,
        "budget": budget,
        "n_per_gap": int(df["n"].median()),
        "escape_at_z0_%": first,
        "escape_at_z1_%": last,
        "low_gap_mean_escape_%": round(low_escape, 2),
        "mid_gap_mean_escape_%": round(mid_escape, 2),
        "high_gap_mean_escape_%": round(high_escape, 2),
        "low_minus_high_pp": round(low_escape - high_escape, 2),
        "low_minus_mid_pp": round(low_escape - mid_escape, 2),
        "spearman_zgap_escape": round(float(corr), 4),
    })

if not all_rows:
    raise SystemExit("No gap sweep summaries found.")

all_df = pd.concat(all_rows, ignore_index=True)
sum_df = pd.DataFrame(summary_rows)

all_save = out_dir / "table14_gap_sweep_all_points.csv"
sum_save = out_dir / "table14_gap_sweep_mechanism_summary.csv"

all_df.to_csv(all_save, index=False)
sum_df.to_csv(sum_save, index=False)

print("=" * 100)
print("TABLE 14 — Gap sweep mechanism summary")
print("=" * 100)
print(sum_df.to_string(index=False))

print("\nSaved:")
print(all_save)
print(sum_save)
