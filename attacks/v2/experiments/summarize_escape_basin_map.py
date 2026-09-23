#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import numpy as np

ITEMS = [
    (
        "UI-TARS",
        "B30",
        "z_gap_z_duration",
        "results/mechanism_analysis/escape_basin_map_UITARS_B30_zgap_zduration/escape_basin_points.csv",
    ),
    (
        "UI-TARS",
        "B30",
        "z_gap_z_spatial",
        "results/mechanism_analysis/escape_basin_map_UITARS_B30_zgap_zspatial/escape_basin_points.csv",
    ),
]

out_dir = Path("results/mechanism_analysis")
out_dir.mkdir(parents=True, exist_ok=True)

rows = []

for agent, budget, view, path in ITEMS:
    p = Path(path)
    if not p.exists():
        print("missing:", p)
        continue

    df = pd.read_csv(p)
    df["escaped_bool"] = df["escaped_bool"].astype(bool)

    total = len(df)
    escaped = df[df["escaped_bool"]]
    detected = df[~df["escaped_bool"]]

    low_gap = df["z_gap"] <= 0.20
    compressed_gap = df["z_gap"] <= 0.30
    neutral_high = df["z_gap"] >= 0.50

    def share(mask, subset):
        if len(subset) == 0:
            return np.nan
        return float(mask.loc[subset.index].mean()) * 100

    # Odds ratio for low gap among escaped vs detected.
    a = int((escaped["z_gap"] <= 0.20).sum())
    b = int((escaped["z_gap"] > 0.20).sum())
    c = int((detected["z_gap"] <= 0.20).sum())
    d = int((detected["z_gap"] > 0.20).sum())

    odds_ratio = ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))

    rows.append({
        "agent": agent,
        "budget": budget,
        "view": view,
        "n_total": total,
        "n_escaped": len(escaped),
        "escape_rate_%": round(len(escaped) / total * 100, 2),
        "escaped_median_z_gap": round(float(escaped["z_gap"].median()), 4),
        "detected_median_z_gap": round(float(detected["z_gap"].median()), 4),
        "escaped_share_z_gap_le_0.20_%": round(share(low_gap, escaped), 2),
        "detected_share_z_gap_le_0.20_%": round(share(low_gap, detected), 2),
        "escaped_share_z_gap_le_0.30_%": round(share(compressed_gap, escaped), 2),
        "detected_share_z_gap_le_0.30_%": round(share(compressed_gap, detected), 2),
        "escaped_share_z_gap_ge_0.50_%": round(share(neutral_high, escaped), 2),
        "detected_share_z_gap_ge_0.50_%": round(share(neutral_high, detected), 2),
        "low_gap_odds_ratio_escaped_vs_detected": round(float(odds_ratio), 3),
    })

if not rows:
    raise SystemExit("No basin map points found.")

out = pd.DataFrame(rows)
save = out_dir / "table16_escape_basin_map_quantification.csv"
out.to_csv(save, index=False)

print("=" * 100)
print("TABLE 16 — Escape basin map quantification")
print("=" * 100)
print(out.to_string(index=False))

print("\nSaved:", save)
