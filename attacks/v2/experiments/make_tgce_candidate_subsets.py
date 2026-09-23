#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import pandas as pd


def parse_bool(x):
    return str(x).strip().lower() in {"true", "1", "yes", "escaped", "success"}


def main():
    in_path = Path("results/escape_basin_analysis/candidate_trials_flat.csv")
    out_dir = Path("results/tgce_candidate_subsets")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path, low_memory=False)

    df = df[
        (df["method"].astype(str) == "hybrid")
        & (pd.to_numeric(df["budget"], errors="coerce").isin([10, 30]))
    ].copy()

    df["escaped_bool"] = df["escaped"].apply(parse_bool)
    df["z_gap_float"] = pd.to_numeric(df["z_gap"], errors="coerce")

    esc = df[df["escaped_bool"]].dropna(subset=["z_gap_float"]).copy()

    rows_low = []
    rows_high = []
    rows_mid = []

    for b, g in esc.groupby(pd.to_numeric(esc["budget"], errors="coerce").astype(int)):
        low = g.sort_values("z_gap_float", ascending=True).head(200)
        high = g.sort_values("z_gap_float", ascending=False).head(200)

        q40 = g["z_gap_float"].quantile(0.40)
        q60 = g["z_gap_float"].quantile(0.60)
        mid = g[(g["z_gap_float"] >= q40) & (g["z_gap_float"] <= q60)].head(200)

        rows_low.append(low)
        rows_high.append(high)
        rows_mid.append(mid)

    low_df = pd.concat(rows_low, ignore_index=True)
    high_df = pd.concat(rows_high, ignore_index=True)
    mid_df = pd.concat(rows_mid, ignore_index=True)

    low_df.to_csv(out_dir / "hybrid_escaped_low_gap.csv", index=False)
    high_df.to_csv(out_dir / "hybrid_escaped_high_gap.csv", index=False)
    mid_df.to_csv(out_dir / "hybrid_escaped_mid_gap.csv", index=False)

    print("low gap:")
    print(low_df.groupby("budget").agg(n=("z_gap_float","size"), median_z_gap=("z_gap_float","median")).to_string())

    print("\nhigh gap:")
    print(high_df.groupby("budget").agg(n=("z_gap_float","size"), median_z_gap=("z_gap_float","median")).to_string())

    print("\nmid gap:")
    print(mid_df.groupby("budget").agg(n=("z_gap_float","size"), median_z_gap=("z_gap_float","median")).to_string())

    print("\nSaved:")
    print(out_dir / "hybrid_escaped_low_gap.csv")
    print(out_dir / "hybrid_escaped_high_gap.csv")
    print(out_dir / "hybrid_escaped_mid_gap.csv")


if __name__ == "__main__":
    main()
