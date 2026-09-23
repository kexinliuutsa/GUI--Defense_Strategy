#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_bool(x):
    return str(x).strip().lower() in {"true", "1", "yes", "y", "escaped", "success"}


def normalize_agent(x):
    s = str(x).lower()
    if "ui-tars" in s or "uitars" in s:
        return "UI-TARS"
    if "gpt4o" in s or "gpt-4o" in s:
        return "GPT-4o"
    if "claude" in s:
        return "Claude"
    if "cpm" in s:
        return "AgentCPM"
    if "autoglm" in s or "glm" in s:
        return "AutoGLM"
    return str(x)


def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--candidate-csv",
        default="results/escape_basin_analysis/candidate_trials_flat.csv",
    )
    ap.add_argument("--method", default="hybrid")
    ap.add_argument("--budget", type=int, default=30)
    ap.add_argument("--agents", default="UI-TARS")
    ap.add_argument("--x", default="z_gap")
    ap.add_argument("--y", default="z_duration")
    ap.add_argument("--max-points", type=int, default=20000)
    ap.add_argument("--output-dir", default="results/mechanism_analysis/escape_basin_map_UITARS_B30")

    return ap.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.candidate_csv, low_memory=False)
    df["escaped_bool"] = df["escaped"].apply(parse_bool)
    df["budget_int"] = pd.to_numeric(df["budget"], errors="coerce").astype("Int64")
    df["agent"] = df["participant"].apply(normalize_agent)

    agents = {x.strip() for x in args.agents.split(",") if x.strip()}

    sub = df[
        (df["method"].astype(str) == args.method)
        & (df["budget_int"] == args.budget)
        & (df["agent"].isin(agents))
    ].copy()

    if sub.empty:
        raise SystemExit("No rows after filter.")

    for c in [args.x, args.y]:
        if c not in sub.columns:
            raise SystemExit(f"Missing column: {c}")

    sub[args.x] = pd.to_numeric(sub[args.x], errors="coerce")
    sub[args.y] = pd.to_numeric(sub[args.y], errors="coerce")
    sub = sub.dropna(subset=[args.x, args.y, "escaped_bool"])

    if len(sub) > args.max_points:
        sub = sub.sample(args.max_points, random_state=20260918)

    sub.to_csv(out_dir / "escape_basin_points.csv", index=False)

    detected = sub[~sub["escaped_bool"]]
    escaped = sub[sub["escaped_bool"]]

    plt.figure(figsize=(7, 5.5))
    plt.scatter(detected[args.x], detected[args.y], s=6, alpha=0.25, label="detected")
    plt.scatter(escaped[args.x], escaped[args.y], s=10, alpha=0.75, label="escaped")
    plt.xlabel(args.x)
    plt.ylabel(args.y)
    plt.title(f"Escape basin map ({args.method}, B{args.budget}, {args.agents})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"figure_escape_basin_{args.x}_{args.y}.png", dpi=300)
    plt.savefig(out_dir / f"figure_escape_basin_{args.x}_{args.y}.pdf")
    plt.close()

    # Binned heatmap: escape rate over x/y bins
    sub["_x_bin"] = pd.cut(sub[args.x], bins=np.linspace(0, 1, 21), include_lowest=True)
    sub["_y_bin"] = pd.cut(sub[args.y], bins=np.linspace(0, 1, 21), include_lowest=True)

    heat = (
        sub.groupby(["_x_bin", "_y_bin"], observed=True)
        .agg(
            n=("escaped_bool", "size"),
            escape_rate=("escaped_bool", "mean"),
        )
        .reset_index()
    )

    heat.to_csv(out_dir / "escape_basin_heatmap_bins.csv", index=False)

    pivot = heat.pivot(index="_y_bin", columns="_x_bin", values="escape_rate")
    mat = pivot.to_numpy(dtype=float)

    plt.figure(figsize=(7, 5.5))
    plt.imshow(mat, origin="lower", aspect="auto")
    plt.colorbar(label="escape rate")
    plt.xlabel(args.x + " bins")
    plt.ylabel(args.y + " bins")
    plt.title(f"Binned escape rate ({args.method}, B{args.budget}, {args.agents})")
    plt.tight_layout()
    plt.savefig(out_dir / f"figure_escape_rate_heatmap_{args.x}_{args.y}.png", dpi=300)
    plt.savefig(out_dir / f"figure_escape_rate_heatmap_{args.x}_{args.y}.pdf")
    plt.close()

    print("=" * 100)
    print("ESCAPE BASIN MAP")
    print("=" * 100)
    print("rows:", len(sub))
    print("escaped:", int(sub["escaped_bool"].sum()))
    print("escape rate:", round(float(sub["escaped_bool"].mean()) * 100, 2), "%")

    print("\nSaved:")
    print(out_dir / "escape_basin_points.csv")
    print(out_dir / f"figure_escape_basin_{args.x}_{args.y}.png")
    print(out_dir / f"figure_escape_rate_heatmap_{args.x}_{args.y}.png")


if __name__ == "__main__":
    main()
