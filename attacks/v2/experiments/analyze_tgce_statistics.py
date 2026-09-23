#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


CORE_PARAMS = [
    "gap_scale",
    "z_gap",
    "duration_scale",
    "z_duration",
    "jitter_frac",
    "z_jitter",
    "heterogeneity",
    "z_heterogeneity",
    "spatial_amp_px",
    "z_spatial",
    "frequency",
    "z_frequency",
    "use_spatial",
    "use_temporal",
    "z_use_spatial",
    "z_use_temporal",
]


MAIN_PARAMS = [
    "gap_scale",
    "z_gap",
    "duration_scale",
    "jitter_frac",
    "heterogeneity",
    "spatial_amp_px",
]


def to_float_series(x):
    s = pd.to_numeric(x, errors="coerce")
    return s.astype(float)


def try_mannwhitneyu(x, y):
    """
    Returns U and p-value for x vs y.
    x = escaped values, y = failed values.
    Cliff's delta is computed from U:
      delta = 2U / (nx * ny) - 1
    Negative delta means escaped values tend to be smaller.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]

    if len(x) < 2 or len(y) < 2:
        return np.nan, np.nan, np.nan

    try:
        from scipy.stats import mannwhitneyu

        res = mannwhitneyu(x, y, alternative="two-sided", method="asymptotic")
        u = float(res.statistic)
        p = float(res.pvalue)
    except Exception:
        # Fallback: compute U from ranks, p unavailable.
        combined = np.concatenate([x, y])
        ranks = pd.Series(combined).rank(method="average").to_numpy()
        rx = ranks[: len(x)].sum()
        u = float(rx - len(x) * (len(x) + 1) / 2.0)
        p = np.nan

    delta = 2.0 * u / (len(x) * len(y)) - 1.0
    return u, p, delta


def bh_adjust(pvals):
    """
    Benjamini-Hochberg FDR correction.
    """
    p = np.asarray(pvals, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)

    mask = np.isfinite(p)
    if mask.sum() == 0:
        return q

    idx = np.where(mask)[0]
    p_valid = p[idx]
    order = np.argsort(p_valid)

    ranked = p_valid[order]
    m = len(ranked)

    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)

    q[idx[order]] = adjusted
    return q


def bootstrap_median_diff_ci(
    x,
    y,
    n_boot=1000,
    seed=0,
    ci=95,
    max_per_group=20000,
):
    """
    Bootstrap CI for median(x) - median(y).
    Uses cap for speed on very large query-level tables.
    """
    rng = np.random.default_rng(seed)

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]

    if len(x) < 2 or len(y) < 2:
        return np.nan, np.nan

    if len(x) > max_per_group:
        x = rng.choice(x, size=max_per_group, replace=False)
    if len(y) > max_per_group:
        y = rng.choice(y, size=max_per_group, replace=False)

    vals = np.empty(n_boot, dtype=float)

    nx = len(x)
    ny = len(y)

    for i in range(n_boot):
        xb = rng.choice(x, size=nx, replace=True)
        yb = rng.choice(y, size=ny, replace=True)
        vals[i] = np.median(xb) - np.median(yb)

    alpha = (100 - ci) / 2.0
    lo = np.percentile(vals, alpha)
    hi = np.percentile(vals, 100 - alpha)

    return float(lo), float(hi)


def fmt_p(p):
    if not np.isfinite(p):
        return ""
    if p < 1e-300:
        return "<1e-300"
    if p < 0.001:
        return "<0.001"
    return f"{p:.4g}"


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--candidate-csv",
        default="results/escape_basin_analysis/candidate_trials_flat.csv",
    )
    ap.add_argument(
        "--methods",
        default="hybrid",
        help="Comma-separated methods to analyze. Use all for all methods.",
    )
    ap.add_argument(
        "--budgets",
        default="10,30,100",
        help="Comma-separated budgets to analyze.",
    )
    ap.add_argument(
        "--params",
        default=",".join(CORE_PARAMS),
    )
    ap.add_argument(
        "--n-bootstrap",
        type=int,
        default=1000,
    )
    ap.add_argument(
        "--output-dir",
        default="results/tgce_statistics",
    )

    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.candidate_csv)

    if args.methods.strip().lower() != "all":
        methods = [x.strip() for x in args.methods.split(",") if x.strip()]
        df = df[df["method"].astype(str).isin(methods)].copy()

    budgets = [int(float(x.strip())) for x in args.budgets.split(",") if x.strip()]
    df["budget_int"] = pd.to_numeric(df["budget"], errors="coerce").astype("Int64")
    df = df[df["budget_int"].isin(budgets)].copy()

    params = [x.strip() for x in args.params.split(",") if x.strip()]
    params = [p for p in params if p in df.columns]

    rows = []

    print("=" * 100)
    print("TGCE STATISTICAL ANALYSIS")
    print("=" * 100)
    print("candidate csv:", args.candidate_csv)
    print("methods:", sorted(df["method"].astype(str).unique()))
    print("budgets:", budgets)
    print("params:", params)
    print("n rows:", len(df))

    for (method, budget), g in df.groupby(["method", "budget_int"], dropna=False):
        esc_df = g[g["escaped"] == True]
        fail_df = g[g["escaped"] == False]

        for p in params:
            x = to_float_series(esc_df[p]).dropna().to_numpy()
            y = to_float_series(fail_df[p]).dropna().to_numpy()

            if len(x) < 2 or len(y) < 2:
                continue

            u, pval, delta = try_mannwhitneyu(x, y)
            ci_lo, ci_hi = bootstrap_median_diff_ci(
                x,
                y,
                n_boot=args.n_bootstrap,
                seed=int(budget) + abs(hash(str(method))) % 100000,
            )

            row = {
                "method": method,
                "budget": int(budget),
                "parameter": p,
                "n_escaped": int(len(x)),
                "n_failed": int(len(y)),
                "escaped_median": float(np.median(x)),
                "failed_median": float(np.median(y)),
                "median_diff_escape_minus_failed": float(np.median(x) - np.median(y)),
                "bootstrap_95ci_low": ci_lo,
                "bootstrap_95ci_high": ci_hi,
                "escaped_mean": float(np.mean(x)),
                "failed_mean": float(np.mean(y)),
                "mean_diff_escape_minus_failed": float(np.mean(x) - np.mean(y)),
                "mannwhitney_u": u,
                "p_value": pval,
                "cliffs_delta": delta,
                "escaped_q10": float(np.quantile(x, 0.10)),
                "escaped_q25": float(np.quantile(x, 0.25)),
                "escaped_q75": float(np.quantile(x, 0.75)),
                "escaped_q90": float(np.quantile(x, 0.90)),
                "failed_q10": float(np.quantile(y, 0.10)),
                "failed_q25": float(np.quantile(y, 0.25)),
                "failed_q75": float(np.quantile(y, 0.75)),
                "failed_q90": float(np.quantile(y, 0.90)),
            }

            rows.append(row)

    stats = pd.DataFrame(rows)

    if stats.empty:
        raise SystemExit("No valid parameter statistics produced.")

    stats["p_bh"] = bh_adjust(stats["p_value"].to_numpy())
    stats["p_value_text"] = stats["p_value"].apply(fmt_p)
    stats["p_bh_text"] = stats["p_bh"].apply(fmt_p)

    # Useful ordering: prioritize gap signals, then effect size.
    param_order = {p: i for i, p in enumerate(CORE_PARAMS)}
    stats["_param_order"] = stats["parameter"].map(param_order).fillna(999)
    stats["_abs_delta"] = stats["cliffs_delta"].abs()
    stats = stats.sort_values(
        ["method", "budget", "_param_order", "_abs_delta"],
        ascending=[True, True, True, False],
    ).drop(columns=["_param_order", "_abs_delta"])

    stats.to_csv(out_dir / "table5_tgce_stats_full.csv", index=False)

    main = stats[
        stats["parameter"].isin(MAIN_PARAMS)
    ].copy()

    main = main[
        [
            "method",
            "budget",
            "parameter",
            "n_escaped",
            "n_failed",
            "escaped_median",
            "failed_median",
            "median_diff_escape_minus_failed",
            "bootstrap_95ci_low",
            "bootstrap_95ci_high",
            "cliffs_delta",
            "p_value_text",
            "p_bh_text",
        ]
    ].copy()

    for c in [
        "escaped_median",
        "failed_median",
        "median_diff_escape_minus_failed",
        "bootstrap_95ci_low",
        "bootstrap_95ci_high",
        "cliffs_delta",
    ]:
        main[c] = main[c].astype(float).round(4)

    main.to_csv(out_dir / "table5_tgce_stats_main.csv", index=False)

    with open(out_dir / "table5_tgce_stats_main.md", "w") as f:
        f.write(main.to_markdown(index=False))
        f.write("\n")

    # Gap-only compact table for paper text.
    gap = main[main["parameter"].isin(["gap_scale", "z_gap"])].copy()
    gap.to_csv(out_dir / "table5_tgce_gap_only.csv", index=False)

    with open(out_dir / "table5_tgce_gap_only.md", "w") as f:
        f.write(gap.to_markdown(index=False))
        f.write("\n")

    print("\n" + "=" * 100)
    print("Main parameter table")
    print("=" * 100)
    print(main.to_string(index=False))

    print("\n" + "=" * 100)
    print("Gap-only table")
    print("=" * 100)
    print(gap.to_string(index=False))

    print("\nSaved:")
    print(out_dir / "table5_tgce_stats_full.csv")
    print(out_dir / "table5_tgce_stats_main.csv")
    print(out_dir / "table5_tgce_gap_only.csv")
    print(out_dir / "table5_tgce_stats_main.md")
    print(out_dir / "table5_tgce_gap_only.md")


if __name__ == "__main__":
    main()
