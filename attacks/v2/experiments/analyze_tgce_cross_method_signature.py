#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


OUT_DIR = Path("results/tgce_cross_method")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_bool(x):
    return str(x).strip().lower() in {"true", "1", "yes", "escaped", "success"}


def cliffs_delta(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) == 0 or len(y) == 0:
        return np.nan

    # Efficient enough for sampled large arrays.
    max_n = 5000
    rng = np.random.default_rng(0)

    if len(x) > max_n:
        x = rng.choice(x, size=max_n, replace=False)
    if len(y) > max_n:
        y = rng.choice(y, size=max_n, replace=False)

    gt = 0
    lt = 0

    for xi in x:
        gt += np.sum(xi > y)
        lt += np.sum(xi < y)

    return float((gt - lt) / (len(x) * len(y)))


def mannwhitney_p(x, y):
    try:
        from scipy.stats import mannwhitneyu
        return float(mannwhitneyu(x, y, alternative="two-sided").pvalue)
    except Exception:
        return np.nan


def bh_adjust(pvals):
    pvals = np.asarray(pvals, dtype=float)
    out = np.full_like(pvals, np.nan, dtype=float)

    valid = np.isfinite(pvals)
    pv = pvals[valid]
    m = len(pv)

    if m == 0:
        return out

    order = np.argsort(pv)
    ranked = pv[order]

    adj = np.empty(m)
    running = 1.0

    for i in range(m - 1, -1, -1):
        rank = i + 1
        running = min(running, ranked[i] * m / rank)
        adj[i] = running

    restored = np.empty(m)
    restored[order] = adj
    out[valid] = restored

    return out


def fmt_p(p):
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "<0.001"
    return f"{p:.4g}"


def main():
    in_path = Path("results/escape_basin_analysis/candidate_trials_flat.csv")

    if not in_path.exists():
        raise SystemExit(f"Missing file: {in_path}")

    df = pd.read_csv(in_path, low_memory=False)

    required = ["method", "budget", "escaped", "z_gap", "gap_scale"]
    for c in required:
        if c not in df.columns:
            raise SystemExit(f"Missing required column: {c}")

    df["escaped_bool"] = df["escaped"].apply(parse_bool)
    df["budget_int"] = pd.to_numeric(df["budget"], errors="coerce").astype("Int64")
    df["z_gap_float"] = pd.to_numeric(df["z_gap"], errors="coerce")
    df["gap_scale_float"] = pd.to_numeric(df["gap_scale"], errors="coerce")

    methods = ["hybrid", "random", "tpe"]
    budgets = [10, 30, 100, 300]

    df = df[
        df["method"].astype(str).isin(methods)
        & df["budget_int"].isin(budgets)
    ].dropna(subset=["z_gap_float", "gap_scale_float"]).copy()

    rows = []

    for (method, budget), g in df.groupby(["method", "budget_int"]):
        esc = g[g["escaped_bool"]].copy()
        fail = g[~g["escaped_bool"]].copy()

        if len(esc) == 0 or len(fail) == 0:
            continue

        x = esc["gap_scale_float"].to_numpy()
        y = fail["gap_scale_float"].to_numpy()

        p = mannwhitney_p(x, y)
        delta = cliffs_delta(x, y)

        rows.append({
            "method": method,
            "budget": int(budget),
            "n_trials": int(len(g)),
            "n_escaped": int(len(esc)),
            "n_failed": int(len(fail)),
            "query_escape_rate_%": round(float(len(esc) / len(g) * 100), 3),

            "escaped_gap_scale_median": round(float(np.median(x)), 4),
            "failed_gap_scale_median": round(float(np.median(y)), 4),
            "median_diff_escaped_minus_failed": round(float(np.median(x) - np.median(y)), 4),

            "escaped_z_gap_median": round(float(esc["z_gap_float"].median()), 4),
            "failed_z_gap_median": round(float(fail["z_gap_float"].median()), 4),

            "escaped_share_gap_scale_le_090_%": round(float((esc["gap_scale_float"] <= 0.90).mean() * 100), 2),
            "failed_share_gap_scale_le_090_%": round(float((fail["gap_scale_float"] <= 0.90).mean() * 100), 2),

            "escaped_share_z_gap_le_020_%": round(float((esc["z_gap_float"] <= 0.20).mean() * 100), 2),
            "failed_share_z_gap_le_020_%": round(float((fail["z_gap_float"] <= 0.20).mean() * 100), 2),

            "cliffs_delta_gap_scale": round(delta, 4),
            "mannwhitney_p": p,
        })

    out = pd.DataFrame(rows)

    if out.empty:
        raise SystemExit("No rows generated.")

    out["mannwhitney_p_bh"] = bh_adjust(out["mannwhitney_p"].to_numpy())
    out["mannwhitney_p_text"] = out["mannwhitney_p"].apply(fmt_p)
    out["mannwhitney_p_bh_text"] = out["mannwhitney_p_bh"].apply(fmt_p)

    out = out.sort_values(["budget", "method"]).reset_index(drop=True)

    out.to_csv(OUT_DIR / "table10a_cross_method_tgce_gap_signature.csv", index=False)

    with open(OUT_DIR / "table10a_cross_method_tgce_gap_signature.md", "w") as f:
        f.write(out.to_markdown(index=False))
        f.write("\n")

    print("=" * 100)
    print("TABLE 10A — Cross-method TGCE gap signature")
    print("=" * 100)
    print(out.to_string(index=False))

    print("\nSaved:")
    print(OUT_DIR / "table10a_cross_method_tgce_gap_signature.csv")
    print(OUT_DIR / "table10a_cross_method_tgce_gap_signature.md")


if __name__ == "__main__":
    main()
