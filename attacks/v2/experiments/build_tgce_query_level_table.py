#!/usr/bin/env python3
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


def find_col(df, candidates, required=True):
    cols = list(df.columns)
    lower_map = {c.lower(): c for c in cols}

    for cand in candidates:
        if cand in cols:
            return cand
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]

    if required:
        raise ValueError(
            f"Cannot find any of columns {candidates}. Available columns:\n{cols}"
        )
    return None


def normalize_budget(x):
    if pd.isna(x):
        return x
    s = str(x)
    if s.startswith("B"):
        return s
    try:
        return f"B{int(float(s))}"
    except Exception:
        return s


def as_bool_series(s):
    if s.dtype == bool:
        return s

    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float) > 0

    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .map({
            "true": True,
            "false": False,
            "1": True,
            "0": False,
            "yes": True,
            "no": False,
            "escape": True,
            "escaped": True,
            "detected": False,
            "not_detected": True,
        })
    )


def cliffs_delta(x, y):
    """
    Efficient Cliff's delta:
    delta = P(x > y) - P(x < y)
    Negative means x tends to be smaller than y.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]

    if len(x) == 0 or len(y) == 0:
        return np.nan

    y_sorted = np.sort(y)

    less = np.searchsorted(y_sorted, x, side="left")
    leq = np.searchsorted(y_sorted, x, side="right")

    n_less = less.sum()
    n_greater = len(y) * len(x) - leq.sum()

    return (n_greater - n_less) / (len(x) * len(y))


def bootstrap_median_diff_ci(x, y, n_boot=2000, seed=0, max_per_group=5000):
    """
    Bootstrap CI for median(x) - median(y).
    Uses capped resampling for speed on very large query logs.
    """
    rng = np.random.default_rng(seed)

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]

    if len(x) == 0 or len(y) == 0:
        return np.nan, np.nan

    if len(x) > max_per_group:
        x = rng.choice(x, size=max_per_group, replace=False)
    if len(y) > max_per_group:
        y = rng.choice(y, size=max_per_group, replace=False)

    diffs = np.empty(n_boot, dtype=float)

    for i in range(n_boot):
        xb = rng.choice(x, size=len(x), replace=True)
        yb = rng.choice(y, size=len(y), replace=True)
        diffs[i] = np.median(xb) - np.median(yb)

    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return lo, hi


def bh_correct(pvals):
    """
    Benjamini-Hochberg correction.
    Returns adjusted p-values in original order.
    """
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]

    adjusted = np.empty(n, dtype=float)
    prev = 1.0

    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * n / rank
        prev = min(prev, val)
        adjusted[order[i]] = min(prev, 1.0)

    return adjusted


def fmt_p(p):
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "<0.001"
    return f"{p:.4f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate-csv",
        default="results/escape_basin_analysis/candidate_trials_flat.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="results/tgce_query_level_statistics",
    )
    parser.add_argument(
        "--params",
        default="gap_scale",
        help="Comma-separated parameters, e.g. gap_scale,z_gap",
    )
    parser.add_argument(
        "--methods",
        default="hybrid,random,tpe",
        help="Comma-separated methods to include, or ALL",
    )
    parser.add_argument(
        "--budgets",
        default="10,30,100",
        help="Comma-separated budgets to include, or ALL",
    )
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260920)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.candidate_csv, low_memory=False)

    method_col = find_col(df, ["method", "attack_method"])
    budget_col = find_col(df, ["budget", "B", "query_budget"])
    escaped_col = find_col(
        df,
        [
            "escaped",
            "escape",
            "is_escape",
            "candidate_escaped",
            "frozen_escape",
            "success",
            "is_success",
        ],
    )

    df = df.copy()
    df["_method"] = df[method_col].astype(str).str.lower()
    df["_budget"] = df[budget_col].map(normalize_budget)
    df["_escaped"] = as_bool_series(df[escaped_col])

    before = len(df)
    df = df[df["_escaped"].notna()].copy()
    print(f"Loaded {before:,} rows; kept {len(df):,} rows with valid escape labels.")
    print("Detected columns:")
    print(f"  method  : {method_col}")
    print(f"  budget  : {budget_col}")
    print(f"  escaped : {escaped_col}")

    params = [p.strip() for p in args.params.split(",") if p.strip()]

    methods = None
    if args.methods.upper() != "ALL":
        methods = [m.strip().lower() for m in args.methods.split(",") if m.strip()]

    budgets = None
    if args.budgets.upper() != "ALL":
        budgets = [normalize_budget(b.strip()) for b in args.budgets.split(",") if b.strip()]

    if methods is not None:
        df = df[df["_method"].isin(methods)].copy()

    if budgets is not None:
        df = df[df["_budget"].isin(budgets)].copy()

    rows = []

    for method in sorted(df["_method"].dropna().unique()):
        for budget in sorted(df["_budget"].dropna().unique(), key=lambda x: int(str(x).replace("B", "")) if str(x).replace("B", "").isdigit() else 999999):
            sub = df[(df["_method"] == method) & (df["_budget"] == budget)]

            if len(sub) == 0:
                continue

            for param in params:
                if param not in sub.columns:
                    print(f"[WARN] Missing param column: {param}")
                    continue

                vals = pd.to_numeric(sub[param], errors="coerce")
                tmp = sub.assign(_param_value=vals).dropna(subset=["_param_value"])

                esc = tmp.loc[tmp["_escaped"], "_param_value"].to_numpy(dtype=float)
                fail = tmp.loc[~tmp["_escaped"], "_param_value"].to_numpy(dtype=float)

                if len(esc) == 0 or len(fail) == 0:
                    continue

                esc_median = float(np.median(esc))
                fail_median = float(np.median(fail))
                median_diff = esc_median - fail_median

                ci_lo, ci_hi = bootstrap_median_diff_ci(
                    esc,
                    fail,
                    n_boot=args.n_boot,
                    seed=args.seed,
                )

                delta = cliffs_delta(esc, fail)

                try:
                    _, p = mannwhitneyu(
                        esc,
                        fail,
                        alternative="two-sided",
                        method="asymptotic",
                    )
                except Exception:
                    p = np.nan

                rows.append(
                    {
                        "method": method,
                        "budget": budget,
                        "parameter": param,
                        "n_escaped": int(len(esc)),
                        "n_failed": int(len(fail)),
                        "escaped_median": esc_median,
                        "failed_median": fail_median,
                        "median_diff": median_diff,
                        "ci95_low": ci_lo,
                        "ci95_high": ci_hi,
                        "cliffs_delta": delta,
                        "mannwhitney_p": p,
                    }
                )

    table = pd.DataFrame(rows)

    if table.empty:
        raise RuntimeError("No rows generated. Check column names / method / budget filters.")

    table["bh_corrected_p"] = bh_correct(table["mannwhitney_p"].fillna(1.0).to_numpy())

    # Pretty version for paper.
    pretty = table.copy()
    for c in ["escaped_median", "failed_median", "median_diff", "ci95_low", "ci95_high", "cliffs_delta"]:
        pretty[c] = pretty[c].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")

    pretty["95% CI"] = "[" + pretty["ci95_low"].astype(str) + ", " + pretty["ci95_high"].astype(str) + "]"
    pretty["BH-corrected p"] = table["bh_corrected_p"].map(fmt_p)

    pretty = pretty[
        [
            "method",
            "budget",
            "parameter",
            "n_escaped",
            "n_failed",
            "escaped_median",
            "failed_median",
            "median_diff",
            "95% CI",
            "cliffs_delta",
            "BH-corrected p",
        ]
    ].rename(
        columns={
            "method": "Method",
            "budget": "Budget",
            "parameter": "Parameter",
            "n_escaped": "Escaped n",
            "n_failed": "Failed n",
            "escaped_median": "Escaped median",
            "failed_median": "Failed median",
            "median_diff": "Median diff.",
            "cliffs_delta": "Cliff's δ",
        }
    )

    raw_path = out_dir / "table_tgce_query_level_statistics_raw.csv"
    pretty_path = out_dir / "table_tgce_query_level_statistics_pretty.csv"
    md_path = out_dir / "table_tgce_query_level_statistics.md"

    table.to_csv(raw_path, index=False)
    pretty.to_csv(pretty_path, index=False)

    with open(md_path, "w") as f:
        f.write("# Table X. Query-level evidence of Temporal Gap Compression Escape\n\n")
        f.write(pretty.to_markdown(index=False))
        f.write("\n")

    print("\nSaved:")
    print(raw_path)
    print(pretty_path)
    print(md_path)

    print("\nPreview:")
    print(pretty.to_string(index=False))


if __name__ == "__main__":
    main()
