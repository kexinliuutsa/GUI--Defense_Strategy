#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_AHB_ROOT = Path("/storage/cjh/self-evolving-safety/Passing-the-Turing-Test-on-Screen-Agent-Humanization-Benchmark-main")
if _AHB_ROOT.exists() and str(_AHB_ROOT) not in sys.path:
    sys.path.insert(0, str(_AHB_ROOT))

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def to_float_series(x):
    return pd.to_numeric(x, errors="coerce").astype(float)


def load_existing_candidates(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p)
    return df


def load_loop_candidates_from_dirs(input_dirs):
    """
    Reuse the previous parser if available.
    It can read attack_history.jsonl and sometimes best candidates from session_results.csv.
    """
    try:
        from attacks.v2.experiments.analyze_escape_basin_parameters import load_all_trials
    except Exception as e:
        print("Could not import load_all_trials from analyze_escape_basin_parameters.py")
        print("error:", repr(e))
        return pd.DataFrame()

    return load_all_trials(input_dirs)


def mannwhitney_delta(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]

    if len(x) < 2 or len(y) < 2:
        return np.nan, np.nan

    try:
        from scipy.stats import mannwhitneyu
        res = mannwhitneyu(x, y, alternative="two-sided", method="asymptotic")
        u = float(res.statistic)
        p = float(res.pvalue)
        delta = 2.0 * u / (len(x) * len(y)) - 1.0
        return p, delta
    except Exception:
        return np.nan, np.nan


def summarize_escape_gap(df, label, budgets, compressed_threshold):
    rows = []

    if df.empty:
        return pd.DataFrame()

    df = df.copy()

    df["budget_int"] = pd.to_numeric(df["budget"], errors="coerce").astype("Int64")
    df = df[df["budget_int"].isin(budgets)].copy()

    if "gap_scale" not in df.columns:
        return pd.DataFrame()

    if "escaped" not in df.columns:
        return pd.DataFrame()

    esc = df[df["escaped"] == True].copy()

    for b, g in esc.groupby("budget_int", dropna=False):
        gap = to_float_series(g["gap_scale"]).dropna()

        if len(gap) == 0:
            continue

        rows.append({
            "condition": label,
            "budget": int(b),
            "n_escaped_candidates_with_gap": int(len(gap)),
            "escaped_gap_median": float(gap.median()),
            "escaped_gap_q25": float(gap.quantile(0.25)),
            "escaped_gap_q75": float(gap.quantile(0.75)),
            "compressed_gap_threshold": float(compressed_threshold),
            "compressed_gap_escape_share_%": float((gap <= compressed_threshold).mean() * 100),
            "escaped_cost_median": (
                float(to_float_series(g.loc[gap.index, "cost"]).dropna().median())
                if "cost" in g.columns else np.nan
            ),
            "source_level": "query_or_best_candidate",
        })

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--frozen-candidate-csv",
        default="results/escape_basin_analysis/candidate_trials_flat.csv",
    )
    ap.add_argument(
        "--loop-fpr5-dirs",
        default="results/v2_defense_loop_v1_FPR5_B10_B30_N100_3seeds",
    )
    ap.add_argument(
        "--loop-fpr10-dirs",
        default="results/v2_defense_loop_v1_FPR10_B10_B30_N100_3seeds",
    )
    ap.add_argument(
        "--budgets",
        default="10,30",
    )
    ap.add_argument(
        "--compressed-threshold",
        type=float,
        default=0.90,
        help="Defines the compressed-gap basin. 0.90 is conservative based on escaped q75 near 0.88-0.90.",
    )
    ap.add_argument(
        "--output-dir",
        default="results/tgce_loop_shift",
    )

    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    budgets = [int(float(x.strip())) for x in args.budgets.split(",") if x.strip()]

    print("=" * 100)
    print("TGCE LOOP SHIFT ANALYSIS")
    print("=" * 100)

    frozen = load_existing_candidates(args.frozen_candidate_csv)
    frozen = frozen[frozen["method"].astype(str) == "hybrid"].copy()

    fpr5_dirs = [x.strip() for x in args.loop_fpr5_dirs.split(",") if x.strip()]
    fpr10_dirs = [x.strip() for x in args.loop_fpr10_dirs.split(",") if x.strip()]

    loop5 = load_loop_candidates_from_dirs(fpr5_dirs)
    loop10 = load_loop_candidates_from_dirs(fpr10_dirs)

    if loop5.empty:
        print("\nWARNING: No loop FPR5 candidate parameters found.")
        print("You may need to rerun FPR5 with --save-history full.")
    if loop10.empty:
        print("\nWARNING: No loop FPR10 candidate parameters found.")
        print("You may need to rerun FPR10 with --save-history full.")

    if not loop5.empty:
        loop5 = loop5[loop5["method"].astype(str) == "hybrid"].copy()
    if not loop10.empty:
        loop10 = loop10[loop10["method"].astype(str) == "hybrid"].copy()

    rows = []

    rows.append(
        summarize_escape_gap(
            frozen,
            "Frozen defense",
            budgets,
            args.compressed_threshold,
        )
    )

    if not loop5.empty:
        rows.append(
            summarize_escape_gap(
                loop5,
                "Hardcase loop FPR5",
                budgets,
                args.compressed_threshold,
            )
        )

    if not loop10.empty:
        rows.append(
            summarize_escape_gap(
                loop10,
                "Hardcase loop FPR10",
                budgets,
                args.compressed_threshold,
            )
        )

    table = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    if table.empty:
        raise SystemExit("No valid loop-shift table produced.")

    table = table.sort_values(["budget", "condition"])

    for c in [
        "escaped_gap_median",
        "escaped_gap_q25",
        "escaped_gap_q75",
        "compressed_gap_threshold",
        "compressed_gap_escape_share_%",
        "escaped_cost_median",
    ]:
        if c in table.columns:
            table[c] = table[c].astype(float).round(4)

    table.to_csv(out_dir / "table6_tgce_loop_gap_shift.csv", index=False)

    with open(out_dir / "table6_tgce_loop_gap_shift.md", "w") as f:
        f.write(table.to_markdown(index=False))
        f.write("\n")

    print("\n" + "=" * 100)
    print("Table 6: gap shift before/after hardcase loop")
    print("=" * 100)
    print(table.to_string(index=False))

    # Pairwise tests: Frozen escaped gap vs loop escaped gap.
    tests = []

    for b in budgets:
        f = frozen[
            (pd.to_numeric(frozen["budget"], errors="coerce").astype("Int64") == b)
            & (frozen["escaped"] == True)
        ]

        for label, df in [
            ("Hardcase loop FPR5", loop5),
            ("Hardcase loop FPR10", loop10),
        ]:
            if df.empty or "gap_scale" not in df.columns:
                continue

            l = df[
                (pd.to_numeric(df["budget"], errors="coerce").astype("Int64") == b)
                & (df["escaped"] == True)
            ]

            x = to_float_series(f["gap_scale"]).dropna().to_numpy()
            y = to_float_series(l["gap_scale"]).dropna().to_numpy()

            p, delta = mannwhitney_delta(y, x)

            tests.append({
                "budget": b,
                "comparison": f"{label} escaped gap vs Frozen escaped gap",
                "n_loop": int(len(y)),
                "n_frozen": int(len(x)),
                "loop_gap_median": float(np.median(y)) if len(y) else np.nan,
                "frozen_gap_median": float(np.median(x)) if len(x) else np.nan,
                "median_shift_loop_minus_frozen": (
                    float(np.median(y) - np.median(x)) if len(y) and len(x) else np.nan
                ),
                "mannwhitney_p": p,
                "cliffs_delta_loop_vs_frozen": delta,
            })

    tests = pd.DataFrame(tests)

    if not tests.empty:
        for c in [
            "loop_gap_median",
            "frozen_gap_median",
            "median_shift_loop_minus_frozen",
            "cliffs_delta_loop_vs_frozen",
        ]:
            tests[c] = tests[c].astype(float).round(4)

        tests.to_csv(out_dir / "table6_tgce_loop_gap_shift_tests.csv", index=False)

        with open(out_dir / "table6_tgce_loop_gap_shift_tests.md", "w") as f:
            f.write(tests.to_markdown(index=False))
            f.write("\n")

        print("\n" + "=" * 100)
        print("Loop-vs-frozen escaped gap tests")
        print("=" * 100)
        print(tests.to_string(index=False))

    print("\nSaved:")
    print(out_dir / "table6_tgce_loop_gap_shift.csv")
    print(out_dir / "table6_tgce_loop_gap_shift_tests.csv")


if __name__ == "__main__":
    main()
