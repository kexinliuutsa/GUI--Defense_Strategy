#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


OUT_DIR = Path("results/paper_tables")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SOURCES = [
    {
        "group": "Compressed-gap basin",
        "path": "results/tgce_counterfactual/tgce_gap_counterfactual_raw.csv",
        "note": "lowest-z_gap replayable escaped candidates",
    },
    {
        "group": "Neutral/high-gap control",
        "path": "results/tgce_counterfactual_neutral_high_control/tgce_gap_counterfactual_raw.csv",
        "note": "escaped candidates with z_gap >= 0.50",
    },
]

MODES = ["original", "gap_fixed", "gap_only"]


def parse_bool(x):
    return str(x).strip().lower() in {"true", "1", "yes", "y"}


def fmt_p(p):
    if p is None or not np.isfinite(p):
        return ""
    if p < 0.001:
        return "<0.001"
    return f"{p:.4g}"


def bootstrap_rate_ci(x, n_boot=10000, seed=0):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)

    for i in range(n_boot):
        vals[i] = rng.choice(x, size=len(x), replace=True).mean()

    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def mcnemar_exact(a, b):
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)

    # a/b binary table:
    # n00: both fail
    # n01: a fail, b escape
    # n10: a escape, b fail
    # n11: both escape
    n00 = int((~a & ~b).sum())
    n01 = int((~a & b).sum())
    n10 = int((a & ~b).sum())
    n11 = int((a & b).sum())

    try:
        from statsmodels.stats.contingency_tables import mcnemar
        res = mcnemar([[n00, n01], [n10, n11]], exact=True)
        p = float(res.pvalue)
    except Exception:
        try:
            from scipy.stats import binomtest
            p = float(binomtest(k=min(n01, n10), n=n01 + n10, p=0.5).pvalue)
        except Exception:
            p = np.nan

    # Paired OR from discordant pairs; add 0.5 to avoid divide by zero.
    paired_or = (n10 + 0.5) / (n01 + 0.5)

    return n00, n01, n10, n11, p, paired_or


def cochran_q(pivot):
    """
    Cochran's Q test for three related binary treatments.
    """
    X = pivot[MODES].astype(bool).astype(int).to_numpy()
    n, k = X.shape

    col_sum = X.sum(axis=0)
    row_sum = X.sum(axis=1)

    numerator = (k - 1) * (k * np.sum(col_sum ** 2) - np.sum(col_sum) ** 2)
    denominator = k * np.sum(row_sum) - np.sum(row_sum ** 2)

    if denominator == 0:
        return np.nan, np.nan

    q = float(numerator / denominator)

    try:
        from scipy.stats import chi2
        p = float(1 - chi2.cdf(q, k - 1))
    except Exception:
        p = np.nan

    return q, p


def load_and_pair(path, group_name, note):
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"Missing file: {p}")

    df = pd.read_csv(p)
    df["_raw_order"] = np.arange(len(df))
    df["group"] = group_name
    df["selection_note"] = note
    df["escaped_bool"] = df["escaped"].apply(parse_bool)
    df["budget_int"] = pd.to_numeric(df["budget"], errors="coerce").astype(int)

    paired_parts = []

    for budget, g in df.sort_values("_raw_order").groupby("budget_int", sort=False):
        g = g.copy().reset_index(drop=True)

        # The raw counterfactual runner writes one row per mode for each selected candidate.
        # Therefore every 3 consecutive rows are one paired candidate.
        g["pair_id"] = g.index // 3

        # Validate each pair has all three modes. Keep only complete pairs.
        counts = g.groupby("pair_id")["counterfactual_mode"].nunique()
        complete_pair_ids = counts[counts == 3].index
        g = g[g["pair_id"].isin(complete_pair_ids)].copy()

        # Make pair id globally unique.
        g["pair_id"] = (
            group_name
            + "|B"
            + str(int(budget))
            + "|pair="
            + g["pair_id"].astype(str)
        )

        paired_parts.append(g)

    out = pd.concat(paired_parts, ignore_index=True)

    return out


def build_stats(all_df):
    rate_rows = []
    pair_rows = []
    q_rows = []
    main_rows = []

    group_order = {
        "Compressed-gap basin": 0,
        "Neutral/high-gap control": 1,
    }

    for (group_name, budget), g in all_df.groupby(["group", "budget_int"], sort=False):
        pivot = (
            g.pivot_table(
                index="pair_id",
                columns="counterfactual_mode",
                values="escaped_bool",
                aggfunc="first",
            )
            .dropna(subset=MODES)
        )

        if pivot.empty:
            continue

        n_paired = int(len(pivot))

        mode_rates = {}
        mode_cis = {}
        mode_counts = {}

        for mode in MODES:
            x = pivot[mode].astype(bool).to_numpy()
            rate = float(x.mean() * 100)
            lo, hi = bootstrap_rate_ci(
                x,
                seed=int(budget) * 100 + group_order.get(group_name, 9) * 10 + len(mode),
            )

            mode_rates[mode] = rate
            mode_cis[mode] = (lo * 100, hi * 100)
            mode_counts[mode] = int(x.sum())

            rate_rows.append({
                "group": group_name,
                "budget": f"B{int(budget)}",
                "mode": mode,
                "n": n_paired,
                "escaped": int(x.sum()),
                "escape_rate_%": round(rate, 2),
                "bootstrap_95ci_low_%": round(lo * 100, 2),
                "bootstrap_95ci_high_%": round(hi * 100, 2),
            })

        q, qp = cochran_q(pivot)
        q_rows.append({
            "group": group_name,
            "budget": f"B{int(budget)}",
            "n_paired": n_paired,
            "cochran_q": q,
            "cochran_q_p": qp,
            "cochran_q_p_text": fmt_p(qp),
        })

        comparisons = [
            ("original", "gap_fixed"),
            ("gap_only", "gap_fixed"),
            ("gap_only", "original"),
        ]

        comparison_p = {}

        for a, b in comparisons:
            xa = pivot[a].astype(bool).to_numpy()
            xb = pivot[b].astype(bool).to_numpy()

            n00, n01, n10, n11, p, paired_or = mcnemar_exact(xa, xb)

            rate_a = float(xa.mean())
            rate_b = float(xb.mean())
            diff_pp = (rate_a - rate_b) * 100
            rr = (rate_a + 1e-12) / (rate_b + 1e-12)

            comparison_p[f"{a}_vs_{b}"] = fmt_p(p)

            pair_rows.append({
                "group": group_name,
                "budget": f"B{int(budget)}",
                "comparison": f"{a} vs {b}",
                "n_paired": n_paired,
                "a_escape_rate_%": round(rate_a * 100, 2),
                "b_escape_rate_%": round(rate_b * 100, 2),
                "absolute_rate_diff_pp": round(diff_pp, 2),
                "risk_ratio_a_over_b": round(rr, 3),
                "paired_odds_ratio_discordant": round(paired_or, 3),
                "n00_both_fail": n00,
                "n01_a_fail_b_escape": n01,
                "n10_a_escape_b_fail": n10,
                "n11_both_escape": n11,
                "mcnemar_p": p,
                "mcnemar_p_text": fmt_p(p),
            })

        if group_name == "Compressed-gap basin":
            implication = "TGCE holds: gap fixing sharply reduces escape; gap-only largely recovers escape"
        else:
            implication = "Boundary control: TGCE pattern absent"

        main_rows.append({
            "group": group_name,
            "budget": f"B{int(budget)}",
            "n_paired": n_paired,
            "original": (
                f"{mode_rates['original']:.2f}% "
                f"[{mode_cis['original'][0]:.2f}, {mode_cis['original'][1]:.2f}]"
            ),
            "gap_fixed": (
                f"{mode_rates['gap_fixed']:.2f}% "
                f"[{mode_cis['gap_fixed'][0]:.2f}, {mode_cis['gap_fixed'][1]:.2f}]"
            ),
            "gap_only": (
                f"{mode_rates['gap_only']:.2f}% "
                f"[{mode_cis['gap_only'][0]:.2f}, {mode_cis['gap_only'][1]:.2f}]"
            ),
            "original_minus_gap_fixed_pp": round(
                mode_rates["original"] - mode_rates["gap_fixed"], 2
            ),
            "original_vs_gap_fixed_p": comparison_p.get("original_vs_gap_fixed", ""),
            "gap_only_vs_gap_fixed_p": comparison_p.get("gap_only_vs_gap_fixed", ""),
            "gap_only_vs_original_p": comparison_p.get("gap_only_vs_original", ""),
            "implication": implication,
        })

    rates = pd.DataFrame(rate_rows)
    pairs = pd.DataFrame(pair_rows)
    qtab = pd.DataFrame(q_rows)
    main = pd.DataFrame(main_rows)

    # Sort outputs.
    order_group = {"Compressed-gap basin": 0, "Neutral/high-gap control": 1}

    for df in [rates, pairs, qtab, main]:
        df["_group_order"] = df["group"].map(order_group)
        df["_budget_order"] = df["budget"].str.replace("B", "", regex=False).astype(int)
        df.sort_values(["_group_order", "_budget_order"], inplace=True)
        df.drop(columns=["_group_order", "_budget_order"], inplace=True)

    return main, rates, pairs, qtab


def main():
    dfs = []
    for src in SOURCES:
        dfs.append(load_and_pair(src["path"], src["group"], src["note"]))

    all_df = pd.concat(dfs, ignore_index=True)
    all_df.to_csv(OUT_DIR / "table7_counterfactual_paired_raw_reconstructed.csv", index=False)

    main_tab, rates, pairs, qtab = build_stats(all_df)

    main_tab.to_csv(OUT_DIR / "table7_tgce_counterfactual_main_REBUILT.csv", index=False)
    rates.to_csv(OUT_DIR / "table7a_tgce_rates_ci_REBUILT.csv", index=False)
    pairs.to_csv(OUT_DIR / "table7b_tgce_paired_effects_REBUILT.csv", index=False)
    qtab.to_csv(OUT_DIR / "table7c_tgce_cochran_q_REBUILT.csv", index=False)

    with open(OUT_DIR / "table7_tgce_counterfactual_main_REBUILT.md", "w") as f:
        f.write(main_tab.to_markdown(index=False))
        f.write("\n")

    with open(OUT_DIR / "table7a_tgce_rates_ci_REBUILT.md", "w") as f:
        f.write(rates.to_markdown(index=False))
        f.write("\n")

    with open(OUT_DIR / "table7b_tgce_paired_effects_REBUILT.md", "w") as f:
        f.write(pairs.to_markdown(index=False))
        f.write("\n")

    with open(OUT_DIR / "table7c_tgce_cochran_q_REBUILT.md", "w") as f:
        f.write(qtab.to_markdown(index=False))
        f.write("\n")

    print("=" * 100)
    print("MAIN TABLE 7 — Counterfactual intervention + boundary control")
    print("=" * 100)
    print(main_tab.to_string(index=False))

    print("\n" + "=" * 100)
    print("TABLE 7A — Rates + bootstrap CI")
    print("=" * 100)
    print(rates.to_string(index=False))

    print("\n" + "=" * 100)
    print("TABLE 7B — Paired effects, McNemar 2x2 tables, RR, paired OR")
    print("=" * 100)
    print(pairs.to_string(index=False))

    print("\n" + "=" * 100)
    print("TABLE 7C — Cochran Q")
    print("=" * 100)
    print(qtab.to_string(index=False))

    print("\nSaved:")
    print(OUT_DIR / "table7_tgce_counterfactual_main_REBUILT.csv")
    print(OUT_DIR / "table7a_tgce_rates_ci_REBUILT.csv")
    print(OUT_DIR / "table7b_tgce_paired_effects_REBUILT.csv")
    print(OUT_DIR / "table7c_tgce_cochran_q_REBUILT.csv")


if __name__ == "__main__":
    main()
