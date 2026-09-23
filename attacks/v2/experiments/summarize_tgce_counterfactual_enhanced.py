#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import math
import numpy as np
import pandas as pd


def fmt_p(p):
    if p is None or not np.isfinite(p):
        return ""
    if p < 0.001:
        return "<0.001"
    return f"{p:.4g}"


def bootstrap_rate_ci(x, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return np.nan, np.nan
    vals = np.empty(n_boot)
    for i in range(n_boot):
        vals[i] = rng.choice(x, size=len(x), replace=True).mean()
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def mcnemar_exact(a, b):
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)

    # table:
    # a=0,b=0 | a=0,b=1
    # a=1,b=0 | a=1,b=1
    n00 = int((~a & ~b).sum())
    n01 = int((~a & b).sum())
    n10 = int((a & ~b).sum())
    n11 = int((a & b).sum())

    try:
        from statsmodels.stats.contingency_tables import mcnemar
        res = mcnemar([[n00, n01], [n10, n11]], exact=True)
        p = float(res.pvalue)
    except Exception:
        # Fallback binomial exact via scipy.
        try:
            from scipy.stats import binomtest
            p = float(binomtest(k=min(n01, n10), n=n01+n10, p=0.5).pvalue)
        except Exception:
            p = np.nan

    # Paired odds ratio for discordant pairs.
    paired_or = (n10 + 0.5) / (n01 + 0.5)

    return n00, n01, n10, n11, p, paired_or


def cochran_q_test(pivot, modes):
    """
    Cochran's Q for related binary samples across >2 modes.
    """
    X = pivot[modes].astype(bool).astype(int).to_numpy()
    n, k = X.shape

    col_sum = X.sum(axis=0)
    row_sum = X.sum(axis=1)

    numerator = (k - 1) * (k * np.sum(col_sum ** 2) - np.sum(col_sum) ** 2)
    denominator = k * np.sum(row_sum) - np.sum(row_sum ** 2)

    if denominator == 0:
        return np.nan, np.nan

    q = numerator / denominator

    try:
        from scipy.stats import chi2
        p = float(1 - chi2.cdf(q, k - 1))
    except Exception:
        p = np.nan

    return float(q), p


def main():
    in_path = Path("results/tgce_counterfactual/tgce_gap_counterfactual_raw.csv")
    out_dir = Path("results/tgce_counterfactual")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)

    df["escaped_bool"] = df["escaped"].astype(str).str.lower().isin(["true", "1"])

    id_cols = ["budget", "record_index", "source_q", "source_z_gap", "source_gap_scale"]
    for c in id_cols:
        if c not in df.columns:
            raise SystemExit(f"Missing required column: {c}")

    df["candidate_id"] = df[id_cols].astype(str).agg("|".join, axis=1)

    rate_rows = []
    pair_rows = []
    q_rows = []

    modes = ["original", "gap_fixed", "gap_only"]

    for budget, g in df.groupby("budget"):
        pivot = (
            g.pivot_table(
                index="candidate_id",
                columns="counterfactual_mode",
                values="escaped_bool",
                aggfunc="first",
            )
            .dropna(subset=modes)
        )

        if pivot.empty:
            continue

        # Per-mode rates.
        for mode in modes:
            x = pivot[mode].astype(bool).to_numpy()
            lo, hi = bootstrap_rate_ci(x, seed=int(budget) + len(mode))
            rate_rows.append({
                "budget": int(budget),
                "mode": mode,
                "n": int(len(x)),
                "escaped": int(x.sum()),
                "escape_rate_%": round(float(x.mean() * 100), 2),
                "bootstrap_95ci_low_%": round(lo * 100, 2),
                "bootstrap_95ci_high_%": round(hi * 100, 2),
            })

        # Cochran's Q across all 3 modes.
        q, qp = cochran_q_test(pivot, modes)
        q_rows.append({
            "budget": int(budget),
            "n_paired": int(len(pivot)),
            "cochran_q": q,
            "cochran_q_p": qp,
            "cochran_q_p_text": fmt_p(qp),
        })

        # Pairwise paired tests.
        comparisons = [
            ("original", "gap_fixed"),
            ("gap_only", "gap_fixed"),
            ("gap_only", "original"),
        ]

        for a, b in comparisons:
            xa = pivot[a].astype(bool).to_numpy()
            xb = pivot[b].astype(bool).to_numpy()

            n00, n01, n10, n11, p, paired_or = mcnemar_exact(xa, xb)

            rate_a = xa.mean()
            rate_b = xb.mean()

            rr = (rate_a + 1e-12) / (rate_b + 1e-12)
            diff = rate_a - rate_b

            pair_rows.append({
                "budget": int(budget),
                "comparison": f"{a} vs {b}",
                "n_paired": int(len(xa)),
                "a_escape_rate_%": round(rate_a * 100, 2),
                "b_escape_rate_%": round(rate_b * 100, 2),
                "absolute_rate_diff_pp": round(diff * 100, 2),
                "risk_ratio_a_over_b": round(rr, 3),
                "paired_odds_ratio_discordant": round(paired_or, 3),
                "n00_both_fail": n00,
                "n01_a_fail_b_escape": n01,
                "n10_a_escape_b_fail": n10,
                "n11_both_escape": n11,
                "mcnemar_p": p,
                "mcnemar_p_text": fmt_p(p),
            })

    rates = pd.DataFrame(rate_rows)
    pairs = pd.DataFrame(pair_rows)
    qtab = pd.DataFrame(q_rows)

    rates.to_csv(out_dir / "table7a_tgce_counterfactual_rates_ci.csv", index=False)
    pairs.to_csv(out_dir / "table7b_tgce_counterfactual_paired_effects.csv", index=False)
    qtab.to_csv(out_dir / "table7c_tgce_counterfactual_cochran_q.csv", index=False)

    with open(out_dir / "table7a_tgce_counterfactual_rates_ci.md", "w") as f:
        f.write(rates.to_markdown(index=False) + "\n")
    with open(out_dir / "table7b_tgce_counterfactual_paired_effects.md", "w") as f:
        f.write(pairs.to_markdown(index=False) + "\n")
    with open(out_dir / "table7c_tgce_counterfactual_cochran_q.md", "w") as f:
        f.write(qtab.to_markdown(index=False) + "\n")

    print("=" * 100)
    print("Table 7A: rates + bootstrap CI")
    print("=" * 100)
    print(rates.to_string(index=False))

    print("\n" + "=" * 100)
    print("Table 7B: paired effects")
    print("=" * 100)
    print(pairs.to_string(index=False))

    print("\n" + "=" * 100)
    print("Table 7C: Cochran's Q")
    print("=" * 100)
    print(qtab.to_string(index=False))

    print("\nSaved:")
    print(out_dir / "table7a_tgce_counterfactual_rates_ci.csv")
    print(out_dir / "table7b_tgce_counterfactual_paired_effects.csv")
    print(out_dir / "table7c_tgce_counterfactual_cochran_q.csv")


if __name__ == "__main__":
    main()
