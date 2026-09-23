#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def bootstrap_rate_ci(x, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return np.nan, np.nan
    vals = np.empty(n_boot)
    for i in range(n_boot):
        vals[i] = rng.choice(x, size=len(x), replace=True).mean()
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def mcnemar_test(a, b):
    """
    Paired binary test.
    a and b are boolean arrays for two counterfactual modes.
    Returns discordant counts and approximate McNemar p-value.
    """
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)

    # b01: a failed, b escaped
    # b10: a escaped, b failed
    b01 = int((~a & b).sum())
    b10 = int((a & ~b).sum())

    n = b01 + b10

    if n == 0:
        return b01, b10, np.nan

    try:
        from statsmodels.stats.contingency_tables import mcnemar
        table = [[int((~a & ~b).sum()), b01],
                 [b10, int((a & b).sum())]]
        res = mcnemar(table, exact=True)
        p = float(res.pvalue)
    except Exception:
        # Simple chi-square approximation with continuity correction.
        stat = (abs(b01 - b10) - 1) ** 2 / n
        try:
            from scipy.stats import chi2
            p = float(1 - chi2.cdf(stat, 1))
        except Exception:
            p = np.nan

    return b01, b10, p


def fmt_p(p):
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "<0.001"
    return f"{p:.4g}"


def main():
    in_path = Path("results/tgce_counterfactual/tgce_gap_counterfactual_raw.csv")
    out_dir = Path("results/tgce_counterfactual")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)

    # Build candidate id for paired comparison.
    # source_q helps distinguish multiple escaped candidates from same record.
    id_cols = ["budget", "record_index", "source_q", "source_z_gap", "source_gap_scale"]
    for c in id_cols:
        if c not in df.columns:
            raise SystemExit(f"Missing column: {c}")

    df["candidate_id"] = df[id_cols].astype(str).agg("|".join, axis=1)
    df["escaped_bool"] = df["escaped"].astype(str).str.lower().isin(["true", "1"])

    rate_rows = []
    pair_rows = []

    for budget, g in df.groupby("budget"):
        pivot = (
            g.pivot_table(
                index="candidate_id",
                columns="counterfactual_mode",
                values="escaped_bool",
                aggfunc="first",
            )
            .dropna()
        )

        for mode in ["original", "gap_fixed", "gap_only"]:
            if mode not in pivot.columns:
                continue
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

        comparisons = [
            ("original", "gap_fixed"),
            ("gap_only", "gap_fixed"),
            ("gap_only", "original"),
        ]

        for a, b in comparisons:
            if a not in pivot.columns or b not in pivot.columns:
                continue

            xa = pivot[a].astype(bool).to_numpy()
            xb = pivot[b].astype(bool).to_numpy()

            b01, b10, p = mcnemar_test(xa, xb)

            pair_rows.append({
                "budget": int(budget),
                "comparison": f"{a} vs {b}",
                "n_paired": int(len(xa)),
                f"{a}_escape_rate_%": round(float(xa.mean() * 100), 2),
                f"{b}_escape_rate_%": round(float(xb.mean() * 100), 2),
                "absolute_rate_diff_%": round(float((xa.mean() - xb.mean()) * 100), 2),
                "discordant_a_fail_b_escape": b01,
                "discordant_a_escape_b_fail": b10,
                "mcnemar_p": p,
                "mcnemar_p_text": fmt_p(p),
            })

    rate = pd.DataFrame(rate_rows)
    pair = pd.DataFrame(pair_rows)

    rate.to_csv(out_dir / "table7_tgce_counterfactual_rates_with_ci.csv", index=False)
    pair.to_csv(out_dir / "table7_tgce_counterfactual_paired_tests.csv", index=False)

    with open(out_dir / "table7_tgce_counterfactual_rates_with_ci.md", "w") as f:
        f.write(rate.to_markdown(index=False))
        f.write("\n")

    with open(out_dir / "table7_tgce_counterfactual_paired_tests.md", "w") as f:
        f.write(pair.to_markdown(index=False))
        f.write("\n")

    print("=" * 100)
    print("Counterfactual rates with bootstrap CI")
    print("=" * 100)
    print(rate.to_string(index=False))

    print("\n" + "=" * 100)
    print("Paired McNemar tests")
    print("=" * 100)
    print(pair.to_string(index=False))

    print("\nSaved:")
    print(out_dir / "table7_tgce_counterfactual_rates_with_ci.csv")
    print(out_dir / "table7_tgce_counterfactual_paired_tests.csv")


if __name__ == "__main__":
    main()
