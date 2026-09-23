#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


OUT_DIR = Path("results/tgce_cross_method")
OUT_DIR.mkdir(parents=True, exist_ok=True)

METHODS = ["hybrid", "random", "tpe"]
MODES = ["original", "gap_fixed", "gap_only"]


def parse_bool(x):
    return str(x).strip().lower() in {"true", "1", "yes", "y"}


def bootstrap_ci(x, n_boot=10000, seed=0):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)

    for i in range(n_boot):
        vals[i] = rng.choice(x, size=len(x), replace=True).mean()

    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def fmt_p(p):
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "<0.001"
    return f"{p:.4g}"


def mcnemar_exact(a, b):
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)

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

    paired_or = (n10 + 0.5) / (n01 + 0.5)

    return n00, n01, n10, n11, p, paired_or


def load_method_raw(method):
    p = OUT_DIR / f"counterfactual_{method}" / "tgce_gap_counterfactual_raw.csv"

    if not p.exists():
        print("missing:", p)
        return None

    df = pd.read_csv(p)
    df["_raw_order"] = np.arange(len(df))
    df["method"] = method
    df["escaped_bool"] = df["escaped"].apply(parse_bool)
    df["budget_int"] = pd.to_numeric(df["budget"], errors="coerce").astype(int)

    paired_parts = []

    for budget, g in df.sort_values("_raw_order").groupby("budget_int", sort=False):
        g = g.copy().reset_index(drop=True)

        # run_tgce_gap_counterfactual writes 3 rows per candidate:
        # original, gap_fixed, gap_only.
        g["pair_local_id"] = g.index // 3

        counts = g.groupby("pair_local_id")["counterfactual_mode"].nunique()
        complete_ids = counts[counts == 3].index
        g = g[g["pair_local_id"].isin(complete_ids)].copy()

        g["pair_id"] = (
            method
            + "|B"
            + str(int(budget))
            + "|pair="
            + g["pair_local_id"].astype(str)
        )

        paired_parts.append(g)

    if not paired_parts:
        return None

    return pd.concat(paired_parts, ignore_index=True)


def main():
    all_parts = []

    for method in METHODS:
        df = load_method_raw(method)
        if df is not None:
            all_parts.append(df)

    if not all_parts:
        raise SystemExit("No counterfactual raw files found.")

    all_df = pd.concat(all_parts, ignore_index=True)
    all_df.to_csv(OUT_DIR / "table10b_cross_method_counterfactual_raw_paired.csv", index=False)

    rate_rows = []
    effect_rows = []
    main_rows = []

    for (method, budget), g in all_df.groupby(["method", "budget_int"]):
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

        n = int(len(pivot))
        rates = {}
        cis = {}

        for mode in MODES:
            x = pivot[mode].astype(bool).to_numpy()
            rate = float(x.mean() * 100)
            lo, hi = bootstrap_ci(
                x,
                seed=len(method) * 1000 + int(budget) * 10 + len(mode),
            )

            rates[mode] = rate
            cis[mode] = (lo * 100, hi * 100)

            rate_rows.append({
                "method": method,
                "budget": f"B{int(budget)}",
                "mode": mode,
                "n": n,
                "escaped": int(x.sum()),
                "escape_rate_%": round(rate, 2),
                "bootstrap_95ci_low_%": round(lo * 100, 2),
                "bootstrap_95ci_high_%": round(hi * 100, 2),
            })

        comparisons = [
            ("original", "gap_fixed"),
            ("gap_only", "gap_fixed"),
            ("gap_only", "original"),
        ]

        p_texts = {}

        for a, b in comparisons:
            xa = pivot[a].astype(bool).to_numpy()
            xb = pivot[b].astype(bool).to_numpy()

            n00, n01, n10, n11, p, paired_or = mcnemar_exact(xa, xb)

            rate_a = float(xa.mean())
            rate_b = float(xb.mean())
            diff_pp = (rate_a - rate_b) * 100
            rr = (rate_a + 1e-12) / (rate_b + 1e-12)

            p_texts[f"{a}_vs_{b}"] = fmt_p(p)

            effect_rows.append({
                "method": method,
                "budget": f"B{int(budget)}",
                "comparison": f"{a} vs {b}",
                "n_paired": n,
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

        tgce_holds = (
            rates["original"] > rates["gap_fixed"] + 20
            and rates["gap_only"] > rates["gap_fixed"] + 20
        )

        main_rows.append({
            "method": method,
            "budget": f"B{int(budget)}",
            "n_paired": n,
            "original": f"{rates['original']:.2f}% [{cis['original'][0]:.2f}, {cis['original'][1]:.2f}]",
            "gap_fixed": f"{rates['gap_fixed']:.2f}% [{cis['gap_fixed'][0]:.2f}, {cis['gap_fixed'][1]:.2f}]",
            "gap_only": f"{rates['gap_only']:.2f}% [{cis['gap_only'][0]:.2f}, {cis['gap_only'][1]:.2f}]",
            "original_minus_gap_fixed_pp": round(rates["original"] - rates["gap_fixed"], 2),
            "gap_only_minus_gap_fixed_pp": round(rates["gap_only"] - rates["gap_fixed"], 2),
            "original_vs_gap_fixed_p": p_texts.get("original_vs_gap_fixed", ""),
            "gap_only_vs_gap_fixed_p": p_texts.get("gap_only_vs_gap_fixed", ""),
            "gap_only_vs_original_p": p_texts.get("gap_only_vs_original", ""),
            "tgce_pattern": "holds" if tgce_holds else "weak/absent",
        })

    main = pd.DataFrame(main_rows)
    rates = pd.DataFrame(rate_rows)
    effects = pd.DataFrame(effect_rows)

    method_order = {"hybrid": 0, "random": 1, "tpe": 2}

    for d in [main, rates, effects]:
        d["_method_order"] = d["method"].map(method_order)
        d["_budget_order"] = d["budget"].str.replace("B", "", regex=False).astype(int)
        d.sort_values(["_budget_order", "_method_order"], inplace=True)
        d.drop(columns=["_method_order", "_budget_order"], inplace=True)

    main.to_csv(OUT_DIR / "table10b_cross_method_tgce_counterfactual_main.csv", index=False)
    rates.to_csv(OUT_DIR / "table10c_cross_method_tgce_counterfactual_rates_ci.csv", index=False)
    effects.to_csv(OUT_DIR / "table10d_cross_method_tgce_counterfactual_paired_effects.csv", index=False)

    with open(OUT_DIR / "table10b_cross_method_tgce_counterfactual_main.md", "w") as f:
        f.write(main.to_markdown(index=False))
        f.write("\n")

    with open(OUT_DIR / "table10c_cross_method_tgce_counterfactual_rates_ci.md", "w") as f:
        f.write(rates.to_markdown(index=False))
        f.write("\n")

    with open(OUT_DIR / "table10d_cross_method_tgce_counterfactual_paired_effects.md", "w") as f:
        f.write(effects.to_markdown(index=False))
        f.write("\n")

    print("=" * 100)
    print("TABLE 10B — Cross-method TGCE counterfactual")
    print("=" * 100)
    print(main.to_string(index=False))

    print("\nSaved:")
    print(OUT_DIR / "table10b_cross_method_tgce_counterfactual_main.csv")
    print(OUT_DIR / "table10c_cross_method_tgce_counterfactual_rates_ci.csv")
    print(OUT_DIR / "table10d_cross_method_tgce_counterfactual_paired_effects.csv")


if __name__ == "__main__":
    main()
