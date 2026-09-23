#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--raw",
        default="results/tgce_gap_sweep_by_basin_hybrid_B10_B30/gap_sweep_raw.csv",
    )
    ap.add_argument(
        "--output-dir",
        default="results/tgce_gap_sweep_by_basin_hybrid_B10_B30/fixed_summary",
    )
    return ap.parse_args()


def bootstrap_ci_binary(x, n_boot=5000, seed=0):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)

    for i in range(n_boot):
        vals[i] = rng.choice(x, size=len(x), replace=True).mean()

    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def stable_seed(*parts):
    import hashlib
    s = "|".join(str(x) for x in parts)
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)


def mcnemar_exact(a, b):
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)

    n00 = int((~a & ~b).sum())
    n01 = int((~a & b).sum())
    n10 = int((a & ~b).sum())
    n11 = int((a & b).sum())

    try:
        from statsmodels.stats.contingency_tables import mcnemar
        p = float(mcnemar([[n00, n01], [n10, n11]], exact=True).pvalue)
    except Exception:
        try:
            from scipy.stats import binomtest
            p = float(binomtest(min(n01, n10), n01 + n10, p=0.5).pvalue)
        except Exception:
            p = np.nan

    OR = (n10 + 0.5) / (n01 + 0.5)
    return n00, n01, n10, n11, p, OR


def fmt_p(p):
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "<0.001"
    return f"{p:.4g}"


def summarize_rates(df: pd.DataFrame, out_dir: Path):
    rows = []

    keys = [
        "method",
        "budget",
        "basin",
        "intervention",
        "intervention_label",
        "fixed_gap_scale",
        "fixed_z_gap",
    ]

    for key, g in df.groupby(keys, dropna=False):
        d = dict(zip(keys, key))

        x = g["escaped"].astype(bool).to_numpy()
        lo, hi = bootstrap_ci_binary(
            x,
            seed=stable_seed(*key),
        )

        rows.append({
            **d,
            "n": len(g),
            "escaped": int(x.sum()),
            "escape_rate_%": round(float(x.mean()) * 100, 2),
            "escape_rate_95ci": f"[{lo*100:.2f}, {hi*100:.2f}]",
        })

    out = pd.DataFrame(rows)

    def sort_gap(x):
        if str(x) == "logged":
            return -1.0
        return float(x)

    out["_sort_gap"] = out["fixed_gap_scale"].apply(sort_gap)
    out = out.sort_values(
        ["method", "budget", "basin", "_sort_gap"]
    ).drop(columns=["_sort_gap"])

    out.to_csv(out_dir / "table_gap_sweep_rates_fixed.csv", index=False)
    return out


def summarize_best_gap(rate: pd.DataFrame, out_dir: Path):
    fixed = rate[rate["intervention"].eq("fixed_gap")].copy()
    fixed["fixed_gap_scale_float"] = fixed["fixed_gap_scale"].astype(float)

    rows = []

    for (method, budget, basin), g in fixed.groupby(["method", "budget", "basin"]):
        g = g.sort_values("fixed_gap_scale_float")

        maxrow = g.sort_values("escape_rate_%", ascending=False).iloc[0]
        minrow = g.sort_values("escape_rate_%", ascending=True).iloc[0]

        rows.append({
            "method": method,
            "budget": budget,
            "basin": basin,
            "n_per_gap": int(g["n"].iloc[0]),
            "attack_best_gap_scale_max_escape": float(maxrow["fixed_gap_scale"]),
            "max_escape_rate_%": float(maxrow["escape_rate_%"]),
            "defense_best_gap_scale_min_escape": float(minrow["fixed_gap_scale"]),
            "min_escape_rate_%": float(minrow["escape_rate_%"]),
            "range_pp": round(float(maxrow["escape_rate_%"] - minrow["escape_rate_%"]), 2),
            "escape_at_gap_0.6_%": lookup(g, 0.6),
            "escape_at_gap_0.8_%": lookup(g, 0.8),
            "escape_at_gap_1.0_%": lookup(g, 1.0),
            "escape_at_gap_1.175_%": lookup(g, 1.175),
            "escape_at_gap_1.5_%": lookup(g, 1.5),
            "escape_at_gap_1.7_%": lookup(g, 1.7),
        })

    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "table_gap_sweep_best_gap_fixed.csv", index=False)
    return out


def lookup(g, gap):
    sub = g[np.isclose(g["fixed_gap_scale"].astype(float), gap, atol=1e-9)]
    if sub.empty:
        return np.nan
    return float(sub.iloc[0]["escape_rate_%"])


def summarize_paired(df: pd.DataFrame, out_dir: Path):
    rows = []

    for (method, budget, basin), g in df.groupby(["method", "budget", "basin"]):
        pivot = g.pivot_table(
            index="candidate_key",
            columns="intervention_label",
            values="escaped",
            aggfunc="first",
        )

        labels = list(pivot.columns)

        gap_labels = []
        for lab in labels:
            if str(lab).startswith("gap_"):
                try:
                    gap_labels.append((float(str(lab).replace("gap_", "")), lab))
                except Exception:
                    pass

        gap_labels = sorted(gap_labels)

        comparisons = []

        if "logged_original" in labels:
            for _, lab in gap_labels:
                comparisons.append(("logged_original", lab))

        # adjacent gap comparisons
        for (_, lab_a), (_, lab_b) in zip(gap_labels[:-1], gap_labels[1:]):
            comparisons.append((lab_a, lab_b))

        # compare all gaps against no-op if present
        noop = None
        for val, lab in gap_labels:
            if abs(val - 1.0) < 1e-9:
                noop = lab

        if noop is not None:
            for _, lab in gap_labels:
                if lab != noop:
                    comparisons.append((lab, noop))

        seen = set()

        for a_lab, b_lab in comparisons:
            if (a_lab, b_lab) in seen:
                continue
            seen.add((a_lab, b_lab))

            sub = pivot[[a_lab, b_lab]].dropna()
            if sub.empty:
                continue

            a = sub[a_lab].astype(bool).to_numpy()
            b = sub[b_lab].astype(bool).to_numpy()

            n00, n01, n10, n11, p, OR = mcnemar_exact(a, b)

            rows.append({
                "method": method,
                "budget": budget,
                "basin": basin,
                "comparison": f"{a_lab}_vs_{b_lab}",
                "n_paired": len(sub),
                f"{a_lab}_rate_%": round(a.mean() * 100, 2),
                f"{b_lab}_rate_%": round(b.mean() * 100, 2),
                "risk_diff_pp": round((a.mean() - b.mean()) * 100, 2),
                "paired_discordant_OR": round(float(OR), 3),
                "n00_both_no_escape": n00,
                "n01_a_no_b_yes": n01,
                "n10_a_yes_b_no": n10,
                "n11_both_escape": n11,
                "mcnemar_p": p,
                "mcnemar_p_text": fmt_p(p),
            })

    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "table_gap_sweep_paired_effects_fixed.csv", index=False)
    return out


def make_plot(rate: pd.DataFrame, out_dir: Path):
    import matplotlib.pyplot as plt

    fixed = rate[rate["intervention"].eq("fixed_gap")].copy()
    fixed["gap"] = fixed["fixed_gap_scale"].astype(float)

    for (method, budget), g in fixed.groupby(["method", "budget"]):
        plt.figure(figsize=(7, 4.5))

        for basin, gb in g.groupby("basin"):
            gb = gb.sort_values("gap")
            plt.plot(
                gb["gap"],
                gb["escape_rate_%"],
                marker="o",
                label=basin,
            )

        plt.xlabel("Fixed gap_scale")
        plt.ylabel("Escape rate (%)")
        plt.title(f"TGCE gap sweep: {method} {budget}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        name = f"figure_gap_sweep_fixed_{method}_{budget}".replace("/", "_")
        plt.savefig(out_dir / f"{name}.png", dpi=220)
        plt.savefig(out_dir / f"{name}.pdf")
        plt.close()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.raw, low_memory=False)

    rate = summarize_rates(df, out_dir)
    best = summarize_best_gap(rate, out_dir)
    paired = summarize_paired(df, out_dir)
    make_plot(rate, out_dir)

    print("=" * 120)
    print("FIXED RATE TABLE")
    print("=" * 120)
    print(rate.to_string(index=False))

    print("\n" + "=" * 120)
    print("FIXED BEST GAP TABLE")
    print("=" * 120)
    print(best.to_string(index=False))

    print("\nSaved:")
    print(out_dir / "table_gap_sweep_rates_fixed.csv")
    print(out_dir / "table_gap_sweep_best_gap_fixed.csv")
    print(out_dir / "table_gap_sweep_paired_effects_fixed.csv")


if __name__ == "__main__":
    main()
