#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


OUT_DIR = Path("results/tgce_cross_agent")
MANIFEST = Path("results/tgce_cross_agent/candidate_subsets/manifest_cross_agent_compressed_candidates.csv")


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
            p = float(binomtest(k=min(n01, n10), n=n01+n10, p=0.5).pvalue)
        except Exception:
            p = np.nan

    paired_or = (n10 + 0.5) / (n01 + 0.5)

    return n00, n01, n10, n11, p, paired_or


def safe_agent_name(agent):
    return str(agent).replace(" ", "_").replace("/", "_")


def main():
    if not MANIFEST.exists():
        raise SystemExit(f"Missing manifest: {MANIFEST}")

    man = pd.read_csv(MANIFEST)

    rows = []
    effect_rows = []

    for _, m in man.iterrows():
        agent = str(m["agent"])
        budget = int(m["budget"])

        out_dir = OUT_DIR / "counterfactual" / f"{safe_agent_name(agent)}_B{budget}"
        raw_path = out_dir / "tgce_gap_counterfactual_raw.csv"

        if not raw_path.exists():
            print("missing:", raw_path)
            continue

        df = pd.read_csv(raw_path)
        df["_raw_order"] = np.arange(len(df))
        df["escaped_bool"] = df["escaped"].apply(parse_bool)

        # Reconstruct paired candidate: 3 rows per candidate.
        df = df.sort_values("_raw_order").reset_index(drop=True)
        df["pair_id"] = df.index // 3

        pivot = (
            df.pivot_table(
                index="pair_id",
                columns="counterfactual_mode",
                values="escaped_bool",
                aggfunc="first",
            )
            .dropna(subset=["original", "gap_fixed", "gap_only"])
        )

        if pivot.empty:
            continue

        n = int(len(pivot))

        rates = {}
        cis = {}

        for mode in ["original", "gap_fixed", "gap_only"]:
            x = pivot[mode].astype(bool).to_numpy()
            rate = float(x.mean() * 100)
            lo, hi = bootstrap_ci(x, seed=budget + len(agent) + len(mode))
            rates[mode] = rate
            cis[mode] = (lo * 100, hi * 100)

        for a, b in [
            ("original", "gap_fixed"),
            ("gap_only", "gap_fixed"),
            ("gap_only", "original"),
        ]:
            xa = pivot[a].astype(bool).to_numpy()
            xb = pivot[b].astype(bool).to_numpy()

            n00, n01, n10, n11, p, paired_or = mcnemar_exact(xa, xb)

            rate_a = float(xa.mean())
            rate_b = float(xb.mean())

            effect_rows.append({
                "agent": agent,
                "budget": f"B{budget}",
                "comparison": f"{a} vs {b}",
                "n_paired": n,
                "a_escape_rate_%": round(rate_a * 100, 2),
                "b_escape_rate_%": round(rate_b * 100, 2),
                "absolute_rate_diff_pp": round((rate_a - rate_b) * 100, 2),
                "risk_ratio_a_over_b": round((rate_a + 1e-12) / (rate_b + 1e-12), 3),
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

        rows.append({
            "agent": agent,
            "budget": f"B{budget}",
            "n_paired": n,
            "original": f"{rates['original']:.2f}% [{cis['original'][0]:.2f}, {cis['original'][1]:.2f}]",
            "gap_fixed": f"{rates['gap_fixed']:.2f}% [{cis['gap_fixed'][0]:.2f}, {cis['gap_fixed'][1]:.2f}]",
            "gap_only": f"{rates['gap_only']:.2f}% [{cis['gap_only'][0]:.2f}, {cis['gap_only'][1]:.2f}]",
            "original_minus_gap_fixed_pp": round(rates["original"] - rates["gap_fixed"], 2),
            "gap_only_minus_gap_fixed_pp": round(rates["gap_only"] - rates["gap_fixed"], 2),
            "tgce_pattern": "holds" if tgce_holds else "weak/absent",
        })

    main = pd.DataFrame(rows)
    effects = pd.DataFrame(effect_rows)

    if main.empty:
        raise SystemExit("No rows generated.")

    main = main.sort_values(["budget", "agent"])
    effects = effects.sort_values(["budget", "agent", "comparison"])

    main.to_csv(OUT_DIR / "table11b_cross_agent_tgce_counterfactual_main.csv", index=False)
    effects.to_csv(OUT_DIR / "table11c_cross_agent_tgce_counterfactual_paired_effects.csv", index=False)

    with open(OUT_DIR / "table11b_cross_agent_tgce_counterfactual_main.md", "w") as f:
        f.write(main.to_markdown(index=False))
        f.write("\n")

    with open(OUT_DIR / "table11c_cross_agent_tgce_counterfactual_paired_effects.md", "w") as f:
        f.write(effects.to_markdown(index=False))
        f.write("\n")

    print("=" * 100)
    print("TABLE 11B — Cross-agent TGCE counterfactual")
    print("=" * 100)
    print(main.to_string(index=False))

    print("\nSaved:")
    print(OUT_DIR / "table11b_cross_agent_tgce_counterfactual_main.csv")
    print(OUT_DIR / "table11c_cross_agent_tgce_counterfactual_paired_effects.csv")


if __name__ == "__main__":
    main()
