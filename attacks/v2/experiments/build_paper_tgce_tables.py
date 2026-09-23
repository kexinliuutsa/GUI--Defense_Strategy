#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


OUT_DIR = Path("results/paper_tables")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_bool(x):
    return str(x).strip().lower() in {"true", "1", "yes", "escaped", "success"}


def bootstrap_ci(x, n_boot=10000, seed=0):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)

    for i in range(n_boot):
        vals[i] = rng.choice(x, size=len(x), replace=True).mean()

    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def fmt_rate_ci(rate, lo, hi):
    return f"{rate:.2f}% [{lo:.2f}, {hi:.2f}]"


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

    return n00, n01, n10, n11, p


def load_counterfactual_raw(path, group_name):
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"Missing file: {p}")

    df = pd.read_csv(p)

    required = [
        "budget",
        "record_index",
        "source_q",
        "source_z_gap",
        "source_gap_scale",
        "counterfactual_mode",
        "escaped",
    ]

    for c in required:
        if c not in df.columns:
            raise SystemExit(f"Missing required column {c} in {p}")

    df["escaped_bool"] = df["escaped"].apply(parse_bool)
    df["group"] = group_name

    id_cols = ["group", "budget", "record_index", "source_q", "source_z_gap", "source_gap_scale"]
    df["candidate_id"] = df[id_cols].astype(str).agg("|".join, axis=1)

    return df


def summarize_group(df):
    modes = ["original", "gap_fixed", "gap_only"]

    rows = []
    stat_rows = []

    for (group_name, budget), g in df.groupby(["group", "budget"]):
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

        rates = {}
        cis = {}

        for mode in modes:
            x = pivot[mode].astype(bool).to_numpy()
            rate = float(x.mean() * 100)
            lo, hi = bootstrap_ci(x, seed=int(float(budget)) + len(mode))
            rates[mode] = rate
            cis[mode] = (lo * 100, hi * 100)

        # Paired tests
        o = pivot["original"].astype(bool).to_numpy()
        gf = pivot["gap_fixed"].astype(bool).to_numpy()
        go = pivot["gap_only"].astype(bool).to_numpy()

        n00_ogf, n01_ogf, n10_ogf, n11_ogf, p_ogf = mcnemar_exact(o, gf)
        n00_ggf, n01_ggf, n10_ggf, n11_ggf, p_ggf = mcnemar_exact(go, gf)
        n00_go, n01_go, n10_go, n11_go, p_go = mcnemar_exact(go, o)

        implication = (
            "TGCE pattern holds"
            if group_name.startswith("Compressed")
            else "TGCE pattern absent / boundary control"
        )

        rows.append({
            "group": group_name,
            "budget": f"B{int(float(budget))}",
            "n_paired": int(len(pivot)),
            "original_escape_rate_95ci": fmt_rate_ci(
                rates["original"], *cis["original"]
            ),
            "gap_fixed_escape_rate_95ci": fmt_rate_ci(
                rates["gap_fixed"], *cis["gap_fixed"]
            ),
            "gap_only_escape_rate_95ci": fmt_rate_ci(
                rates["gap_only"], *cis["gap_only"]
            ),
            "original_vs_gap_fixed_pp_drop": round(
                rates["original"] - rates["gap_fixed"], 2
            ),
            "original_vs_gap_fixed_mcnemar_p": fmt_p(p_ogf),
            "gap_only_vs_gap_fixed_mcnemar_p": fmt_p(p_ggf),
            "gap_only_vs_original_mcnemar_p": fmt_p(p_go),
            "main_implication": implication,
        })

        stat_rows.extend([
            {
                "group": group_name,
                "budget": f"B{int(float(budget))}",
                "comparison": "original vs gap_fixed",
                "n_paired": int(len(pivot)),
                "a_escape_rate_%": round(rates["original"], 2),
                "b_escape_rate_%": round(rates["gap_fixed"], 2),
                "absolute_rate_diff_pp": round(rates["original"] - rates["gap_fixed"], 2),
                "n00_both_fail": n00_ogf,
                "n01_a_fail_b_escape": n01_ogf,
                "n10_a_escape_b_fail": n10_ogf,
                "n11_both_escape": n11_ogf,
                "mcnemar_p": p_ogf,
                "mcnemar_p_text": fmt_p(p_ogf),
            },
            {
                "group": group_name,
                "budget": f"B{int(float(budget))}",
                "comparison": "gap_only vs gap_fixed",
                "n_paired": int(len(pivot)),
                "a_escape_rate_%": round(rates["gap_only"], 2),
                "b_escape_rate_%": round(rates["gap_fixed"], 2),
                "absolute_rate_diff_pp": round(rates["gap_only"] - rates["gap_fixed"], 2),
                "n00_both_fail": n00_ggf,
                "n01_a_fail_b_escape": n01_ggf,
                "n10_a_escape_b_fail": n10_ggf,
                "n11_both_escape": n11_ggf,
                "mcnemar_p": p_ggf,
                "mcnemar_p_text": fmt_p(p_ggf),
            },
            {
                "group": group_name,
                "budget": f"B{int(float(budget))}",
                "comparison": "gap_only vs original",
                "n_paired": int(len(pivot)),
                "a_escape_rate_%": round(rates["gap_only"], 2),
                "b_escape_rate_%": round(rates["original"], 2),
                "absolute_rate_diff_pp": round(rates["gap_only"] - rates["original"], 2),
                "n00_both_fail": n00_go,
                "n01_a_fail_b_escape": n01_go,
                "n10_a_escape_b_fail": n10_go,
                "n11_both_escape": n11_go,
                "mcnemar_p": p_go,
                "mcnemar_p_text": fmt_p(p_go),
            },
        ])

    return pd.DataFrame(rows), pd.DataFrame(stat_rows)


def build_table7():
    compressed = load_counterfactual_raw(
        "results/tgce_counterfactual/tgce_gap_counterfactual_raw.csv",
        "Compressed-gap basin",
    )

    neutral_high = load_counterfactual_raw(
        "results/tgce_counterfactual_neutral_high_control/tgce_gap_counterfactual_raw.csv",
        "Neutral/high-gap control",
    )

    all_df = pd.concat([compressed, neutral_high], ignore_index=True)

    main, stats = summarize_group(all_df)

    group_order = {
        "Compressed-gap basin": 0,
        "Neutral/high-gap control": 1,
    }

    main["_group_order"] = main["group"].map(group_order)
    main["_budget_order"] = main["budget"].str.replace("B", "", regex=False).astype(int)
    main = main.sort_values(["_group_order", "_budget_order"]).drop(
        columns=["_group_order", "_budget_order"]
    )

    stats["_group_order"] = stats["group"].map(group_order)
    stats["_budget_order"] = stats["budget"].str.replace("B", "", regex=False).astype(int)
    stats = stats.sort_values(["_group_order", "_budget_order", "comparison"]).drop(
        columns=["_group_order", "_budget_order"]
    )

    main.to_csv(OUT_DIR / "table7_tgce_counterfactual_main.csv", index=False)
    stats.to_csv(OUT_DIR / "table7_tgce_counterfactual_paired_stats.csv", index=False)

    with open(OUT_DIR / "table7_tgce_counterfactual_main.md", "w") as f:
        f.write(main.to_markdown(index=False))
        f.write("\n")

    with open(OUT_DIR / "table7_tgce_counterfactual_paired_stats.md", "w") as f:
        f.write(stats.to_markdown(index=False))
        f.write("\n")

    print("=" * 100)
    print("TABLE 7 — TGCE counterfactual intervention and boundary control")
    print("=" * 100)
    print(main.to_string(index=False))

    print("\nSaved:")
    print(OUT_DIR / "table7_tgce_counterfactual_main.csv")
    print(OUT_DIR / "table7_tgce_counterfactual_main.md")
    print(OUT_DIR / "table7_tgce_counterfactual_paired_stats.csv")


def build_table8():
    src = Path("results/tgce_counterfactual/table7d_tgce_counterfactual_across_defenses.csv")

    if not src.exists():
        print(f"Skipping Table 8; missing {src}")
        return

    df = pd.read_csv(src)

    # Compact table: focus on original and gap-only because they show TGCE suppression.
    compact = df[df["mode"].isin(["original", "gap_only"])].copy()

    compact["budget_mode"] = compact["budget"].astype(str) + "_" + compact["mode"]

    wide = (
        compact
        .pivot_table(
            index=["defense", "human_FPR_target"],
            columns="budget_mode",
            values="escape_rate_%",
            aggfunc="first",
        )
        .reset_index()
    )

    for c in ["10_original", "10_gap_only", "30_original", "30_gap_only"]:
        if c not in wide.columns:
            wide[c] = np.nan

    wide = wide[
        [
            "defense",
            "human_FPR_target",
            "10_original",
            "10_gap_only",
            "30_original",
            "30_gap_only",
        ]
    ]

    wide = wide.rename(columns={
        "10_original": "B10_original_escape_%",
        "10_gap_only": "B10_gap_only_escape_%",
        "30_original": "B30_original_escape_%",
        "30_gap_only": "B30_gap_only_escape_%",
    })

    fpr_order = {"original": 0, "5%": 1, "10%": 2}
    wide["_order"] = wide["human_FPR_target"].map(fpr_order)
    wide = wide.sort_values(["_order", "defense"]).drop(columns=["_order"])

    wide.to_csv(OUT_DIR / "table8_tgce_suppression_across_defenses.csv", index=False)

    with open(OUT_DIR / "table8_tgce_suppression_across_defenses.md", "w") as f:
        f.write(wide.to_markdown(index=False))
        f.write("\n")

    print("\n" + "=" * 100)
    print("TABLE 8 — TGCE suppression under hardcase-loop defenses")
    print("=" * 100)
    print(wide.to_string(index=False))

    print("\nSaved:")
    print(OUT_DIR / "table8_tgce_suppression_across_defenses.csv")
    print(OUT_DIR / "table8_tgce_suppression_across_defenses.md")


def main():
    build_table7()
    build_table8()


if __name__ == "__main__":
    main()
