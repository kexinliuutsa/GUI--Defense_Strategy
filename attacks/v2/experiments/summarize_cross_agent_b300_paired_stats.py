#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import numpy as np


AGENTS = ["AgentCPM", "AutoGLM", "Claude", "GPT-4o", "UI-TARS"]

RAW_DIR = Path("results/tgce_cross_agent_b300/counterfactual")
OUT_DIR = Path("results/tgce_cross_agent_b300")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_bool(x):
    return str(x).strip().lower() in {"true", "1", "yes", "y", "escaped", "success"}


def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def bootstrap_ci_binary(x, n_boot=5000, seed=0):
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
            p = float(binomtest(min(n01, n10), n01 + n10, p=0.5).pvalue)
        except Exception:
            p = np.nan

    paired_or = (n10 + 0.5) / (n01 + 0.5)
    return n00, n01, n10, n11, p, paired_or


def fmt_p(p):
    if p is None or not np.isfinite(p):
        return ""
    if p < 0.001:
        return "<0.001"
    return f"{p:.4g}"


def load_agent_raw(agent):
    p = RAW_DIR / agent / "tgce_gap_counterfactual_raw.csv"
    if not p.exists():
        print("missing:", p)
        return None

    df = pd.read_csv(p)

    mode_col = find_col(df, ["counterfactual_mode", "mode", "variant"])
    escaped_col = find_col(df, ["escaped", "escape", "is_escape", "accepted"])
    budget_col = find_col(df, ["budget"])

    if mode_col is None or escaped_col is None:
        raise SystemExit(f"Cannot identify mode/escaped columns in {p}. columns={list(df.columns)}")

    df = df.copy()
    df["agent"] = agent
    df["_mode"] = df[mode_col].astype(str)
    df["_escaped"] = df[escaped_col].apply(parse_bool)

    if budget_col:
        df["_budget"] = df[budget_col].astype(str)
    else:
        df["_budget"] = "300"

    # Identify candidate id. If absent, pair by row order within each mode.
    id_col = find_col(df, [
        "candidate_id",
        "candidate_i",
        "pair_id",
        "row_id",
        "source_candidate_id",
        "original_candidate_id",
    ])

    if id_col is not None:
        df["_pair_id"] = df[id_col].astype(str)
    else:
        df["_pair_id"] = df.groupby("_mode").cumcount().astype(str)

    keep = df[["agent", "_budget", "_pair_id", "_mode", "_escaped"]].copy()
    return keep


def summarize_agent(agent, df):
    piv = (
        df.pivot_table(
            index=["agent", "_budget", "_pair_id"],
            columns="_mode",
            values="_escaped",
            aggfunc="first",
        )
        .reset_index()
    )

    required = ["original", "gap_fixed", "gap_only"]
    missing = [c for c in required if c not in piv.columns]
    if missing:
        raise SystemExit(f"{agent}: missing modes {missing}. Found columns: {list(piv.columns)}")

    piv = piv.dropna(subset=required).copy()

    rows_main = []
    rows_pair = []

    n = len(piv)

    rate_info = {}
    for mode in required:
        x = piv[mode].astype(bool).to_numpy()
        rate = float(x.mean())
        lo, hi = bootstrap_ci_binary(x, seed=abs(hash((agent, mode))) % (2**32 - 1))
        rate_info[mode] = {
            "rate": rate,
            "ci": f"[{lo*100:.2f}, {hi*100:.2f}]",
            "text": f"{rate*100:.2f}% [{lo*100:.2f}, {hi*100:.2f}]",
            "n_escape": int(x.sum()),
        }

    comparisons = [
        ("original", "gap_fixed"),
        ("gap_only", "gap_fixed"),
        ("gap_only", "original"),
    ]

    for a_name, b_name in comparisons:
        a = piv[a_name].astype(bool).to_numpy()
        b = piv[b_name].astype(bool).to_numpy()

        n00, n01, n10, n11, p, paired_or = mcnemar_exact(a, b)

        rate_a = float(a.mean())
        rate_b = float(b.mean())

        risk_diff_pp = (rate_a - rate_b) * 100
        risk_ratio = (rate_a + 1e-12) / (rate_b + 1e-12)

        # Unpaired OR-style effect with Haldane correction, useful descriptive effect.
        a_pos = int(a.sum())
        a_neg = int((~a).sum())
        b_pos = int(b.sum())
        b_neg = int((~b).sum())
        odds_ratio = ((a_pos + 0.5) * (b_neg + 0.5)) / ((a_neg + 0.5) * (b_pos + 0.5))

        rows_pair.append({
            "agent": agent,
            "budget": "B300",
            "comparison": f"{a_name}_vs_{b_name}",
            "n_paired": n,
            f"{a_name}_rate_%": round(rate_a * 100, 2),
            f"{b_name}_rate_%": round(rate_b * 100, 2),
            "risk_diff_pp": round(risk_diff_pp, 2),
            "risk_ratio": round(risk_ratio, 3),
            "odds_ratio_descriptive": round(float(odds_ratio), 3),
            "paired_discordant_OR": round(float(paired_or), 3),
            "n00_both_no_escape": n00,
            "n01_a_no_b_yes": n01,
            "n10_a_yes_b_no": n10,
            "n11_both_escape": n11,
            "mcnemar_p": p,
            "mcnemar_p_text": fmt_p(p),
        })

    # Conservative interpretation label.
    original_rate = rate_info["original"]["rate"]
    gf_rate = rate_info["gap_fixed"]["rate"]
    go_rate = rate_info["gap_only"]["rate"]

    if original_rate >= 0.20 and go_rate >= gf_rate + 0.15:
        label = "holds"
    elif original_rate < 0.10:
        label = "low_original_replay"
    else:
        label = "weak_or_mixed"

    rows_main.append({
        "agent": agent,
        "budget": "B300",
        "n_paired": n,
        "original": rate_info["original"]["text"],
        "gap_fixed": rate_info["gap_fixed"]["text"],
        "gap_only": rate_info["gap_only"]["text"],
        "original_minus_gap_fixed_pp": round((original_rate - gf_rate) * 100, 2),
        "gap_only_minus_gap_fixed_pp": round((go_rate - gf_rate) * 100, 2),
        "original_vs_gap_fixed_p": rows_pair[0]["mcnemar_p_text"],
        "gap_only_vs_gap_fixed_p": rows_pair[1]["mcnemar_p_text"],
        "gap_only_vs_original_p": rows_pair[2]["mcnemar_p_text"],
        "interpretation": label,
    })

    return pd.DataFrame(rows_main), pd.DataFrame(rows_pair), piv


def main():
    all_main = []
    all_pair = []
    all_piv = []

    for agent in AGENTS:
        df = load_agent_raw(agent)
        if df is None:
            continue

        main_df, pair_df, piv = summarize_agent(agent, df)
        all_main.append(main_df)
        all_pair.append(pair_df)
        all_piv.append(piv)

    if not all_main:
        raise SystemExit("No agent raw files found.")

    main = pd.concat(all_main, ignore_index=True)
    pair = pd.concat(all_pair, ignore_index=True)
    piv_all = pd.concat(all_piv, ignore_index=True)

    main_path = OUT_DIR / "table11e_cross_agent_B300_counterfactual_paired_main.csv"
    pair_path = OUT_DIR / "table11f_cross_agent_B300_counterfactual_paired_effects.csv"
    piv_path = OUT_DIR / "cross_agent_B300_counterfactual_paired_matrix.csv"

    main.to_csv(main_path, index=False)
    pair.to_csv(pair_path, index=False)
    piv_all.to_csv(piv_path, index=False)

    print("=" * 100)
    print("TABLE 11E — B300 cross-agent paired counterfactual main")
    print("=" * 100)
    print(main.to_string(index=False))

    print("\n" + "=" * 100)
    print("TABLE 11F — B300 cross-agent paired effects")
    print("=" * 100)
    print(pair.to_string(index=False))

    print("\nSaved:")
    print(main_path)
    print(pair_path)
    print(piv_path)


if __name__ == "__main__":
    main()
