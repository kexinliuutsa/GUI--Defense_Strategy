#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd


OUT_DIR = Path("results/tgce_cross_agent")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_bool(x):
    return str(x).strip().lower() in {"true", "1", "yes", "escaped", "success"}


def normalize_agent(x):
    s = str(x).lower()

    if "ui-tars" in s or "uitars" in s:
        return "UI-TARS"
    if "gpt4o" in s or "gpt-4o" in s or "gpt_4o" in s:
        return "GPT-4o"
    if "claude" in s or "sonnet" in s:
        return "Claude"
    if "cpm" in s:
        return "AgentCPM"
    if "autoglm" in s or "auto-glm" in s or "glm" in s:
        return "AutoGLM"
    if "mobileagent" in s or "mobile-agent" in s:
        return "MobileAgent"
    if "human" in s:
        return "Human-related"

    return str(x)


def cliffs_delta(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) == 0 or len(y) == 0:
        return np.nan

    rng = np.random.default_rng(0)
    max_n = 5000

    if len(x) > max_n:
        x = rng.choice(x, size=max_n, replace=False)
    if len(y) > max_n:
        y = rng.choice(y, size=max_n, replace=False)

    gt = 0
    lt = 0

    for xi in x:
        gt += np.sum(xi > y)
        lt += np.sum(xi < y)

    return float((gt - lt) / (len(x) * len(y)))


def mannwhitney_p(x, y):
    try:
        from scipy.stats import mannwhitneyu
        return float(mannwhitneyu(x, y, alternative="two-sided").pvalue)
    except Exception:
        return np.nan


def bh_adjust(pvals):
    pvals = np.asarray(pvals, dtype=float)
    out = np.full_like(pvals, np.nan, dtype=float)

    valid = np.isfinite(pvals)
    pv = pvals[valid]
    m = len(pv)

    if m == 0:
        return out

    order = np.argsort(pv)
    ranked = pv[order]

    adj = np.empty(m)
    running = 1.0

    for i in range(m - 1, -1, -1):
        rank = i + 1
        running = min(running, ranked[i] * m / rank)
        adj[i] = running

    restored = np.empty(m)
    restored[order] = adj
    out[valid] = restored

    return out


def fmt_p(p):
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "<0.001"
    return f"{p:.4g}"


def main():
    in_path = Path("results/escape_basin_analysis/candidate_trials_flat.csv")

    if not in_path.exists():
        raise SystemExit(f"Missing {in_path}")

    df = pd.read_csv(in_path, low_memory=False)

    for c in ["method", "budget", "escaped", "gap_scale", "z_gap"]:
        if c not in df.columns:
            raise SystemExit(f"Missing column: {c}")

    if "participant" not in df.columns:
        raise SystemExit("Missing participant column; inspect CSV columns first.")

    df["agent"] = df["participant"].apply(normalize_agent)
    df["escaped_bool"] = df["escaped"].apply(parse_bool)
    df["budget_int"] = pd.to_numeric(df["budget"], errors="coerce").astype("Int64")
    df["gap_scale_float"] = pd.to_numeric(df["gap_scale"], errors="coerce")
    df["z_gap_float"] = pd.to_numeric(df["z_gap"], errors="coerce")

    df = df[
        (df["method"].astype(str).isin(["hybrid", "random", "tpe"]))
        & (df["budget_int"].isin([10, 30, 100, 300]))
    ].dropna(subset=["gap_scale_float", "z_gap_float"]).copy()

    rows = []

    for (method, budget, agent), g in df.groupby(["method", "budget_int", "agent"]):
        esc = g[g["escaped_bool"]]
        fail = g[~g["escaped_bool"]]

        if len(esc) < 30 or len(fail) < 30:
            continue

        x = esc["gap_scale_float"].to_numpy()
        y = fail["gap_scale_float"].to_numpy()

        p = mannwhitney_p(x, y)
        delta = cliffs_delta(x, y)

        rows.append({
            "method": method,
            "budget": int(budget),
            "agent": agent,
            "n_trials": int(len(g)),
            "n_escaped": int(len(esc)),
            "n_failed": int(len(fail)),
            "query_escape_rate_%": round(float(len(esc) / len(g) * 100), 3),

            "escaped_gap_scale_median": round(float(np.median(x)), 4),
            "failed_gap_scale_median": round(float(np.median(y)), 4),
            "median_diff_escaped_minus_failed": round(float(np.median(x) - np.median(y)), 4),

            "escaped_z_gap_median": round(float(esc["z_gap_float"].median()), 4),
            "failed_z_gap_median": round(float(fail["z_gap_float"].median()), 4),

            "escaped_share_gap_scale_le_090_%": round(float((esc["gap_scale_float"] <= 0.90).mean() * 100), 2),
            "failed_share_gap_scale_le_090_%": round(float((fail["gap_scale_float"] <= 0.90).mean() * 100), 2),

            "escaped_share_z_gap_le_020_%": round(float((esc["z_gap_float"] <= 0.20).mean() * 100), 2),
            "failed_share_z_gap_le_020_%": round(float((fail["z_gap_float"] <= 0.20).mean() * 100), 2),

            "cliffs_delta_gap_scale": round(delta, 4),
            "mannwhitney_p": p,
        })

    out = pd.DataFrame(rows)

    if out.empty:
        raise SystemExit("No rows generated. Maybe escaped counts per agent are too small.")

    out["mannwhitney_p_bh"] = bh_adjust(out["mannwhitney_p"].to_numpy())
    out["mannwhitney_p_text"] = out["mannwhitney_p"].apply(fmt_p)
    out["mannwhitney_p_bh_text"] = out["mannwhitney_p_bh"].apply(fmt_p)

    out = out.sort_values(["budget", "method", "agent"]).reset_index(drop=True)

    out.to_csv(OUT_DIR / "table11a_cross_agent_tgce_gap_signature.csv", index=False)

    with open(OUT_DIR / "table11a_cross_agent_tgce_gap_signature.md", "w") as f:
        f.write(out.to_markdown(index=False))
        f.write("\n")

    print("=" * 100)
    print("TABLE 11A — Cross-agent TGCE gap signature")
    print("=" * 100)
    print(out.to_string(index=False))

    # Compact summary: one row per agent across all methods/budgets.
    rows2 = []

    for agent, g in df.groupby("agent"):
        esc = g[g["escaped_bool"]]
        fail = g[~g["escaped_bool"]]

        if len(esc) < 50 or len(fail) < 50:
            continue

        x = esc["gap_scale_float"].to_numpy()
        y = fail["gap_scale_float"].to_numpy()

        rows2.append({
            "agent": agent,
            "n_trials": int(len(g)),
            "n_escaped": int(len(esc)),
            "n_failed": int(len(fail)),
            "escaped_gap_scale_median": round(float(np.median(x)), 4),
            "failed_gap_scale_median": round(float(np.median(y)), 4),
            "median_diff": round(float(np.median(x) - np.median(y)), 4),
            "escaped_share_gap_scale_le_090_%": round(float((esc["gap_scale_float"] <= 0.90).mean() * 100), 2),
            "failed_share_gap_scale_le_090_%": round(float((fail["gap_scale_float"] <= 0.90).mean() * 100), 2),
            "cliffs_delta_gap_scale": round(cliffs_delta(x, y), 4),
            "mannwhitney_p": mannwhitney_p(x, y),
        })

    compact = pd.DataFrame(rows2)

    if not compact.empty:
        compact["mannwhitney_p_bh"] = bh_adjust(compact["mannwhitney_p"].to_numpy())
        compact["mannwhitney_p_text"] = compact["mannwhitney_p"].apply(fmt_p)
        compact["mannwhitney_p_bh_text"] = compact["mannwhitney_p_bh"].apply(fmt_p)
        compact = compact.sort_values("agent")
        compact.to_csv(OUT_DIR / "table11a_cross_agent_tgce_gap_signature_compact.csv", index=False)

        with open(OUT_DIR / "table11a_cross_agent_tgce_gap_signature_compact.md", "w") as f:
            f.write(compact.to_markdown(index=False))
            f.write("\n")

        print("\n" + "=" * 100)
        print("COMPACT CROSS-AGENT SUMMARY")
        print("=" * 100)
        print(compact.to_string(index=False))

    print("\nSaved:")
    print(OUT_DIR / "table11a_cross_agent_tgce_gap_signature.csv")
    print(OUT_DIR / "table11a_cross_agent_tgce_gap_signature_compact.csv")


if __name__ == "__main__":
    main()
