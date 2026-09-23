#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import re
import subprocess
import os
import numpy as np
import pandas as pd


ROOT = Path("results/tgce_cross_agent_b300")
SUBSET_DIR = ROOT / "candidate_subsets"
CF_DIR = ROOT / "counterfactual"
SUBSET_DIR.mkdir(parents=True, exist_ok=True)
CF_DIR.mkdir(parents=True, exist_ok=True)


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

    return str(x)


def safe_name(x):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(x)).strip("_")


def make_subsets():
    df = pd.read_csv("results/escape_basin_analysis/candidate_trials_flat.csv", low_memory=False)

    df["escaped_bool"] = df["escaped"].apply(parse_bool)
    df["budget_int"] = pd.to_numeric(df["budget"], errors="coerce").astype("Int64")
    df["z_gap_float"] = pd.to_numeric(df["z_gap"], errors="coerce")
    df["agent"] = df["participant"].apply(normalize_agent)

    df = df[
        (df["method"].astype(str) == "hybrid")
        & (df["budget_int"] == 300)
        & (df["escaped_bool"])
    ].dropna(subset=["z_gap_float"]).copy()

    manifest = []

    for agent, g in df.groupby("agent"):
        g = g.sort_values("z_gap_float", ascending=True).head(100).copy()

        if len(g) < 50:
            print(f"skip {agent}: only {len(g)} escaped candidates")
            continue

        out_csv = SUBSET_DIR / f"hybrid_{safe_name(agent)}_B300_compressed.csv"
        g.to_csv(out_csv, index=False)

        manifest.append({
            "agent": agent,
            "budget": 300,
            "n_candidates": int(len(g)),
            "median_z_gap": float(g["z_gap_float"].median()),
            "min_z_gap": float(g["z_gap_float"].min()),
            "max_z_gap": float(g["z_gap_float"].max()),
            "csv": str(out_csv),
        })

    man = pd.DataFrame(manifest)
    man.to_csv(SUBSET_DIR / "manifest_cross_agent_B300_compressed_candidates.csv", index=False)

    print("=" * 100)
    print("B300 cross-agent candidate subsets")
    print("=" * 100)
    print(man.to_string(index=False))

    return man


def run_counterfactuals(man):
    env = os.environ.copy()
    repo = str(Path.cwd())
    ahb = "/storage/cjh/self-evolving-safety/Passing-the-Turing-Test-on-Screen-Agent-Humanization-Benchmark-main"

    env["PYTHONPATH"] = repo + ":" + ahb + ":" + env.get("PYTHONPATH", "")
    env["GUI_DEFENSE_MODULE"] = "evaluation.frozen_v1v2v3_defense_hardened"

    for _, r in man.iterrows():
        agent = str(r["agent"])
        csv = str(r["csv"])
        out_dir = CF_DIR / safe_name(agent)

        print("=" * 100)
        print(f"Running B300 counterfactual: agent={agent}")
        print("=" * 100)

        cmd = [
            "python", "-u", "attacks/v2/experiments/run_tgce_gap_counterfactual.py",
            "--candidate-csv", csv,
            "--method", "hybrid",
            "--budgets", "300",
            "--max-candidates-per-budget", "100",
            "--output-dir", str(out_dir),
        ]

        subprocess.run(cmd, check=True, env=env)


def parse_bool2(x):
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


def summarize(man):
    rows = []

    for _, r in man.iterrows():
        agent = str(r["agent"])
        raw = CF_DIR / safe_name(agent) / "tgce_gap_counterfactual_raw.csv"

        if not raw.exists():
            print("missing:", raw)
            continue

        df = pd.read_csv(raw)
        df["_raw_order"] = np.arange(len(df))
        df["escaped_bool"] = df["escaped"].apply(parse_bool2)

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

        rates = {}
        cis = {}

        for mode in ["original", "gap_fixed", "gap_only"]:
            x = pivot[mode].astype(bool).to_numpy()
            rate = float(x.mean() * 100)
            lo, hi = bootstrap_ci(x, seed=len(agent) * 100 + len(mode))
            rates[mode] = rate
            cis[mode] = (lo * 100, hi * 100)

        rows.append({
            "agent": agent,
            "budget": "B300",
            "n_paired": int(len(pivot)),
            "original": f"{rates['original']:.2f}% [{cis['original'][0]:.2f}, {cis['original'][1]:.2f}]",
            "gap_fixed": f"{rates['gap_fixed']:.2f}% [{cis['gap_fixed'][0]:.2f}, {cis['gap_fixed'][1]:.2f}]",
            "gap_only": f"{rates['gap_only']:.2f}% [{cis['gap_only'][0]:.2f}, {cis['gap_only'][1]:.2f}]",
            "original_minus_gap_fixed_pp": round(rates["original"] - rates["gap_fixed"], 2),
            "gap_only_minus_gap_fixed_pp": round(rates["gap_only"] - rates["gap_fixed"], 2),
            "tgce_pattern": (
                "holds"
                if rates["original"] > rates["gap_fixed"] + 20
                and rates["gap_only"] > rates["gap_fixed"] + 20
                else "weak/absent"
            ),
        })

    out = pd.DataFrame(rows)

    if out.empty:
        raise SystemExit("No summary rows generated.")

    out = out.sort_values("agent")
    out.to_csv(ROOT / "table11d_cross_agent_B300_tgce_counterfactual_main.csv", index=False)

    with open(ROOT / "table11d_cross_agent_B300_tgce_counterfactual_main.md", "w") as f:
        f.write(out.to_markdown(index=False))
        f.write("\n")

    print("=" * 100)
    print("TABLE 11D — B300 cross-agent TGCE counterfactual")
    print("=" * 100)
    print(out.to_string(index=False))

    print("\nSaved:")
    print(ROOT / "table11d_cross_agent_B300_tgce_counterfactual_main.csv")


def main():
    man = make_subsets()
    if man.empty:
        raise SystemExit("No B300 subsets to run.")
    run_counterfactuals(man)
    summarize(man)


if __name__ == "__main__":
    main()
