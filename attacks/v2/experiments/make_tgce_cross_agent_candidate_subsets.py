#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import re
import pandas as pd


OUT_DIR = Path("results/tgce_cross_agent/candidate_subsets")
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

    return str(x)


def safe_name(x):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(x)).strip("_")


def main():
    in_path = Path("results/escape_basin_analysis/candidate_trials_flat.csv")

    df = pd.read_csv(in_path, low_memory=False)

    if "participant" not in df.columns:
        raise SystemExit("Missing participant column.")

    df["agent"] = df["participant"].apply(normalize_agent)
    df["escaped_bool"] = df["escaped"].apply(parse_bool)
    df["budget_int"] = pd.to_numeric(df["budget"], errors="coerce").astype("Int64")
    df["z_gap_float"] = pd.to_numeric(df["z_gap"], errors="coerce")

    # Use hybrid first. You can later switch to random/tpe if needed.
    df = df[
        (df["method"].astype(str) == "hybrid")
        & (df["budget_int"].isin([10, 30]))
        & (df["escaped_bool"])
    ].dropna(subset=["z_gap_float"]).copy()

    manifest = []

    for (agent, budget), g in df.groupby(["agent", "budget_int"]):
        g = g.sort_values("z_gap_float", ascending=True).head(100).copy()

        if len(g) < 20:
            continue

        fn = OUT_DIR / f"hybrid_{safe_name(agent)}_B{int(budget)}_compressed.csv"
        g.to_csv(fn, index=False)

        manifest.append({
            "agent": agent,
            "budget": int(budget),
            "n_candidates": int(len(g)),
            "median_z_gap": float(g["z_gap_float"].median()),
            "min_z_gap": float(g["z_gap_float"].min()),
            "max_z_gap": float(g["z_gap_float"].max()),
            "csv": str(fn),
        })

    man = pd.DataFrame(manifest)
    man.to_csv(OUT_DIR / "manifest_cross_agent_compressed_candidates.csv", index=False)

    print("=" * 100)
    print("Cross-agent compressed candidate subsets")
    print("=" * 100)

    if man.empty:
        print("No agent/budget subsets with enough candidates.")
    else:
        print(man.to_string(index=False))

    print("\nSaved:")
    print(OUT_DIR / "manifest_cross_agent_compressed_candidates.csv")


if __name__ == "__main__":
    main()
