#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[3]
_AHB_ROOT = Path("/storage/cjh/self-evolving-safety/Passing-the-Turing-Test-on-Screen-Agent-Humanization-Benchmark-main")

for p in [_REPO_ROOT, _AHB_ROOT]:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))


from attacks.v2.search_space import LatentVector
from attacks.v2.mutation import mutate
from attacks.v2.oracle import load_defense, load_long_tap_records
from attacks.v2.experiments.run_tgce_gap_counterfactual import (
    fill_record_index,
    candidate_to_detection,
    get_task_cluster_from_row,
)


LATENT_ORDER = [
    "z_spatial",
    "z_frequency",
    "z_duration",
    "z_gap",
    "z_jitter",
    "z_heterogeneity",
    "z_use_spatial",
    "z_use_temporal",
]


def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--candidate-csv",
        default="results/escape_basin_analysis/candidate_trials_flat.csv",
    )
    ap.add_argument("--method", default="hybrid")
    ap.add_argument("--budget", type=int, default=30)
    ap.add_argument("--agents", default="UI-TARS")
    ap.add_argument("--max-candidates", type=int, default=150)
    ap.add_argument("--selection", default="escaped_low_gap",
                    choices=["escaped_low_gap", "escaped_random", "all_random"])

    ap.add_argument(
        "--z-grid",
        default="0.00,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00",
    )

    ap.add_argument("--defense-module", default="evaluation.frozen_v1v2v3_defense_hardened")
    ap.add_argument("--seed", type=int, default=20260918)
    ap.add_argument("--output-dir", default="results/mechanism_analysis/gap_sweep_boundary")

    return ap.parse_args()


def stable_int(*parts, mod=2**32 - 1):
    s = "|".join(str(p) for p in parts)
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h[:12], 16) % mod


def parse_bool(x):
    return str(x).strip().lower() in {"true", "1", "yes", "y", "escaped", "success"}


def normalize_agent(x):
    s = str(x).lower()
    if "ui-tars" in s or "uitars" in s:
        return "UI-TARS"
    if "gpt4o" in s or "gpt-4o" in s:
        return "GPT-4o"
    if "claude" in s or "sonnet" in s:
        return "Claude"
    if "cpm" in s:
        return "AgentCPM"
    if "autoglm" in s or "glm" in s:
        return "AutoGLM"
    return str(x)


def z_gap_to_gap_scale(z):
    # Same mapping used in previous TGCE notes.
    return 0.55 + 1.25 * float(z)


def get_base_session(original_record):
    if isinstance(original_record, dict) and "gestures" in original_record:
        return original_record["gestures"]
    return original_record


def build_latent_with_gap(row, z_gap):
    vals = []
    for k in LATENT_ORDER:
        if k == "z_gap":
            vals.append(float(z_gap))
        else:
            vals.append(float(row[k]))
    vals = np.asarray(vals, dtype=float)
    return LatentVector(values=vals)


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_long_tap_records()

    df = pd.read_csv(args.candidate_csv, low_memory=False)
    df["_raw_order"] = np.arange(len(df))
    df["escaped_bool"] = df["escaped"].apply(parse_bool)
    df["budget_int"] = pd.to_numeric(df["budget"], errors="coerce").astype("Int64")
    df["agent"] = df["participant"].apply(normalize_agent)
    df = fill_record_index(df, records)

    agents = {x.strip() for x in args.agents.split(",") if x.strip()}
    sub = df[
        (df["method"].astype(str) == args.method)
        & (df["budget_int"] == args.budget)
        & (df["agent"].isin(agents))
    ].copy()

    if sub.empty:
        raise SystemExit("No candidates after method/budget/agent filter.")

    for k in LATENT_ORDER:
        if k not in sub.columns:
            raise SystemExit(f"Missing latent column: {k}")

    if args.selection == "escaped_low_gap":
        sub = sub[sub["escaped_bool"]].copy()
        sub["z_gap_float"] = pd.to_numeric(sub["z_gap"], errors="coerce")
        sub = sub.sort_values(["z_gap_float", "_raw_order"], ascending=[True, True])
    elif args.selection == "escaped_random":
        sub = sub[sub["escaped_bool"]].sample(frac=1.0, random_state=args.seed)
    else:
        sub = sub.sample(frac=1.0, random_state=args.seed)

    sub = sub.head(args.max_candidates).copy()

    print("=" * 100)
    print("GAP SWEEP BOUNDARY")
    print("=" * 100)
    print("method:", args.method)
    print("budget:", args.budget)
    print("agents:", args.agents)
    print("selection:", args.selection)
    print("n candidates:", len(sub))

    os.environ["GUI_DEFENSE_MODULE"] = args.defense_module
    defense = load_defense()

    z_grid = [float(x) for x in args.z_grid.split(",") if x.strip()]

    rows = []

    for i, (_, row) in enumerate(sub.iterrows()):
        record_index = int(float(row["record_index_filled"]))
        original_record = records[record_index]
        base_session = get_base_session(original_record)
        task_cluster = get_task_cluster_from_row(row)

        for z in z_grid:
            try:
                latent = build_latent_with_gap(row, z)

                rng = np.random.default_rng(
                    stable_int(args.seed, i, row.get("session_id"), row.get("_raw_order"), z)
                )

                mutated = mutate(base_session, latent, rng)

                detected = candidate_to_detection(
                    defense,
                    original_record,
                    mutated,
                    task_cluster=task_cluster,
                )

                rows.append({
                    "candidate_i": i,
                    "record_index": record_index,
                    "participant": row.get("participant"),
                    "agent": row.get("agent"),
                    "session_id": row.get("session_id"),
                    "method": args.method,
                    "budget": args.budget,
                    "original_z_gap": row.get("z_gap"),
                    "sweep_z_gap": z,
                    "sweep_gap_scale": z_gap_to_gap_scale(z),
                    "detected": bool(detected),
                    "escaped": not bool(detected),
                    "error": "",
                })

            except Exception as e:
                rows.append({
                    "candidate_i": i,
                    "record_index": row.get("record_index_filled"),
                    "participant": row.get("participant"),
                    "agent": row.get("agent"),
                    "session_id": row.get("session_id"),
                    "method": args.method,
                    "budget": args.budget,
                    "original_z_gap": row.get("z_gap"),
                    "sweep_z_gap": z,
                    "sweep_gap_scale": z_gap_to_gap_scale(z),
                    "detected": None,
                    "escaped": None,
                    "error": repr(e),
                })

    raw = pd.DataFrame(rows)
    raw.to_csv(out_dir / "gap_sweep_raw.csv", index=False)

    valid = raw[raw["error"].eq("")].copy()
    valid["escaped_int"] = valid["escaped"].astype(int)
    valid["detected_int"] = valid["detected"].astype(int)

    summary = (
        valid.groupby(["sweep_z_gap", "sweep_gap_scale"])
        .agg(
            n=("escaped_int", "size"),
            escape_rate_pct=("escaped_int", lambda x: round(float(x.mean()) * 100, 2)),
            detection_rate_pct=("detected_int", lambda x: round(float(x.mean()) * 100, 2)),
        )
        .reset_index()
    )

    summary.to_csv(out_dir / "gap_sweep_summary.csv", index=False)

    plt.figure(figsize=(7, 4.5))
    plt.plot(summary["sweep_gap_scale"], summary["escape_rate_pct"], marker="o")
    plt.xlabel("gap scale")
    plt.ylabel("escape rate (%)")
    plt.title(f"Escape rate vs gap scale ({args.method}, B{args.budget}, {args.agents})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "figure_gap_scale_vs_escape_rate.png", dpi=300)
    plt.savefig(out_dir / "figure_gap_scale_vs_escape_rate.pdf")
    plt.close()

    plt.figure(figsize=(7, 4.5))
    plt.plot(summary["sweep_z_gap"], summary["escape_rate_pct"], marker="o")
    plt.xlabel("z_gap")
    plt.ylabel("escape rate (%)")
    plt.title(f"Escape rate vs z_gap ({args.method}, B{args.budget}, {args.agents})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "figure_z_gap_vs_escape_rate.png", dpi=300)
    plt.savefig(out_dir / "figure_z_gap_vs_escape_rate.pdf")
    plt.close()

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(summary.to_string(index=False))

    print("\nSaved:")
    print(out_dir / "gap_sweep_raw.csv")
    print(out_dir / "gap_sweep_summary.csv")
    print(out_dir / "figure_gap_scale_vs_escape_rate.png")
    print(out_dir / "figure_gap_scale_vs_escape_rate.pdf")


if __name__ == "__main__":
    main()
