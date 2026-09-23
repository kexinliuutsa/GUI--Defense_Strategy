#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[3]
_AHB_ROOT = Path("/storage/cjh/self-evolving-safety/Passing-the-Turing-Test-on-Screen-Agent-Humanization-Benchmark-main")

for p in [_REPO_ROOT, _AHB_ROOT]:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))


from attacks.v2.experiments.run_hardcase_loop_heldout_replay import (
    load_candidate_df,
    load_defense_with_env,
    evaluate_test_spec,
    summarize_runs,
    parse_test_specs,
)
from attacks.v2.oracle import load_long_tap_records


DEFAULT_TEST_SPECS = [
    "seen_agent_UITARS_B30_hybrid|hybrid|30|UI-TARS|150|30",
    "heldout_budget_UITARS_B100_hybrid|hybrid|100|UI-TARS|150|100",
    "heldout_method_UITARS_random_B30|random|30|UI-TARS|150|30",
    "heldout_method_UITARS_tpe_B30|tpe|30|UI-TARS|150|30",
    "heldout_agent_AutoGLM_B30_hybrid|hybrid|30|AutoGLM|100|30",
    "heldout_agent_AgentCPM_B30_hybrid|hybrid|30|AgentCPM|150|30",
]


def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--candidate-csv",
        default="results/escape_basin_analysis/candidate_trials_flat.csv",
    )
    ap.add_argument(
        "--hardcase-pkl",
        default="results/strict_hardcase_memories/strict_train_UITARS_B10_B20_existing_methods_all31.pkl",
    )
    ap.add_argument(
        "--radii",
        default="0.0,0.4,0.6,0.8,1.0263925586627112,1.245990721485415,1.6,2.0",
    )
    ap.add_argument("--test-spec", action="append", default=None)
    ap.add_argument("--output-dir", default="results/mechanism_analysis/hardcase_radius_coverage")
    ap.add_argument("--seed", type=int, default=20260918)

    return ap.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_long_tap_records()
    df = load_candidate_df(args.candidate_csv, records)

    specs = parse_test_specs(args.test_spec or DEFAULT_TEST_SPECS)
    radii = [float(x) for x in args.radii.split(",") if x.strip()]

    frozen_defense = load_defense_with_env("evaluation.frozen_v1v2v3_defense_hardened")

    all_summary = []
    all_runs = []

    for radius in radii:
        print("\n" + "=" * 100)
        print("RADIUS:", radius)
        print("=" * 100)

        loop_defense = load_defense_with_env(
            "evaluation.defense_loop_v1_hardcase_memory",
            hardcase_pkl=args.hardcase_pkl,
            radius=radius,
        )

        run_parts = []
        query_parts = []

        for spec in specs:
            print("\nTEST:", spec)

            run_df, query_df = evaluate_test_spec(
                spec=spec,
                df=df,
                records=records,
                frozen_defense=frozen_defense,
                loop_defense=loop_defense,
                args=args,
            )

            if run_df is None or run_df.empty:
                continue

            run_df.insert(0, "radius", radius)
            query_df.insert(0, "radius", radius)

            run_parts.append(run_df)
            query_parts.append(query_df)

        if not run_parts:
            continue

        runs = pd.concat(run_parts, ignore_index=True)
        runs.to_csv(out_dir / f"run_level_radius_{radius:.4f}.csv", index=False)

        if query_parts:
            queries = pd.concat(query_parts, ignore_index=True)
            queries.to_csv(out_dir / f"query_level_radius_{radius:.4f}.csv", index=False)

        # summarize_runs expects args.radius because it was reused from
        # run_hardcase_loop_heldout_replay.py. Here radius is swept, so we
        # attach the current radius before summarizing.
        args.radius = radius
        summary = summarize_runs(runs, args, args.hardcase_pkl)
        summary.insert(0, "radius", radius)

        summary.to_csv(out_dir / f"summary_radius_{radius:.4f}.csv", index=False)

        print(summary[
            [
                "radius",
                "test_name",
                "frozen_ASR_%",
                "loop_ASR_%",
                "ASR_drop_pp",
                "relative_reduction_%",
                "mcnemar_p_text",
            ]
        ].to_string(index=False))

        all_summary.append(summary)
        all_runs.append(runs)

    if not all_summary:
        raise SystemExit("No summary generated.")

    final = pd.concat(all_summary, ignore_index=True)
    final.to_csv(out_dir / "radius_coverage_summary_all.csv", index=False)

    run_all = pd.concat(all_runs, ignore_index=True)
    run_all.to_csv(out_dir / "radius_coverage_run_level_all.csv", index=False)

    # Plot 1: loop ASR vs radius
    plt.figure(figsize=(8, 5))
    for test_name, g in final.groupby("test_name", sort=False):
        g = g.sort_values("radius")
        plt.plot(g["radius"], g["loop_ASR_%"], marker="o", label=test_name)

    plt.xlabel("hardcase memory radius")
    plt.ylabel("loop ASR (%)")
    plt.title("Hardcase radius coverage curve")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(out_dir / "figure_radius_vs_loop_ASR.png", dpi=300)
    plt.savefig(out_dir / "figure_radius_vs_loop_ASR.pdf")
    plt.close()

    # Plot 2: ASR drop vs radius
    plt.figure(figsize=(8, 5))
    for test_name, g in final.groupby("test_name", sort=False):
        g = g.sort_values("radius")
        plt.plot(g["radius"], g["ASR_drop_pp"], marker="o", label=test_name)

    plt.xlabel("hardcase memory radius")
    plt.ylabel("ASR drop vs frozen (percentage points)")
    plt.title("ASR drop as hardcase coverage expands")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(out_dir / "figure_radius_vs_ASR_drop.png", dpi=300)
    plt.savefig(out_dir / "figure_radius_vs_ASR_drop.pdf")
    plt.close()

    print("\n" + "=" * 100)
    print("FINAL RADIUS COVERAGE SUMMARY")
    print("=" * 100)
    print(final[
        [
            "radius",
            "test_name",
            "frozen_ASR_%",
            "loop_ASR_%",
            "ASR_drop_pp",
            "relative_reduction_%",
            "mcnemar_p_text",
        ]
    ].to_string(index=False))

    print("\nSaved:")
    print(out_dir / "radius_coverage_summary_all.csv")
    print(out_dir / "radius_coverage_run_level_all.csv")
    print(out_dir / "figure_radius_vs_loop_ASR.png")
    print(out_dir / "figure_radius_vs_ASR_drop.png")


if __name__ == "__main__":
    main()
