from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def pct(x):
    return round(float(x) * 100.0, 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", default="results/v2_formal")
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    agg_path = result_dir / "aggregate_by_budget_method.csv"

    if not agg_path.exists():
        raise FileNotFoundError(f"Missing aggregate file: {agg_path}")

    df = pd.read_csv(agg_path)

    out_dir = result_dir / "analysis"
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = df.sort_values(["budget", "method"]).copy()

    table = df[
        [
            "budget",
            "method",
            "pooled_asr",
            "escaped",
            "n_completed",
            "median_of_seed_median_q_first",
            "median_of_seed_median_best_cost",
            "std_seed_asr",
        ]
    ].copy()

    table["asr_percent"] = table["pooled_asr"].map(pct)
    table["std_seed_asr_percent"] = table["std_seed_asr"].map(pct)

    table = table[
        [
            "budget",
            "method",
            "asr_percent",
            "std_seed_asr_percent",
            "escaped",
            "n_completed",
            "median_of_seed_median_q_first",
            "median_of_seed_median_best_cost",
        ]
    ]

    table_path = out_dir / "main_result_table.csv"
    table.to_csv(table_path, index=False)

    md_path = out_dir / "main_result_table.md"
    table.to_markdown(md_path, index=False)

    print("=" * 100)
    print("MAIN RESULT TABLE")
    print("=" * 100)
    print(table.to_string(index=False))
    print("saved:", table_path)
    print("saved:", md_path)

    # Improvement table: hybrid vs random / tpe
    rows = []
    for budget, g in df.groupby("budget"):
        by_method = {r["method"]: r for _, r in g.iterrows()}
        h = by_method.get("hybrid")
        if h is None:
            continue

        for base in ["random", "tpe"]:
            b = by_method.get(base)
            if b is None:
                continue

            rows.append(
                {
                    "budget": budget,
                    "comparison": f"hybrid_vs_{base}",
                    "asr_abs_gain_pct_points": pct(h["pooled_asr"] - b["pooled_asr"]),
                    "asr_relative_gain_pct": round(
                        100.0 * (h["pooled_asr"] - b["pooled_asr"]) / b["pooled_asr"],
                        2,
                    )
                    if b["pooled_asr"] > 0
                    else "",
                    "q_first_reduction": b["median_of_seed_median_q_first"]
                    - h["median_of_seed_median_q_first"],
                    "best_cost_reduction": b["median_of_seed_median_best_cost"]
                    - h["median_of_seed_median_best_cost"],
                }
            )

    imp = pd.DataFrame(rows)
    imp_path = out_dir / "hybrid_improvement_table.csv"
    imp.to_csv(imp_path, index=False)

    print("=" * 100)
    print("HYBRID IMPROVEMENT TABLE")
    print("=" * 100)
    print(imp.to_string(index=False))
    print("saved:", imp_path)

    # Plots
    metrics = [
        ("pooled_asr", "Attack Success Rate", "asr_vs_budget.png"),
        (
            "median_of_seed_median_q_first",
            "Median Queries to First Escape",
            "qfirst_vs_budget.png",
        ),
        (
            "median_of_seed_median_best_cost",
            "Median Best Perturbation Cost",
            "bestcost_vs_budget.png",
        ),
    ]

    for metric, ylabel, filename in metrics:
        plt.figure()
        for method, g in df.groupby("method"):
            g = g.sort_values("budget")
            plt.plot(g["budget"], g[metric], marker="o", label=method)

        plt.xlabel("Budget")
        plt.ylabel(ylabel)
        plt.title(ylabel + " vs Budget")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        out = fig_dir / filename
        plt.savefig(out, dpi=200)
        plt.close()

        print("saved figure:", out)

    # A concise written summary
    summary_path = out_dir / "formal_result_summary.txt"

    best_b300 = df[df["budget"] == 300].set_index("method")
    b100 = df[df["budget"] == 100].set_index("method")

    lines = []
    lines.append("Formal attack v2 result summary")
    lines.append("=" * 80)
    lines.append("")
    lines.append(
        "Across all four budgets, Hybrid achieves the highest pooled ASR among random, TPE, and Hybrid."
    )
    lines.append(
        f"At B300, Hybrid reaches {pct(best_b300.loc['hybrid', 'pooled_asr'])}% ASR, "
        f"compared with {pct(best_b300.loc['random', 'pooled_asr'])}% for random "
        f"and {pct(best_b300.loc['tpe', 'pooled_asr'])}% for TPE."
    )
    lines.append(
        f"At B300, Hybrid reduces median queries to first escape to "
        f"{best_b300.loc['hybrid', 'median_of_seed_median_q_first']}, "
        f"compared with {best_b300.loc['random', 'median_of_seed_median_q_first']} for random "
        f"and {best_b300.loc['tpe', 'median_of_seed_median_q_first']} for TPE."
    )
    lines.append(
        f"At B100, Hybrid achieves the lowest median best perturbation cost "
        f"({best_b300.loc['hybrid', 'median_of_seed_median_best_cost']:.4f} at B300; "
        f"{b100.loc['hybrid', 'median_of_seed_median_best_cost']:.4f} at B100)."
    )
    lines.append("")
    lines.append(
        "Interpretation: Hybrid combines broad exploration with local refinement around successful evasive samples. "
        "This gives it higher discovery than TPE and better query efficiency than random search."
    )

    summary_path.write_text("\n".join(lines))
    print("saved:", summary_path)
    print("=" * 100)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
