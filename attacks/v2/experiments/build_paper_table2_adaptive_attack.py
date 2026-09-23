#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_BUDGETS = {10, 30, 100, 300}
PREFERRED_METHODS = {"random", "tpe", "hybrid"}

METHOD_NAMES = {
    "random": "Random search",
    "tpe": "TPE search",
    "hybrid": "Hybrid adaptive search",
    "multi-basin-hybrid": "Multi-basin hybrid",
    "llm-batch-strict-qwen3-8b": "Qwen batch strict",
    "llm-batch-hybrid-qwen3-8b": "Qwen batch hybrid",
    "llm-arm-bandit-qwen3-8b": "Qwen arm bandit",
    "llm-api-surrogate-deepseek-chat": "DeepSeek surrogate",
    "llm-api-batch-hybrid-deepseek-chat": "DeepSeek batch hybrid",
}


def read_aggregate(path: Path):
    try:
        df = pd.read_csv(path)
    except Exception:
        return None

    needed = {"method", "budget", "n_total", "escaped", "pooled_asr"}
    if not needed.issubset(df.columns):
        return None

    return df


def score_candidate(path: Path, df: pd.DataFrame):
    methods = set(df["method"].astype(str))
    budgets = set(df["budget"].astype(int))

    required_combos = 0
    for m in PREFERRED_METHODS:
        for b in REQUIRED_BUDGETS:
            hit = df[
                (df["method"].astype(str) == m)
                & (df["budget"].astype(int) == b)
            ]
            if len(hit):
                required_combos += 1

    budget_score = len(REQUIRED_BUDGETS.intersection(budgets))
    method_score = len(PREFERRED_METHODS.intersection(methods))
    n_total = int(df["n_total"].sum()) if "n_total" in df.columns else 0

    return {
        "path": str(path),
        "rows": len(df),
        "methods": ",".join(sorted(methods)),
        "budgets": ",".join(map(str, sorted(budgets))),
        "required_combos": required_combos,
        "budget_score": budget_score,
        "method_score": method_score,
        "n_total_sum": n_total,
        "score": required_combos * 1000000 + budget_score * 10000 + method_score * 1000 + n_total,
    }


def auto_find_aggregate(results_root: Path):
    candidates = []

    for p in results_root.rglob("aggregate_by_budget_method.csv"):
        df = read_aggregate(p)
        if df is None:
            continue
        candidates.append(score_candidate(p, df))

    inv = pd.DataFrame(candidates)

    if inv.empty:
        raise SystemExit("No aggregate_by_budget_method.csv found under results/")

    inv = inv.sort_values("score", ascending=False)
    return inv


def make_long_table(df: pd.DataFrame):
    out = df.copy()

    out["method_name"] = out["method"].map(METHOD_NAMES).fillna(out["method"])
    out["ASR (%)"] = (out["pooled_asr"] * 100).round(2)

    if "median_of_seed_median_q_first" in out.columns:
        out["Median queries to first escape"] = out["median_of_seed_median_q_first"]
    else:
        out["Median queries to first escape"] = None

    if "median_of_seed_median_best_cost" in out.columns:
        out["Median best cost"] = out["median_of_seed_median_best_cost"].round(4)
    else:
        out["Median best cost"] = None

    cols = [
        "method_name",
        "method",
        "budget",
        "n_total",
        "n_completed",
        "escaped",
        "ASR (%)",
        "Median queries to first escape",
        "Median best cost",
    ]

    cols = [c for c in cols if c in out.columns]

    out = out[cols].sort_values(["method_name", "budget"])
    return out


def make_wide_table(long_df: pd.DataFrame):
    rows = []

    for method, g in long_df.groupby("method"):
        row = {
            "Attack method": METHOD_NAMES.get(method, method),
            "method_id": method,
        }

        for b in sorted(g["budget"].astype(int).unique()):
            sub = g[g["budget"].astype(int) == b].iloc[0]

            asr = sub["ASR (%)"]
            escaped = int(sub["escaped"])
            n_total = int(sub["n_total"])
            row[f"B{b} ASR (%)"] = asr
            row[f"B{b} escaped/n"] = f"{escaped}/{n_total}"

            if "Median queries to first escape" in sub.index:
                row[f"B{b} median q"] = sub["Median queries to first escape"]

            if "Median best cost" in sub.index:
                row[f"B{b} median cost"] = sub["Median best cost"]

        rows.append(row)

    out = pd.DataFrame(rows)

    # Prefer main attack methods first.
    order = {
        "random": 0,
        "tpe": 1,
        "hybrid": 2,
        "multi-basin-hybrid": 3,
    }
    out["_order"] = out["method_id"].map(order).fillna(99)
    out = out.sort_values(["_order", "Attack method"]).drop(columns=["_order"])

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aggregate", default=None, help="Path to aggregate_by_budget_method.csv")
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--output-dir", default="results/paper_tables")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.aggregate:
        aggregate_path = Path(args.aggregate)
        df = read_aggregate(aggregate_path)
        if df is None:
            raise SystemExit(f"Cannot read valid aggregate file: {aggregate_path}")
        inventory = None
    else:
        inventory = auto_find_aggregate(Path(args.results_root))
        inventory_path = out_dir / "table2_candidate_aggregate_inventory.csv"
        inventory.to_csv(inventory_path, index=False)

        print("=" * 100)
        print("Candidate aggregate files")
        print("=" * 100)
        print(inventory.head(20).to_string(index=False))
        print("\nsaved:", inventory_path)

        aggregate_path = Path(inventory.iloc[0]["path"])
        df = read_aggregate(aggregate_path)

    print("\n" + "=" * 100)
    print("Using aggregate file")
    print("=" * 100)
    print(aggregate_path)

    long_table = make_long_table(df)
    wide_table = make_wide_table(long_table)

    long_csv = out_dir / "table2_adaptive_attack_budget_curve_long.csv"
    wide_csv = out_dir / "table2_adaptive_attack_budget_curve_wide.csv"
    long_md = out_dir / "table2_adaptive_attack_budget_curve_long.md"
    wide_md = out_dir / "table2_adaptive_attack_budget_curve_wide.md"

    long_table.to_csv(long_csv, index=False)
    wide_table.to_csv(wide_csv, index=False)

    with long_md.open("w") as f:
        f.write(long_table.to_markdown(index=False))
        f.write("\n")

    with wide_md.open("w") as f:
        f.write(wide_table.to_markdown(index=False))
        f.write("\n")

    print("\n" + "=" * 100)
    print("Table 2 long")
    print("=" * 100)
    print(long_table.to_string(index=False))

    print("\n" + "=" * 100)
    print("Table 2 wide")
    print("=" * 100)
    print(wide_table.to_string(index=False))

    print("\nsaved:")
    print(long_csv)
    print(wide_csv)
    print(long_md)
    print(wide_md)


if __name__ == "__main__":
    main()
