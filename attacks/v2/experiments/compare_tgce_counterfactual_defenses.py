#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import pandas as pd


SOURCES = [
    ("Frozen defense", "original", "results/tgce_counterfactual/table7_tgce_gap_counterfactual_summary.csv"),
    ("Hardcase loop", "5%", "results/tgce_counterfactual_loop_fpr5/table7_tgce_gap_counterfactual_summary.csv"),
    ("Hardcase loop", "10%", "results/tgce_counterfactual_loop_fpr10/table7_tgce_gap_counterfactual_summary.csv"),
]


def main():
    out_dir = Path("results/tgce_counterfactual")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for defense, fpr, path in SOURCES:
        p = Path(path)
        if not p.exists():
            print("missing:", p)
            continue

        df = pd.read_csv(p)

        for _, r in df.iterrows():
            rows.append({
                "defense": defense,
                "human_FPR_target": fpr,
                "budget": int(r["budget"]),
                "mode": r["counterfactual_mode"],
                "n": int(r["n"]),
                "escaped": int(r["escaped"]),
                "escape_rate_%": float(r["escape_rate_pct"]),
            })

    out = pd.DataFrame(rows)

    if out.empty:
        raise SystemExit("No rows loaded.")

    order = {"original": 0, "gap_fixed": 1, "gap_only": 2}
    fpr_order = {"original": 0, "5%": 1, "10%": 2}
    out["_mode_order"] = out["mode"].map(order)
    out["_fpr_order"] = out["human_FPR_target"].map(fpr_order)

    out = out.sort_values(["budget", "_fpr_order", "_mode_order"]).drop(
        columns=["_mode_order", "_fpr_order"]
    )

    out.to_csv(out_dir / "table7d_tgce_counterfactual_across_defenses.csv", index=False)

    with open(out_dir / "table7d_tgce_counterfactual_across_defenses.md", "w") as f:
        f.write(out.to_markdown(index=False) + "\n")

    print(out.to_string(index=False))

    print("\nSaved:")
    print(out_dir / "table7d_tgce_counterfactual_across_defenses.csv")


if __name__ == "__main__":
    main()
