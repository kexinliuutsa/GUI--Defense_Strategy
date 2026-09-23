#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


OUT_DIR = Path("results/paper_tables")
OUT_DIR.mkdir(parents=True, exist_ok=True)


SOURCES = [
    {
        "condition": "Frozen detector",
        "human_FPR_target": "not calibrated in loop",
        "path": "results/tgce_counterfactual/table7_tgce_gap_counterfactual_summary.csv",
    },
    {
        "condition": "Hardcase loop",
        "human_FPR_target": "5%",
        "path": "results/tgce_counterfactual_loop_fpr5/table7_tgce_gap_counterfactual_summary.csv",
    },
    {
        "condition": "Hardcase loop",
        "human_FPR_target": "10%",
        "path": "results/tgce_counterfactual_loop_fpr10/table7_tgce_gap_counterfactual_summary.csv",
    },
]


def load_actual_human_fpr():
    """
    Optional metadata from calibration files.
    """
    rows = []

    calib = Path("results/v2_defense_loop_v1_human_fpr/human_calibrated_radius_table.csv")
    if calib.exists():
        df = pd.read_csv(calib)
        for _, r in df.iterrows():
            target = None
            for c in ["target_human_FPR_%", "target_FPR_%", "target_fpr_pct"]:
                if c in df.columns:
                    target = float(r[c])
                    break

            radius = None
            for c in ["calibrated_radius", "radius"]:
                if c in df.columns:
                    radius = float(r[c])
                    break

            actual = None
            for c in ["actual_human_FPR_%", "actual_FPR_%", "calibration_human_FPR_%"]:
                if c in df.columns:
                    actual = float(r[c])
                    break

            rows.append({
                "target_human_FPR_%": target,
                "calibrated_radius": radius,
                "calibration_human_FPR_%": actual,
            })

    return pd.DataFrame(rows)


def main():
    rows = []

    for src in SOURCES:
        p = Path(src["path"])
        if not p.exists():
            print("missing:", p)
            continue

        df = pd.read_csv(p)

        for _, r in df.iterrows():
            mode = str(r["counterfactual_mode"])
            budget = int(r["budget"])
            n = int(r["n"])
            escaped = int(r["escaped"])
            acceptance = float(r["escape_rate_pct"])

            rows.append({
                "condition": src["condition"],
                "human_FPR_target": src["human_FPR_target"],
                "budget": f"B{budget}",
                "variant": mode,
                "n": n,
                "accepted_as_human_like": escaped,
                "acceptance_rate_%": acceptance,
                "detected_rate_%": round(100.0 - acceptance, 2),
            })

    out = pd.DataFrame(rows)

    if out.empty:
        raise SystemExit("No rows loaded.")

    condition_order = {
        "Frozen detector": 0,
        "Hardcase loop": 1,
    }
    fpr_order = {
        "not calibrated in loop": 0,
        "5%": 1,
        "10%": 2,
    }
    variant_order = {
        "original": 0,
        "gap_fixed": 1,
        "gap_only": 2,
    }

    out["_condition_order"] = out["condition"].map(condition_order)
    out["_fpr_order"] = out["human_FPR_target"].map(fpr_order)
    out["_budget_order"] = out["budget"].str.replace("B", "", regex=False).astype(int)
    out["_variant_order"] = out["variant"].map(variant_order)

    out = out.sort_values(
        ["_budget_order", "_condition_order", "_fpr_order", "_variant_order"]
    ).drop(columns=["_condition_order", "_fpr_order", "_budget_order", "_variant_order"])

    out.to_csv(OUT_DIR / "table9_tgce_human_fpr_acceptance_long.csv", index=False)

    # Wide table for paper.
    wide = (
        out
        .pivot_table(
            index=["condition", "human_FPR_target", "budget"],
            columns="variant",
            values="acceptance_rate_%",
            aggfunc="first",
        )
        .reset_index()
    )

    for c in ["original", "gap_fixed", "gap_only"]:
        if c not in wide.columns:
            wide[c] = np.nan

    wide = wide[
        [
            "condition",
            "human_FPR_target",
            "budget",
            "original",
            "gap_fixed",
            "gap_only",
        ]
    ]

    wide = wide.rename(columns={
        "original": "original_acceptance_%",
        "gap_fixed": "gap_fixed_acceptance_%",
        "gap_only": "gap_only_acceptance_%",
    })

    wide["_condition_order"] = wide["condition"].map(condition_order)
    wide["_fpr_order"] = wide["human_FPR_target"].map(fpr_order)
    wide["_budget_order"] = wide["budget"].str.replace("B", "", regex=False).astype(int)

    wide = wide.sort_values(
        ["_budget_order", "_condition_order", "_fpr_order"]
    ).drop(columns=["_condition_order", "_fpr_order", "_budget_order"])

    wide.to_csv(OUT_DIR / "table9_tgce_human_fpr_acceptance_wide.csv", index=False)

    with open(OUT_DIR / "table9_tgce_human_fpr_acceptance_long.md", "w") as f:
        f.write(out.to_markdown(index=False))
        f.write("\n")

    with open(OUT_DIR / "table9_tgce_human_fpr_acceptance_wide.md", "w") as f:
        f.write(wide.to_markdown(index=False))
        f.write("\n")

    print("=" * 100)
    print("TABLE 9 — Human-FPR-calibrated acceptance of TGCE variants")
    print("=" * 100)
    print(wide.to_string(index=False))

    calib = load_actual_human_fpr()
    if not calib.empty:
        calib.to_csv(OUT_DIR / "table9_human_fpr_calibration_metadata.csv", index=False)
        print("\nCalibration metadata:")
        print(calib.to_string(index=False))
        print("\nSaved calibration metadata:")
        print(OUT_DIR / "table9_human_fpr_calibration_metadata.csv")

    print("\nSaved:")
    print(OUT_DIR / "table9_tgce_human_fpr_acceptance_long.csv")
    print(OUT_DIR / "table9_tgce_human_fpr_acceptance_wide.csv")


if __name__ == "__main__":
    main()
