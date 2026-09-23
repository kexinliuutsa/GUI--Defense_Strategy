#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from evaluation.defense_loop_v1_hardcase_memory import FrozenV1V2V3Defense


def main():
    out_dir = Path("results/v2_defense_loop_v1_human_fpr_user_split")
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ["LOOP_HARDCASE_PKL"] = (
        "results/v2_loop_hard_cases/mutated_records/escaped_mutated_records.pkl"
    )

    defense = FrozenV1V2V3Defense()

    inner = getattr(defense, "inner", {})
    human_records = list(inner["human_records"])

    rows = []

    for r in human_records:
        participant = str(r.get("participant", "UNKNOWN"))
        d = float(defense.hardcase_distance(r))
        rows.append({
            "participant": participant,
            "distance": d,
        })

    df = pd.DataFrame(rows)

    targets = [2.5, 5.0, 10.0]

    split_rows = []

    for heldout_user in sorted(df["participant"].unique()):
        train = df[df["participant"] != heldout_user].copy()
        test = df[df["participant"] == heldout_user].copy()

        for target in targets:
            # radius calibrated on train humans
            q = target / 100.0
            radius = float(train["distance"].quantile(q, interpolation="higher"))

            train_fpr = float((train["distance"] <= radius).mean() * 100)
            test_fpr = float((test["distance"] <= radius).mean() * 100)

            split_rows.append({
                "heldout_user": heldout_user,
                "target_FPR_%": target,
                "calibrated_radius": radius,
                "train_human_n": len(train),
                "test_human_n": len(test),
                "train_actual_FPR_%": round(train_fpr, 2),
                "test_actual_FPR_%": round(test_fpr, 2),
            })

    split_df = pd.DataFrame(split_rows)

    print("=" * 100)
    print("Leave-one-user-out human FPR calibration")
    print("=" * 100)
    print(split_df.to_string(index=False))

    split_df.to_csv(out_dir / "leave_one_user_out_human_fpr.csv", index=False)

    summary = (
        split_df
        .groupby("target_FPR_%")
        .agg(
            n_splits=("heldout_user", "nunique"),
            mean_test_FPR_pct=("test_actual_FPR_%", "mean"),
            min_test_FPR_pct=("test_actual_FPR_%", "min"),
            max_test_FPR_pct=("test_actual_FPR_%", "max"),
            median_test_FPR_pct=("test_actual_FPR_%", "median"),
            mean_radius=("calibrated_radius", "mean"),
            min_radius=("calibrated_radius", "min"),
            max_radius=("calibrated_radius", "max"),
        )
        .reset_index()
    )

    for c in summary.columns:
        if c != "target_FPR_%":
            try:
                summary[c] = summary[c].round(3)
            except Exception:
                pass

    print("\n" + "=" * 100)
    print("Summary")
    print("=" * 100)
    print(summary.to_string(index=False))

    summary.to_csv(out_dir / "leave_one_user_out_human_fpr_summary.csv", index=False)

    print("\nsaved:")
    print(out_dir / "leave_one_user_out_human_fpr.csv")
    print(out_dir / "leave_one_user_out_human_fpr_summary.csv")


if __name__ == "__main__":
    main()
