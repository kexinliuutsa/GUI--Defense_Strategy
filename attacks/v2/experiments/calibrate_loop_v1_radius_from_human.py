#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

from evaluation.defense_loop_v1_hardcase_memory import FrozenV1V2V3Defense


OUT = Path("results/v2_defense_loop_v1_human_fpr")
OUT.mkdir(parents=True, exist_ok=True)

HARDCASE_PKL = (
    "results/v2_loop_hard_cases/mutated_records/"
    "escaped_mutated_records.pkl"
)


def looks_like_record_list(x):
    return (
        isinstance(x, list)
        and len(x) > 0
        and isinstance(x[0], dict)
        and "gestures" in x[0]
    )


def main():
    os.environ["LOOP_HARDCASE_PKL"] = HARDCASE_PKL
    os.environ["LOOP_MEMORY_RADIUS"] = "1.0"

    d0 = FrozenV1V2V3Defense()

    candidates = []

    for space_name in ["ns", "inner"]:
        space = getattr(d0, space_name, {})
        if not isinstance(space, dict):
            continue

        for key, val in sorted(space.items()):
            if looks_like_record_list(val):
                candidates.append((space_name, key, val))

    human_like = []

    for space_name, key, records in candidates:
        if "human" in key.lower():
            human_like.append((space_name, key, records))

    if not human_like:
        for space_name, key, records in candidates:
            parts = {
                str(r.get("participant", "unknown")).lower()
                for r in records[:200]
            }
            if any(p.startswith("user") or p == "human" for p in parts):
                human_like.append((space_name, key, records))

    if not human_like:
        raise SystemExit("No human-like record list found.")

    space_name, key, human_records = max(human_like, key=lambda x: len(x[2]))

    print("using human records:", space_name, key, "n=", len(human_records))

    rows = []

    for rec in human_records:
        participant = str(rec.get("participant", "unknown"))
        session_id = str(rec.get("session_id", "unknown"))
        dist = d0.hardcase_distance(rec)

        rows.append({
            "participant": participant,
            "session_id": session_id,
            "distance_to_hardcase": float(dist),
        })

    dist_df = pd.DataFrame(rows)
    dist_path = OUT / "human_distance_to_hardcase_memory.csv"
    dist_df.to_csv(dist_path, index=False)

    ds = dist_df["distance_to_hardcase"].to_numpy(dtype=float)

    calib_rows = []

    for target_fpr in [1.0, 2.5, 5.0, 10.0, 15.0, 20.0, 25.0]:
        radius = float(np.percentile(ds, target_fpr))
        actual_fpr = float(np.mean(ds <= radius) * 100)

        calib_rows.append({
            "target_human_FPR_%": target_fpr,
            "calibrated_radius": radius,
            "actual_human_FPR_%": actual_fpr,
            "human_detected": int(np.sum(ds <= radius)),
            "human_total": int(len(ds)),
        })

    calib_df = pd.DataFrame(calib_rows)
    calib_path = OUT / "human_calibrated_radius_table.csv"
    calib_df.to_csv(calib_path, index=False)

    print("\nHuman calibrated radius table")
    print(calib_df.to_string(index=False))
    print("\nsaved:", calib_path)
    print("saved:", dist_path)


if __name__ == "__main__":
    main()
