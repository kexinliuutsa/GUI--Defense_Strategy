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

RADII = [3.0, 3.1, 3.2, 3.25, 3.3, 3.5, 4.0, 5.0]


def looks_like_record_list(x):
    return (
        isinstance(x, list)
        and len(x) > 0
        and isinstance(x[0], dict)
        and "gestures" in x[0]
    )


def main():
    os.environ["LOOP_HARDCASE_PKL"] = HARDCASE_PKL
    os.environ["LOOP_MEMORY_RADIUS"] = "3.25"

    d0 = FrozenV1V2V3Defense()

    candidates = []

    print("=" * 100)
    print("Candidate record lists")
    print("=" * 100)

    for space_name in ["ns", "inner"]:
        space = getattr(d0, space_name, {})
        if not isinstance(space, dict):
            continue

        for key, val in sorted(space.items()):
            if looks_like_record_list(val):
                participants = sorted({
                    str(r.get("participant", "unknown"))
                    for r in val[:100]
                })

                print(
                    space_name,
                    key,
                    "len=",
                    len(val),
                    "participants_sample=",
                    participants[:10],
                )

                candidates.append((space_name, key, val))

    human_like = []

    # 先找变量名里带 human 的 list
    for space_name, key, records in candidates:
        if "human" in key.lower():
            human_like.append((space_name, key, records))

    # 如果变量名没写 human，就找 participant 是 user1/user2/... 的 list
    if not human_like:
        for space_name, key, records in candidates:
            parts = sorted({
                str(r.get("participant", "unknown")).lower()
                for r in records[:200]
            })

            if any(p.startswith("user") or p == "human" for p in parts):
                human_like.append((space_name, key, records))

    if not human_like:
        raise SystemExit(
            "No human-like records found. Check candidate lists printed above."
        )

    space_name, key, human_records = max(
        human_like,
        key=lambda x: len(x[2]),
    )

    print("\n" + "=" * 100)
    print("Using human records:", space_name, key, "n=", len(human_records))
    print("=" * 100)

    cnt = defaultdict(int)
    for rec in human_records:
        cnt[str(rec.get("participant", "unknown"))] += 1

    print("Human participant counts:")
    for k, v in sorted(cnt.items()):
        print(k, v)

    rows = []

    for R in RADII:
        print("\n" + "=" * 100)
        print("radius =", R)
        print("=" * 100)

        os.environ["LOOP_MEMORY_RADIUS"] = str(R)
        os.environ["LOOP_HARDCASE_PKL"] = HARDCASE_PKL

        d = FrozenV1V2V3Defense()

        stats = defaultdict(lambda: {
            "n": 0,
            "memory_detect": 0,
            "distances": [],
        })

        for i, rec in enumerate(human_records):
            participant = str(rec.get("participant", "unknown"))

            # IMPORTANT:
            # Do NOT call score_record() here.
            # Human participants such as user1 do not have frozen V2 cross-fit policy.
            dist = d.hardcase_distance(rec)
            memory_detect = bool(dist <= R)

            stats[participant]["n"] += 1
            stats[participant]["memory_detect"] += int(memory_detect)
            stats[participant]["distances"].append(dist)

            if (i + 1) % 100 == 0:
                print("processed", i + 1, "/", len(human_records))

        for participant, s in sorted(stats.items()):
            n = s["n"]
            ds = np.asarray(s["distances"], dtype=float)

            rows.append({
                "radius": R,
                "participant": participant,
                "n": n,
                "memory_detect": s["memory_detect"],
                "memory_FPR_%": round(s["memory_detect"] / n * 100, 2),
                "distance_min": round(float(np.min(ds)), 4),
                "distance_p05": round(float(np.percentile(ds, 5)), 4),
                "distance_p25": round(float(np.percentile(ds, 25)), 4),
                "distance_median": round(float(np.median(ds)), 4),
                "distance_p75": round(float(np.percentile(ds, 75)), 4),
                "distance_p95": round(float(np.percentile(ds, 95)), 4),
            })

    df = pd.DataFrame(rows)

    out = OUT / "human_memory_fpr_by_radius.csv"
    df.to_csv(out, index=False)

    print("\n" + "=" * 100)
    print("Human memory FPR by radius")
    print("=" * 100)
    print(df.to_string(index=False))
    print("\nsaved:", out)


if __name__ == "__main__":
    main()
