#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path
from collections import defaultdict

import pandas as pd

from attacks.v2.oracle import load_long_tap_records
from evaluation.defense_loop_v1_hardcase_memory import FrozenV1V2V3Defense


OUT = Path("results/v2_defense_loop_v1_participant_check")
OUT.mkdir(parents=True, exist_ok=True)

CSV = OUT / "detect_rate_by_participant_radius.csv"
LOG_TXT = OUT / "participant_counts.txt"

RADII = [3.0, 3.1, 3.2, 3.25, 3.3, 3.5, 4.0, 5.0]

HARDCASE_PKL = (
    "results/v2_loop_hard_cases/mutated_records/"
    "escaped_mutated_records.pkl"
)


def load_records():
    try:
        records = load_long_tap_records(max_scan=499)
    except TypeError:
        records = load_long_tap_records()

    if isinstance(records, tuple):
        for item in records:
            if isinstance(item, list):
                records = item
                break

    return records


def load_existing():
    if CSV.exists():
        return pd.read_csv(CSV)
    return pd.DataFrame()


def save_rows(rows):
    df_new = pd.DataFrame(rows)

    if CSV.exists():
        old = pd.read_csv(CSV)
        df = pd.concat([old, df_new], ignore_index=True)
        df = df.drop_duplicates(
            subset=["radius", "participant"],
            keep="last",
        )
    else:
        df = df_new

    df = df.sort_values(["radius", "participant"])
    df.to_csv(CSV, index=False)
    return df


def main():
    records = load_records()

    participants = sorted(
        {str(r.get("participant", "unknown")) for r in records}
    )

    with LOG_TXT.open("w") as f:
        f.write("Participant counts\n")
        f.write("=" * 80 + "\n")
        cnt = defaultdict(int)
        for r in records:
            cnt[str(r.get("participant", "unknown"))] += 1

        for k, v in sorted(cnt.items()):
            line = f"{repr(k)} {v}"
            print(line)
            f.write(line + "\n")

    existing = load_existing()

    done_radii = set()
    if not existing.empty:
        for R in RADII:
            sub = existing[existing["radius"].astype(float) == float(R)]
            done_participants = set(sub["participant"].astype(str))
            if set(participants).issubset(done_participants):
                done_radii.add(float(R))

    print("=" * 100)
    print("Resume participant detect-rate check")
    print("=" * 100)
    print("output:", CSV)
    print("done radii:", sorted(done_radii))
    print("remaining:", [R for R in RADII if float(R) not in done_radii])

    for R in RADII:
        if float(R) in done_radii:
            print(f"[skip] radius={R} already complete")
            continue

        print("=" * 100)
        print(f"[run] radius={R}")
        print("=" * 100)

        os.environ["LOOP_MEMORY_RADIUS"] = str(R)
        os.environ["LOOP_HARDCASE_PKL"] = HARDCASE_PKL

        defense = FrozenV1V2V3Defense()

        stats = defaultdict(lambda: {
            "n": 0,
            "memory_detect": 0,
            "frozen_detect": 0,
            "final_detect": 0,
        })

        for i, rec in enumerate(records):
            participant = str(rec.get("participant", "unknown"))

            out = defense.score_record(rec)

            stats[participant]["n"] += 1
            stats[participant]["memory_detect"] += int(
                out.get("loop_memory_detect", False)
            )
            stats[participant]["frozen_detect"] += int(
                out.get("frozen_final_detect", False)
            )
            stats[participant]["final_detect"] += int(
                out.get("final_detect", False)
            )

            if (i + 1) % 100 == 0:
                print(f"  processed {i + 1}/{len(records)}")

        rows = []
        for participant, s in sorted(stats.items()):
            n = s["n"]
            rows.append({
                "radius": R,
                "participant": participant,
                "n": n,
                "memory_detect": s["memory_detect"],
                "memory_detect_rate_%": round(
                    s["memory_detect"] / n * 100, 2
                ) if n else None,
                "frozen_detect": s["frozen_detect"],
                "frozen_detect_rate_%": round(
                    s["frozen_detect"] / n * 100, 2
                ) if n else None,
                "final_detect": s["final_detect"],
                "final_detect_rate_%": round(
                    s["final_detect"] / n * 100, 2
                ) if n else None,
            })

        df = save_rows(rows)

        print(f"[saved] radius={R} -> {CSV}")
        print(df[df["radius"].astype(float) == float(R)].to_string(index=False))

    print("=" * 100)
    print("DONE")
    print("=" * 100)
    print("saved:", CSV)


if __name__ == "__main__":
    main()
