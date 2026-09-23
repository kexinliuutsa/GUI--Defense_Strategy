from __future__ import annotations

import argparse
import copy
import csv
import json
import pickle
from pathlib import Path

import numpy as np

from attacks.v2.search_space import LatentVector
from attacks.v2.mutation import mutate
from attacks.v2.oracle import load_long_tap_records


KEYS = [
    "z_spatial",
    "z_frequency",
    "z_duration",
    "z_gap",
    "z_jitter",
    "z_heterogeneity",
    "z_use_spatial",
    "z_use_temporal",
]


def latent_from_json(s):
    d = json.loads(s)
    arr = np.array([float(d[k]) for k in KEYS], dtype=float)
    arr = np.clip(arr, 0.0, 1.0)
    return LatentVector.from_numpy(arr)


def load_records():
    try:
        records = load_long_tap_records(max_scan=499)
    except TypeError:
        records = load_long_tap_records()

    if isinstance(records, tuple):
        for item in records:
            if isinstance(item, list):
                return item
        return records[0]

    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--dedup", default="record_idx_lowest_cost")
    ap.add_argument("--seed", type=int, default=20260916)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load manifest.
    rows = []
    with Path(args.manifest).open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r.get("best_latent_json"):
                rows.append(r)

    print("manifest rows:", len(rows))

    # Dedup by record_idx, keep lowest cost.
    if args.dedup == "record_idx_lowest_cost":
        by_idx = {}
        for r in rows:
            idx = int(r["record_idx"])
            cost = float(r["best_cost"]) if r.get("best_cost") else 999999.0

            if idx not in by_idx:
                by_idx[idx] = r
            else:
                old = by_idx[idx]
                old_cost = float(old["best_cost"]) if old.get("best_cost") else 999999.0
                if cost < old_cost:
                    by_idx[idx] = r

        rows = list(by_idx.values())

    rows.sort(key=lambda r: float(r["best_cost"]) if r.get("best_cost") else 999999.0)

    print("rows after dedup:", len(rows))

    records = load_records()
    print("loaded original records:", len(records))

    # Build session_id index.
    sid_to_record = {}
    for rec in records:
        if isinstance(rec, dict) and rec.get("session_id") is not None:
            sid_to_record[str(rec["session_id"])] = rec

    rng = np.random.default_rng(args.seed)

    mutated_records = []
    metadata = []
    missing = []

    for r in rows:
        idx = int(r["record_idx"])
        sid = str(r.get("session_id"))

        original = sid_to_record.get(sid)

        if original is None and 0 <= idx < len(records):
            original = records[idx]

        if original is None:
            missing.append((idx, sid, "cannot_find_original"))
            continue

        if not isinstance(original, dict):
            missing.append((idx, sid, f"original_not_dict_type={type(original)}"))
            continue

        if "gestures" not in original:
            missing.append((idx, sid, f"no_gestures_key_keys={list(original.keys())}"))
            continue

        gestures = original["gestures"]

        # Important: mutate must receive gestures, not the whole record dict.
        latent = latent_from_json(r["best_latent_json"])

        try:
            mutated_gestures = mutate(copy.deepcopy(gestures), latent, rng)
        except Exception as e:
            missing.append((idx, sid, f"mutate_error={repr(e)}"))
            continue

        if isinstance(mutated_gestures, tuple):
            mutated_gestures = mutated_gestures[0]

        meta = {
            "source": r.get("source"),
            "method": r.get("method"),
            "budget": int(r["budget"]) if r.get("budget") else None,
            "seed": int(r["seed"]) if r.get("seed") else None,
            "record_idx": idx,
            "participant": r.get("participant"),
            "session_id": sid,
            "queries_to_first_escape": int(r["queries_to_first_escape"]) if r.get("queries_to_first_escape") else None,
            "best_cost": float(r["best_cost"]) if r.get("best_cost") else None,
            "best_latent": json.loads(r["best_latent_json"]),
        }

        new_record = copy.deepcopy(original)
        new_record["gestures"] = mutated_gestures
        new_record["_attack_metadata"] = meta

        mutated_records.append(new_record)
        metadata.append(meta)

    out_pkl = out_dir / "escaped_mutated_records.pkl"
    out_jsonl = out_dir / "escaped_mutated_metadata.jsonl"
    out_missing = out_dir / "missing_or_failed.csv"

    with out_pkl.open("wb") as f:
        pickle.dump(
            {
                "mutated_records": mutated_records,
                "metadata": metadata,
            },
            f,
        )

    with out_jsonl.open("w") as f:
        for m in metadata:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    with out_missing.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["record_idx", "session_id", "reason"])
        writer.writerows(missing)

    print("=" * 90)
    print("Export escaped mutated records")
    print("=" * 90)
    print("mutated records:", len(mutated_records))
    print("metadata:", out_jsonl)
    print("pickle:", out_pkl)
    print("missing/failed:", len(missing), out_missing)


if __name__ == "__main__":
    main()
