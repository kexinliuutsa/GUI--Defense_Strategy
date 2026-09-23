#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd


_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_AHB_ROOT = Path("/storage/cjh/self-evolving-safety/Passing-the-Turing-Test-on-Screen-Agent-Humanization-Benchmark-main")
if _AHB_ROOT.exists() and str(_AHB_ROOT) not in sys.path:
    sys.path.insert(0, str(_AHB_ROOT))


def parse_bool_series(s):
    def one(x):
        if isinstance(x, bool):
            return x
        sx = str(x).strip().lower()
        if sx in {"true", "1", "yes", "y", "escaped", "success"}:
            return True
        return False
    return s.apply(one).astype(bool)


def make_record_lookup(records):
    lookup = {}

    for i, r in enumerate(records):
        if not isinstance(r, dict):
            continue

        sid = r.get("session_id")
        participant = r.get("participant")
        group = r.get("group")

        keys = [
            (str(participant), str(sid)),
            (str(group), str(participant), str(sid)),
            str(sid),
        ]

        for k in keys:
            if k is not None and "None" not in str(k):
                lookup[k] = i

    return lookup


def fill_record_index(df, records):
    df = df.copy()

    # Normalize existing or alternate index columns.
    candidate_cols = [
        "record_index",
        "record_idx",
        "idx",
        "session_index",
        "source_record_index",
        "original_record_index",
    ]

    if "record_index_filled" not in df.columns:
        df["record_index_filled"] = np.nan

    for c in candidate_cols:
        if c in df.columns:
            v = pd.to_numeric(df[c], errors="coerce")
            df["record_index_filled"] = df["record_index_filled"].fillna(v)

    # Map from participant/session_id if possible.
    lookup = make_record_lookup(records)

    def by_metadata(row):
        if pd.notna(row.get("record_index_filled", np.nan)):
            return row["record_index_filled"]

        sid = row.get("session_id", None)
        participant = row.get("participant", None)
        group = row.get("group", None)

        keys = [
            (str(participant), str(sid)),
            (str(group), str(participant), str(sid)),
            str(sid),
        ]

        for k in keys:
            if k in lookup:
                return lookup[k]

        return np.nan

    df["record_index_filled"] = df.apply(by_metadata, axis=1)

    return df


LATENT_KEYS = [
    "z_spatial",
    "z_frequency",
    "z_duration",
    "z_gap",
    "z_jitter",
    "z_heterogeneity",
    "z_use_spatial",
    "z_use_temporal",
]


def make_latent_values(row, mode):
    """
    Build normalized latent dictionary from candidate row.

    LatentVector in this repo expects:
        LatentVector(values=np.ndarray)

    Order:
        z_spatial, z_frequency, z_duration, z_gap,
        z_jitter, z_heterogeneity, z_use_spatial, z_use_temporal
    """
    defaults = {
        "z_spatial": 0.5,
        "z_frequency": 0.5,
        "z_duration": 0.5,
        "z_gap": 0.5,
        "z_jitter": 0.5,
        "z_heterogeneity": 0.5,
        "z_use_spatial": 0.0,
        "z_use_temporal": 1.0,
    }

    vals = {}

    for k in LATENT_KEYS:
        if k in row.index and pd.notna(row[k]):
            vals[k] = float(row[k])
        else:
            vals[k] = float(defaults[k])

    if mode == "original":
        pass

    elif mode == "gap_fixed":
        # Neutralize gap compression only.
        vals["z_gap"] = 0.50

    elif mode == "gap_only":
        # Keep only original gap compression.
        original_gap = float(vals["z_gap"])

        vals = dict(defaults)
        vals["z_gap"] = original_gap
        vals["z_use_temporal"] = 1.0
        vals["z_use_spatial"] = 0.0

    else:
        raise ValueError(mode)

    # Clip to valid latent range.
    arr = np.array([vals[k] for k in LATENT_KEYS], dtype=float)
    arr = np.clip(arr, 0.0, 1.0)

    return arr


def build_latent(LatentVector, row, mode):
    arr = make_latent_values(row, mode)
    return LatentVector(values=arr)


def get_task_cluster_from_row(row):
    for c in [
        "task_cluster",
        "cluster",
        "cid",
        "original_task_cluster",
        "frozen_task_cluster",
    ]:
        if c in row.index and pd.notna(row[c]):
            try:
                return int(float(row[c]))
            except Exception:
                pass
    return None


def candidate_to_detection(defense, original_record, mutated, task_cluster=None):
    """
    Evaluate mutated trajectory.

    mutate() may return either:
    - a full record dict with gestures
    - a gestures list
    """
    if isinstance(mutated, dict) and "gestures" in mutated:
        try:
            return bool(defense.detect_record(mutated, task_cluster=task_cluster))
        except TypeError:
            return bool(defense.detect_record(mutated))

    if isinstance(mutated, list):
        try:
            oracle = defense.make_oracle(original_record, task_cluster=task_cluster)
        except TypeError:
            oracle = defense.make_oracle(original_record)
        return bool(oracle(mutated))

    if hasattr(mutated, "gestures"):
        rec = dict(original_record)
        rec["gestures"] = getattr(mutated, "gestures")
        try:
            return bool(defense.detect_record(rec, task_cluster=task_cluster))
        except TypeError:
            return bool(defense.detect_record(rec))

    raise TypeError(f"Unsupported mutate output type: {type(mutated)!r}")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--candidate-csv",
        default="results/escape_basin_analysis/candidate_trials_flat.csv",
    )
    ap.add_argument("--method", default="hybrid")
    ap.add_argument("--budgets", default="10,30")
    ap.add_argument("--max-candidates-per-budget", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260917)
    ap.add_argument("--output-dir", default="results/tgce_counterfactual")

    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("TGCE GAP COUNTERFACTUAL")
    print("=" * 100)

    from attacks.v2.oracle import load_defense, load_long_tap_records
    from attacks.v2.search_space import LatentVector
    from attacks.v2.mutation import mutate

    rng = np.random.default_rng(args.seed)

    defense = load_defense()
    records = load_long_tap_records()

    df = pd.read_csv(args.candidate_csv, low_memory=False)

    df = df[df["method"].astype(str) == args.method].copy()
    df["budget_int"] = pd.to_numeric(df["budget"], errors="coerce").astype("Int64")

    budgets = [int(float(x.strip())) for x in args.budgets.split(",") if x.strip()]
    df = df[df["budget_int"].isin(budgets)].copy()

    for c in ["z_gap", "escaped"]:
        if c not in df.columns:
            raise SystemExit(f"Missing required column: {c}")

    df["escaped_bool"] = parse_bool_series(df["escaped"])
    df = fill_record_index(df, records)

    print("candidate rows after method/budget filter:", len(df))
    print("escaped_bool counts:")
    print(df["escaped_bool"].value_counts(dropna=False).to_string())

    esc0 = df[df["escaped_bool"]].copy()

    print("\nmissing among escaped before drop:")
    for c in ["record_index", "record_index_filled", "session_id", "participant", "z_gap", "gap_scale"]:
        if c in esc0.columns:
            print(c, "non-null", esc0[c].notna().sum(), "missing", esc0[c].isna().sum())
        else:
            print(c, "MISSING")

    esc = esc0.dropna(subset=["record_index_filled", "z_gap"]).copy()

    print("\nusable escaped candidates:", len(esc))

    if esc.empty:
        print("\nFirst escaped rows for debugging:")
        cols = [
            c for c in [
                "source_file",
                "line_no",
                "method",
                "budget",
                "seed",
                "record_index",
                "record_index_filled",
                "record_idx",
                "session_id",
                "participant",
                "z_gap",
                "gap_scale",
                "q",
                "cost",
            ]
            if c in esc0.columns
        ]
        print(esc0[cols].head(30).to_string(index=False))
        raise SystemExit("No escaped candidates selected after filling record index.")

    esc["z_gap_float"] = pd.to_numeric(esc["z_gap"], errors="coerce")
    esc = esc.sort_values(["budget_int", "z_gap_float"])

    rows = []

    for budget, g in esc.groupby("budget_int"):
        g = g.head(args.max_candidates_per_budget).copy()

        print(f"\nbudget={budget}, testing {len(g)} escaped candidates")

        for _, row in g.iterrows():
            record_index = int(float(row["record_index_filled"]))

            if record_index < 0 or record_index >= len(records):
                continue

            original_record = records[record_index]
            task_cluster = get_task_cluster_from_row(row)

            for mode in ["original", "gap_fixed", "gap_only"]:
                try:
                    latent = build_latent(LatentVector, row, mode)

                    # mutate() expects the gesture/session list, not the full record dict.
                    if isinstance(original_record, dict) and "gestures" in original_record:
                        base_session = original_record["gestures"]
                    else:
                        base_session = original_record

                    mutated = mutate(base_session, latent, rng)

                    detected = candidate_to_detection(
                        defense,
                        original_record,
                        mutated,
                        task_cluster=task_cluster,
                    )

                    rows.append({
                        "method": args.method,
                        "budget": int(budget),
                        "record_index": record_index,
                        "participant": original_record.get("participant", None)
                            if isinstance(original_record, dict) else None,
                        "session_id": original_record.get("session_id", None)
                            if isinstance(original_record, dict) else None,
                        "source_q": row.get("q", None),
                        "source_cost": row.get("cost", None),
                        "source_z_gap": row.get("z_gap", None),
                        "source_gap_scale": row.get("gap_scale", None),
                        "counterfactual_mode": mode,
                        "detected": bool(detected),
                        "escaped": bool(not detected),
                        "error": "",
                    })

                except Exception as e:
                    rows.append({
                        "method": args.method,
                        "budget": int(budget),
                        "record_index": record_index,
                        "participant": original_record.get("participant", None)
                            if isinstance(original_record, dict) else None,
                        "session_id": original_record.get("session_id", None)
                            if isinstance(original_record, dict) else None,
                        "source_q": row.get("q", None),
                        "source_cost": row.get("cost", None),
                        "source_z_gap": row.get("z_gap", None),
                        "source_gap_scale": row.get("gap_scale", None),
                        "counterfactual_mode": mode,
                        "detected": None,
                        "escaped": None,
                        "error": repr(e),
                    })

    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "tgce_gap_counterfactual_raw.csv", index=False)

    valid = out[out["escaped"].notna()].copy()

    if valid.empty:
        print("\nNo valid counterfactual evaluations.")
        print("First errors:")
        print(out[out["error"].astype(str) != ""].head(30).to_string(index=False))
        raise SystemExit(1)

    summary = (
        valid
        .groupby(["budget", "counterfactual_mode"], dropna=False)
        .agg(
            n=("escaped", "size"),
            escaped=("escaped", "sum"),
            escape_rate_pct=("escaped", lambda x: round(float(np.mean(x)) * 100, 2)),
        )
        .reset_index()
    )

    summary.to_csv(out_dir / "table7_tgce_gap_counterfactual_summary.csv", index=False)

    with open(out_dir / "table7_tgce_gap_counterfactual_summary.md", "w") as f:
        f.write(summary.to_markdown(index=False))
        f.write("\n")

    print("\n" + "=" * 100)
    print("Counterfactual summary")
    print("=" * 100)
    print(summary.to_string(index=False))

    n_errors = int(out["escaped"].isna().sum())
    print("\nerrors:", n_errors)

    if n_errors:
        print(out[out["escaped"].isna()].head(20).to_string(index=False))

    print("\nSaved:")
    print(out_dir / "tgce_gap_counterfactual_raw.csv")
    print(out_dir / "table7_tgce_gap_counterfactual_summary.csv")


if __name__ == "__main__":
    main()
