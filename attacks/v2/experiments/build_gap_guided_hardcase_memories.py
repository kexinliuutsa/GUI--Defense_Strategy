#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import os
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[3]
_AHB_ROOT = Path("/storage/cjh/self-evolving-safety/Passing-the-Turing-Test-on-Screen-Agent-Humanization-Benchmark-main")

for p in [_REPO_ROOT, _AHB_ROOT]:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))


from attacks.v2.search_space import LatentVector
from attacks.v2.mutation import mutate
from attacks.v2.oracle import load_defense, load_long_tap_records
from attacks.v2.experiments.run_tgce_gap_counterfactual import (
    fill_record_index,
    candidate_to_detection,
    get_task_cluster_from_row,
)


LATENT_ORDER = [
    "z_spatial",
    "z_frequency",
    "z_duration",
    "z_gap",
    "z_jitter",
    "z_heterogeneity",
    "z_use_spatial",
    "z_use_temporal",
]


def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--candidate-csv",
        default="results/escape_basin_analysis/candidate_trials_flat.csv",
    )
    ap.add_argument("--method", default="hybrid")
    ap.add_argument("--budgets", default="10,30")
    ap.add_argument("--n-hardcases", type=int, default=31)

    ap.add_argument(
        "--participant-regex",
        default="",
        help="Optional regex filter on participant/agent/source columns, e.g. UI|TARS",
    )

    ap.add_argument(
        "--strategies",
        default="random,tgce_lowgap,gap_stratified,highgap_control",
    )

    ap.add_argument("--seed", type=int, default=20260920)

    ap.add_argument(
        "--verify-escape",
        action="store_true",
        help="Rerun detector on reconstructed hardcases and keep only currently escaping ones.",
    )

    ap.add_argument(
        "--defense-module",
        default="evaluation.frozen_v1v2v3_defense_hardened",
    )

    ap.add_argument(
        "--output-dir",
        default="results/gap_guided_hardcase_memories",
    )

    return ap.parse_args()


def stable_int(*parts, mod=2**32 - 1):
    s = "|".join(str(p) for p in parts)
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h[:12], 16) % mod


def parse_bool(x):
    return str(x).strip().lower() in {
        "true",
        "1",
        "yes",
        "y",
        "escaped",
        "success",
    }


def get_base_session(original_record):
    if isinstance(original_record, dict) and "gestures" in original_record:
        return original_record["gestures"]
    return original_record


def build_latent_logged(row):
    vals = []
    for name in LATENT_ORDER:
        if name not in row.index:
            raise KeyError(f"Missing latent column: {name}")
        vals.append(float(row[name]))
    return LatentVector(values=np.asarray(vals, dtype=float))


def filter_participant(df: pd.DataFrame, pattern: str):
    if not pattern:
        return df

    candidate_cols = [
        c for c in df.columns
        if c.lower() in {"participant", "agent", "source", "generator", "model"}
        or "participant" in c.lower()
        or "agent" in c.lower()
        or "source" in c.lower()
    ]

    if not candidate_cols:
        print("[WARN] No participant/source-like columns found. Skipping participant-regex filter.")
        return df

    mask = np.zeros(len(df), dtype=bool)
    rx = re.compile(pattern, flags=re.I)

    for c in candidate_cols:
        mask |= df[c].astype(str).apply(lambda x: bool(rx.search(x))).to_numpy()

    return df[mask].copy()


def dedupe_by_session(df: pd.DataFrame):
    """
    Prefer one hardcase per source trajectory when possible.
    """
    key_cols = []
    for c in ["record_index_filled", "record_index", "session_id", "participant"]:
        if c in df.columns:
            key_cols.append(c)

    if not key_cols:
        return df

    return df.drop_duplicates(subset=key_cols, keep="first")


def select_strategy(df: pd.DataFrame, strategy: str, n: int, seed: int):
    d = df.copy()

    if strategy == "random":
        return d.sample(n=min(n, len(d)), random_state=seed)

    if strategy == "tgce_lowgap":
        return d.sort_values(
            ["z_gap_float", "_raw_order"],
            ascending=[True, True],
        ).head(n)

    if strategy == "highgap_control":
        return d.sort_values(
            ["z_gap_float", "_raw_order"],
            ascending=[False, True],
        ).head(n)

    if strategy == "gap_stratified":
        # Stratify escaped hardcases by z_gap bins.
        # This tests whether covering the whole gap space is better than only low-gap.
        bins = [
            (0.00, 0.20, "very_low"),
            (0.20, 0.40, "low_mid"),
            (0.40, 0.70, "mid_high"),
            (0.70, 1.01, "very_high"),
        ]

        parts = []
        base_quota = max(1, n // len(bins))

        for lo, hi, name in bins:
            sub = d[(d["z_gap_float"] >= lo) & (d["z_gap_float"] < hi)].copy()
            sub = sub.sort_values(["z_gap_float", "_raw_order"], ascending=[True, True])
            parts.append(sub.head(base_quota))

        out = pd.concat(parts, ignore_index=False) if parts else pd.DataFrame()

        if len(out) < n:
            used = set(out["_raw_order"].tolist()) if len(out) else set()
            rest = d[~d["_raw_order"].isin(used)].copy()
            rest = rest.sort_values(["z_gap_float", "_raw_order"], ascending=[True, True])
            out = pd.concat([out, rest.head(n - len(out))], ignore_index=False)

        return out.head(n)

    raise ValueError(f"Unknown strategy: {strategy}")


def reconstruct_hardcases(selected: pd.DataFrame, records, defense=None, verify_escape=False, seed=0):
    hardcase_records = []
    meta_rows = []

    for i, (_, row) in enumerate(selected.iterrows()):
        record_index = int(float(row["record_index_filled"]))
        original_record = records[record_index]
        base_session = get_base_session(original_record)

        latent = build_latent_logged(row)

        rng = np.random.default_rng(
            stable_int(seed, row.get("method", ""), row.get("budget", ""), i, row["_raw_order"])
        )

        mutated = mutate(base_session, latent, rng)

        detected = None
        escaped_now = None

        if verify_escape:
            task_cluster = get_task_cluster_from_row(row)
            detected = bool(
                candidate_to_detection(
                    defense,
                    original_record,
                    mutated,
                    task_cluster=task_cluster,
                )
            )
            escaped_now = not detected

            if not escaped_now:
                continue

        hardcase_records.append(mutated)

        meta = {
            "hardcase_i": len(hardcase_records) - 1,
            "raw_order": int(row["_raw_order"]),
            "record_index": record_index,
            "method": row.get("method", ""),
            "budget": row.get("budget", ""),
            "participant": row.get("participant", ""),
            "session_id": row.get("session_id", ""),
            "z_gap": float(row.get("z_gap", np.nan)),
            "gap_scale": float(row.get("gap_scale", np.nan)) if "gap_scale" in row.index else np.nan,
            "z_duration": float(row.get("z_duration", np.nan)) if "z_duration" in row.index else np.nan,
            "z_spatial": float(row.get("z_spatial", np.nan)) if "z_spatial" in row.index else np.nan,
            "cost": float(row.get("cost", np.nan)) if "cost" in row.index else np.nan,
            "verified_detected": detected,
            "verified_escaped": escaped_now,
        }

        meta_rows.append(meta)

    return hardcase_records, pd.DataFrame(meta_rows)


def main():
    args = parse_args()

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    os.environ["GUI_DEFENSE_MODULE"] = args.defense_module

    print("=" * 120)
    print("BUILD GAP-GUIDED HARDCASE MEMORIES")
    print("=" * 120)
    print("candidate_csv:", args.candidate_csv)
    print("method:", args.method)
    print("budgets:", args.budgets)
    print("n_hardcases:", args.n_hardcases)
    print("participant_regex:", args.participant_regex)
    print("strategies:", args.strategies)
    print("verify_escape:", args.verify_escape)
    print("output_dir:", out_root)

    records = load_long_tap_records()
    defense = load_defense() if args.verify_escape else None

    df = pd.read_csv(args.candidate_csv, low_memory=False)
    df["_raw_order"] = np.arange(len(df))

    df["escaped_bool"] = df["escaped"].apply(parse_bool)
    df["budget_int"] = pd.to_numeric(df["budget"], errors="coerce").astype("Int64")
    df["z_gap_float"] = pd.to_numeric(df["z_gap"], errors="coerce")

    df = fill_record_index(df, records)

    budgets = [int(x.strip()) for x in args.budgets.split(",") if x.strip()]
    strategies = [x.strip() for x in args.strategies.split(",") if x.strip()]

    base = df[
        (df["method"].astype(str) == args.method)
        & (df["budget_int"].isin(budgets))
        & (df["escaped_bool"])
        & (df["z_gap_float"].notna())
    ].copy()

    base = filter_participant(base, args.participant_regex)

    if base.empty:
        raise SystemExit("No escaped candidates found after filtering.")

    print("candidate escaped pool:", len(base))

    all_summary = []

    for strategy in strategies:
        selected = select_strategy(
            base,
            strategy=strategy,
            n=args.n_hardcases * 3,
            seed=stable_int(args.seed, strategy),
        )

        selected = dedupe_by_session(selected).head(args.n_hardcases)

        strategy_dir = out_root / strategy
        strategy_dir.mkdir(parents=True, exist_ok=True)

        hardcase_records, meta = reconstruct_hardcases(
            selected,
            records=records,
            defense=defense,
            verify_escape=args.verify_escape,
            seed=stable_int(args.seed, strategy),
        )

        with open(strategy_dir / "hardcases_records.pkl", "wb") as f:
            pickle.dump(hardcase_records, f)

        meta.to_csv(strategy_dir / "hardcases_metadata.csv", index=False)

        selected.to_csv(strategy_dir / "selected_source_rows.csv", index=False)

        summary = {
            "strategy": strategy,
            "n_selected_source_rows": len(selected),
            "n_hardcase_records_saved": len(hardcase_records),
            "median_z_gap": float(meta["z_gap"].median()) if len(meta) else np.nan,
            "median_gap_scale": float(meta["gap_scale"].median()) if len(meta) else np.nan,
            "min_gap_scale": float(meta["gap_scale"].min()) if len(meta) else np.nan,
            "max_gap_scale": float(meta["gap_scale"].max()) if len(meta) else np.nan,
            "path": str(strategy_dir / "hardcases_records.pkl"),
        }

        all_summary.append(summary)

        print("\n" + "-" * 120)
        print("strategy:", strategy)
        print(pd.DataFrame([summary]).to_string(index=False))
        print("saved:", strategy_dir)

    summary_df = pd.DataFrame(all_summary)
    summary_df.to_csv(out_root / "hardcase_memory_summary.csv", index=False)

    print("\n" + "=" * 120)
    print("SUMMARY")
    print("=" * 120)
    print(summary_df.to_string(index=False))
    print("\nSaved:", out_root / "hardcase_memory_summary.csv")


if __name__ == "__main__":
    main()
