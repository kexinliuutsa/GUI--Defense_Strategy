#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import os
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

GAP_SCALE_MIN = 0.55
GAP_SCALE_RANGE = 1.25


def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--candidate-csv",
        default="results/escape_basin_analysis/candidate_trials_flat.csv",
        help="Used only to define the evaluation session set and task_cluster metadata.",
    )
    ap.add_argument("--budgets", default="10,30,100")
    ap.add_argument("--n-sessions", type=int, default=300)

    ap.add_argument(
        "--session-source-method",
        default="hybrid",
        help="Use sessions appearing in this method from candidate-csv.",
    )
    ap.add_argument(
        "--session-source-budget",
        type=int,
        default=30,
        help="Use sessions appearing in this budget from candidate-csv.",
    )

    ap.add_argument(
        "--participant-regex",
        default="",
        help="Optional regex filter on participant/agent/source columns.",
    )

    ap.add_argument(
        "--gap-policy",
        choices=["low_beta", "low_grid", "mixed_low_uniform"],
        default="mixed_low_uniform",
    )

    ap.add_argument("--seed", type=int, default=20260920)

    ap.add_argument(
        "--defense-module",
        default="evaluation.frozen_v1v2v3_defense_hardened",
    )

    ap.add_argument(
        "--output-dir",
        default="results/gap_aware_attacker",
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


def gap_scale_from_z(z_gap: float) -> float:
    return GAP_SCALE_MIN + GAP_SCALE_RANGE * float(z_gap)


def get_base_session(original_record):
    if isinstance(original_record, dict) and "gestures" in original_record:
        return original_record["gestures"]
    return original_record


def filter_participant(df: pd.DataFrame, pattern: str):
    if not pattern:
        return df

    import re

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

    rx = re.compile(pattern, flags=re.I)
    mask = np.zeros(len(df), dtype=bool)

    for c in candidate_cols:
        mask |= df[c].astype(str).apply(lambda x: bool(rx.search(x))).to_numpy()

    return df[mask].copy()


def choose_eval_sessions(df: pd.DataFrame, records, args):
    d = df[
        (df["method"].astype(str) == args.session_source_method)
        & (df["budget_int"].eq(args.session_source_budget))
    ].copy()

    d = filter_participant(d, args.participant_regex)

    if d.empty:
        raise SystemExit("No candidate rows found for session set.")

    d = d.drop_duplicates(subset=["record_index_filled"], keep="first")
    d = d.sort_values("_raw_order").head(args.n_sessions)

    return d


def sample_z_gap(rng: np.random.Generator, policy: str) -> float:
    """
    Low z_gap means compressed inter-gesture gaps.
    """
    if policy == "low_beta":
        # Strongly concentrated near 0.
        return float(rng.beta(1.2, 6.0))

    if policy == "low_grid":
        grid = np.array([0.02, 0.04, 0.08, 0.12, 0.16, 0.20, 0.28, 0.36])
        return float(rng.choice(grid))

    if policy == "mixed_low_uniform":
        u = rng.random()

        if u < 0.65:
            return float(rng.beta(1.2, 6.0))
        elif u < 0.90:
            grid = np.array([0.02, 0.04, 0.08, 0.12, 0.16, 0.20, 0.28, 0.36])
            return float(rng.choice(grid))
        else:
            return float(rng.random())

    raise ValueError(policy)


def sample_latent(rng: np.random.Generator, policy: str) -> LatentVector:
    """
    Gap-aware attacker:
    - z_gap biased low.
    - temporal mutation usually enabled.
    - other dimensions are moderately randomized.
    """
    z_gap = sample_z_gap(rng, policy)

    # Keep other dimensions broad enough so this is still an adaptive attack,
    # not only a one-dimensional sweep.
    z_spatial = float(rng.beta(2.0, 2.0))
    z_frequency = float(rng.random())
    z_duration = float(rng.beta(2.0, 2.5))
    z_jitter = float(rng.beta(2.0, 2.0))
    z_heterogeneity = float(rng.beta(2.0, 2.0))

    # Prefer temporal enabled; spatial mixed.
    z_use_temporal = 1.0 if rng.random() < 0.90 else 0.0
    z_use_spatial = 1.0 if rng.random() < 0.60 else 0.0

    vals = [
        z_spatial,
        z_frequency,
        z_duration,
        z_gap,
        z_jitter,
        z_heterogeneity,
        z_use_spatial,
        z_use_temporal,
    ]

    return LatentVector(values=np.asarray(vals, dtype=float))


def eval_candidate(defense, original_record, mutated, task_cluster):
    detected = candidate_to_detection(
        defense,
        original_record,
        mutated,
        task_cluster=task_cluster,
    )
    detected = bool(detected)
    escaped = not detected
    return detected, escaped


def bootstrap_ci_binary(x, n_boot=3000, seed=0):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)

    for i in range(n_boot):
        vals[i] = rng.choice(x, size=len(x), replace=True).mean()

    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def summarize(raw: pd.DataFrame, out_dir: Path):
    session_rows = []

    for (budget, record_index), g in raw.groupby(["budget", "record_index"]):
        escaped_any = bool(g["escaped"].any())

        if escaped_any:
            q_first = int(g[g["escaped"]]["query_i"].iloc[0])
            best_gap_scale = float(g[g["escaped"]]["gap_scale"].iloc[0])
            best_z_gap = float(g[g["escaped"]]["z_gap"].iloc[0])
        else:
            q_first = np.nan
            best_gap_scale = np.nan
            best_z_gap = np.nan

        session_rows.append({
            "budget": budget,
            "record_index": record_index,
            "escaped": escaped_any,
            "queries_used": int(g["query_i"].max()),
            "q_first_escape": q_first,
            "first_escape_z_gap": best_z_gap,
            "first_escape_gap_scale": best_gap_scale,
        })

    session = pd.DataFrame(session_rows)
    session.to_csv(out_dir / "gap_aware_session_results.csv", index=False)

    agg_rows = []

    for budget, g in session.groupby("budget"):
        x = g["escaped"].astype(bool).to_numpy()
        lo, hi = bootstrap_ci_binary(x, seed=stable_int("agg", budget))

        q = g.loc[g["escaped"], "q_first_escape"].dropna()

        agg_rows.append({
            "attack": "gap_aware",
            "budget": budget,
            "n_sessions": len(g),
            "escaped": int(x.sum()),
            "ASR_%": round(float(x.mean()) * 100, 2),
            "ASR_95ci": f"[{lo*100:.2f}, {hi*100:.2f}]",
            "median_q_first_escape": float(q.median()) if len(q) else np.nan,
            "median_first_escape_z_gap": float(g.loc[g["escaped"], "first_escape_z_gap"].median()) if x.sum() else np.nan,
            "median_first_escape_gap_scale": float(g.loc[g["escaped"], "first_escape_gap_scale"].median()) if x.sum() else np.nan,
        })

    agg = pd.DataFrame(agg_rows)
    agg.to_csv(out_dir / "gap_aware_aggregate.csv", index=False)

    return session, agg


def main():
    args = parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ["GUI_DEFENSE_MODULE"] = args.defense_module

    print("=" * 120)
    print("GAP-AWARE ATTACKER")
    print("=" * 120)
    print("candidate_csv:", args.candidate_csv)
    print("budgets:", args.budgets)
    print("n_sessions:", args.n_sessions)
    print("session_source_method:", args.session_source_method)
    print("session_source_budget:", args.session_source_budget)
    print("gap_policy:", args.gap_policy)
    print("participant_regex:", args.participant_regex)
    print("output_dir:", out_dir)

    defense = load_defense()
    records = load_long_tap_records()

    df = pd.read_csv(args.candidate_csv, low_memory=False)
    df["_raw_order"] = np.arange(len(df))
    df["escaped_bool"] = df["escaped"].apply(parse_bool)
    df["budget_int"] = pd.to_numeric(df["budget"], errors="coerce").astype("Int64")
    df = fill_record_index(df, records)

    sessions = choose_eval_sessions(df, records, args)

    print("eval sessions:", len(sessions))

    budgets = [int(x.strip()) for x in args.budgets.split(",") if x.strip()]

    raw_rows = []

    for budget in budgets:
        print("\n" + "=" * 120)
        print(f"BUDGET B{budget}")
        print("=" * 120)

        for si, (_, row) in enumerate(sessions.iterrows()):
            record_index = int(float(row["record_index_filled"]))
            original_record = records[record_index]
            base_session = get_base_session(original_record)
            task_cluster = get_task_cluster_from_row(row)

            escaped_this = False

            for q in range(1, budget + 1):
                rng = np.random.default_rng(
                    stable_int(args.seed, "gap_aware", budget, si, record_index, q)
                )

                latent = sample_latent(rng, args.gap_policy)
                mutated = mutate(base_session, latent, rng)

                detected, escaped = eval_candidate(
                    defense,
                    original_record,
                    mutated,
                    task_cluster,
                )

                z_gap = float(latent.values[3])
                gap_scale = gap_scale_from_z(z_gap)

                raw_rows.append({
                    "attack": "gap_aware",
                    "budget": f"B{budget}",
                    "session_i": si,
                    "record_index": record_index,
                    "participant": row.get("participant", ""),
                    "session_id": row.get("session_id", ""),
                    "query_i": q,
                    "detected": bool(detected),
                    "escaped": bool(escaped),
                    "z_spatial": float(latent.values[0]),
                    "z_frequency": float(latent.values[1]),
                    "z_duration": float(latent.values[2]),
                    "z_gap": z_gap,
                    "gap_scale": gap_scale,
                    "z_jitter": float(latent.values[4]),
                    "z_heterogeneity": float(latent.values[5]),
                    "z_use_spatial": float(latent.values[6]),
                    "z_use_temporal": float(latent.values[7]),
                })

                if escaped:
                    escaped_this = True
                    break

            if (si + 1) % 25 == 0:
                print(f"B{budget}: processed {si + 1}/{len(sessions)}")

    raw = pd.DataFrame(raw_rows)
    raw.to_csv(out_dir / "gap_aware_query_results.csv", index=False)

    session, agg = summarize(raw, out_dir)

    print("\n" + "=" * 120)
    print("AGGREGATE")
    print("=" * 120)
    print(agg.to_string(index=False))

    print("\nSaved:")
    print(out_dir / "gap_aware_query_results.csv")
    print(out_dir / "gap_aware_session_results.csv")
    print(out_dir / "gap_aware_aggregate.csv")


if __name__ == "__main__":
    main()
