#!/usr/bin/env python3
from __future__ import annotations

import argparse
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


from attacks.v2.experiments.run_tgce_gap_counterfactual import (
    parse_bool_series,
    fill_record_index,
    build_latent,
    candidate_to_detection,
    get_task_cluster_from_row,
)


def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--candidate-csv",
        default="results/escape_basin_analysis/candidate_trials_flat.csv",
    )
    ap.add_argument("--method", default="hybrid")
    ap.add_argument("--budgets", default="10,30")
    ap.add_argument("--max-candidates-per-budget", type=int, default=50)
    ap.add_argument("--replay-seeds", type=int, default=20)
    ap.add_argument("--detector-repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260917)
    ap.add_argument("--output-dir", default="results/tgce_replay_protocol")

    return ap.parse_args()


def make_candidate_id(row):
    parts = [
        row.get("budget_int", ""),
        row.get("record_index_filled", ""),
        row.get("participant", ""),
        row.get("session_id", ""),
        row.get("q", ""),
        row.get("z_gap", ""),
        row.get("gap_scale", ""),
    ]
    return "|".join(str(x) for x in parts)


def select_candidates(df, args, records):
    budgets = [int(float(x.strip())) for x in args.budgets.split(",") if x.strip()]

    df = df[df["method"].astype(str) == args.method].copy()
    df["budget_int"] = pd.to_numeric(df["budget"], errors="coerce").astype("Int64")
    df = df[df["budget_int"].isin(budgets)].copy()

    if "escaped" not in df.columns:
        raise SystemExit("Missing column: escaped")
    if "z_gap" not in df.columns:
        raise SystemExit("Missing column: z_gap")

    df["escaped_bool"] = parse_bool_series(df["escaped"])
    df = fill_record_index(df, records)

    esc = df[df["escaped_bool"]].dropna(
        subset=["record_index_filled", "z_gap"]
    ).copy()

    esc["z_gap_float"] = pd.to_numeric(esc["z_gap"], errors="coerce")
    esc = esc.dropna(subset=["z_gap_float"])

    # Same logic as compressed-gap counterfactual:
    # select lowest-z_gap replayable escaped candidates per budget.
    selected = []

    for b, g in esc.groupby("budget_int"):
        gg = g.sort_values("z_gap_float", ascending=True).head(
            args.max_candidates_per_budget
        )
        selected.append(gg)

    if not selected:
        raise SystemExit("No escaped candidates selected.")

    out = pd.concat(selected, ignore_index=True)
    out["candidate_id"] = out.apply(make_candidate_id, axis=1)

    return out


def main():
    args = parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from attacks.v2.oracle import load_defense, load_long_tap_records
    from attacks.v2.search_space import LatentVector
    from attacks.v2.mutation import mutate

    print("=" * 100)
    print("TGCE REPLAY PROTOCOL DIAGNOSTIC")
    print("=" * 100)

    defense = load_defense()
    records = load_long_tap_records()

    df = pd.read_csv(args.candidate_csv, low_memory=False)
    selected = select_candidates(df, args, records)

    print("selected candidates:", len(selected))
    print(
        selected.groupby("budget_int")
        .agg(
            n=("candidate_id", "size"),
            median_z_gap=("z_gap_float", "median"),
            min_z_gap=("z_gap_float", "min"),
            max_z_gap=("z_gap_float", "max"),
        )
        .to_string()
    )

    raw_rows = []

    for cand_i, row in selected.reset_index(drop=True).iterrows():
        record_index = int(float(row["record_index_filled"]))
        original_record = records[record_index]

        if isinstance(original_record, dict) and "gestures" in original_record:
            base_session = original_record["gestures"]
        else:
            base_session = original_record

        task_cluster = get_task_cluster_from_row(row)

        for mode in ["original", "gap_fixed", "gap_only"]:
            latent = build_latent(LatentVector, row, mode)

            for replay_i in range(args.replay_seeds):
                # Different seeds test mutation replay stochasticity.
                replay_seed = (
                    args.seed
                    + int(row["budget_int"]) * 1000000
                    + cand_i * 1000
                    + replay_i
                )

                rng = np.random.default_rng(replay_seed)

                try:
                    mutated = mutate(base_session, latent, rng)

                    # Same mutated object, repeated detector calls.
                    # This tests detector determinism conditional on an identical trajectory.
                    dets = []
                    for _ in range(args.detector_repeats):
                        detected = candidate_to_detection(
                            defense,
                            original_record,
                            mutated,
                            task_cluster=task_cluster,
                        )
                        dets.append(bool(detected))

                    unique_labels = sorted(set(dets))
                    detector_nondeterministic = len(unique_labels) > 1

                    raw_rows.append({
                        "budget": int(row["budget_int"]),
                        "candidate_id": row["candidate_id"],
                        "candidate_rank": cand_i,
                        "record_index": record_index,
                        "participant": row.get("participant", None),
                        "session_id": row.get("session_id", None),
                        "source_q": row.get("q", None),
                        "source_z_gap": row.get("z_gap", None),
                        "source_gap_scale": row.get("gap_scale", None),
                        "mode": mode,
                        "replay_i": replay_i,
                        "replay_seed": replay_seed,
                        "detector_repeats": args.detector_repeats,
                        "detector_unique_labels": str(unique_labels),
                        "detector_nondeterministic": detector_nondeterministic,
                        "detected": dets[0] if not detector_nondeterministic else None,
                        "escaped": (not dets[0]) if not detector_nondeterministic else None,
                        "error": "",
                    })

                except Exception as e:
                    raw_rows.append({
                        "budget": int(row["budget_int"]),
                        "candidate_id": row["candidate_id"],
                        "candidate_rank": cand_i,
                        "record_index": record_index,
                        "participant": row.get("participant", None),
                        "session_id": row.get("session_id", None),
                        "source_q": row.get("q", None),
                        "source_z_gap": row.get("z_gap", None),
                        "source_gap_scale": row.get("gap_scale", None),
                        "mode": mode,
                        "replay_i": replay_i,
                        "replay_seed": replay_seed,
                        "detector_repeats": args.detector_repeats,
                        "detector_unique_labels": "",
                        "detector_nondeterministic": None,
                        "detected": None,
                        "escaped": None,
                        "error": repr(e),
                    })

    raw = pd.DataFrame(raw_rows)
    raw.to_csv(out_dir / "tgce_replay_protocol_raw.csv", index=False)

    valid = raw[raw["escaped"].notna()].copy()
    valid["escaped_bool"] = valid["escaped"].astype(bool)

    if valid.empty:
        print("No valid replay rows.")
        print(raw.head(30).to_string(index=False))
        raise SystemExit(1)

    summary = (
        valid
        .groupby(["budget", "mode"], dropna=False)
        .agg(
            n_candidates=("candidate_id", "nunique"),
            n_replays=("escaped_bool", "size"),
            accepted_as_human_like=("escaped_bool", "sum"),
            acceptance_rate_pct=("escaped_bool", lambda x: round(float(np.mean(x)) * 100, 2)),
            detector_nondeterministic_replays=("detector_nondeterministic", "sum"),
            errors=("error", lambda x: int((x.astype(str) != "").sum())),
        )
        .reset_index()
    )

    per_candidate = (
        valid
        .groupby(["budget", "mode", "candidate_id"], dropna=False)
        .agg(
            n_replays=("escaped_bool", "size"),
            candidate_acceptance_rate=("escaped_bool", "mean"),
        )
        .reset_index()
    )

    stability = (
        per_candidate
        .groupby(["budget", "mode"], dropna=False)
        .agg(
            n_candidates=("candidate_id", "size"),
            median_candidate_acceptance_rate=("candidate_acceptance_rate", "median"),
            mean_candidate_acceptance_rate=("candidate_acceptance_rate", "mean"),
            unstable_candidates=(
                "candidate_acceptance_rate",
                lambda x: int(((x > 0) & (x < 1)).sum()),
            ),
            unstable_candidate_rate_pct=(
                "candidate_acceptance_rate",
                lambda x: round(float(((x > 0) & (x < 1)).mean()) * 100, 2),
            ),
            always_escape_candidates=(
                "candidate_acceptance_rate",
                lambda x: int((x == 1).sum()),
            ),
            never_escape_candidates=(
                "candidate_acceptance_rate",
                lambda x: int((x == 0).sum()),
            ),
        )
        .reset_index()
    )

    summary.to_csv(out_dir / "table_replay_protocol_summary.csv", index=False)
    stability.to_csv(out_dir / "table_replay_candidate_stability.csv", index=False)

    with open(out_dir / "table_replay_protocol_summary.md", "w") as f:
        f.write(summary.to_markdown(index=False))
        f.write("\n")

    with open(out_dir / "table_replay_candidate_stability.md", "w") as f:
        f.write(stability.to_markdown(index=False))
        f.write("\n")

    # Small protocol note.
    total_nondet = int(valid["detector_nondeterministic"].sum())
    total_valid = int(len(valid))
    total_errors = int((raw["error"].astype(str) != "").sum())

    protocol_note = f"""# TGCE replay protocol diagnostic

Selected candidates are replayable escaped candidates from the low-z_gap compressed-gap basin.

Detector determinism check:
- Repeated detector calls per identical mutated trajectory: {args.detector_repeats}
- Valid replay rows: {total_valid}
- Detector-nondeterministic replay rows: {total_nondet}
- Errors: {total_errors}

Interpretation:
- If detector-nondeterministic replay rows are zero, the detector is deterministic conditional on an identical mutated trajectory.
- If candidate acceptance varies across replay seeds, then replay variability is caused by stochastic mutation reconstruction rather than detector randomness.
- Therefore, original replay acceptance below 100% should be described as a consequence of reconstructing mutations from logged latent parameters without exact low-level random choices, not as detector nondeterminism.
"""

    with open(out_dir / "replay_protocol_note.md", "w") as f:
        f.write(protocol_note)

    print("\n" + "=" * 100)
    print("Replay-level summary")
    print("=" * 100)
    print(summary.to_string(index=False))

    print("\n" + "=" * 100)
    print("Candidate-level stability")
    print("=" * 100)
    print(stability.to_string(index=False))

    print("\nDetector-nondeterministic replay rows:", total_nondet)
    print("Errors:", total_errors)

    print("\nSaved:")
    print(out_dir / "tgce_replay_protocol_raw.csv")
    print(out_dir / "table_replay_protocol_summary.csv")
    print(out_dir / "table_replay_candidate_stability.csv")
    print(out_dir / "replay_protocol_note.md")


if __name__ == "__main__":
    main()
