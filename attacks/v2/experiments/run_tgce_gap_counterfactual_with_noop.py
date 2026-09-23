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
NOOP_GAP_SCALE = 1.0
NOOP_Z_GAP = (NOOP_GAP_SCALE - GAP_SCALE_MIN) / GAP_SCALE_RANGE  # 0.36
MIDPOINT_Z_GAP = 0.5


def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--candidate-csv",
        default="results/escape_basin_analysis/candidate_trials_flat.csv",
    )
    ap.add_argument("--method", default="hybrid")
    ap.add_argument("--budgets", default="10,30")
    ap.add_argument("--max-candidates-per-budget", type=int, default=100)

    ap.add_argument(
        "--selection",
        choices=["compressed", "neutral_high", "all_escaped"],
        default="compressed",
        help="Which escaped candidates to select for counterfactual replay.",
    )

    ap.add_argument(
        "--compressed-z-gap-threshold",
        type=float,
        default=0.30,
        help="Used for compressed basin selection.",
    )
    ap.add_argument(
        "--neutral-high-z-gap-threshold",
        type=float,
        default=0.45,
        help="Used for neutral/high-gap control selection.",
    )

    ap.add_argument("--seed", type=int, default=20260919)
    ap.add_argument(
        "--defense-module",
        default="evaluation.frozen_v1v2v3_defense_hardened",
    )
    ap.add_argument(
        "--output-dir",
        default="results/tgce_counterfactual_with_noop",
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


def build_latent(row, mode: str):
    """
    Four modes:

    original:
        Use all latent coordinates from the logged attack candidate.

    gap_fixed_midpoint:
        Replace only z_gap with search midpoint 0.5.
        This gives gap_scale = 1.175, so it removes compression but is not no-op.

    gap_fixed_noop:
        Replace only z_gap with 0.36.
        This gives gap_scale = 1.0, strict no-op for inter-gesture gap scaling.

    gap_only:
        Keep original z_gap, neutralize other continuous dimensions,
        disable spatial, enable temporal.
    """
    vals = []

    for name in LATENT_ORDER:
        if name not in row.index:
            raise KeyError(f"Missing latent column: {name}")

        v = float(row[name])

        if mode == "original":
            pass

        elif mode == "gap_fixed_midpoint":
            if name == "z_gap":
                v = MIDPOINT_Z_GAP

        elif mode == "gap_fixed_noop":
            if name == "z_gap":
                v = NOOP_Z_GAP

        elif mode == "gap_only":
            if name == "z_gap":
                v = float(row[name])
            elif name == "z_use_spatial":
                v = 0.0
            elif name == "z_use_temporal":
                v = 1.0
            else:
                v = 0.5

        else:
            raise ValueError(f"Unknown mode: {mode}")

        vals.append(v)

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


def bootstrap_ci_binary(x, n_boot=5000, seed=0):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)

    for i in range(n_boot):
        vals[i] = rng.choice(x, size=len(x), replace=True).mean()

    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def mcnemar_exact(a, b):
    """
    a and b are boolean arrays where True means escaped.
    """
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)

    n00 = int((~a & ~b).sum())
    n01 = int((~a & b).sum())
    n10 = int((a & ~b).sum())
    n11 = int((a & b).sum())

    try:
        from statsmodels.stats.contingency_tables import mcnemar
        res = mcnemar([[n00, n01], [n10, n11]], exact=True)
        p = float(res.pvalue)
    except Exception:
        try:
            from scipy.stats import binomtest
            p = float(binomtest(min(n01, n10), n01 + n10, p=0.5).pvalue)
        except Exception:
            p = np.nan

    paired_or = (n10 + 0.5) / (n01 + 0.5)
    return n00, n01, n10, n11, p, paired_or


def fmt_p(p):
    if p is None or not np.isfinite(p):
        return ""
    if p < 0.001:
        return "<0.001"
    return f"{p:.4g}"


def summarize(raw: pd.DataFrame, out_dir: Path):
    rows = []

    modes = [
        "original",
        "gap_fixed_midpoint",
        "gap_fixed_noop",
        "gap_only",
    ]

    for (method, budget, selection), g in raw.groupby(["method", "budget", "selection"]):
        for mode in modes:
            col = f"{mode}_escaped"
            x = g[col].astype(bool).to_numpy()
            lo, hi = bootstrap_ci_binary(
                x,
                seed=stable_int(method, budget, selection, mode),
            )

            rows.append({
                "method": method,
                "budget": budget,
                "selection": selection,
                "mode": mode,
                "n": len(g),
                "escaped": int(x.sum()),
                "escape_rate_%": round(float(x.mean()) * 100, 2),
                "escape_rate_95ci": f"[{lo*100:.2f}, {hi*100:.2f}]",
                "z_gap_setting": (
                    "logged" if mode in ["original", "gap_only"]
                    else MIDPOINT_Z_GAP if mode == "gap_fixed_midpoint"
                    else NOOP_Z_GAP
                ),
                "gap_scale_setting": (
                    "logged" if mode in ["original", "gap_only"]
                    else gap_scale_from_z(MIDPOINT_Z_GAP) if mode == "gap_fixed_midpoint"
                    else gap_scale_from_z(NOOP_Z_GAP)
                ),
            })

    rate_table = pd.DataFrame(rows)

    pair_rows = []

    comparisons = [
        ("original", "gap_fixed_midpoint"),
        ("original", "gap_fixed_noop"),
        ("gap_only", "gap_fixed_midpoint"),
        ("gap_only", "gap_fixed_noop"),
        ("gap_fixed_midpoint", "gap_fixed_noop"),
        ("gap_only", "original"),
    ]

    for (method, budget, selection), g in raw.groupby(["method", "budget", "selection"]):
        for a_name, b_name in comparisons:
            a = g[f"{a_name}_escaped"].astype(bool).to_numpy()
            b = g[f"{b_name}_escaped"].astype(bool).to_numpy()

            n00, n01, n10, n11, p, paired_or = mcnemar_exact(a, b)
            rate_a = float(a.mean())
            rate_b = float(b.mean())

            pair_rows.append({
                "method": method,
                "budget": budget,
                "selection": selection,
                "comparison": f"{a_name}_vs_{b_name}",
                "n_paired": len(g),
                f"{a_name}_rate_%": round(rate_a * 100, 2),
                f"{b_name}_rate_%": round(rate_b * 100, 2),
                "risk_diff_pp": round((rate_a - rate_b) * 100, 2),
                "paired_discordant_OR": round(float(paired_or), 3),
                "n00_both_no_escape": n00,
                "n01_a_no_b_yes": n01,
                "n10_a_yes_b_no": n10,
                "n11_both_escape": n11,
                "mcnemar_p": p,
                "mcnemar_p_text": fmt_p(p),
            })

    pair_table = pd.DataFrame(pair_rows)

    rate_table.to_csv(out_dir / "table_tgce_noop_counterfactual_rates.csv", index=False)
    pair_table.to_csv(out_dir / "table_tgce_noop_counterfactual_paired_effects.csv", index=False)

    return rate_table, pair_table


def main():
    args = parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ["GUI_DEFENSE_MODULE"] = args.defense_module

    print("=" * 120)
    print("TGCE COUNTERFACTUAL WITH STRICT NO-OP GAP CONTROL")
    print("=" * 120)
    print("candidate_csv:", args.candidate_csv)
    print("method:", args.method)
    print("budgets:", args.budgets)
    print("selection:", args.selection)
    print("max_candidates_per_budget:", args.max_candidates_per_budget)
    print("midpoint z_gap:", MIDPOINT_Z_GAP)
    print("midpoint gap_scale:", gap_scale_from_z(MIDPOINT_Z_GAP))
    print("noop z_gap:", NOOP_Z_GAP)
    print("noop gap_scale:", gap_scale_from_z(NOOP_Z_GAP))
    print("output_dir:", out_dir)

    defense = load_defense()
    records = load_long_tap_records()

    df = pd.read_csv(args.candidate_csv, low_memory=False)
    df["_raw_order"] = np.arange(len(df))

    df["escaped_bool"] = df["escaped"].apply(parse_bool)
    df["budget_int"] = pd.to_numeric(df["budget"], errors="coerce").astype("Int64")
    df["z_gap_float"] = pd.to_numeric(df["z_gap"], errors="coerce")
    df = fill_record_index(df, records)

    budgets = [int(x.strip()) for x in args.budgets.split(",") if x.strip()]

    base = df[
        (df["method"].astype(str) == args.method)
        & (df["budget_int"].isin(budgets))
        & (df["escaped_bool"])
    ].copy()

    if base.empty:
        raise SystemExit("No escaped candidates found after filtering.")

    rows = []

    for budget in budgets:
        sub = base[base["budget_int"].eq(budget)].copy()

        if args.selection == "compressed":
            sub = sub[sub["z_gap_float"] <= args.compressed_z_gap_threshold]
            sub = sub.sort_values(["z_gap_float", "_raw_order"], ascending=[True, True])

        elif args.selection == "neutral_high":
            sub = sub[sub["z_gap_float"] >= args.neutral_high_z_gap_threshold]
            sub = sub.sort_values(["z_gap_float", "_raw_order"], ascending=[False, True])

        elif args.selection == "all_escaped":
            sub = sub.sort_values(["_raw_order"], ascending=True)

        sub = sub.head(args.max_candidates_per_budget)

        print("\n" + "=" * 120)
        print(f"BUDGET B{budget} | selected rows: {len(sub)}")
        print("=" * 120)

        for i, (_, row) in enumerate(sub.iterrows()):
            record_index = int(float(row["record_index_filled"]))
            original_record = records[record_index]
            base_session = get_base_session(original_record)
            task_cluster = get_task_cluster_from_row(row)

            result = {
                "method": args.method,
                "budget": f"B{budget}",
                "selection": args.selection,
                "candidate_i": i,
                "raw_order": int(row["_raw_order"]),
                "participant": row.get("participant", ""),
                "session_id": row.get("session_id", ""),
                "record_index": record_index,
                "source_z_gap": float(row.get("z_gap", np.nan)),
                "source_gap_scale": float(row.get("gap_scale", np.nan)) if "gap_scale" in row.index else np.nan,
                "midpoint_z_gap": MIDPOINT_Z_GAP,
                "midpoint_gap_scale": gap_scale_from_z(MIDPOINT_Z_GAP),
                "noop_z_gap": NOOP_Z_GAP,
                "noop_gap_scale": gap_scale_from_z(NOOP_Z_GAP),
            }

            for mode in [
                "original",
                "gap_fixed_midpoint",
                "gap_fixed_noop",
                "gap_only",
            ]:
                try:
                    latent = build_latent(row, mode)
                    rng = np.random.default_rng(
                        stable_int(args.seed, args.method, budget, args.selection, i, mode, row["_raw_order"])
                    )
                    mutated = mutate(base_session, latent, rng)
                    detected, escaped = eval_candidate(
                        defense,
                        original_record,
                        mutated,
                        task_cluster,
                    )
                    result[f"{mode}_detected"] = bool(detected)
                    result[f"{mode}_escaped"] = bool(escaped)
                    result[f"{mode}_error"] = ""

                except Exception as e:
                    result[f"{mode}_detected"] = np.nan
                    result[f"{mode}_escaped"] = False
                    result[f"{mode}_error"] = repr(e)

            rows.append(result)

            if (i + 1) % 20 == 0:
                print(f"B{budget}: processed {i + 1}/{len(sub)}")

    raw = pd.DataFrame(rows)
    raw_path = out_dir / "tgce_noop_counterfactual_raw.csv"
    raw.to_csv(raw_path, index=False)

    rate_table, pair_table = summarize(raw, out_dir)

    print("\n" + "=" * 120)
    print("RATE TABLE")
    print("=" * 120)
    print(rate_table.to_string(index=False))

    print("\n" + "=" * 120)
    print("PAIRED EFFECTS")
    print("=" * 120)
    print(pair_table.to_string(index=False))

    print("\nSaved:")
    print(raw_path)
    print(out_dir / "table_tgce_noop_counterfactual_rates.csv")
    print(out_dir / "table_tgce_noop_counterfactual_paired_effects.csv")


if __name__ == "__main__":
    main()
