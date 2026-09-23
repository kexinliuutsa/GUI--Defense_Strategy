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
    )
    ap.add_argument("--method", default="hybrid")
    ap.add_argument("--budgets", default="10,30")

    ap.add_argument(
        "--gap-scales",
        default="0.60,0.70,0.80,0.90,1.00,1.10,1.175,1.30,1.50,1.70",
    )

    ap.add_argument("--max-candidates-per-basin-budget", type=int, default=100)

    ap.add_argument("--compressed-z-gap-threshold", type=float, default=0.30)
    ap.add_argument("--neutral-high-z-gap-threshold", type=float, default=0.45)

    ap.add_argument(
        "--include-all-escaped",
        action="store_true",
        help="Also run an all_escaped basin besides compressed and neutral_high.",
    )

    ap.add_argument("--seed", type=int, default=20260919)

    ap.add_argument(
        "--defense-module",
        default="evaluation.frozen_v1v2v3_defense_hardened",
    )

    ap.add_argument(
        "--output-dir",
        default="results/tgce_gap_sweep_by_basin",
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


def gap_scale_to_z(gap_scale: float) -> float:
    z = (float(gap_scale) - GAP_SCALE_MIN) / GAP_SCALE_RANGE
    return float(np.clip(z, 0.0, 1.0))


def z_to_gap_scale(z_gap: float) -> float:
    return GAP_SCALE_MIN + GAP_SCALE_RANGE * float(z_gap)


def get_base_session(original_record):
    if isinstance(original_record, dict) and "gestures" in original_record:
        return original_record["gestures"]
    return original_record


def build_latent_with_gap_scale(row, fixed_gap_scale: float):
    """
    Keep every latent coordinate from the logged candidate, except z_gap.
    z_gap is set so that decoded gap_scale equals fixed_gap_scale.
    """
    fixed_z_gap = gap_scale_to_z(fixed_gap_scale)

    vals = []

    for name in LATENT_ORDER:
        if name not in row.index:
            raise KeyError(f"Missing latent column: {name}")

        if name == "z_gap":
            v = fixed_z_gap
        else:
            v = float(row[name])

        vals.append(v)

    return LatentVector(values=np.asarray(vals, dtype=float))


def build_latent_logged(row):
    vals = []
    for name in LATENT_ORDER:
        if name not in row.index:
            raise KeyError(f"Missing latent column: {name}")
        vals.append(float(row[name]))
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


def select_basin_rows(base: pd.DataFrame, basin: str, budget: int, args):
    sub = base[base["budget_int"].eq(budget)].copy()

    if basin == "compressed":
        sub = sub[sub["z_gap_float"] <= args.compressed_z_gap_threshold]
        sub = sub.sort_values(["z_gap_float", "_raw_order"], ascending=[True, True])

    elif basin == "neutral_high":
        sub = sub[sub["z_gap_float"] >= args.neutral_high_z_gap_threshold]
        sub = sub.sort_values(["z_gap_float", "_raw_order"], ascending=[False, True])

    elif basin == "all_escaped":
        sub = sub.sort_values(["_raw_order"], ascending=True)

    else:
        raise ValueError(f"Unknown basin: {basin}")

    return sub.head(args.max_candidates_per_basin_budget)


def summarize_rates(raw: pd.DataFrame, out_dir: Path):
    rows = []

    for (method, budget, basin, intervention), g in raw.groupby(
        ["method", "budget", "basin", "intervention"]
    ):
        x = g["escaped"].astype(bool).to_numpy()
        lo, hi = bootstrap_ci_binary(
            x,
            seed=stable_int(method, budget, basin, intervention),
        )

        rows.append({
            "method": method,
            "budget": budget,
            "basin": basin,
            "intervention": intervention,
            "n": len(g),
            "escaped": int(x.sum()),
            "escape_rate_%": round(float(x.mean()) * 100, 2),
            "escape_rate_95ci": f"[{lo*100:.2f}, {hi*100:.2f}]",
            "gap_scale": g["fixed_gap_scale"].iloc[0],
            "z_gap": g["fixed_z_gap"].iloc[0],
        })

    table = pd.DataFrame(rows)

    # Sort logged first, then fixed gap scales.
    table["_sort_gap"] = table["gap_scale"].apply(
        lambda x: -1 if str(x) == "logged" else float(x)
    )
    table = table.sort_values(
        ["method", "budget", "basin", "_sort_gap"],
        ascending=True,
    ).drop(columns=["_sort_gap"])

    table.to_csv(out_dir / "table_gap_sweep_rates.csv", index=False)

    return table


def summarize_best_gap(rate_table: pd.DataFrame, out_dir: Path):
    rows = []

    fixed = rate_table[rate_table["intervention"].eq("fixed_gap")].copy()

    for (method, budget, basin), g in fixed.groupby(["method", "budget", "basin"]):
        g2 = g.sort_values("escape_rate_%", ascending=True)
        best = g2.iloc[0]

        rows.append({
            "method": method,
            "budget": budget,
            "basin": basin,
            "n_per_gap": int(best["n"]),
            "best_gap_scale_min_escape": float(best["gap_scale"]),
            "best_z_gap": float(best["z_gap"]),
            "min_escape_rate_%": float(best["escape_rate_%"]),
            "max_escape_rate_%": float(g["escape_rate_%"].max()),
            "range_pp": round(float(g["escape_rate_%"].max() - g["escape_rate_%"].min()), 2),
            "escape_at_gap_0.8_%": _lookup_rate(g, 0.8),
            "escape_at_gap_1.0_%": _lookup_rate(g, 1.0),
            "escape_at_gap_1.175_%": _lookup_rate(g, 1.175),
            "escape_at_gap_1.5_%": _lookup_rate(g, 1.5),
        })

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "table_gap_sweep_best_gap.csv", index=False)

    return table


def _lookup_rate(g: pd.DataFrame, gap_scale: float):
    sub = g[np.isclose(g["gap_scale"].astype(float), gap_scale, atol=1e-6)]
    if sub.empty:
        return np.nan
    return float(sub.iloc[0]["escape_rate_%"])


def summarize_paired(raw: pd.DataFrame, gap_scales: list[float], out_dir: Path):
    """
    Paired comparisons:
    - fixed gap values vs logged original.
    - consecutive fixed gap values.
    - fixed gap values vs gap=1.0.
    """
    rows = []

    for (method, budget, basin), g in raw.groupby(["method", "budget", "basin"]):
        pivot = g.pivot_table(
            index="candidate_key",
            columns="intervention_label",
            values="escaped",
            aggfunc="first",
        )

        labels = list(pivot.columns)

        comparison_pairs = []

        # fixed vs logged
        if "logged_original" in labels:
            for gs in gap_scales:
                lab = f"gap_{gs:g}"
                if lab in labels:
                    comparison_pairs.append(("logged_original", lab))

        # consecutive fixed gap scales
        for a, b in zip(gap_scales[:-1], gap_scales[1:]):
            la = f"gap_{a:g}"
            lb = f"gap_{b:g}"
            if la in labels and lb in labels:
                comparison_pairs.append((la, lb))

        # vs strict no-op gap=1.0
        if any(abs(gs - 1.0) < 1e-9 for gs in gap_scales):
            noop_lab = "gap_1"
            for gs in gap_scales:
                lab = f"gap_{gs:g}"
                if lab != noop_lab and lab in labels and noop_lab in labels:
                    comparison_pairs.append((lab, noop_lab))

        seen = set()

        for a_lab, b_lab in comparison_pairs:
            key = (a_lab, b_lab)
            if key in seen:
                continue
            seen.add(key)

            sub = pivot[[a_lab, b_lab]].dropna()

            if sub.empty:
                continue

            a = sub[a_lab].astype(bool).to_numpy()
            b = sub[b_lab].astype(bool).to_numpy()

            n00, n01, n10, n11, p, paired_or = mcnemar_exact(a, b)

            rate_a = float(a.mean())
            rate_b = float(b.mean())

            rows.append({
                "method": method,
                "budget": budget,
                "basin": basin,
                "comparison": f"{a_lab}_vs_{b_lab}",
                "n_paired": len(sub),
                f"{a_lab}_rate_%": round(rate_a * 100, 2),
                f"{b_lab}_rate_%": round(rate_b * 100, 2),
                "risk_diff_pp": round((rate_a - rate_b) * 100, 2),
                "paired_discordant_OR": round(float(paired_or), 3),
                "n00_both_no_escape": n00,
                "n01_a_no_b_yes": n01,
                "n10_a_yes_b_no": n10,
                "n11_both_escape": n11,
                "mcnemar_p": p,
                "mcnemar_p_text": fmt_p(p),
            })

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "table_gap_sweep_paired_effects.csv", index=False)

    return table


def make_plot(rate_table: pd.DataFrame, out_dir: Path):
    try:
        import matplotlib.pyplot as plt

        fixed = rate_table[rate_table["intervention"].eq("fixed_gap")].copy()
        fixed["gap_scale"] = fixed["gap_scale"].astype(float)

        for (method, budget), g in fixed.groupby(["method", "budget"]):
            plt.figure(figsize=(7, 4.5))

            for basin, gb in g.groupby("basin"):
                gb = gb.sort_values("gap_scale")
                plt.plot(
                    gb["gap_scale"],
                    gb["escape_rate_%"],
                    marker="o",
                    label=basin,
                )

            plt.xlabel("Fixed gap_scale")
            plt.ylabel("Escape rate (%)")
            plt.title(f"TGCE gap sweep: {method} {budget}")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            safe = f"{method}_{budget}".replace("/", "_")
            plt.savefig(out_dir / f"figure_gap_sweep_{safe}.png", dpi=200)
            plt.savefig(out_dir / f"figure_gap_sweep_{safe}.pdf")
            plt.close()

    except Exception as e:
        print("[WARN] Plot failed:", repr(e))


def main():
    args = parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ["GUI_DEFENSE_MODULE"] = args.defense_module

    budgets = [int(x.strip()) for x in args.budgets.split(",") if x.strip()]
    gap_scales = [float(x.strip()) for x in args.gap_scales.split(",") if x.strip()]

    basins = ["compressed", "neutral_high"]
    if args.include_all_escaped:
        basins.append("all_escaped")

    print("=" * 120)
    print("TGCE GAP SWEEP BY BASIN")
    print("=" * 120)
    print("candidate_csv:", args.candidate_csv)
    print("method:", args.method)
    print("budgets:", budgets)
    print("basins:", basins)
    print("gap_scales:", gap_scales)
    print("max_candidates_per_basin_budget:", args.max_candidates_per_basin_budget)
    print("compressed_z_gap_threshold:", args.compressed_z_gap_threshold)
    print("neutral_high_z_gap_threshold:", args.neutral_high_z_gap_threshold)
    print("output_dir:", out_dir)

    defense = load_defense()
    records = load_long_tap_records()

    df = pd.read_csv(args.candidate_csv, low_memory=False)
    df["_raw_order"] = np.arange(len(df))
    df["escaped_bool"] = df["escaped"].apply(parse_bool)
    df["budget_int"] = pd.to_numeric(df["budget"], errors="coerce").astype("Int64")
    df["z_gap_float"] = pd.to_numeric(df["z_gap"], errors="coerce")

    df = fill_record_index(df, records)

    base = df[
        (df["method"].astype(str) == args.method)
        & (df["budget_int"].isin(budgets))
        & (df["escaped_bool"])
    ].copy()

    if base.empty:
        raise SystemExit("No escaped candidates after filtering.")

    rows = []

    for budget in budgets:
        for basin in basins:
            sub = select_basin_rows(base, basin, budget, args)

            print("\n" + "=" * 120)
            print(f"BUDGET B{budget} | BASIN {basin} | selected rows: {len(sub)}")
            print("=" * 120)

            for i, (_, row) in enumerate(sub.iterrows()):
                record_index = int(float(row["record_index_filled"]))
                original_record = records[record_index]
                base_session = get_base_session(original_record)
                task_cluster = get_task_cluster_from_row(row)

                candidate_key = f"{args.method}_B{budget}_{basin}_{i}_row{int(row['_raw_order'])}"

                common = {
                    "method": args.method,
                    "budget": f"B{budget}",
                    "basin": basin,
                    "candidate_i": i,
                    "candidate_key": candidate_key,
                    "raw_order": int(row["_raw_order"]),
                    "participant": row.get("participant", ""),
                    "session_id": row.get("session_id", ""),
                    "record_index": record_index,
                    "source_z_gap": float(row.get("z_gap", np.nan)),
                    "source_gap_scale": float(row.get("gap_scale", np.nan)) if "gap_scale" in row.index else np.nan,
                }

                # Logged original replay
                try:
                    latent_logged = build_latent_logged(row)
                    rng_logged = np.random.default_rng(
                        stable_int(args.seed, args.method, budget, basin, i, "logged", row["_raw_order"])
                    )
                    mutated_logged = mutate(base_session, latent_logged, rng_logged)
                    detected, escaped = eval_candidate(
                        defense,
                        original_record,
                        mutated_logged,
                        task_cluster,
                    )

                    rows.append({
                        **common,
                        "intervention": "logged_original",
                        "intervention_label": "logged_original",
                        "fixed_gap_scale": "logged",
                        "fixed_z_gap": "logged",
                        "detected": bool(detected),
                        "escaped": bool(escaped),
                        "error": "",
                    })

                except Exception as e:
                    rows.append({
                        **common,
                        "intervention": "logged_original",
                        "intervention_label": "logged_original",
                        "fixed_gap_scale": "logged",
                        "fixed_z_gap": "logged",
                        "detected": np.nan,
                        "escaped": False,
                        "error": repr(e),
                    })

                # Fixed gap sweep.
                for gs in gap_scales:
                    fixed_z = gap_scale_to_z(gs)
                    label = f"gap_{gs:g}"

                    try:
                        latent = build_latent_with_gap_scale(row, gs)

                        # Use same seed across all gap values for same candidate.
                        # This reduces noise from non-gap stochastic mutation components.
                        rng = np.random.default_rng(
                            stable_int(args.seed, args.method, budget, basin, i, "fixed_gap_shared", row["_raw_order"])
                        )

                        mutated = mutate(base_session, latent, rng)
                        detected, escaped = eval_candidate(
                            defense,
                            original_record,
                            mutated,
                            task_cluster,
                        )

                        rows.append({
                            **common,
                            "intervention": "fixed_gap",
                            "intervention_label": label,
                            "fixed_gap_scale": float(gs),
                            "fixed_z_gap": float(fixed_z),
                            "detected": bool(detected),
                            "escaped": bool(escaped),
                            "error": "",
                        })

                    except Exception as e:
                        rows.append({
                            **common,
                            "intervention": "fixed_gap",
                            "intervention_label": label,
                            "fixed_gap_scale": float(gs),
                            "fixed_z_gap": float(fixed_z),
                            "detected": np.nan,
                            "escaped": False,
                            "error": repr(e),
                        })

                if (i + 1) % 20 == 0:
                    print(f"B{budget} {basin}: processed {i + 1}/{len(sub)}")

    raw = pd.DataFrame(rows)

    raw_path = out_dir / "gap_sweep_raw.csv"
    raw.to_csv(raw_path, index=False)

    rate_table = summarize_rates(raw, out_dir)
    best_gap_table = summarize_best_gap(rate_table, out_dir)
    paired_table = summarize_paired(raw, gap_scales, out_dir)
    make_plot(rate_table, out_dir)

    print("\n" + "=" * 120)
    print("RATE TABLE")
    print("=" * 120)
    print(rate_table.to_string(index=False))

    print("\n" + "=" * 120)
    print("BEST GAP TABLE")
    print("=" * 120)
    print(best_gap_table.to_string(index=False))

    print("\n" + "=" * 120)
    print("PAIRED TABLE HEAD")
    print("=" * 120)
    if not paired_table.empty:
        print(paired_table.head(40).to_string(index=False))
    else:
        print("(empty)")

    print("\nSaved:")
    print(raw_path)
    print(out_dir / "table_gap_sweep_rates.csv")
    print(out_dir / "table_gap_sweep_best_gap.csv")
    print(out_dir / "table_gap_sweep_paired_effects.csv")
    print(out_dir / "figure_gap_sweep_hybrid_B10.png")
    print(out_dir / "figure_gap_sweep_hybrid_B30.png")


if __name__ == "__main__":
    main()
