#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import os
import pickle
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


# Keep this consistent with attacks/v2/search_space.py
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
    ap.add_argument("--budget", type=int, default=300)
    ap.add_argument("--agents", default="Claude,GPT-4o")

    ap.add_argument("--target-escapes-per-agent", type=int, default=100)
    ap.add_argument("--max-source-rows-per-agent", type=int, default=5000)

    ap.add_argument("--seed", type=int, default=20260919)
    ap.add_argument("--best-of-n", type=int, default=10)
    ap.add_argument("--exact-repeat-checks", type=int, default=3)

    ap.add_argument("--fixed-z-gap", type=float, default=0.50)

    ap.add_argument(
        "--defense-module",
        default="evaluation.frozen_v1v2v3_defense_hardened",
    )

    ap.add_argument(
        "--output-dir",
        default="results/byte_exact_cross_agent_counterfactual_B300",
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


def normalize_agent(x):
    s = str(x).lower()
    if "ui-tars" in s or "uitars" in s:
        return "UI-TARS"
    if "gpt4o" in s or "gpt-4o" in s or "gpt_4o" in s:
        return "GPT-4o"
    if "claude" in s or "sonnet" in s:
        return "Claude"
    if "cpm" in s:
        return "AgentCPM"
    if "autoglm" in s or "auto-glm" in s or "glm" in s:
        return "AutoGLM"
    return str(x)


def get_base_session(original_record):
    if isinstance(original_record, dict) and "gestures" in original_record:
        return original_record["gestures"]
    return original_record


def latent_from_row_fallback(row, mode: str, fixed_z_gap: float = 0.50):
    """
    Fallback latent builder.
    This keeps the previous TGCE convention:
    - original: use all logged latent coordinates.
    - gap_fixed: keep all coordinates but set z_gap to fixed_z_gap.
    - gap_only: keep z_gap from original, neutralize other continuous coordinates,
      and disable spatial where possible while keeping temporal enabled.
    """
    vals = []
    for name in LATENT_ORDER:
        if name not in row.index:
            raise KeyError(f"Missing latent column: {name}")

        v = float(row[name])

        if mode == "gap_fixed":
            if name == "z_gap":
                v = fixed_z_gap

        elif mode == "gap_only":
            if name == "z_gap":
                v = float(row[name])
            elif name == "z_use_spatial":
                v = 0.0
            elif name == "z_use_temporal":
                v = 1.0
            else:
                v = 0.5

        elif mode == "original":
            pass

        else:
            raise ValueError(f"Unknown mode: {mode}")

        vals.append(v)

    return LatentVector(values=np.asarray(vals, dtype=float))


def build_latent(row, mode: str, fixed_z_gap: float):
    """
    Try to use the existing project helper if present. Fall back to local version.
    """
    try:
        from attacks.v2.experiments.run_tgce_gap_counterfactual import build_latent as project_build_latent

        try:
            return project_build_latent(row, mode=mode)
        except TypeError:
            return project_build_latent(row, mode)

    except Exception:
        return latent_from_row_fallback(row, mode=mode, fixed_z_gap=fixed_z_gap)


def eval_detection(defense, original_record, mutated, task_cluster):
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
    Returns contingency and exact McNemar p.
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


def summarize_agent(raw: pd.DataFrame, agent: str, out_dir: Path):
    sub = raw[raw["agent"].eq(agent)].copy()
    if sub.empty:
        return None, None

    modes = [
        "original_exact",
        "gap_fixed",
        "gap_only",
        "reconstructed_original_single",
        "reconstructed_original_best_of_n",
    ]

    main_rows = []

    for mode in modes:
        col = f"{mode}_escaped"
        if col not in sub.columns:
            continue

        x = sub[col].astype(bool).to_numpy()
        rate = float(x.mean())
        lo, hi = bootstrap_ci_binary(x, seed=stable_int(agent, mode))

        main_rows.append({
            "agent": agent,
            "mode": mode,
            "n": len(x),
            "escaped": int(x.sum()),
            "escape_rate_%": round(rate * 100, 2),
            "escape_rate_95ci": f"[{lo*100:.2f}, {hi*100:.2f}]",
        })

    main = pd.DataFrame(main_rows)

    pair_rows = []

    comparisons = [
        ("original_exact", "gap_fixed"),
        ("gap_only", "gap_fixed"),
        ("gap_only", "original_exact"),
        ("reconstructed_original_best_of_n", "reconstructed_original_single"),
    ]

    for a_name, b_name in comparisons:
        ca = f"{a_name}_escaped"
        cb = f"{b_name}_escaped"

        if ca not in sub.columns or cb not in sub.columns:
            continue

        a = sub[ca].astype(bool).to_numpy()
        b = sub[cb].astype(bool).to_numpy()

        n00, n01, n10, n11, p, paired_or = mcnemar_exact(a, b)

        rate_a = float(a.mean())
        rate_b = float(b.mean())

        pair_rows.append({
            "agent": agent,
            "comparison": f"{a_name}_vs_{b_name}",
            "n_paired": len(sub),
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

    pairs = pd.DataFrame(pair_rows)

    main.to_csv(out_dir / f"summary_{agent}_rates.csv", index=False)
    pairs.to_csv(out_dir / f"summary_{agent}_paired_effects.csv", index=False)

    return main, pairs


def main():
    args = parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    serialized_dir = out_dir / "serialized_mutants"
    serialized_dir.mkdir(parents=True, exist_ok=True)

    os.environ["GUI_DEFENSE_MODULE"] = args.defense_module
    defense = load_defense()
    records = load_long_tap_records()

    df = pd.read_csv(args.candidate_csv, low_memory=False)
    df["_raw_order"] = np.arange(len(df))
    df["escaped_bool_logged"] = df["escaped"].apply(parse_bool)
    df["budget_int"] = pd.to_numeric(df["budget"], errors="coerce").astype("Int64")
    df["agent"] = df["participant"].apply(normalize_agent)
    df = fill_record_index(df, records)

    agents = [x.strip() for x in args.agents.split(",") if x.strip()]

    # Start from historically escaped rows to make rerun efficient.
    base = df[
        (df["method"].astype(str) == args.method)
        & (df["budget_int"] == args.budget)
        & (df["agent"].isin(agents))
        & (df["escaped_bool_logged"])
    ].copy()

    if base.empty:
        raise SystemExit("No logged escaped candidate rows after filter.")

    print("=" * 120)
    print("BYTE-EXACT CROSS-AGENT COUNTERFACTUAL RERUN")
    print("=" * 120)
    print("candidate_csv:", args.candidate_csv)
    print("method:", args.method)
    print("budget:", args.budget)
    print("agents:", agents)
    print("target_escapes_per_agent:", args.target_escapes_per_agent)
    print("max_source_rows_per_agent:", args.max_source_rows_per_agent)
    print("best_of_n:", args.best_of_n)
    print("output_dir:", out_dir)

    all_rows = []

    for agent in agents:
        agent_dir = serialized_dir / agent.replace("/", "_")
        agent_dir.mkdir(parents=True, exist_ok=True)

        cand = base[base["agent"].eq(agent)].copy()

        if cand.empty:
            print(f"\n[WARN] No candidates for {agent}")
            continue

        # Prefer low-gap historical escapes because old evidence says TGCE basin sits there.
        cand["z_gap_float"] = pd.to_numeric(cand["z_gap"], errors="coerce")
        cand = cand.sort_values(
            ["z_gap_float", "_raw_order"],
            ascending=[True, True],
        ).head(args.max_source_rows_per_agent)

        print("\n" + "=" * 120)
        print("AGENT:", agent)
        print("source rows:", len(cand))
        print("=" * 120)

        kept = 0
        tried = 0

        for _, row in cand.iterrows():
            if kept >= args.target_escapes_per_agent:
                break

            tried += 1

            record_index = int(float(row["record_index_filled"]))
            original_record = records[record_index]
            base_session = get_base_session(original_record)
            task_cluster = get_task_cluster_from_row(row)

            row_id = str(row.get("_raw_order"))
            session_id = str(row.get("session_id"))
            participant = str(row.get("participant"))

            try:
                # ============================================================
                # 1. Generate original candidate and save BYTE-EXACT mutant
                # ============================================================
                latent_original = build_latent(row, "original", args.fixed_z_gap)
                rng_original = np.random.default_rng(
                    stable_int(args.seed, agent, "original_exact", row_id, session_id)
                )
                mutated_original = mutate(base_session, latent_original, rng_original)

                original_detected, original_escaped = eval_detection(
                    defense,
                    original_record,
                    mutated_original,
                    task_cluster,
                )

                if not original_escaped:
                    continue

                pair_id = f"{agent.replace('-', '').replace(' ', '')}_pair{kept:04d}_row{row_id}"
                pkl_path = agent_dir / f"{pair_id}.pkl"

                payload = {
                    "pair_id": pair_id,
                    "agent": agent,
                    "method": args.method,
                    "budget": args.budget,
                    "participant": participant,
                    "session_id": session_id,
                    "record_index": record_index,
                    "raw_order": int(row["_raw_order"]),
                    "source_row": row.to_dict(),
                    "mutated_original": mutated_original,
                    "task_cluster": task_cluster,
                    "note": "Byte-exact original mutant generated and serialized by run_byte_exact_cross_agent_counterfactual.py",
                }

                with open(pkl_path, "wb") as f:
                    pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

                # ============================================================
                # 2. Exact repeat checks: same object, repeated detector calls
                # ============================================================
                exact_repeat_escaped = []
                for rep in range(args.exact_repeat_checks):
                    _, e = eval_detection(
                        defense,
                        original_record,
                        mutated_original,
                        task_cluster,
                    )
                    exact_repeat_escaped.append(bool(e))

                exact_repeat_stable = len(set(exact_repeat_escaped)) == 1

                # ============================================================
                # 3. Reconstructed original single replay
                #    This approximates the old protocol.
                # ============================================================
                rng_rec_single = np.random.default_rng(
                    stable_int(args.seed, agent, "reconstructed_single", row_id, session_id)
                )
                rec_single = mutate(base_session, latent_original, rng_rec_single)
                _, rec_single_escaped = eval_detection(
                    defense,
                    original_record,
                    rec_single,
                    task_cluster,
                )

                # ============================================================
                # 4. Reconstructed original best-of-n
                # ============================================================
                rec_best_escaped = False
                rec_best_first_success_j = None

                for j in range(args.best_of_n):
                    rng_rec = np.random.default_rng(
                        stable_int(args.seed, agent, "reconstructed_best", row_id, session_id, j)
                    )
                    rec_j = mutate(base_session, latent_original, rng_rec)
                    _, e_j = eval_detection(
                        defense,
                        original_record,
                        rec_j,
                        task_cluster,
                    )
                    if e_j:
                        rec_best_escaped = True
                        rec_best_first_success_j = j
                        break

                # ============================================================
                # 5. Gap-fixed counterfactual
                # ============================================================
                latent_gap_fixed = build_latent(row, "gap_fixed", args.fixed_z_gap)
                rng_gap_fixed = np.random.default_rng(
                    stable_int(args.seed, agent, "gap_fixed", row_id, session_id)
                )
                mutated_gap_fixed = mutate(base_session, latent_gap_fixed, rng_gap_fixed)
                gap_fixed_detected, gap_fixed_escaped = eval_detection(
                    defense,
                    original_record,
                    mutated_gap_fixed,
                    task_cluster,
                )

                # ============================================================
                # 6. Gap-only counterfactual
                # ============================================================
                latent_gap_only = build_latent(row, "gap_only", args.fixed_z_gap)
                rng_gap_only = np.random.default_rng(
                    stable_int(args.seed, agent, "gap_only", row_id, session_id)
                )
                mutated_gap_only = mutate(base_session, latent_gap_only, rng_gap_only)
                gap_only_detected, gap_only_escaped = eval_detection(
                    defense,
                    original_record,
                    mutated_gap_only,
                    task_cluster,
                )

                all_rows.append({
                    "pair_id": pair_id,
                    "agent": agent,
                    "method": args.method,
                    "budget": args.budget,
                    "participant": participant,
                    "session_id": session_id,
                    "record_index": record_index,
                    "raw_order": int(row["_raw_order"]),
                    "source_logged_escaped": bool(row["escaped_bool_logged"]),
                    "source_z_gap": float(row.get("z_gap", np.nan)),
                    "source_gap_scale": float(row.get("gap_scale", np.nan)) if "gap_scale" in row.index else np.nan,
                    "serialized_path": str(pkl_path),

                    "original_exact_detected": bool(original_detected),
                    "original_exact_escaped": bool(original_escaped),

                    "exact_repeat_checks": args.exact_repeat_checks,
                    "exact_repeat_stable": bool(exact_repeat_stable),
                    "exact_repeat_escaped_values": ",".join(str(x) for x in exact_repeat_escaped),

                    "reconstructed_original_single_escaped": bool(rec_single_escaped),
                    "reconstructed_original_best_of_n_escaped": bool(rec_best_escaped),
                    "reconstructed_original_best_first_success_j": rec_best_first_success_j,

                    "gap_fixed_detected": bool(gap_fixed_detected),
                    "gap_fixed_escaped": bool(gap_fixed_escaped),

                    "gap_only_detected": bool(gap_only_detected),
                    "gap_only_escaped": bool(gap_only_escaped),
                })

                kept += 1

                if kept % 10 == 0:
                    print(f"[{agent}] kept {kept}/{args.target_escapes_per_agent}; tried={tried}")

            except Exception as e:
                all_rows.append({
                    "pair_id": "",
                    "agent": agent,
                    "method": args.method,
                    "budget": args.budget,
                    "participant": participant,
                    "session_id": session_id,
                    "record_index": record_index,
                    "raw_order": int(row["_raw_order"]),
                    "source_logged_escaped": bool(row["escaped_bool_logged"]),
                    "source_z_gap": row.get("z_gap", np.nan),
                    "source_gap_scale": row.get("gap_scale", np.nan) if "gap_scale" in row.index else np.nan,
                    "serialized_path": "",
                    "error": repr(e),
                })

        print(f"[{agent}] DONE. kept={kept}, tried={tried}")

    raw = pd.DataFrame(all_rows)
    raw_path = out_dir / "byte_exact_counterfactual_raw.csv"
    raw.to_csv(raw_path, index=False)

    valid = raw[
        raw.get("original_exact_escaped", False).astype(str).str.lower().eq("true")
    ].copy()

    main_parts = []
    pair_parts = []

    for agent in agents:
        m, p = summarize_agent(valid, agent, out_dir)
        if m is not None:
            main_parts.append(m)
        if p is not None:
            pair_parts.append(p)

    if main_parts:
        main_summary = pd.concat(main_parts, ignore_index=True)
    else:
        main_summary = pd.DataFrame()

    if pair_parts:
        pair_summary = pd.concat(pair_parts, ignore_index=True)
    else:
        pair_summary = pd.DataFrame()

    main_summary_path = out_dir / "table_byte_exact_cross_agent_main.csv"
    pair_summary_path = out_dir / "table_byte_exact_cross_agent_paired_effects.csv"

    main_summary.to_csv(main_summary_path, index=False)
    pair_summary.to_csv(pair_summary_path, index=False)

    print("\n" + "=" * 120)
    print("BYTE-EXACT CROSS-AGENT SUMMARY")
    print("=" * 120)

    if not main_summary.empty:
        print("\nMAIN RATES")
        print(main_summary.to_string(index=False))

    if not pair_summary.empty:
        print("\nPAIRED EFFECTS")
        print(pair_summary.to_string(index=False))

    print("\nSaved:")
    print(raw_path)
    print(main_summary_path)
    print(pair_summary_path)
    print(serialized_dir)


if __name__ == "__main__":
    main()
