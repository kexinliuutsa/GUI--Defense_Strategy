#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import os
import pickle
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


from attacks.v2.search_space import LatentVector
from attacks.v2.mutation import mutate
from attacks.v2.oracle import load_defense, load_long_tap_records
from attacks.v2.experiments.run_tgce_gap_counterfactual import (
    fill_record_index,
    build_latent,
    candidate_to_detection,
    get_task_cluster_from_row,
)


DEFAULT_TEST_SPECS = [
    # matched sanity check
    "matched_hybrid_B30_seen_agents|hybrid|30|AgentCPM,UI-TARS|150|30",

    # unseen attack method
    "unseen_method_random_B30|random|30|ALL|150|30",
    "unseen_method_tpe_B30|tpe|30|ALL|150|30",

    # unseen budget
    "unseen_budget_hybrid_B100|hybrid|100|ALL|100|100",

    # unseen agent under low budget
    "unseen_agent_AutoGLM_B30|hybrid|30|AutoGLM|100|30",

    # unseen low-ASR agents need larger budget
    "unseen_agent_GPT4o_B300|hybrid|300|GPT-4o|50|300",
    "unseen_agent_Claude_B300|hybrid|300|Claude|50|300",
]


def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--candidate-csv",
        default="results/escape_basin_analysis/candidate_trials_flat.csv",
    )
    ap.add_argument(
        "--hardcase-pkl",
        default="results/v2_loop_hard_cases/mutated_records/escaped_mutated_records.pkl",
    )
    ap.add_argument("--radius", type=float, default=1.0263925586627112)
    ap.add_argument("--output-dir", default="results/hardcase_loop_heldout_replay")

    ap.add_argument(
        "--test-spec",
        action="append",
        default=None,
        help=(
            "Format: name|method|budget|agents|max_runs|max_queries. "
            "agents can be ALL or comma-separated normalized names."
        ),
    )

    ap.add_argument("--seed", type=int, default=20260918)
    ap.add_argument("--build-split-memory", action="store_true")
    ap.add_argument("--train-method", default="hybrid")
    ap.add_argument("--train-budget", type=int, default=30)
    ap.add_argument("--train-agents", default="AgentCPM,UI-TARS")
    ap.add_argument("--max-hardcases", type=int, default=500)
    ap.add_argument(
        "--split-memory-output",
        default="results/hardcase_loop_heldout_replay/split_memory/hybrid_B30_AgentCPM_UITARS_hardcases.pkl",
    )

    return ap.parse_args()


def parse_bool(x):
    return str(x).strip().lower() in {"true", "1", "yes", "y", "escaped", "success"}


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
    if "mobileagent" in s or "mobile-agent" in s:
        return "MobileAgent"

    return str(x)


def stable_int(*parts, mod=2**32 - 1):
    s = "|".join(str(p) for p in parts)
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h[:12], 16) % mod


def fmt_p(p):
    if p is None or not np.isfinite(p):
        return ""
    if p < 0.001:
        return "<0.001"
    return f"{p:.4g}"


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
            p = float(binomtest(k=min(n01, n10), n=n01 + n10, p=0.5).pvalue)
        except Exception:
            p = np.nan

    paired_or = (n10 + 0.5) / (n01 + 0.5)

    return n00, n01, n10, n11, p, paired_or


def load_candidate_df(path, records):
    df = pd.read_csv(path, low_memory=False)
    df["_raw_order"] = np.arange(len(df))
    df["escaped_bool"] = df["escaped"].apply(parse_bool)
    df["budget_int"] = pd.to_numeric(df["budget"], errors="coerce").astype("Int64")
    df["q_int"] = pd.to_numeric(df.get("q", df["_raw_order"]), errors="coerce").fillna(df["_raw_order"]).astype(int)
    df["agent"] = df["participant"].apply(normalize_agent)

    df = fill_record_index(df, records)

    return df


def filter_agents(df, agents):
    if agents == "ALL":
        return df

    allowed = {x.strip() for x in str(agents).split(",") if x.strip()}
    return df[df["agent"].isin(allowed)].copy()


def get_base_session(original_record):
    if isinstance(original_record, dict) and "gestures" in original_record:
        return original_record["gestures"]
    return original_record


def format_hardcase_for_memory(mutated, original_record, row, sample_format=None):
    """
    Try to preserve the same rough object shape as the existing hardcase memory.
    If existing memory stores dict records with gestures, return a dict.
    Otherwise return the mutated trajectory object directly.
    """
    if isinstance(sample_format, dict):
        rec = {}
        if isinstance(original_record, dict):
            rec.update(original_record)

        rec["gestures"] = mutated
        rec["participant"] = row.get("participant", rec.get("participant", None))
        rec["session_id"] = row.get("session_id", rec.get("session_id", None))
        rec["source_method"] = row.get("method", None)
        rec["source_budget"] = int(row.get("budget_int", -1))
        rec["source_q"] = int(row.get("q_int", -1))
        rec["source_z_gap"] = row.get("z_gap", None)
        rec["source_gap_scale"] = row.get("gap_scale", None)
        return rec

    return mutated


def load_defense_with_env(module_name, hardcase_pkl=None, radius=None):
    os.environ["GUI_DEFENSE_MODULE"] = module_name

    if hardcase_pkl is not None:
        os.environ["LOOP_HARDCASE_PKL"] = str(hardcase_pkl)

    if radius is not None:
        os.environ["LOOP_MEMORY_RADIUS"] = str(radius)

    return load_defense()


def build_split_memory(args, df, records):
    out_path = Path(args.split_memory_output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    train_agents = {x.strip() for x in args.train_agents.split(",") if x.strip()}

    train = df[
        (df["method"].astype(str) == args.train_method)
        & (df["budget_int"] == args.train_budget)
        & (df["agent"].isin(train_agents))
        & (df["escaped_bool"])
    ].copy()

    if train.empty:
        raise SystemExit("No training escaped candidates found for split memory.")

    # Prefer diverse low-z_gap hardcases if available.
    if "z_gap" in train.columns:
        train["z_gap_float"] = pd.to_numeric(train["z_gap"], errors="coerce")
        train = train.sort_values(["z_gap_float", "_raw_order"], ascending=[True, True])
    else:
        train = train.sample(frac=1.0, random_state=args.seed)

    train = train.head(args.max_hardcases).copy()

    sample_format = None
    base_pkl = Path(args.hardcase_pkl)
    if base_pkl.exists():
        try:
            with open(base_pkl, "rb") as f:
                base_obj = pickle.load(f)
            if isinstance(base_obj, list) and len(base_obj) > 0:
                sample_format = base_obj[0]
        except Exception as e:
            print("[warn] could not inspect base hardcase pkl:", repr(e))

    frozen = load_defense_with_env("evaluation.frozen_v1v2v3_defense_hardened")

    hardcases = []
    meta_rows = []

    for i, (_, row) in enumerate(train.iterrows()):
        try:
            record_index = int(float(row["record_index_filled"]))
            original_record = records[record_index]
            base_session = get_base_session(original_record)

            latent = build_latent(LatentVector, row, "original")
            rng = np.random.default_rng(stable_int(args.seed, "train_memory", i, row.get("session_id"), row.get("q_int")))
            mutated = mutate(base_session, latent, rng)

            task_cluster = get_task_cluster_from_row(row)
            detected = candidate_to_detection(
                frozen,
                original_record,
                mutated,
                task_cluster=task_cluster,
            )

            # Keep only reconstructed hardcases that still escape frozen.
            if detected:
                continue

            obj = format_hardcase_for_memory(mutated, original_record, row, sample_format)
            hardcases.append(obj)

            meta_rows.append({
                "memory_index": len(hardcases) - 1,
                "method": row.get("method", None),
                "budget": int(row.get("budget_int", -1)),
                "agent": row.get("agent", None),
                "participant": row.get("participant", None),
                "session_id": row.get("session_id", None),
                "q": int(row.get("q_int", -1)),
                "z_gap": row.get("z_gap", None),
                "gap_scale": row.get("gap_scale", None),
            })

        except Exception as e:
            print("[warn] failed to build hardcase", i, repr(e))

    if len(hardcases) == 0:
        raise SystemExit("Built zero verified hardcases. Try increasing max-hardcases or use existing memory.")

    with open(out_path, "wb") as f:
        pickle.dump(hardcases, f)

    meta_path = out_path.with_suffix(".metadata.csv")
    pd.DataFrame(meta_rows).to_csv(meta_path, index=False)

    print("=" * 100)
    print("Built split hardcase memory")
    print("=" * 100)
    print("path:", out_path)
    print("n_hardcases:", len(hardcases))
    print("metadata:", meta_path)

    return out_path


def parse_test_specs(specs):
    if not specs:
        specs = DEFAULT_TEST_SPECS

    parsed = []

    for s in specs:
        parts = s.split("|")
        if len(parts) != 6:
            raise SystemExit(f"Bad test spec: {s}")

        name, method, budget, agents, max_runs, max_queries = parts

        parsed.append({
            "test_name": name,
            "method": method,
            "budget": int(budget),
            "agents": agents,
            "max_runs": int(max_runs),
            "max_queries": int(max_queries),
        })

    return parsed


def choose_group_cols(df):
    cols = []

    for c in ["participant", "agent", "session_id", "record_index_filled"]:
        if c in df.columns:
            cols.append(c)

    # Include seed-like columns if they exist.
    for c in [
        "seed", "run_seed", "attack_seed", "source_seed", "replicate",
        "trial_seed", "search_seed", "run_id", "attack_run_id"
    ]:
        if c in df.columns:
            cols.append(c)

    # Remove duplicates while preserving order.
    out = []
    for c in cols:
        if c not in out:
            out.append(c)

    if not out:
        raise SystemExit("Could not infer grouping columns for attack runs.")

    return out


def evaluate_test_spec(spec, df, records, frozen_defense, loop_defense, args):
    method = spec["method"]
    budget = spec["budget"]
    agents = spec["agents"]
    max_runs = spec["max_runs"]
    max_queries = spec["max_queries"]

    sub = df[
        (df["method"].astype(str) == method)
        & (df["budget_int"] == budget)
    ].copy()

    sub = filter_agents(sub, agents)

    if sub.empty:
        print("[warn] empty test split:", spec)
        return None, pd.DataFrame()

    group_cols = choose_group_cols(sub)

    groups = list(sub.groupby(group_cols, dropna=False, sort=False))
    rng = np.random.default_rng(stable_int(args.seed, spec["test_name"]))
    rng.shuffle(groups)

    groups = groups[:max_runs]

    run_rows = []
    query_rows = []

    for run_i, (key, g) in enumerate(groups):
        g = g.sort_values(["q_int", "_raw_order"]).head(max_queries).copy()

        frozen_first_q = None
        loop_first_q = None
        frozen_first_cost = None
        loop_first_cost = None

        n_eval = 0
        n_errors = 0

        for local_i, (_, row) in enumerate(g.iterrows()):
            try:
                record_index = int(float(row["record_index_filled"]))
                original_record = records[record_index]
                base_session = get_base_session(original_record)

                latent = build_latent(LatentVector, row, "original")
                seed = stable_int(
                    args.seed,
                    spec["test_name"],
                    run_i,
                    local_i,
                    row.get("session_id"),
                    row.get("q_int"),
                    row.get("_raw_order"),
                )
                rng2 = np.random.default_rng(seed)

                mutated = mutate(base_session, latent, rng2)
                task_cluster = get_task_cluster_from_row(row)

                frozen_detected = candidate_to_detection(
                    frozen_defense,
                    original_record,
                    mutated,
                    task_cluster=task_cluster,
                )
                loop_detected = candidate_to_detection(
                    loop_defense,
                    original_record,
                    mutated,
                    task_cluster=task_cluster,
                )

                frozen_escape = not bool(frozen_detected)
                loop_escape = not bool(loop_detected)

                q_val = int(row.get("q_int", local_i + 1))

                cost_val = np.nan
                for cc in ["cost", "mutation_cost", "best_cost", "total_cost"]:
                    if cc in row.index:
                        try:
                            cost_val = float(row[cc])
                            break
                        except Exception:
                            pass

                if frozen_escape and frozen_first_q is None:
                    frozen_first_q = q_val
                    frozen_first_cost = cost_val

                if loop_escape and loop_first_q is None:
                    loop_first_q = q_val
                    loop_first_cost = cost_val

                n_eval += 1

                query_rows.append({
                    "test_name": spec["test_name"],
                    "method": method,
                    "budget": budget,
                    "agents": agents,
                    "run_i": run_i,
                    "q": q_val,
                    "agent": row.get("agent", None),
                    "participant": row.get("participant", None),
                    "session_id": row.get("session_id", None),
                    "record_index": row.get("record_index_filled", None),
                    "frozen_escape": frozen_escape,
                    "loop_escape": loop_escape,
                    "cost": cost_val,
                    "error": "",
                })

            except Exception as e:
                n_errors += 1
                query_rows.append({
                    "test_name": spec["test_name"],
                    "method": method,
                    "budget": budget,
                    "agents": agents,
                    "run_i": run_i,
                    "q": int(row.get("q_int", local_i + 1)),
                    "agent": row.get("agent", None),
                    "participant": row.get("participant", None),
                    "session_id": row.get("session_id", None),
                    "record_index": row.get("record_index_filled", None),
                    "frozen_escape": None,
                    "loop_escape": None,
                    "cost": np.nan,
                    "error": repr(e),
                })

        frozen_run_escape = frozen_first_q is not None
        loop_run_escape = loop_first_q is not None

        run_rows.append({
            "test_name": spec["test_name"],
            "method": method,
            "budget": budget,
            "agents": agents,
            "run_i": run_i,
            "run_key": str(key),
            "n_queries_evaluated": n_eval,
            "n_errors": n_errors,
            "frozen_escape": frozen_run_escape,
            "loop_escape": loop_run_escape,
            "frozen_q_first_escape": frozen_first_q,
            "loop_q_first_escape": loop_first_q,
            "frozen_first_escape_cost": frozen_first_cost,
            "loop_first_escape_cost": loop_first_cost,
        })

    run_df = pd.DataFrame(run_rows)
    query_df = pd.DataFrame(query_rows)

    return run_df, query_df


def summarize_runs(run_df, args, memory_pkl):
    rows = []

    for test_name, g in run_df.groupby("test_name", sort=False):
        frozen = g["frozen_escape"].astype(bool).to_numpy()
        loop = g["loop_escape"].astype(bool).to_numpy()

        frozen_rate = float(frozen.mean())
        loop_rate = float(loop.mean())

        f_lo, f_hi = bootstrap_ci_binary(frozen, seed=stable_int(args.seed, test_name, "frozen"))
        l_lo, l_hi = bootstrap_ci_binary(loop, seed=stable_int(args.seed, test_name, "loop"))

        n00, n01, n10, n11, p, paired_or = mcnemar_exact(frozen, loop)

        median_f_q = (
            float(pd.to_numeric(g["frozen_q_first_escape"], errors="coerce").dropna().median())
            if g["frozen_q_first_escape"].notna().any()
            else np.nan
        )
        median_l_q = (
            float(pd.to_numeric(g["loop_q_first_escape"], errors="coerce").dropna().median())
            if g["loop_q_first_escape"].notna().any()
            else np.nan
        )

        method = g["method"].iloc[0]
        budget = int(g["budget"].iloc[0])
        agents = g["agents"].iloc[0]

        rows.append({
            "test_name": test_name,
            "method": method,
            "budget": f"B{budget}",
            "agents": agents,
            "n_runs": int(len(g)),
            "mean_queries_evaluated": round(float(g["n_queries_evaluated"].mean()), 2),
            "frozen_ASR_%": round(frozen_rate * 100, 2),
            "frozen_ASR_95ci": f"[{f_lo*100:.2f}, {f_hi*100:.2f}]",
            "loop_ASR_%": round(loop_rate * 100, 2),
            "loop_ASR_95ci": f"[{l_lo*100:.2f}, {l_hi*100:.2f}]",
            "ASR_drop_pp": round((frozen_rate - loop_rate) * 100, 2),
            "relative_reduction_%": round(
                ((frozen_rate - loop_rate) / frozen_rate * 100) if frozen_rate > 0 else np.nan,
                2,
            ),
            "frozen_median_q_first_escape": median_f_q,
            "loop_median_q_first_escape": median_l_q,
            "n00_both_not_escape": n00,
            "n01_frozen_no_loop_yes": n01,
            "n10_frozen_yes_loop_no": n10,
            "n11_both_escape": n11,
            "paired_odds_ratio_discordant": round(paired_or, 3),
            "mcnemar_p": p,
            "mcnemar_p_text": fmt_p(p),
            "loop_memory_pkl": str(memory_pkl),
            "loop_radius": args.radius,
        })

    out = pd.DataFrame(rows)
    return out


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("HARDCASE LOOP HELD-OUT REPLAY EVALUATION")
    print("=" * 100)

    records = load_long_tap_records()
    df = load_candidate_df(args.candidate_csv, records)

    if args.build_split_memory:
        memory_pkl = build_split_memory(args, df, records)
    else:
        memory_pkl = Path(args.hardcase_pkl)

    if not memory_pkl.exists():
        raise SystemExit(f"Missing hardcase memory pkl: {memory_pkl}")

    print("Using hardcase memory:", memory_pkl)
    print("Using loop radius:", args.radius)

    frozen_defense = load_defense_with_env("evaluation.frozen_v1v2v3_defense_hardened")

    loop_defense = load_defense_with_env(
        "evaluation.defense_loop_v1_hardcase_memory",
        hardcase_pkl=memory_pkl,
        radius=args.radius,
    )

    specs = parse_test_specs(args.test_spec)

    all_run_rows = []
    all_query_rows = []

    for spec in specs:
        print("\n" + "=" * 100)
        print("TEST SPLIT:", spec)
        print("=" * 100)

        run_df, query_df = evaluate_test_spec(
            spec,
            df,
            records,
            frozen_defense,
            loop_defense,
            args,
        )

        if run_df is None or run_df.empty:
            continue

        print(
            run_df[["test_name", "run_i", "n_queries_evaluated", "frozen_escape", "loop_escape"]]
            .head()
            .to_string(index=False)
        )

        all_run_rows.append(run_df)
        all_query_rows.append(query_df)

    if not all_run_rows:
        raise SystemExit("No held-out test rows generated.")

    runs = pd.concat(all_run_rows, ignore_index=True)
    queries = pd.concat(all_query_rows, ignore_index=True)

    summary = summarize_runs(runs, args, memory_pkl)

    runs.to_csv(out_dir / "heldout_replay_run_level.csv", index=False)
    queries.to_csv(out_dir / "heldout_replay_query_level.csv", index=False)
    summary.to_csv(out_dir / "table12_hardcase_loop_heldout_replay_summary.csv", index=False)

    with open(out_dir / "table12_hardcase_loop_heldout_replay_summary.md", "w") as f:
        f.write(summary.to_markdown(index=False))
        f.write("\n")

    print("\n" + "=" * 100)
    print("TABLE 12 — Hardcase loop held-out replay summary")
    print("=" * 100)
    print(summary.to_string(index=False))

    print("\nSaved:")
    print(out_dir / "heldout_replay_run_level.csv")
    print(out_dir / "heldout_replay_query_level.csv")
    print(out_dir / "table12_hardcase_loop_heldout_replay_summary.csv")


if __name__ == "__main__":
    main()
