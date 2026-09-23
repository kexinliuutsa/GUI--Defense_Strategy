from __future__ import annotations

import argparse
import os
import csv
import json
import sys
import time
import traceback
from pathlib import Path
from statistics import median, mean, pstdev

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attacks.v2.runner import AttackRunner
from attacks.v2.mutation import mutate
from attacks.v2.metrics import cost_fn

from attacks.v2.optimizers.random import RandomOptimizer
from attacks.v2.optimizers.tpe import TPEOptimizer
from attacks.v2.optimizers.hybrid import HybridOptimizer
from attacks.v2.optimizers.multibasin_hybrid import MultiBasinHybridOptimizer
from attacks.v2.optimizers.llm_api_nextgen_pack import DeepSeekNextGenOptimizer
from attacks.v2.optimizers.llm_api_anonymous_batch import StrictAnonymousDeepSeekBatchOptimizer
from attacks.v2.optimizers.llm_advanced import LLMRegionHybridOptimizer, LLMDirectionHybridOptimizer, LLMArmBanditOptimizer
from attacks.v2.optimizers.llm import OllamaLLMOptimizer

from attacks.v2.oracle import (
    load_defense,
    load_long_tap_records,
    make_record_oracle,
    extract_session,
    participant,
    session_id,
)


def parse_csv_list(s: str, cast=str):
    return [cast(x.strip()) for x in s.split(",") if x.strip()]


def make_optimizer(method: str, seed: int, budget: int):
    method = method.lower()

    if method == "random":
        return RandomOptimizer(seed=seed)

    if method == "tpe":
        return TPEOptimizer(
            seed=seed,
            n_startup_trials=min(5, max(1, budget)),
        )

    if method == "hybrid":
        return HybridOptimizer(
            seed=seed,
            warmup=min(10, max(3, budget // 4)),
            p_random_before_escape=1.0,
            p_random_after_escape=0.30,
            sigma0=0.18,
            sigma_min=0.035,
            elite_k=5,
            temporal_bias=0.85,
            spatial_bias=0.65,
        )

    if method.startswith("llm-region-hybrid"):
        model = "qwen3:8b"
        if method == "llm-region-hybrid-qwen3-32b":
            model = "qwen3:32b"

        return LLMRegionHybridOptimizer(
            model=model,
            seed=seed,
            candidate_count=8,
            region_count=4,
            max_history=40,
            timeout=120,
            temperature=0.8,
            p_llm_after_escape=0.20,
            p_random_after_escape=0.10,
            sigma0=0.16,
            sigma_min=0.025,
            elite_k=8,
        )

    if method.startswith("llm-direction-hybrid"):
        model = "qwen3:8b"
        if method == "llm-direction-hybrid-qwen3-32b":
            model = "qwen3:32b"

        return LLMDirectionHybridOptimizer(
            model=model,
            seed=seed,
            candidate_count=8,
            max_history=40,
            timeout=120,
            temperature=0.7,
            sigma=0.08,
        )

    if method.startswith("llm-arm-bandit"):
        model = "qwen3:8b"
        if method == "llm-arm-bandit-qwen3-32b":
            model = "qwen3:32b"

        return LLMArmBanditOptimizer(
            model=model,
            seed=seed,
            arm_count=6,
            refresh_every=16,
            max_history=40,
            timeout=120,
            temperature=0.8,
        )

    if method.startswith("llm"):
        model = "qwen3:8b"
        strict = False

        if method == "llm-qwen3-8b":
            model = "qwen3:8b"
        elif method == "llm-qwen3-32b":
            model = "qwen3:32b"
        elif method == "llm-deepseek-r1-7b":
            model = "deepseek-r1:7b"
        elif method == "llm-strict-qwen3-8b":
            model = "qwen3:8b"
            strict = True
        elif method == "llm-strict-qwen3-32b":
            model = "qwen3:32b"
            strict = True

        if strict:
            # Strict black-box setting:
            # - no empirical temporal prior
            # - no biased random cold start
            # - LLM proposes from the first query
            return OllamaLLMOptimizer(
                model=model,
                seed=seed,
                max_history=30,
                timeout=90,
                temperature=0.6,
                fallback_to_random=True,
                temporal_bias=0.50,
                spatial_bias=0.50,
                cold_start=0,
            )

        # Non-strict pilot version.
        # Keeps biased random cold start, useful as an upper-bound style probe.
        return OllamaLLMOptimizer(
            model=model,
            seed=seed,
            max_history=30,
            timeout=90,
            temperature=0.6,
            fallback_to_random=True,
            temporal_bias=0.85,
            spatial_bias=0.65,
            cold_start=3,
        )

    if method.startswith("strict-anon-llm-api-batch"):
        model = "deepseek-chat"

        if method == "strict-anon-llm-api-batch-deepseek-chat":
            model = "deepseek-chat"
        elif method == "strict-anon-llm-api-batch-deepseek-reasoner":
            model = "deepseek-reasoner"

        return StrictAnonymousDeepSeekBatchOptimizer(
            model=model,
            seed=seed,
            candidate_count=8,
            max_history=40,
            timeout=120,
            temperature=0.8,
            min_distance=0.06,
        )

    if method.startswith("llm-api-cem"):
        return DeepSeekNextGenOptimizer(
            mode="cem",
            model="deepseek-chat",
            seed=seed,
            candidate_count=8,
            max_history=40,
            timeout=120,
            temperature=0.8,
        )

    if method.startswith("llm-api-bo"):
        return DeepSeekNextGenOptimizer(
            mode="bo",
            model="deepseek-chat",
            seed=seed,
            candidate_count=8,
            max_history=40,
            timeout=120,
            temperature=0.8,
        )

    if method.startswith("llm-api-segment-square"):
        return DeepSeekNextGenOptimizer(
            mode="segment_square",
            model="deepseek-chat",
            seed=seed,
            candidate_count=8,
            max_history=40,
            timeout=120,
            temperature=0.8,
        )

    if method.startswith("llm-api-transfer-archive"):
        return DeepSeekNextGenOptimizer(
            mode="transfer_archive",
            model="deepseek-chat",
            seed=seed,
            candidate_count=8,
            max_history=40,
            timeout=120,
            temperature=0.8,
        )

    if method.startswith("llm-api-multifidelity"):
        return DeepSeekNextGenOptimizer(
            mode="multifidelity",
            model="deepseek-chat",
            seed=seed,
            candidate_count=8,
            max_history=40,
            timeout=120,
            temperature=0.8,
        )

    if method.startswith("multi-basin-hybrid"):
        return MultiBasinHybridOptimizer(
            seed=seed,
            explore_fraction=0.60,
            basin_threshold=0.38,
            top_k_basins=3,
            candidate_pool=128,
            random_restart_prob=0.10,
        )

    raise ValueError(f"Unknown method: {method}")


def collect_detected_records(defense, records, n_sessions: int, max_scan: int | None):
    selected = []
    scanned = 0

    for idx, record in enumerate(records):
        if max_scan is not None and scanned >= max_scan:
            break

        scanned += 1

        try:
            detected = bool(defense.detect_record(record))
        except Exception:
            continue

        if detected:
            selected.append((idx, record))

        if n_sessions > 0 and len(selected) >= n_sessions:
            break

    return selected, scanned


def load_existing_record_indices(path: Path):
    done = set()
    if not path.exists():
        return done

    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue

            if row.get("ok") is True:
                done.add(int(row["record_idx"]))

    return done


def summarize_rows(rows: list[dict]) -> dict:
    completed = [r for r in rows if r.get("ok")]
    escaped = [r for r in completed if r.get("escaped")]
    errors = [r for r in rows if not r.get("ok")]

    q_first = [
        r["queries_to_first_escape"]
        for r in escaped
        if r.get("queries_to_first_escape") is not None
    ]

    best_costs = [
        r["best_cost"]
        for r in escaped
        if r.get("best_cost") is not None
    ]

    return {
        "n_total": len(rows),
        "n_completed": len(completed),
        "n_errors": len(errors),
        "asr": len(escaped) / len(completed) if completed else "",
        "escaped": len(escaped),
        "median_queries_to_first_escape": median(q_first) if q_first else "",
        "median_best_cost_escaped": median(best_costs) if best_costs else "",
    }


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []

    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def compact_result(result_dict: dict, save_history: str):
    if save_history == "full":
        return result_dict

    if save_history == "none":
        out = dict(result_dict)
        out.pop("history", None)
        return out

    if save_history == "escaped":
        out = dict(result_dict)
        hist = out.get("history", [])
        out["history"] = [h for h in hist if h.get("escaped")]
        return out

    raise ValueError(f"Unknown save_history mode: {save_history}")


def run_one_config(
    *,
    defense,
    selected,
    method: str,
    budget: int,
    seed: int,
    output_dir: Path,
    save_history: str,
):
    detail_path = output_dir / f"details_{method}_B{budget}_seed{seed}.jsonl"
    done_record_indices = load_existing_record_indices(detail_path)

    print("=" * 100)
    print(f"CONFIG method={method} budget={budget} seed={seed}")
    print("detail:", detail_path)
    print("selected sessions:", len(selected))
    print("already completed:", len(done_record_indices))
    print("=" * 100)

    with detail_path.open("a") as fout:
        for local_idx, (record_idx, record) in enumerate(selected):
            if int(record_idx) in done_record_indices:
                print(
                    f"[skip] {method} B{budget} seed={seed} "
                    f"record_idx={record_idx}"
                )
                continue

            run_seed = int(seed) + int(record_idx) + 1000003 * int(budget)

            row = {
                "method": method,
                "budget": budget,
                "seed": seed,
                "run_seed": run_seed,
                "record_idx": int(record_idx),
                "local_idx": int(local_idx),
                "participant": participant(record),
                "session_id": session_id(record),
                "ok": False,
                "error": None,
            }

            print(
                f"[run] method={method} B={budget} seed={seed} "
                f"session {local_idx + 1}/{len(selected)} "
                f"record_idx={record_idx} "
                f"participant={row['participant']} "
                f"session_id={row['session_id']}"
            )

            t0 = time.time()

            try:
                oracle = make_record_oracle(record, defense=defense)
                session = extract_session(record)

                optimizer = make_optimizer(
                    method=method,
                    seed=run_seed,
                    budget=budget,
                )

                runner = AttackRunner(
                    optimizer=optimizer,
                    oracle=oracle,
                    mutate=mutate,
                    cost_fn=cost_fn,
                    budget=budget,
                    seed=run_seed,
                )

                result = runner.run(session)
                result_dict = compact_result(result.to_dict(), save_history)

                row.update(result_dict)
                row["ok"] = True
                row["elapsed_sec"] = round(time.time() - t0, 4)

                print(
                    "  result:",
                    "escaped=", row.get("escaped"),
                    "q_first=", row.get("queries_to_first_escape"),
                    "best_cost=", row.get("best_cost"),
                    "elapsed=", row["elapsed_sec"],
                )

            except Exception:
                row["ok"] = False
                row["error"] = traceback.format_exc()
                row["elapsed_sec"] = round(time.time() - t0, 4)

                print("  ERROR")
                print(row["error"])

            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()

    rows = read_jsonl(detail_path)
    summary = summarize_rows(rows)
    summary.update(
        {
            "method": method,
            "budget": budget,
            "seed": seed,
            "detail_path": str(detail_path),
        }
    )

    return summary


def aggregate_summaries(summary_rows: list[dict]) -> list[dict]:
    groups = {}

    for r in summary_rows:
        key = (r["method"], int(r["budget"]))
        groups.setdefault(key, []).append(r)

    out = []

    for (method, budget), rs in sorted(groups.items(), key=lambda x: (x[0][1], x[0][0])):
        asrs = [float(r["asr"]) for r in rs if r["asr"] != ""]
        q_first = [
            float(r["median_queries_to_first_escape"])
            for r in rs
            if r["median_queries_to_first_escape"] != ""
        ]
        costs = [
            float(r["median_best_cost_escaped"])
            for r in rs
            if r["median_best_cost_escaped"] != ""
        ]

        n_total = sum(int(r["n_total"]) for r in rs)
        n_completed = sum(int(r["n_completed"]) for r in rs)
        escaped = sum(int(r["escaped"]) for r in rs)
        n_errors = sum(int(r["n_errors"]) for r in rs)

        out.append(
            {
                "method": method,
                "budget": budget,
                "n_seeds": len(rs),
                "n_total": n_total,
                "n_completed": n_completed,
                "n_errors": n_errors,
                "pooled_asr": escaped / n_completed if n_completed else "",
                "mean_seed_asr": mean(asrs) if asrs else "",
                "std_seed_asr": pstdev(asrs) if len(asrs) > 1 else 0.0,
                "escaped": escaped,
                "median_of_seed_median_q_first": median(q_first) if q_first else "",
                "median_of_seed_median_best_cost": median(costs) if costs else "",
            }
        )

    return out


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--methods", default="random,tpe,hybrid")
    parser.add_argument("--budgets", default="10,30,100,300")
    parser.add_argument("--seeds", default="20260910,20260911,20260912")

    # n_sessions <= 0 means use all detected sessions found within max_scan.
    parser.add_argument("--n-sessions", type=int, default=0)
    parser.add_argument("--max-scan", type=int, default=499)

    parser.add_argument("--output-dir", default="results/v2_formal_full")
    parser.add_argument("--ahb-root", default=None)

    # full = keep all query history for later layer diagnostics.
    # escaped = keep only escaped query history.
    # none = save only run-level metrics.
    parser.add_argument(
        "--save-history",
        choices=["full", "escaped", "none"],
        default="full",
    )

    args = parser.parse_args()

    methods = parse_csv_list(args.methods, str)
    budgets = parse_csv_list(args.budgets, int)
    seeds = parse_csv_list(args.seeds, int)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("FORMAL ATTACK V2 EXPERIMENT")
    print("=" * 100)
    print("repo:", REPO_ROOT)
    print("methods:", methods)
    print("budgets:", budgets)
    print("seeds:", seeds)
    print("n_sessions:", args.n_sessions, "(<=0 means all detected)")
    print("max_scan:", args.max_scan)
    print("output_dir:", output_dir)
    print("save_history:", args.save_history)

    t_start = time.time()

    defense = load_defense(ahb_root=args.ahb_root)
    records = load_long_tap_records(defense)

    print("loaded hardened defense")
    print("long tap records:", len(records))
    print("manifest:", defense.manifest()["defense_version"])
    print("decision:", defense.manifest()["decision"])

    selected, scanned = collect_detected_records(
        defense=defense,
        records=records,
        n_sessions=args.n_sessions,
        max_scan=args.max_scan,
    )

    print("scanned records:", scanned)
    print("selected initially detected records:", len(selected))

    selected_path = output_dir / "selected_detected_records.csv"
    write_csv(
        selected_path,
        [
            {
                "record_idx": idx,
                "participant": participant(record),
                "session_id": session_id(record),
            }
            for idx, record in selected
        ],
    )
    print("saved selected records:", selected_path)

    all_summary_rows = []

    for budget in budgets:
        for seed in seeds:
            for method in methods:
                summary = run_one_config(
                    defense=defense,
                    selected=selected,
                    method=method,
                    budget=budget,
                    seed=seed,
                    output_dir=output_dir,
                    save_history=args.save_history,
                )

                all_summary_rows.append(summary)

                per_config_summary_path = output_dir / "summary_by_config.csv"
                write_csv(per_config_summary_path, all_summary_rows)

                aggregate_rows = aggregate_summaries(all_summary_rows)
                aggregate_path = output_dir / "aggregate_by_budget_method.csv"
                write_csv(aggregate_path, aggregate_rows)

                print("-" * 100)
                print("updated:", per_config_summary_path)
                print("updated:", aggregate_path)
                print("latest summary:", summary)

    aggregate_rows = aggregate_summaries(all_summary_rows)

    print("=" * 100)
    print("FINAL AGGREGATE")
    print("=" * 100)

    for r in aggregate_rows:
        print(r)

    print("saved output dir:", output_dir)
    print("elapsed_sec:", round(time.time() - t_start, 2))


if __name__ == "__main__":
    main()
