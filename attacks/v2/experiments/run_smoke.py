from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from pathlib import Path
from statistics import median

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attacks.v2.runner import AttackRunner
from attacks.v2.mutation import mutate
from attacks.v2.metrics import cost_fn

from attacks.v2.optimizers.random import RandomOptimizer
from attacks.v2.optimizers.tpe import TPEOptimizer
from attacks.v2.optimizers.cmaes import CMAESOptimizer
from attacks.v2.optimizers.hybrid import HybridOptimizer

from attacks.v2.oracle import (
    load_defense,
    load_long_tap_records,
    make_record_oracle,
    extract_session,
    participant,
    session_id,
)


def parse_methods(s: str) -> list[str]:
    return [x.strip().lower() for x in s.split(",") if x.strip()]


def make_optimizer(method: str, seed: int, budget: int):
    if method == "random":
        return RandomOptimizer(seed=seed)

    if method == "tpe":
        return TPEOptimizer(
            seed=seed,
            n_startup_trials=min(5, max(1, budget)),
        )

    if method == "cmaes":
        return CMAESOptimizer(
            seed=seed,
            popsize=min(8, max(2, budget)),
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

        if len(selected) >= n_sessions:
            break

    return selected, scanned


def summarize(rows: list[dict]) -> list[dict]:
    out = []
    methods = sorted(set(r["method"] for r in rows))

    for method in methods:
        mr = [r for r in rows if r["method"] == method]
        completed = [r for r in mr if r.get("ok")]
        escaped = [r for r in completed if r.get("escaped")]

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

        out.append(
            {
                "method": method,
                "n_total": len(mr),
                "n_completed": len(completed),
                "n_errors": len(mr) - len(completed),
                "asr": (len(escaped) / len(completed)) if completed else None,
                "escaped": len(escaped),
                "median_queries_to_first_escape": median(q_first) if q_first else None,
                "median_best_cost_escaped": median(best_costs) if best_costs else None,
            }
        )

    return out


def write_summary_csv(path: Path, summary_rows: list[dict]):
    if not summary_rows:
        return

    keys = list(summary_rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", default="random,tpe,cmaes")
    parser.add_argument("--budget", type=int, default=10)
    parser.add_argument("--n-sessions", type=int, default=5)
    parser.add_argument("--max-scan", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--output-dir", default="results/v2_smoke")
    parser.add_argument("--ahb-root", default=None)
    args = parser.parse_args()

    methods = parse_methods(args.methods)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    detail_path = output_dir / (
        f"details_methods-{','.join(methods)}"
        f"_B{args.budget}_N{args.n_sessions}_seed{args.seed}.jsonl"
    )
    summary_path = output_dir / (
        f"summary_methods-{','.join(methods)}"
        f"_B{args.budget}_N{args.n_sessions}_seed{args.seed}.csv"
    )

    print("=" * 100)
    print("ATTACK V2 SMOKE TEST")
    print("=" * 100)
    print("repo:", REPO_ROOT)
    print("methods:", methods)
    print("budget:", args.budget)
    print("n_sessions:", args.n_sessions)
    print("output detail:", detail_path)
    print("output summary:", summary_path)

    t0 = time.time()

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
    print("selected detected records:", len(selected))

    if len(selected) == 0:
        raise RuntimeError("No initially detected records found.")

    rows = []

    with detail_path.open("w") as fout:
        for method_idx, method in enumerate(methods):
            print("-" * 100)
            print("method:", method)

            for local_idx, (record_idx, record) in enumerate(selected):
                run_seed = args.seed + method_idx * 100000 + local_idx
                row = {
                    "method": method,
                    "budget": args.budget,
                    "seed": run_seed,
                    "record_idx": record_idx,
                    "local_idx": local_idx,
                    "participant": participant(record),
                    "session_id": session_id(record),
                    "ok": False,
                    "error": None,
                }

                print(
                    f"[{method}] session {local_idx + 1}/{len(selected)} "
                    f"record_idx={record_idx} "
                    f"participant={row['participant']} "
                    f"session_id={row['session_id']}"
                )

                try:
                    oracle = make_record_oracle(record, defense=defense)
                    session = extract_session(record)

                    optimizer = make_optimizer(
                        method=method,
                        seed=run_seed,
                        budget=args.budget,
                    )

                    runner = AttackRunner(
                        optimizer=optimizer,
                        oracle=oracle,
                        mutate=mutate,
                        cost_fn=cost_fn,
                        budget=args.budget,
                        seed=run_seed,
                    )

                    result = runner.run(session)
                    result_dict = result.to_dict()

                    row.update(
                        {
                            "ok": True,
                            "initial_detected": result.initial_detected,
                            "escaped": result.escaped,
                            "queries_used": result.queries_used,
                            "queries_to_first_escape": result.queries_to_first_escape,
                            "best_cost": result.best_cost,
                            "best_latent": result_dict.get("best_latent"),
                            "history": result_dict.get("history"),
                        }
                    )

                    print(
                        "  result:",
                        "escaped=", row["escaped"],
                        "q_first=", row["queries_to_first_escape"],
                        "best_cost=", row["best_cost"],
                    )

                except Exception:
                    row["ok"] = False
                    row["error"] = traceback.format_exc()
                    print("  ERROR")
                    print(row["error"])

                rows.append(row)
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                fout.flush()

    summary_rows = summarize(rows)
    write_summary_csv(summary_path, summary_rows)

    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    for row in summary_rows:
        print(row)

    print("saved detail:", detail_path)
    print("saved summary:", summary_path)
    print("elapsed_sec:", round(time.time() - t0, 2))


if __name__ == "__main__":
    main()
