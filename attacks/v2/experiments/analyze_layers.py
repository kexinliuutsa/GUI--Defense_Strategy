from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attacks.v2.search_space import LatentVector
from attacks.v2.parameterization import parameterize
from attacks.v2.mutation import mutate
from attacks.v2.oracle import load_defense, load_long_tap_records, extract_session


def load_jsonl(path: Path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--details", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--ahb-root", default=None)
    args = parser.parse_args()

    details_path = Path(args.details)

    if args.out is None:
        out_path = details_path.with_name("layer_diagnostics_" + details_path.stem + ".csv")
    else:
        out_path = Path(args.out)

    defense = load_defense(ahb_root=args.ahb_root)
    records = load_long_tap_records(defense)
    rows = load_jsonl(details_path)

    fieldnames = [
        "method",
        "budget",
        "seed",
        "record_idx",
        "participant",
        "session_id",
        "query",
        "history_detected",
        "history_escaped",
        "replay_final_detect",
        "match_history",
        "v1_detect",
        "v2_temporal_detect",
        "v3_detect",
        "v1v2_detect",
        "v2_temporal_margin",
        "v2_temporal_threshold",
        "v3_interval_rate",
        "v3_interval_total",
        "v3_interval_violations",
        "cost",
        "use_spatial",
        "use_temporal",
        "spatial_amp_px",
        "spatial_freq",
        "duration_scale",
        "gap_scale",
        "timing_jitter",
        "heterogeneity",
        "z_spatial",
        "z_frequency",
        "z_duration",
        "z_gap",
        "z_jitter",
        "z_heterogeneity",
        "z_use_spatial",
        "z_use_temporal",
    ]

    diag_rows = []

    for run in rows:
        if not run.get("ok"):
            continue

        record = records[int(run["record_idx"])]
        session = extract_session(record)

        rng = np.random.default_rng(int(run.get("run_seed", run["seed"])))

        for h in run.get("history", []):
            z = LatentVector.from_dict(h["latent"])
            params = parameterize(z)

            candidate = mutate(session, z, rng)
            candidate_record = defense.replace_gestures(record, candidate)
            diag = defense.score_record(candidate_record)

            replay_final = bool(diag["final_detect"])
            history_detected = bool(h["detected"])

            diag_rows.append(
                {
                    "method": run["method"],
                    "budget": run["budget"],
                    "seed": run["seed"],
                    "record_idx": run["record_idx"],
                    "participant": run["participant"],
                    "session_id": run["session_id"],
                    "query": h["query"],
                    "history_detected": history_detected,
                    "history_escaped": bool(h["escaped"]),
                    "replay_final_detect": replay_final,
                    "match_history": replay_final == history_detected,
                    "v1_detect": bool(diag["v1_detect"]),
                    "v2_temporal_detect": bool(diag["v2_temporal_detect"]),
                    "v3_detect": bool(diag["v3_detect"]),
                    "v1v2_detect": bool(diag["v1v2_detect"]),
                    "v2_temporal_margin": float(diag["v2_temporal_margin"]),
                    "v2_temporal_threshold": float(diag["v2_temporal_threshold"]),
                    "v3_interval_rate": float(diag["interval_violation_rate"]),
                    "v3_interval_total": int(diag["interval_total"]),
                    "v3_interval_violations": int(diag["interval_violations"]),
                    "cost": float(h["cost"]),
                    "use_spatial": bool(params.use_spatial),
                    "use_temporal": bool(params.use_temporal),
                    "spatial_amp_px": float(params.spatial_amp_px),
                    "spatial_freq": int(params.spatial_freq),
                    "duration_scale": float(params.duration_scale),
                    "gap_scale": float(params.gap_scale),
                    "timing_jitter": float(params.timing_jitter),
                    "heterogeneity": float(params.heterogeneity),
                    **{k: float(v) for k, v in h["latent"].items()},
                }
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in diag_rows:
            writer.writerow(r)

    print("saved:", out_path)
    print("diagnostic rows:", len(diag_rows))

    by_method = defaultdict(list)
    for r in diag_rows:
        by_method[r["method"]].append(r)

    print("=" * 100)
    print("QUERY-LEVEL LAYER SUMMARY")
    print("=" * 100)

    for method, rs in sorted(by_method.items()):
        n = len(rs)
        if n == 0:
            continue

        def rate(key):
            return sum(bool(r[key]) for r in rs) / n

        escaped = [r for r in rs if not r["replay_final_detect"]]
        print("method:", method)
        print("  queries:", n)
        print("  replay escape query rate:", len(escaped) / n)
        print("  v1 detect rate:", rate("v1_detect"))
        print("  v2 detect rate:", rate("v2_temporal_detect"))
        print("  v3 detect rate:", rate("v3_detect"))
        print("  all replay matched history:", all(r["match_history"] for r in rs))

    print("=" * 100)
    print("ESCAPED QUERY PARAMETER SNAPSHOT")
    print("=" * 100)

    escaped_rows = [r for r in diag_rows if not r["replay_final_detect"]]

    for method in sorted(set(r["method"] for r in escaped_rows)):
        rs = [r for r in escaped_rows if r["method"] == method]
        if not rs:
            continue

        def med(key):
            vals = sorted(float(r[key]) for r in rs)
            return vals[len(vals) // 2]

        print("method:", method, "escaped_queries:", len(rs))
        for key in [
            "use_spatial",
            "use_temporal",
            "spatial_amp_px",
            "spatial_freq",
            "duration_scale",
            "gap_scale",
            "timing_jitter",
            "heterogeneity",
            "cost",
        ]:
            if key.startswith("use_"):
                print(" ", key, "rate:", sum(bool(r[key]) for r in rs) / len(rs))
            else:
                print(" ", key, "median:", med(key))


if __name__ == "__main__":
    main()
