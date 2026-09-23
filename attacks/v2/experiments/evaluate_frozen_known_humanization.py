#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import json
import hashlib
from pathlib import Path
from collections import defaultdict

import pandas as pd


DEFAULT_AHB_ROOT = (
    "/storage/cjh/self-evolving-safety/"
    "Passing-the-Turing-Test-on-Screen-Agent-Humanization-Benchmark-main"
)


def rec_get(record, key, default=None):
    if isinstance(record, dict):
        return record.get(key, default)
    return getattr(record, key, default)


def looks_like_record_list(x):
    return (
        isinstance(x, list)
        and len(x) > 0
        and isinstance(x[0], dict)
        and "gestures" in x[0]
    )


def record_uid(record):
    participant = str(rec_get(record, "participant", "UNKNOWN"))
    session_id = str(rec_get(record, "session_id", "UNKNOWN"))
    group = str(rec_get(record, "group", "UNKNOWN"))
    raw = f"{group}::{participant}::{session_id}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def participant_type(participant: str):
    p = participant.lower().strip()

    # 注意：不要用 "human" in p，因为 ui-tars no humanity 会误判。
    if p.startswith("user") or p in {"human", "real human", "human raw"}:
        return "human"

    return "agent"


def infer_baseline_label(source_key: str, participant: str, group: str):
    s = f"{source_key} {participant} {group}".lower()

    if "human" in source_key.lower() or participant_type(participant) == "human":
        return "clean_human"

    if "raw" in s:
        return "raw_agent"

    if "spline" in s or "bspline" in s or "b_spline" in s:
        return "ahb_bspline_or_motion_transform"

    if "history" in s or "hist" in s:
        return "ahb_history_matching"

    if "fake" in s:
        return "ahb_fake_action"

    if "temporal" in s or "duration" in s or "time" in s:
        return "ahb_temporal_adjustment"

    if "long" in s and "tap" in s:
        return "long_tap_condition"

    if "claude" in s:
        return "claude_agent_condition"

    if "gpt4o" in s or "gpt-4o" in s:
        return "gpt4o_agent_condition"

    if "ui-tars" in s or "uitars" in s:
        return "ui_tars_agent_condition"

    if "cpm" in s:
        return "cpm_gui_agent_condition"

    if "autoglm" in s or "auto glm" in s:
        return "autoglm_agent_condition"

    return "other_or_unlabeled"


def find_record_lists(defense):
    candidates = []

    for space_name in ["ns", "inner"]:
        space = getattr(defense, space_name, {})
        if not isinstance(space, dict):
            continue

        for key, val in sorted(space.items()):
            if looks_like_record_list(val):
                participants = sorted({
                    str(rec_get(r, "participant", "UNKNOWN"))
                    for r in val[:200]
                })
                groups = sorted({
                    str(rec_get(r, "group", "UNKNOWN"))
                    for r in val[:200]
                })

                candidates.append({
                    "space": space_name,
                    "key": key,
                    "records": val,
                    "n_records": len(val),
                    "participants_sample": participants[:20],
                    "groups_sample": groups[:20],
                })

    return candidates


def score_one(defense, record):
    return defense.score_record(
        record,
        task_cluster=None,
        v1_condition="Long Tap Author",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output-dir",
        default="results/frozen_defense_known_humanization",
    )
    ap.add_argument(
        "--max-per-list",
        type=int,
        default=None,
        help="Optional cap per record list for quick smoke test.",
    )
    ap.add_argument(
        "--include-human",
        action="store_true",
        help=(
            "Try scoring human records. Usually this will fail for frozen V2 "
            "cross-fit evaluator, so default is to skip human records."
        ),
    )
    ap.add_argument(
        "--dedup-within-list",
        action="store_true",
        help="Deduplicate records within each source list by group/participant/session_id.",
    )
    args = ap.parse_args()

    os.environ["AHB_ROOT"] = os.environ.get("AHB_ROOT", DEFAULT_AHB_ROOT)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from evaluation.frozen_v1v2v3_defense_hardened import FrozenV1V2V3Defense

    print("=" * 100)
    print("Loading frozen defense")
    print("=" * 100)

    defense = FrozenV1V2V3Defense()

    try:
        manifest = defense.manifest()
        manifest_path = out_dir / "frozen_defense_manifest.json"
        with manifest_path.open("w") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
        print("saved manifest:", manifest_path)
        print("defense_version:", manifest.get("defense_version"))
        print("decision:", manifest.get("decision"))
    except Exception as e:
        print("manifest unavailable:", repr(e))

    candidates = find_record_lists(defense)

    inventory_rows = []
    for c in candidates:
        inventory_rows.append({
            "space": c["space"],
            "key": c["key"],
            "n_records": c["n_records"],
            "participants_sample": json.dumps(c["participants_sample"], ensure_ascii=False),
            "groups_sample": json.dumps(c["groups_sample"], ensure_ascii=False),
        })

    inv_df = pd.DataFrame(inventory_rows).sort_values(
        ["space", "key"],
        ignore_index=True,
    )

    inv_path = out_dir / "record_list_inventory.csv"
    inv_df.to_csv(inv_path, index=False)

    print("\n" + "=" * 100)
    print("Record list inventory")
    print("=" * 100)
    print(inv_df.to_string(index=False))
    print("saved:", inv_path)

    detail_rows = []

    print("\n" + "=" * 100)
    print("Scoring candidate record lists")
    print("=" * 100)

    for c in candidates:
        source_space = c["space"]
        source_key = c["key"]
        records = c["records"]

        if args.max_per_list is not None:
            records = records[: args.max_per_list]

        if args.dedup_within_list:
            seen = set()
            deduped = []
            for r in records:
                uid = record_uid(r)
                if uid in seen:
                    continue
                seen.add(uid)
                deduped.append(r)
            records = deduped

        print("\n" + "-" * 100)
        print(f"source={source_space}.{source_key} n={len(records)}")
        print("-" * 100)

        for i, rec in enumerate(records):
            group = str(rec_get(rec, "group", "UNKNOWN"))
            participant = str(rec_get(rec, "participant", "UNKNOWN"))
            session_id = str(rec_get(rec, "session_id", "UNKNOWN"))
            ptype = participant_type(participant)
            baseline_label = infer_baseline_label(source_key, participant, group)

            if ptype == "human" and not args.include_human:
                detail_rows.append({
                    "source_space": source_space,
                    "source_key": source_key,
                    "baseline_label": baseline_label,
                    "record_index": i,
                    "uid": record_uid(rec),
                    "group": group,
                    "participant": participant,
                    "participant_type": ptype,
                    "session_id": session_id,
                    "ok": False,
                    "skipped": True,
                    "error": "skipped_human_record_by_default",
                    "final_detect": None,
                    "v1_detect": None,
                    "v2_detect": None,
                    "v3_detect": None,
                })
                continue

            try:
                out = score_one(defense, rec)

                detail_rows.append({
                    "source_space": source_space,
                    "source_key": source_key,
                    "baseline_label": baseline_label,
                    "record_index": i,
                    "uid": record_uid(rec),
                    "group": group,
                    "participant": participant,
                    "participant_type": ptype,
                    "session_id": session_id,
                    "ok": True,
                    "skipped": False,
                    "error": "",
                    "final_detect": bool(out.get("final_detect", False)),
                    "v1_detect": bool(out.get("v1_detect", False)),
                    "v2_detect": bool(out.get("v2_detect", False)),
                    "v3_detect": bool(out.get("v3_detect", False)),
                    "v1_score": out.get("v1_score", None),
                    "v2_score": out.get("v2_score", None),
                    "v3_rate": out.get("v3_rate", None),
                    "task_cluster": out.get("task_cluster", None),
                    "raw_score_json": json.dumps(out, default=str, ensure_ascii=False),
                })

            except Exception as e:
                detail_rows.append({
                    "source_space": source_space,
                    "source_key": source_key,
                    "baseline_label": baseline_label,
                    "record_index": i,
                    "uid": record_uid(rec),
                    "group": group,
                    "participant": participant,
                    "participant_type": ptype,
                    "session_id": session_id,
                    "ok": False,
                    "skipped": False,
                    "error": repr(e),
                    "final_detect": None,
                    "v1_detect": None,
                    "v2_detect": None,
                    "v3_detect": None,
                })

            if (i + 1) % 100 == 0:
                print(f"processed {i + 1}/{len(records)}")

    detail_df = pd.DataFrame(detail_rows)
    detail_path = out_dir / "frozen_known_humanization_details.csv"
    detail_df.to_csv(detail_path, index=False)

    print("\n" + "=" * 100)
    print("Saved details")
    print("=" * 100)
    print("saved:", detail_path)

    scored = detail_df[(detail_df["ok"] == True) & (detail_df["skipped"] == False)].copy()

    if scored.empty:
        print("No scored records. Check errors in details CSV.")
        return

    def summarize(g):
        n = len(g)
        final = int(g["final_detect"].sum())
        v1 = int(g["v1_detect"].sum()) if "v1_detect" in g else 0
        v2 = int(g["v2_detect"].sum()) if "v2_detect" in g else 0
        v3 = int(g["v3_detect"].sum()) if "v3_detect" in g else 0

        return pd.Series({
            "n": n,
            "final_detect": final,
            "final_detect_rate_%": round(final / n * 100, 2),
            "v1_detect": v1,
            "v1_detect_rate_%": round(v1 / n * 100, 2),
            "v2_detect": v2,
            "v2_detect_rate_%": round(v2 / n * 100, 2),
            "v3_detect": v3,
            "v3_detect_rate_%": round(v3 / n * 100, 2),
        })

    by_source = (
        scored
        .groupby(["source_space", "source_key", "baseline_label", "participant_type"], dropna=False)
        .apply(summarize)
        .reset_index()
        .sort_values(["baseline_label", "source_key"])
    )

    by_participant = (
        scored
        .groupby(["baseline_label", "participant", "participant_type"], dropna=False)
        .apply(summarize)
        .reset_index()
        .sort_values(["baseline_label", "participant"])
    )

    by_baseline = (
        scored
        .groupby(["baseline_label", "participant_type"], dropna=False)
        .apply(summarize)
        .reset_index()
        .sort_values(["baseline_label", "participant_type"])
    )

    source_path = out_dir / "summary_by_source_list.csv"
    part_path = out_dir / "summary_by_baseline_participant.csv"
    base_path = out_dir / "summary_by_baseline.csv"

    by_source.to_csv(source_path, index=False)
    by_participant.to_csv(part_path, index=False)
    by_baseline.to_csv(base_path, index=False)

    print("\n" + "=" * 100)
    print("Summary by baseline")
    print("=" * 100)
    print(by_baseline.to_string(index=False))
    print("saved:", base_path)

    print("\n" + "=" * 100)
    print("Summary by baseline and participant")
    print("=" * 100)
    print(by_participant.to_string(index=False))
    print("saved:", part_path)

    print("\n" + "=" * 100)
    print("Summary by source list")
    print("=" * 100)
    print(by_source.to_string(index=False))
    print("saved:", source_path)

    # Also write a markdown table for paper notes.
    md_path = out_dir / "summary_by_baseline.md"
    with md_path.open("w") as f:
        f.write("# Frozen defense on known humanization conditions\n\n")
        f.write(by_baseline.to_markdown(index=False))
        f.write("\n")
    print("saved:", md_path)


if __name__ == "__main__":
    main()
