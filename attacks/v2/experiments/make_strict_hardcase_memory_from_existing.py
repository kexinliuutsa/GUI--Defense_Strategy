#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
import re

import pandas as pd

# ---------------------------------------------------------------------
# Pickle compatibility: original hardcase objects may reference modules
# such as "analysis" saved from the AHB/codebase runtime.
# Add likely roots before pickle.load().
# ---------------------------------------------------------------------
import sys

_REPO_ROOT = Path(__file__).resolve().parents[3]
_AHB_ROOT = Path("/storage/cjh/self-evolving-safety/Passing-the-Turing-Test-on-Screen-Agent-Humanization-Benchmark-main")

for _root in [_REPO_ROOT, _AHB_ROOT]:
    if _root.exists() and str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

# If a pickle refers to top-level module "analysis", Python needs the
# directory containing analysis.py in sys.path.
for _root in [_REPO_ROOT, _AHB_ROOT]:
    if _root.exists():
        for _analysis_py in _root.rglob("analysis.py"):
            _parent = str(_analysis_py.parent)
            if _parent not in sys.path:
                sys.path.insert(0, _parent)




def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--hardcase-pkl",
        default="results/v2_loop_hard_cases/mutated_records/escaped_mutated_records.pkl",
    )
    ap.add_argument(
        "--metadata-jsonl",
        default="results/v2_loop_hard_cases/mutated_records/escaped_mutated_metadata.jsonl",
    )
    ap.add_argument(
        "--output-dir",
        default="results/strict_hardcase_memories",
    )

    ap.add_argument("--name", required=True)
    ap.add_argument("--train-methods", default="hybrid")
    ap.add_argument("--train-budgets", default="30")
    ap.add_argument("--train-agents", default="AgentCPM,UI-TARS")
    ap.add_argument("--max-hardcases", type=int, default=500)

    return ap.parse_args()


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


def safe_name(x):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(x)).strip("_")


def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c

    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]

    return None


def main():
    args = parse_args()

    hardcase_pkl = Path(args.hardcase_pkl)
    metadata_jsonl = Path(args.metadata_jsonl)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(hardcase_pkl, "rb") as f:
        hardcase_root = pickle.load(f)

    rows = []
    with open(metadata_jsonl, "r") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            d = json.loads(line)
            d["_memory_index"] = i
            rows.append(d)

    meta = pd.DataFrame(rows)

    def unwrap_hardcases(root, expected_len=None):
        """
        Return:
          hardcases: the actual list of hardcase objects
          root_kind: 'list' or 'dict'
          list_key: key used when root is dict
        """
        if isinstance(root, list):
            return root, "list", None

        if isinstance(root, dict):
            print("=" * 100)
            print("Hardcase pkl root is dict")
            print("=" * 100)
            print("keys:", list(root.keys()))

            preferred_keys = [
                "hardcases",
                "records",
                "mutated_records",
                "escaped_mutated_records",
                "data",
                "items",
                "samples",
                "trajectories",
            ]

            # First try preferred keys.
            for k in preferred_keys:
                if k in root and isinstance(root[k], list):
                    if expected_len is None or len(root[k]) == expected_len:
                        return root[k], "dict", k

            # Then try any list with expected length.
            if expected_len is not None:
                for k, v in root.items():
                    if isinstance(v, list) and len(v) == expected_len:
                        return v, "dict", k

            # Otherwise fall back to the largest non-empty list.
            list_candidates = [
                (k, v) for k, v in root.items()
                if isinstance(v, list) and len(v) > 0
            ]

            if list_candidates:
                list_candidates.sort(key=lambda kv: len(kv[1]), reverse=True)
                k, v = list_candidates[0]
                print("[warn] using largest list field as hardcases:", k, "len=", len(v))
                return v, "dict", k

        print("Unsupported hardcase root type:", type(root))
        if isinstance(root, dict):
            print("available keys:")
            for k, v in root.items():
                try:
                    print(" ", k, type(v), "len=", len(v))
                except Exception:
                    print(" ", k, type(v))
        raise SystemExit("Could not unwrap hardcase list from pkl.")

    hardcases, root_kind, list_key = unwrap_hardcases(hardcase_root, expected_len=len(meta))

    print("=" * 100)
    print("Unwrapped hardcases")
    print("=" * 100)
    print("root_kind:", root_kind)
    print("list_key:", list_key)
    print("n hardcases:", len(hardcases))
    print("metadata rows:", len(meta))

    if len(meta) != len(hardcases):
        print("[warn] metadata length != hardcase length")
        print("metadata:", len(meta))
        print("hardcases:", len(hardcases))
        print("Will only use overlapping indices.")

    method_col = pick_col(meta, [
        "method", "source_method", "attack_method", "search_method"
    ])
    budget_col = pick_col(meta, [
        "budget", "source_budget", "attack_budget"
    ])
    agent_col = pick_col(meta, [
        "agent", "participant", "source_agent", "model", "agent_name"
    ])

    print("=" * 100)
    print("Detected metadata columns")
    print("=" * 100)
    print("method_col:", method_col)
    print("budget_col:", budget_col)
    print("agent_col:", agent_col)

    if method_col is None or budget_col is None or agent_col is None:
        print("\nAvailable columns:")
        for c in meta.columns:
            print(" ", c)
        raise SystemExit("Could not identify method/budget/agent columns.")

    meta["_method_norm"] = meta[method_col].astype(str)
    meta["_budget_norm"] = pd.to_numeric(meta[budget_col], errors="coerce").astype("Int64")
    meta["_agent_norm"] = meta[agent_col].apply(normalize_agent)

    train_methods = {x.strip() for x in args.train_methods.split(",") if x.strip()}
    train_budgets = {int(float(x.strip())) for x in args.train_budgets.split(",") if x.strip()}
    train_agents = {x.strip() for x in args.train_agents.split(",") if x.strip()}

    selected = meta[
        meta["_method_norm"].isin(train_methods)
        & meta["_budget_norm"].isin(train_budgets)
        & meta["_agent_norm"].isin(train_agents)
    ].copy()

    selected = selected.sort_values("_memory_index").head(args.max_hardcases)

    if selected.empty:
        print("\nMetadata preview:")
        print(meta[["_method_norm", "_budget_norm", "_agent_norm"]].value_counts().head(50))
        raise SystemExit("Selected zero hardcases. Check filters.")

    selected_indices = [
        int(i) for i in selected["_memory_index"].tolist()
        if int(i) < len(hardcases)
    ]

    strict_hardcases = [hardcases[i] for i in selected_indices]

    out_pkl = output_dir / f"{safe_name(args.name)}.pkl"
    out_meta = output_dir / f"{safe_name(args.name)}.metadata.csv"

    # Preserve original pkl container format.
    # This matters because the loop defense may expect a dict/wrapper, not a bare list.
    if root_kind == "dict":
        out_root = dict(hardcase_root)
        out_root[list_key] = strict_hardcases

        # If other list-valued fields align one-to-one with hardcases, filter them too.
        for k, v in list(out_root.items()):
            if k == list_key:
                continue
            if isinstance(v, list) and len(v) == len(hardcases):
                out_root[k] = [v[i] for i in selected_indices if i < len(v)]

        with open(out_pkl, "wb") as f:
            pickle.dump(out_root, f)
    else:
        with open(out_pkl, "wb") as f:
            pickle.dump(strict_hardcases, f)

    selected.to_csv(out_meta, index=False)

    print("\n" + "=" * 100)
    print("Strict hardcase memory built from existing pkl")
    print("=" * 100)
    print("name:", args.name)
    print("output pkl:", out_pkl)
    print("output metadata:", out_meta)
    print("n selected:", len(strict_hardcases))

    print("\nSelected counts:")
    print(
        selected.groupby(["_method_norm", "_budget_norm", "_agent_norm"])
        .size()
        .reset_index(name="n")
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
