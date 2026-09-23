#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ======================================================================================
# Utilities
# ======================================================================================

def safe_json_loads(x):
    if x is None:
        return None
    if isinstance(x, dict):
        return x
    if isinstance(x, list):
        return x
    if not isinstance(x, str):
        return None
    s = x.strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return None
    try:
        return json.loads(s)
    except Exception:
        return None


def flatten_dict(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out = {}
    for k, v in d.items():
        kk = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            out.update(flatten_dict(v, kk))
        else:
            out[kk] = v
    return out


def to_numeric_float(x):
    """
    Robust numeric conversion.
    Pandas/numpy quantile can fail on boolean dtype, so cast everything to float.
    True/False become 1.0/0.0.
    """
    s = pd.to_numeric(x, errors="coerce")
    try:
        return s.astype(float)
    except Exception:
        return pd.to_numeric(s.astype(str), errors="coerce").astype(float)


def as_bool(x):
    if isinstance(x, bool):
        return x
    if x is None:
        return None
    if isinstance(x, (int, float)):
        if math.isnan(float(x)):
            return None
        return bool(x)
    if isinstance(x, str):
        s = x.strip().lower()
        if s in {"true", "1", "yes", "y", "escaped", "success"}:
            return True
        if s in {"false", "0", "no", "n", "detected", "fail", "failed"}:
            return False
    return None


def infer_budget_from_path(path: str):
    m = re.search(r"B(\d+)", path)
    if m:
        return int(m.group(1))
    return None


def infer_method_from_path(path: str):
    s = path.lower()
    if "two_stage" in s or "two-stage" in s:
        return "two-stage-param"
    if "hybrid" in s:
        return "hybrid"
    if "random" in s:
        return "random"
    if "tpe" in s:
        return "tpe"
    if "qwen" in s:
        return "qwen-llm"
    if "deepseek" in s:
        return "deepseek-llm"
    return None


def derive_real_params(row: dict[str, Any]) -> dict[str, Any]:
    """
    If a trial only has normalized z_* parameters, derive approximate real mutation params.
    This follows the search-space parameterization used in the current attack code.
    """
    out = dict(row)

    freq_grid = [1, 2, 4, 6, 8, 12, 16]

    def f(name):
        v = out.get(name)
        if v is None:
            return None
        try:
            return float(v)
        except Exception:
            return None

    z_spatial = f("z_spatial")
    z_frequency = f("z_frequency")
    z_duration = f("z_duration")
    z_gap = f("z_gap")
    z_jitter = f("z_jitter")
    z_heterogeneity = f("z_heterogeneity")
    z_use_spatial = f("z_use_spatial")
    z_use_temporal = f("z_use_temporal")

    if "spatial_amp_px" not in out and z_spatial is not None:
        out["spatial_amp_px"] = 10.0 * z_spatial

    if "frequency" not in out and z_frequency is not None:
        idx = int(np.clip(round(z_frequency * (len(freq_grid) - 1)), 0, len(freq_grid) - 1))
        out["frequency"] = freq_grid[idx]

    if "duration_scale" not in out and z_duration is not None:
        out["duration_scale"] = 0.55 + 1.25 * z_duration

    if "gap_scale" not in out and z_gap is not None:
        out["gap_scale"] = 0.55 + 1.25 * z_gap

    if "jitter_frac" not in out and z_jitter is not None:
        out["jitter_frac"] = 0.35 * z_jitter

    if "heterogeneity" not in out and z_heterogeneity is not None:
        out["heterogeneity"] = 0.55 * z_heterogeneity

    if "use_spatial" not in out and z_use_spatial is not None:
        out["use_spatial"] = z_use_spatial >= 0.5

    if "use_temporal" not in out and z_use_temporal is not None:
        out["use_temporal"] = z_use_temporal >= 0.5

    return out


# ======================================================================================
# Load trial-level data
# ======================================================================================

def extract_candidate_from_obj(obj: dict[str, Any]):
    for key in [
        "candidate",
        "latent",
        "z",
        "params",
        "best_candidate",
        "best_latent",
        "best_params",
    ]:
        v = obj.get(key)
        if isinstance(v, dict):
            return v

    # Some rows store z_* directly at the same level.
    z_like = {
        k: v for k, v in obj.items()
        if (
            k.startswith("z_")
            or k in {
                "spatial_amp_px",
                "frequency",
                "duration_scale",
                "gap_scale",
                "jitter_frac",
                "heterogeneity",
                "use_spatial",
                "use_temporal",
                "op_mode",
                "op_prob",
                "op_strength",
                "family",
            }
        )
    }
    if z_like:
        return z_like

    return None


def extract_escaped(obj: dict[str, Any]):
    for k in ["escaped", "success", "is_escape", "attack_success"]:
        if k in obj:
            b = as_bool(obj.get(k))
            if b is not None:
                return b

    # If detected is explicitly present, escaped = not detected.
    for k in ["detected", "final_detect", "is_detected"]:
        if k in obj:
            b = as_bool(obj.get(k))
            if b is not None:
                return not b

    return None


def extract_cost(obj: dict[str, Any]):
    for k in [
        "cost",
        "best_cost",
        "median_best_cost",
        "total_cost",
        "distortion_cost",
        "objective_cost",
    ]:
        if k in obj and obj.get(k) is not None:
            try:
                return float(obj.get(k))
            except Exception:
                pass
    return None


def extract_q(obj: dict[str, Any]):
    for k in ["q", "query", "query_index", "queries", "queries_to_first_escape"]:
        if k in obj and obj.get(k) is not None:
            try:
                return int(float(obj.get(k)))
            except Exception:
                pass
    return None


def extract_trials_from_jsonl(path: Path):
    rows = []

    with path.open("r", errors="ignore") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue

            try:
                parent = json.loads(line)
            except Exception:
                continue

            if not isinstance(parent, dict):
                continue

            parent_method = parent.get("method") or infer_method_from_path(str(path))
            parent_budget = parent.get("budget") or infer_budget_from_path(str(path))
            parent_seed = parent.get("seed")
            parent_record_index = parent.get("record_index")
            parent_session_id = parent.get("session_id")
            parent_participant = parent.get("participant")

            # Case 1: per-session object with a history list.
            history = None
            for hk in ["history", "trials", "queries", "evaluations", "attempts"]:
                if isinstance(parent.get(hk), list):
                    history = parent.get(hk)
                    break

            if history is not None:
                for trial in history:
                    if not isinstance(trial, dict):
                        continue

                    cand = extract_candidate_from_obj(trial)
                    if cand is None:
                        continue

                    escaped = extract_escaped(trial)
                    if escaped is None:
                        continue

                    flat = flatten_dict(cand)
                    flat = derive_real_params(flat)

                    row = {
                        "source_file": str(path),
                        "line_no": line_no,
                        "method": trial.get("method", parent_method),
                        "budget": trial.get("budget", parent_budget),
                        "seed": trial.get("seed", parent_seed),
                        "record_index": trial.get("record_index", parent_record_index),
                        "session_id": trial.get("session_id", parent_session_id),
                        "participant": trial.get("participant", parent_participant),
                        "q": extract_q(trial),
                        "escaped": bool(escaped),
                        "cost": extract_cost(trial),
                    }
                    row.update(flat)
                    rows.append(row)

            # Case 2: each JSONL line is itself a single trial.
            else:
                cand = extract_candidate_from_obj(parent)
                escaped = extract_escaped(parent)

                if cand is not None and escaped is not None:
                    flat = flatten_dict(cand)
                    flat = derive_real_params(flat)

                    row = {
                        "source_file": str(path),
                        "line_no": line_no,
                        "method": parent_method,
                        "budget": parent_budget,
                        "seed": parent_seed,
                        "record_index": parent_record_index,
                        "session_id": parent_session_id,
                        "participant": parent_participant,
                        "q": extract_q(parent),
                        "escaped": bool(escaped),
                        "cost": extract_cost(parent),
                    }
                    row.update(flat)
                    rows.append(row)

    return rows


def extract_trials_from_session_csv(path: Path):
    """
    Fallback: extracts only best successful candidates from session_results.csv.
    This is less useful than attack_history.jsonl because failed candidates are missing.
    """
    rows = []

    try:
        df = pd.read_csv(path)
    except Exception:
        return rows

    json_cols = [
        c for c in df.columns
        if any(x in c.lower() for x in ["candidate", "latent", "params"])
    ]

    if not json_cols:
        return rows

    for idx, r in df.iterrows():
        escaped = as_bool(r.get("escaped"))
        if escaped is None:
            continue

        cand = None
        for c in json_cols:
            obj = safe_json_loads(r.get(c))
            if isinstance(obj, dict):
                cand = obj
                break

        if cand is None:
            continue

        flat = flatten_dict(cand)
        flat = derive_real_params(flat)

        row = {
            "source_file": str(path),
            "line_no": None,
            "method": r.get("method", infer_method_from_path(str(path))),
            "budget": r.get("budget", infer_budget_from_path(str(path))),
            "seed": r.get("seed", None),
            "record_index": r.get("record_index", idx),
            "session_id": r.get("session_id", None),
            "participant": r.get("participant", None),
            "q": r.get("queries_to_first_escape", None),
            "escaped": bool(escaped),
            "cost": r.get("best_cost", None),
        }
        row.update(flat)
        rows.append(row)

    return rows


def load_all_trials(input_dirs: list[str]):
    rows = []

    for d in input_dirs:
        root = Path(d)
        if not root.exists():
            print("missing input:", root)
            continue

        jsonl_files = list(root.rglob("*.jsonl"))
        csv_files = list(root.rglob("session_results.csv"))

        print(f"scanning {root}: {len(jsonl_files)} jsonl, {len(csv_files)} session csv")

        for p in jsonl_files:
            got = extract_trials_from_jsonl(p)
            if got:
                print(f"  loaded {len(got):>8} trial rows from {p}")
            rows.extend(got)

        # Fallback, useful if a run did not save full history.
        for p in csv_files:
            got = extract_trials_from_session_csv(p)
            if got:
                print(f"  loaded {len(got):>8} best-candidate rows from {p}")
            rows.extend(got)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Normalize dtypes.
    for c in ["budget", "seed", "record_index", "q"]:
        if c in df.columns:
            df[c] = to_numeric_float(df[c])

    for c in ["cost"]:
        if c in df.columns:
            df[c] = to_numeric_float(df[c])

    df["escaped"] = df["escaped"].astype(bool)

    return df


# ======================================================================================
# Analysis
# ======================================================================================

CORE_PARAMS = [
    "spatial_amp_px",
    "frequency",
    "duration_scale",
    "gap_scale",
    "jitter_frac",
    "heterogeneity",
    "use_spatial",
    "use_temporal",
    "op_prob",
    "op_strength",
    "z_spatial",
    "z_frequency",
    "z_duration",
    "z_gap",
    "z_jitter",
    "z_heterogeneity",
    "z_use_spatial",
    "z_use_temporal",
]


def numeric_param_columns(df: pd.DataFrame):
    cols = []

    for c in CORE_PARAMS:
        if c in df.columns:
            cols.append(c)

    # Add other numeric candidate columns if available.
    skip = {
        "line_no",
        "budget",
        "seed",
        "record_index",
        "q",
        "cost",
        "escaped",
    }
    for c in df.columns:
        if c in skip or c in cols:
            continue
        if c.startswith("_"):
            continue
        if c in {"source_file", "method", "session_id", "participant", "family", "op_mode"}:
            continue

        s = to_numeric_float(df[c])
        if s.notna().sum() >= max(10, int(0.05 * len(df))):
            cols.append(c)

    # Deduplicate preserving order.
    out = []
    for c in cols:
        if c not in out:
            out.append(c)

    return out


def add_param_bins(df: pd.DataFrame, params: list[str]):
    out = df.copy()

    for p in params:
        s = to_numeric_float(out[p])
        if s.notna().sum() < 10:
            continue

        try:
            out[f"{p}_bin"] = pd.qcut(s, q=5, duplicates="drop")
        except Exception:
            try:
                out[f"{p}_bin"] = pd.cut(s, bins=5, duplicates="drop")
            except Exception:
                pass

    return out


def summarize_params(df: pd.DataFrame, params: list[str]):
    rows = []

    group_cols = ["method", "budget"]

    for keys, g in df.groupby(group_cols, dropna=False):
        method, budget = keys

        esc = g[g["escaped"] == True]
        fail = g[g["escaped"] == False]

        for p in params:
            x_esc = to_numeric_float(esc[p]).dropna() if p in esc.columns else pd.Series(dtype=float)
            x_fail = to_numeric_float(fail[p]).dropna() if p in fail.columns else pd.Series(dtype=float)
            x_all = to_numeric_float(g[p]).dropna() if p in g.columns else pd.Series(dtype=float)

            if len(x_all) < 10 or len(x_esc) < 2 or len(x_fail) < 2:
                continue

            esc_median = float(x_esc.median())
            fail_median = float(x_fail.median())
            esc_mean = float(x_esc.mean())
            fail_mean = float(x_fail.mean())
            pooled_std = float(x_all.std()) if float(x_all.std()) > 1e-12 else np.nan

            rows.append({
                "method": method,
                "budget": int(budget) if pd.notna(budget) else None,
                "parameter": p,
                "n_escape": len(x_esc),
                "n_fail": len(x_fail),
                "escape_median": esc_median,
                "fail_median": fail_median,
                "median_delta_escape_minus_fail": esc_median - fail_median,
                "escape_mean": esc_mean,
                "fail_mean": fail_mean,
                "mean_delta_escape_minus_fail": esc_mean - fail_mean,
                "standardized_mean_delta": (
                    (esc_mean - fail_mean) / pooled_std
                    if pooled_std == pooled_std
                    else None
                ),
                "escape_q10": float(x_esc.quantile(0.10)),
                "escape_q25": float(x_esc.quantile(0.25)),
                "escape_q75": float(x_esc.quantile(0.75)),
                "escape_q90": float(x_esc.quantile(0.90)),
            })

    return pd.DataFrame(rows)


def summarize_arms(df: pd.DataFrame):
    rows = []

    categorical = [c for c in ["family", "op_mode"] if c in df.columns]

    for keys, g in df.groupby(["method", "budget"], dropna=False):
        method, budget = keys
        n_total = len(g)
        n_escape_total = int(g["escaped"].sum())

        for c in categorical:
            for val, h in g.groupby(c, dropna=False):
                n = len(h)
                esc = int(h["escaped"].sum())
                rows.append({
                    "method": method,
                    "budget": int(budget) if pd.notna(budget) else None,
                    "arm_type": c,
                    "arm_value": str(val),
                    "n_trials": n,
                    "escaped": esc,
                    "escape_rate_%": round(esc / n * 100, 2) if n else None,
                    "share_of_all_trials_%": round(n / n_total * 100, 2) if n_total else None,
                    "share_of_all_escapes_%": round(esc / n_escape_total * 100, 2) if n_escape_total else None,
                    "median_cost_if_escape": (
                        float(h[h["escaped"] == True]["cost"].median())
                        if "cost" in h.columns and esc > 0
                        else None
                    ),
                })

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    return out.sort_values(["method", "budget", "arm_type", "escape_rate_%"], ascending=[True, True, True, False])


def bin_effects(df: pd.DataFrame, params: list[str]):
    rows = []
    d = add_param_bins(df, params)

    for keys, g in d.groupby(["method", "budget"], dropna=False):
        method, budget = keys

        for p in params:
            bcol = f"{p}_bin"
            if bcol not in g.columns:
                continue

            for b, h in g.groupby(bcol, dropna=False):
                n = len(h)
                esc = int(h["escaped"].sum())
                if n < 5:
                    continue

                rows.append({
                    "method": method,
                    "budget": int(budget) if pd.notna(budget) else None,
                    "parameter": p,
                    "bin": str(b),
                    "n_trials": n,
                    "escaped": esc,
                    "escape_rate_%": round(esc / n * 100, 2),
                    "median_cost_if_escape": (
                        float(h[h["escaped"] == True]["cost"].median())
                        if "cost" in h.columns and esc > 0
                        else None
                    ),
                })

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    return out.sort_values(["method", "budget", "parameter", "escape_rate_%"], ascending=[True, True, True, False])


def session_level_escape_summary(df: pd.DataFrame):
    rows = []

    keys = ["method", "budget", "seed", "record_index", "session_id", "participant"]

    for key_vals, g in df.groupby(keys, dropna=False):
        d = dict(zip(keys, key_vals))
        esc = g[g["escaped"] == True]

        row = {
            **d,
            "n_queries_observed": len(g),
            "escaped": len(esc) > 0,
            "q_first_escape": float(esc["q"].min()) if len(esc) and "q" in esc.columns else None,
            "best_escape_cost": float(esc["cost"].min()) if len(esc) and "cost" in esc.columns else None,
            "n_escaped_candidates": int(len(esc)),
        }

        if len(esc):
            best = esc.sort_values("cost", na_position="last").iloc[0]
            for p in CORE_PARAMS + ["family", "op_mode"]:
                if p in best.index:
                    row[f"best_{p}"] = best[p]

        rows.append(row)

    out = pd.DataFrame(rows)
    return out.sort_values(["method", "budget", "seed", "record_index"])


def cluster_escaped(df: pd.DataFrame, params: list[str]):
    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
    except Exception as e:
        print("skipping clustering because sklearn is unavailable:", repr(e))
        return pd.DataFrame()

    rows = []

    for keys, g in df.groupby(["method", "budget"], dropna=False):
        method, budget = keys
        esc = g[g["escaped"] == True].copy()

        usable = []
        for p in params:
            if p in esc.columns:
                s = to_numeric_float(esc[p])
                if s.notna().sum() >= 10 and s.nunique(dropna=True) > 1:
                    usable.append(p)

        if len(esc) < 10 or len(usable) < 2:
            continue

        X = esc[usable].apply(to_numeric_float)
        X = X.fillna(X.median(numeric_only=True))

        scaler = StandardScaler()
        Xz = scaler.fit_transform(X)

        max_k = min(5, len(esc) // 5)
        if max_k < 2:
            continue

        # Use a simple k selection by inertia elbow proxy.
        best_k = min(3, max_k)
        km = KMeans(n_clusters=best_k, random_state=0, n_init=20)
        labels = km.fit_predict(Xz)

        esc = esc.copy()
        esc["cluster"] = labels

        for cl, h in esc.groupby("cluster"):
            row = {
                "method": method,
                "budget": int(budget) if pd.notna(budget) else None,
                "cluster": int(cl),
                "n_escaped_candidates": len(h),
                "median_cost": float(h["cost"].median()) if "cost" in h.columns else None,
                "median_q": float(h["q"].median()) if "q" in h.columns else None,
            }

            for p in usable:
                row[f"median_{p}"] = float(to_numeric_float(h[p]).median())

            if "family" in h.columns:
                row["top_family"] = str(h["family"].mode().iloc[0]) if len(h["family"].dropna()) else ""
            if "op_mode" in h.columns:
                row["top_op_mode"] = str(h["op_mode"].mode().iloc[0]) if len(h["op_mode"].dropna()) else ""

            rows.append(row)

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    return out.sort_values(["method", "budget", "n_escaped_candidates"], ascending=[True, True, False])


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--input-dirs",
        default=(
            "results/v2_formal_full,"
            "results/v2_matched_hybrid_B10_B30_B100_N100_seed20260917_20260919,"
            "results/v2_two_stage_param_B10_B30_B100_N100_3seeds_merged,"
            "results/v2_two_stage_param_B10_B30_N100_seed20260917,"
            "results/v2_two_stage_param_B10_B30_N100_seed20260918_20260919,"
            "results/v2_two_stage_param_B100_N100_seed20260917,"
            "results/v2_two_stage_param_B100_N100_seed20260918_20260919"
        ),
    )
    ap.add_argument(
        "--output-dir",
        default="results/escape_basin_analysis",
    )

    args = ap.parse_args()

    input_dirs = [x.strip() for x in args.input_dirs.split(",") if x.strip()]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("ESCAPE BASIN PARAMETER ANALYSIS")
    print("=" * 100)
    print("input dirs:")
    for d in input_dirs:
        print(" -", d)
    print("output:", out_dir)

    df = load_all_trials(input_dirs)

    if df.empty:
        raise SystemExit(
            "No candidate-level trials found. Make sure attack_history.jsonl exists, "
            "or use runs with --save-history full."
        )

    # Drop exact duplicate trial rows caused by scanning merged + original directories.
    dedup_cols = [
        c for c in [
            "source_file",
            "line_no",
            "method",
            "budget",
            "seed",
            "record_index",
            "q",
            "escaped",
            "cost",
        ]
        if c in df.columns
    ]
    df = df.drop_duplicates(subset=dedup_cols).reset_index(drop=True)

    params = numeric_param_columns(df)

    print("\n" + "=" * 100)
    print("Loaded trials")
    print("=" * 100)
    print("n trials:", len(df))
    print("methods:", sorted(map(str, df["method"].dropna().unique())))
    print("budgets:", sorted(to_numeric_float(df["budget"]).dropna().astype(int).unique()))
    print("numeric params:", params)

    df.to_csv(out_dir / "candidate_trials_flat.csv", index=False)

    overview = (
        df.groupby(["method", "budget"], dropna=False)
          .agg(
              n_trials=("escaped", "size"),
              escaped_candidates=("escaped", "sum"),
              candidate_escape_rate_pct=("escaped", lambda x: round(float(np.mean(x)) * 100, 2)),
              median_cost_all=("cost", "median"),
              median_cost_escape=("cost", lambda x: np.nan),
          )
          .reset_index()
    )

    # Fix median_cost_escape separately.
    med_rows = []
    for keys, g in df.groupby(["method", "budget"], dropna=False):
        method, budget = keys
        esc = g[g["escaped"] == True]
        med_rows.append({
            "method": method,
            "budget": budget,
            "median_cost_escape": float(esc["cost"].median()) if len(esc) and "cost" in esc.columns else None,
            "median_q_escape": float(esc["q"].median()) if len(esc) and "q" in esc.columns else None,
        })
    med_df = pd.DataFrame(med_rows)
    overview = overview.drop(columns=["median_cost_escape"], errors="ignore").merge(
        med_df,
        on=["method", "budget"],
        how="left",
    )

    overview.to_csv(out_dir / "trial_level_overview.csv", index=False)

    param_summary = summarize_params(df, params)
    param_summary.to_csv(out_dir / "escaped_vs_failed_parameter_summary.csv", index=False)

    arm_summary = summarize_arms(df)
    arm_summary.to_csv(out_dir / "family_and_operation_escape_rates.csv", index=False)

    bins = bin_effects(df, params)
    bins.to_csv(out_dir / "parameter_bin_escape_rates.csv", index=False)

    session_summary = session_level_escape_summary(df)
    session_summary.to_csv(out_dir / "session_level_escape_summary.csv", index=False)

    clusters = cluster_escaped(df, params)
    clusters.to_csv(out_dir / "escaped_candidate_clusters.csv", index=False)

    print("\n" + "=" * 100)
    print("Trial-level overview")
    print("=" * 100)
    print(overview.to_string(index=False))

    if not param_summary.empty:
        print("\n" + "=" * 100)
        print("Top parameter differences: escaped vs failed")
        print("=" * 100)
        top = param_summary.copy()
        top["abs_standardized_mean_delta"] = top["standardized_mean_delta"].abs()
        top = top.sort_values("abs_standardized_mean_delta", ascending=False)
        print(
            top[
                [
                    "method",
                    "budget",
                    "parameter",
                    "n_escape",
                    "n_fail",
                    "escape_median",
                    "fail_median",
                    "median_delta_escape_minus_fail",
                    "standardized_mean_delta",
                ]
            ].head(40).to_string(index=False)
        )

    if not arm_summary.empty:
        print("\n" + "=" * 100)
        print("Family / operation arms")
        print("=" * 100)
        print(arm_summary.head(60).to_string(index=False))

    if not bins.empty:
        print("\n" + "=" * 100)
        print("Highest escape-rate parameter bins")
        print("=" * 100)
        print(
            bins.sort_values("escape_rate_%", ascending=False)
                .head(60)
                .to_string(index=False)
        )

    if not clusters.empty:
        print("\n" + "=" * 100)
        print("Escaped candidate clusters / candidate basins")
        print("=" * 100)
        print(clusters.to_string(index=False))

    print("\n" + "=" * 100)
    print("Saved")
    print("=" * 100)
    for name in [
        "candidate_trials_flat.csv",
        "trial_level_overview.csv",
        "escaped_vs_failed_parameter_summary.csv",
        "family_and_operation_escape_rates.csv",
        "parameter_bin_escape_rates.csv",
        "session_level_escape_summary.csv",
        "escaped_candidate_clusters.csv",
    ]:
        print(out_dir / name)


if __name__ == "__main__":
    main()
