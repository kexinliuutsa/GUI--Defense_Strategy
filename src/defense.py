#!/usr/bin/env python3
"""
HARDENED WRAPPER FOR THE FROZEN V1 + V2 + V3 DEFENSE

This file preserves the decision semantics of:
    defense_version = "frozen-v1v2v3-action-interval-v1"

Final verdict:
    V1_behavioral OR V2_temporal_crossfit OR V3_action_interval

Important invariants
--------------------
1. V1 is reconstructed from the already-frozen V1 initialization prefix.
2. V2 is loaded from the already-frozen nested generator-disjoint bundle.
3. V2 policy keys identify the held-out generator fold; each selected model
   was trained WITHOUT that generator.
4. V3 uses the exact frozen task-cluster thresholds and exact floating-point
   session threshold 0.6666666666666667.
5. No attack result is used to fit/recalibrate any component at inference.
6. Attacker-facing oracle returns ONLY a bool and carries no diagnostic state.
7. This file intentionally DOES NOT change historical corner-case semantics:
      - V3 with zero valid non-negative intervals => rate 0.0 => no V3 detect.
      - the exact floating threshold is preserved, including tie behavior.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import joblib
import numpy as np
import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

AHB_ROOT = Path(
    os.environ.get(
        "AHB_ROOT",
        str(
            Path.home()
            / "Desktop"
            / "Passing-the-Turing-Test-on-Screen-Agent-Humanization-Benchmark-main"
        ),
    )
).expanduser().resolve()

if not AHB_ROOT.exists():
    raise FileNotFoundError(
        f"AHB repo not found: {AHB_ROOT}\n"
        "Set AHB_ROOT explicitly."
    )

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(AHB_ROOT))

from src.heads.temporal_shape import timing_shape  # noqa: E402
from analysis.lib.feature_library import startT_us, endT_us  # noqa: E402


V1_SRC = ROOT / "evaluation" / "run_blackbox_attack.py"
V2_BUNDLE = ROOT / "results" / "defense_v2_temporal_policy.joblib"
CLUSTER_LOOKUP_CSV = (
    ROOT
    / "results"
    / "ahb_cheap_layer"
    / "session_predictions.csv"
)

EXPECTED_V2_VERSION = "defense-v2-temporal-nested-v4-frozen"
DEFENSE_VERSION = "frozen-v1v2v3-action-interval-v1"

# Historical frozen V3 semantics. DO NOT numerically "simplify" this value.
V3_INTERVAL_RATE_THRESHOLD = 0.6666666666666667

# Exact AHB task-cluster interval thresholds used by the frozen defense.
# Units: microseconds.
V3_INTERVAL_THRESHOLDS_US = {
    0: 6311961.0,
    1: 6008676.0,
    2: 5927296.0,
    3: 5957152.0,
    4: 5670071.0,
}


# =============================================================================
# SMALL HELPERS
# =============================================================================

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rec_get(record, key, default=None):
    if isinstance(record, dict):
        return record.get(key, default)
    return getattr(record, key, default)


def participant(record) -> str:
    return str(rec_get(record, "participant", "UNKNOWN"))


def session_id(record) -> str:
    return str(rec_get(record, "session_id", "UNKNOWN"))


def valid_gestures(record) -> list:
    out = []
    for gesture in rec_get(record, "gestures", []):
        if gesture is None:
            continue
        try:
            if (
                len(gesture) > 0
                and hasattr(gesture[0], "timestamp_us")
            ):
                out.append(gesture)
        except TypeError:
            pass
    return out


def gesture_matrix(record) -> np.ndarray:
    gestures = valid_gestures(record)
    if not gestures:
        return np.zeros((0, 50), dtype=float)

    rows = []
    for gesture in gestures:
        x = np.asarray(timing_shape(gesture), dtype=float).reshape(-1)
        if x.shape != (50,):
            raise RuntimeError(
                f"timing_shape shape={x.shape}, expected (50,)"
            )
        rows.append(x)

    return np.vstack(rows)


def mean_session_margin(model, record) -> float:
    X = gesture_matrix(record)
    if len(X) == 0:
        return float("-inf")
    margins = model.decision_function(X)
    return float(np.mean(margins))


# =============================================================================
# FROZEN V1 LOADER
# =============================================================================

def load_frozen_v1_prefix() -> dict:
    """
    Reconstruct only the already-frozen V1 initialization/scoring environment.

    The adaptive attack loop in evaluation/run_blackbox_attack.py is NEVER run.
    """
    if not V1_SRC.exists():
        raise FileNotFoundError(V1_SRC)

    text = V1_SRC.read_text(encoding="utf-8")

    marker = re.search(
        r"(?m)^for\s+attack_number\s*,\s*record_index\s+in\s+enumerate\(\s*$",
        text,
    )
    if marker is None:
        raise RuntimeError(
            "Could not locate frozen V1 attack-loop marker."
        )

    ns = {
        "__name__": "__main__",
        "__file__": str(V1_SRC),
    }

    capture = io.StringIO()
    old_cwd = Path.cwd()

    try:
        os.chdir(ROOT)
        with contextlib.redirect_stdout(capture):
            exec(
                compile(
                    text[: marker.start()],
                    str(V1_SRC),
                    "exec",
                ),
                ns,
                ns,
            )
    finally:
        os.chdir(old_cwd)

    required = (
        "score_condition",
        "get_gestures",
        "replace_gestures",
    )
    missing = [name for name in required if ns.get(name) is None]
    if missing:
        raise RuntimeError(
            f"Frozen V1 prefix missing required symbols: {missing}"
        )

    return ns


# =============================================================================
# ATTACKER-FACING ORACLE HANDLE
# =============================================================================

@dataclass
class OracleMonitor:
    """
    Runner-owned metadata.

    IMPORTANT:
    Do not pass this object to attack_session().
    """
    queries: int = 0


class LabelOnlyOracle:
    """
    Minimal attacker-facing callable.

    It exposes only __call__(candidate_gestures) -> bool.
    No per-head scores, thresholds, task-cluster id, or last_diag are stored
    on this object.

    This is an API-level black-box boundary for an audited in-process attack.
    It is not intended as a hostile-code sandbox against Python introspection.
    """

    __slots__ = ("_call",)

    def __init__(self, call: Callable[[list], bool]):
        object.__setattr__(self, "_call", call)

    def __call__(self, candidate_gestures) -> bool:
        return bool(self._call(candidate_gestures))

    def __repr__(self) -> str:
        return "<LabelOnlyOracle bool(candidate_gestures)>"


# =============================================================================
# FINAL FROZEN DEFENSE
# =============================================================================

class FrozenV1V2V3Defense:
    def __init__(self):
        # ---------------------------------------------------------------------
        # Artifact existence
        # ---------------------------------------------------------------------
        for path, label in (
            (V1_SRC, "V1 source"),
            (V2_BUNDLE, "V2 bundle"),
            (CLUSTER_LOOKUP_CSV, "V3 task-cluster lookup"),
        ):
            if not path.exists():
                raise FileNotFoundError(f"Missing {label}: {path}")

        # ---------------------------------------------------------------------
        # V2 bundle
        # ---------------------------------------------------------------------
        self.v2_bundle = joblib.load(V2_BUNDLE)

        version = self.v2_bundle.get("policy_version")
        if version != EXPECTED_V2_VERSION:
            raise RuntimeError(
                f"Unexpected frozen V2 version: {version!r}; "
                f"expected {EXPECTED_V2_VERSION!r}"
            )

        if self.v2_bundle.get(
            "threshold_selection_uses_attack_condition",
            True,
        ):
            raise RuntimeError(
                "Frozen V2 provenance says attack data selected threshold."
            )

        policies = self.v2_bundle.get("policies")
        if not isinstance(policies, dict) or not policies:
            raise RuntimeError("Frozen V2 bundle has no policies.")

        self.policies = policies
        self._validate_v2_policies()

        # ---------------------------------------------------------------------
        # V1 source regression
        # ---------------------------------------------------------------------
        current_v1_hash = sha256_file(V1_SRC)
        frozen_v1_hash = self.v2_bundle.get("v1_source_sha256")

        if (
            frozen_v1_hash is not None
            and current_v1_hash != frozen_v1_hash
        ):
            raise RuntimeError(
                "FROZEN V1 SOURCE HASH CHANGED.\n"
                f"bundle:  {frozen_v1_hash}\n"
                f"current: {current_v1_hash}"
            )

        self.v1_source_sha256 = current_v1_hash
        self.v2_bundle_sha256 = sha256_file(V2_BUNDLE)
        self.cluster_lookup_sha256 = sha256_file(CLUSTER_LOOKUP_CSV)

        # ---------------------------------------------------------------------
        # Frozen V1 scorer + record adapter
        # ---------------------------------------------------------------------
        self.ns = load_frozen_v1_prefix()
        self.inner = self.ns.get("namespace", {})

        self.score_condition = self.ns["score_condition"]
        self.get_gestures = self.ns["get_gestures"]
        self.replace_gestures = self.ns["replace_gestures"]

        # ---------------------------------------------------------------------
        # V3 task-cluster lookup
        # ---------------------------------------------------------------------
        self.cluster_lookup = self._load_cluster_lookup()

    # =========================================================================
    # V2 VALIDATION
    # =========================================================================

    def _validate_v2_policies(self) -> None:
        """
        Validate the semantics of the frozen cross-fitted V2 bundle.

        Policy key == held-out generator identity.
        The corresponding model was fitted without that held-out generator.
        """
        for key, policy in self.policies.items():
            if not isinstance(policy, dict):
                raise RuntimeError(
                    f"V2 policy {key!r} is not a dict."
                )

            heldout = str(policy.get("heldout_agent", ""))
            if heldout != str(key):
                raise RuntimeError(
                    "V2 cross-fit policy mismatch: "
                    f"key={key!r}, heldout_agent={heldout!r}"
                )

            if policy.get("model") is None:
                raise RuntimeError(
                    f"V2 policy {key!r} has no model."
                )

            threshold = float(policy.get("threshold", np.nan))
            if not np.isfinite(threshold):
                raise RuntimeError(
                    f"V2 policy {key!r} has invalid threshold: {threshold}"
                )

            fold = int(policy.get("fold", -1))
            if fold <= 0:
                raise RuntimeError(
                    f"V2 policy {key!r} has invalid fold: {fold}"
                )

    # =========================================================================
    # V3 TASK CLUSTER
    # =========================================================================

    def _load_cluster_lookup(self) -> dict:
        df = pd.read_csv(CLUSTER_LOOKUP_CSV, low_memory=False)

        required = {
            "participant",
            "session_id",
            "task_cluster",
        }
        missing = required - set(df.columns)
        if missing:
            raise RuntimeError(
                f"Cluster lookup missing columns: {sorted(missing)}"
            )

        lookup = {}

        rows = (
            df[
                [
                    "participant",
                    "session_id",
                    "task_cluster",
                ]
            ]
            .drop_duplicates()
            .itertuples(index=False)
        )

        for r in rows:
            key = (str(r.participant), str(r.session_id))
            cid = int(r.task_cluster)

            if key in lookup and lookup[key] != cid:
                raise RuntimeError(
                    f"Conflicting task cluster for {key}: "
                    f"{lookup[key]} vs {cid}"
                )

            if cid not in V3_INTERVAL_THRESHOLDS_US:
                raise RuntimeError(
                    f"Unexpected task cluster {cid} for {key}"
                )

            lookup[key] = cid

        if not lookup:
            raise RuntimeError("Frozen task-cluster lookup is empty.")

        return lookup

    def resolve_task_cluster(
        self,
        record,
        task_cluster: Optional[int] = None,
    ) -> int:
        if task_cluster is not None:
            cid = int(task_cluster)
        else:
            key = (
                participant(record),
                session_id(record),
            )
            if key not in self.cluster_lookup:
                raise RuntimeError(
                    "No frozen task-cluster mapping for "
                    f"{key}. Pass task_cluster explicitly only when it is "
                    "runner-owned frozen metadata."
                )
            cid = int(self.cluster_lookup[key])

        if cid not in V3_INTERVAL_THRESHOLDS_US:
            raise RuntimeError(f"Unexpected task cluster: {cid}")

        return cid

    # =========================================================================
    # V3 ACTION INTERVAL
    # =========================================================================

    def interval_violation_rate(
        self,
        record,
        task_cluster: Optional[int] = None,
    ) -> dict:
        cid = self.resolve_task_cluster(record, task_cluster)
        threshold_us = float(V3_INTERVAL_THRESHOLDS_US[cid])

        gestures = valid_gestures(record)

        total = 0
        violations = 0
        values = []
        negative_intervals_skipped = 0

        prev_end = None

        for gesture in gestures:
            s = startT_us(gesture)
            e = endT_us(gesture)

            if s is None or e is None:
                continue

            s = float(s)
            e = float(e)

            if prev_end is not None:
                dt = s - prev_end

                # Historical frozen semantics:
                # negative intervals are ignored.
                if dt >= 0:
                    total += 1
                    values.append(float(dt))
                    if dt >= threshold_us:
                        violations += 1
                else:
                    negative_intervals_skipped += 1

            prev_end = e

        # Historical frozen semantics:
        # no valid non-negative intervals => rate 0.0 => V3 does not fire.
        rate = violations / total if total > 0 else 0.0

        detect = bool(
            rate >= V3_INTERVAL_RATE_THRESHOLD
        )

        return {
            "task_cluster": cid,
            "interval_threshold_us": threshold_us,
            "interval_total": int(total),
            "interval_violations": int(violations),
            "interval_violation_rate": float(rate),
            "negative_intervals_skipped": int(
                negative_intervals_skipped
            ),
            "no_valid_intervals": bool(total == 0),
            "v3_threshold": V3_INTERVAL_RATE_THRESHOLD,
            "v3_detect": detect,
        }

    # =========================================================================
    # OFFLINE / RUNNER-OWNED DIAGNOSTICS
    # =========================================================================

    def score_record(
        self,
        record,
        *,
        task_cluster: Optional[int] = None,
        v1_condition: str = "Long Tap Author",
    ) -> dict:
        """
        Full runner-owned diagnostic scoring.

        Do NOT pass the returned dict to the attacker.

        v1_condition is a logging/dataframe label in the frozen V1 path; the
        audited V1 feature/model/threshold selection does not branch on it.
        """
        pid = participant(record)

        if pid not in self.policies:
            raise RuntimeError(
                "No frozen V2 cross-fit policy for participant/generator "
                f"{pid!r}. This frozen object is an agent-side cross-fitted "
                "benchmark evaluator, not a single universal deployment model."
            )

        # ---------------------------------------------------------------------
        # V1
        # ---------------------------------------------------------------------
        _, detail = self.score_condition(
            v1_condition,
            [record],
            policy="released_executor_rule",
        )

        if len(detail) != 1:
            raise RuntimeError(
                "V1 expected exactly one session result."
            )

        v1_detect = bool(
            detail["final_detect"].iloc[0]
        )

        # ---------------------------------------------------------------------
        # V2 cross-fitted temporal head
        # ---------------------------------------------------------------------
        policy = self.policies[pid]

        if str(policy.get("heldout_agent")) != pid:
            raise RuntimeError(
                "Selected V2 policy is not the correct held-out-generator "
                f"fold for {pid!r}."
            )

        temporal_model = policy["model"]
        temporal_threshold = float(policy["threshold"])

        temporal_margin = mean_session_margin(
            temporal_model,
            record,
        )

        v2_temporal_detect = bool(
            temporal_margin >= temporal_threshold
        )

        v1v2_detect = bool(
            v1_detect or v2_temporal_detect
        )

        # ---------------------------------------------------------------------
        # V3
        # ---------------------------------------------------------------------
        v3 = self.interval_violation_rate(
            record,
            task_cluster=task_cluster,
        )

        # ---------------------------------------------------------------------
        # FINAL
        # ---------------------------------------------------------------------
        final_detect = bool(
            v1v2_detect or v3["v3_detect"]
        )

        return {
            "participant": pid,
            "session_id": session_id(record),

            "v1_detect": v1_detect,

            "v2_fold": int(policy["fold"]),
            "v2_heldout_agent": str(policy["heldout_agent"]),
            "v2_temporal_margin": float(temporal_margin),
            "v2_temporal_threshold": temporal_threshold,
            "v2_temporal_detect": v2_temporal_detect,

            "v1v2_detect": v1v2_detect,

            **v3,

            "v3_rescue": bool(
                v3["v3_detect"] and not v1v2_detect
            ),
            "final_detect": final_detect,
        }

    def detect_record(
        self,
        record,
        *,
        task_cluster: Optional[int] = None,
        v1_condition: str = "Long Tap Author",
    ) -> bool:
        """
        Runner-side single-record boolean decision.
        """
        return bool(
            self.score_record(
                record,
                task_cluster=task_cluster,
                v1_condition=v1_condition,
            )["final_detect"]
        )

    # =========================================================================
    # ATTACKER-FACING LABEL-ONLY ORACLE
    # =========================================================================

    def make_label_only_oracle(
        self,
        original_record,
        *,
        task_cluster: Optional[int] = None,
        v1_condition: str = "Long Tap Author",
    ) -> tuple[LabelOnlyOracle, OracleMonitor]:
        """
        Build a label-only oracle and a separate runner-owned query monitor.

        Correct usage:
            oracle, monitor = defense.make_label_only_oracle(record)
            result = attack_session(..., oracle=oracle, ...)
            print(monitor.queries)  # runner only

        DO NOT pass monitor to attack_session().
        """
        cid = self.resolve_task_cluster(
            original_record,
            task_cluster,
        )

        monitor = OracleMonitor()

        def _query(candidate_gestures) -> bool:
            monitor.queries += 1

            candidate_record = self.replace_gestures(
                original_record,
                candidate_gestures,
            )

            # No diag is stored on the attacker-facing oracle.
            return bool(
                self.detect_record(
                    candidate_record,
                    task_cluster=cid,
                    v1_condition=v1_condition,
                )
            )

        return LabelOnlyOracle(_query), monitor

    # Backward-compatible name, but returns ONLY the oracle.
    # Prefer make_label_only_oracle() in new runner code so query accounting is
    # kept outside attacker reach.
    def make_oracle(
        self,
        original_record,
        *,
        task_cluster: Optional[int] = None,
        v1_condition: str = "Long Tap Author",
    ) -> LabelOnlyOracle:
        oracle, _monitor = self.make_label_only_oracle(
            original_record,
            task_cluster=task_cluster,
            v1_condition=v1_condition,
        )
        return oracle

    # =========================================================================
    # MANIFEST / INTEGRITY
    # =========================================================================

    def manifest(self) -> dict:
        return {
            "defense_version": DEFENSE_VERSION,
            "decision": "V1 OR V2_crossfit OR V3",

            "v1_source": str(V1_SRC),
            "v1_source_sha256": self.v1_source_sha256,

            "v2_bundle": str(V2_BUNDLE),
            "v2_bundle_sha256": self.v2_bundle_sha256,
            "v2_policy_version": self.v2_bundle.get(
                "policy_version"
            ),
            "v2_protocol": (
                "generator-disjoint cross-fitted temporal policy; "
                "policy key identifies the held-out generator fold"
            ),
            "v2_threshold_selection_uses_attack_condition": bool(
                self.v2_bundle.get(
                    "threshold_selection_uses_attack_condition",
                    True,
                )
            ),

            "cluster_lookup_csv": str(CLUSTER_LOOKUP_CSV),
            "cluster_lookup_sha256": self.cluster_lookup_sha256,

            "v3_feature": "AHB action-interval violation rate",
            "v3_session_threshold_exact_float": (
                V3_INTERVAL_RATE_THRESHOLD
            ),
            "v3_author_interval_thresholds_us": dict(
                V3_INTERVAL_THRESHOLDS_US
            ),
            "v3_zero_valid_interval_semantics": (
                "preserve-frozen: rate=0.0, v3_detect=False"
            ),
            "v3_negative_interval_semantics": (
                "preserve-frozen: negative intervals excluded"
            ),

            "oracle_contract": (
                "attacker-facing callable returns one bool only; "
                "diagnostics and query monitor are runner-owned"
            ),
        }

    def integrity_check(self) -> dict:
        """
        Static/runtime consistency checks only. This is not an evaluation.
        """
        issues = []

        if (
            self.v2_bundle.get("policy_version")
            != EXPECTED_V2_VERSION
        ):
            issues.append("V2 policy version mismatch")

        if self.v2_bundle.get(
            "threshold_selection_uses_attack_condition",
            True,
        ):
            issues.append(
                "V2 threshold provenance indicates attack-condition use"
            )

        if not self.cluster_lookup:
            issues.append("empty task-cluster lookup")

        for pid, policy in self.policies.items():
            if str(policy.get("heldout_agent")) != str(pid):
                issues.append(
                    f"V2 heldout policy mismatch for {pid!r}"
                )

        return {
            "ok": not issues,
            "issues": issues,
            "n_v2_crossfit_policies": len(self.policies),
            "n_cluster_mappings": len(self.cluster_lookup),
            "manifest": self.manifest(),
        }


# =============================================================================
# REGRESSION SELF-TEST
# =============================================================================

def self_test() -> None:
    """
    Preserve the historical frozen Long-Tap regression counts.

    This is a regression/integrity test for code packaging, not a new
    evaluation or a threshold-selection procedure.
    """
    defense = FrozenV1V2V3Defense()

    check = defense.integrity_check()
    if not check["ok"]:
        raise RuntimeError(
            f"Integrity check failed: {check['issues']}"
        )

    ns = defense.ns
    inner = defense.inner

    long_records = list(
        ns.get(
            "long_records",
            inner.get("long_records", []),
        )
    )

    if len(long_records) != 499:
        raise RuntimeError(
            f"Expected 499 Long-Tap records, got {len(long_records)}"
        )

    print("=" * 100)
    print("HARDENED FROZEN V1+V2+V3 REGRESSION SELF-TEST")
    print("=" * 100)
    print("Long-Tap sessions:", len(long_records))

    # Batch V1 for speed.
    _, detail = defense.score_condition(
        "Long Tap Author",
        long_records,
        policy="released_executor_rule",
    )

    if len(detail) != 499:
        raise RuntimeError("Unexpected V1 detail size.")

    v1_count = int(
        detail["final_detect"].astype(bool).sum()
    )

    v1v2_count = 0
    v3_rescue_count = 0
    final_count = 0

    for i, record in enumerate(long_records):
        pid = participant(record)
        policy = defense.policies[pid]

        temporal_margin = mean_session_margin(
            policy["model"],
            record,
        )
        temporal_detect = bool(
            temporal_margin >= float(policy["threshold"])
        )

        v1_detect = bool(
            detail["final_detect"].iloc[i]
        )
        v1v2 = bool(
            v1_detect or temporal_detect
        )

        v3 = defense.interval_violation_rate(
            record
        )["v3_detect"]

        final = bool(v1v2 or v3)

        v1v2_count += int(v1v2)
        v3_rescue_count += int(v3 and not v1v2)
        final_count += int(final)

    print("V1:", f"{v1_count}/499")
    print("V1+V2:", f"{v1v2_count}/499")
    print("V3 rescue:", v3_rescue_count)
    print(
        "FINAL:",
        f"{final_count}/499",
        f"= {final_count/499:.4%}",
    )

    # Exact historical regression values.
    assert v1_count == 382
    assert v1v2_count == 413
    assert v3_rescue_count == 72
    assert final_count == 485

    print("\nSELF TEST: PASS")


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--self-test",
        action="store_true",
        help="Run historical frozen regression counts.",
    )

    ap.add_argument(
        "--integrity",
        action="store_true",
        help="Run packaging/integrity checks only.",
    )

    ap.add_argument(
        "--manifest",
        action="store_true",
        help="Print hardened runtime manifest.",
    )

    args = ap.parse_args()

    defense = FrozenV1V2V3Defense()

    if args.self_test:
        self_test()
        return

    if args.integrity:
        print(
            json.dumps(
                defense.integrity_check(),
                indent=2,
                sort_keys=True,
            )
        )
        return

    if args.manifest:
        print(
            json.dumps(
                defense.manifest(),
                indent=2,
                sort_keys=True,
            )
        )
        return

    print("Loaded:", defense.manifest()["defense_version"])
    print("Use --integrity, --manifest, or --self-test.")


if __name__ == "__main__":
    main()
