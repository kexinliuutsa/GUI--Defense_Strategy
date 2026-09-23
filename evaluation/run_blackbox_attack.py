from __future__ import annotations

import copy

import os
import runpy
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

DEFENSE_ROOT = (
    Path.home()
    / "Desktop"
    / "GUI--Defense_Strategy"
)

AHB_ROOT = (
    Path.home()
    / "Desktop"
    / "Passing-the-Turing-Test-on-Screen-Agent-Humanization-Benchmark-main"
)

EXP49 = (
    AHB_ROOT
    / "experiments"
    / "49_joint_clean_calibration"
    / "run.py"
)

sys.path.insert(
    0,
    str(DEFENSE_ROOT),
)

sys.path.insert(
    0,
    str(AHB_ROOT),
)


from attacks.blackbox_adaptive_attack import (
    attack_session,
)


# ============================================================
# CONFIG
#
# First smoke test:
#   3 sessions
#   30 label-only queries each
#
# Do not increase yet.
# ============================================================

N_ATTACK = 3

QUERY_BUDGET = int(os.environ.get("BLACKBOX_BUDGET", "30"))

REFINE_QUERIES = 8

SEED = int(os.environ.get("BLACKBOX_SEED", "20260827"))


# ============================================================
# LOAD THE EXACT FROZEN EXP49 ENVIRONMENT
#
# We execute the frozen experiment rather than reimplementing
# its detector here.
#
# The attacker itself still sees only a single boolean.
# ============================================================

print(
    "=" * 100
)

print(
    "LOADING FROZEN EXP49 ENVIRONMENT"
)

print(
    "=" * 100
)

_previous_cwd = Path.cwd()

try:
    os.chdir(AHB_ROOT)

    namespace = runpy.run_path(
        str(EXP49)
    )

finally:
    os.chdir(_previous_cwd)


required_names = [
    "long_records",
    "score_condition",
]

for name in required_names:

    if name not in namespace:

        raise RuntimeError(
            f"EXP49 did not expose required object: {name}"
        )


long_records = namespace[
    "long_records"
]

score_condition = namespace[
    "score_condition"
]


print(
    "\nLoaded Long-Tap records:",
    len(long_records),
)


if len(long_records) != 499:

    raise RuntimeError(
        "Expected exactly 499 Long-Tap sessions, "
        f"got {len(long_records)}"
    )


# ============================================================
# RECORD / GESTURE HELPERS
#
# EXP49 records may be dict-like, tuple-like, or objects.
# We locate the gesture payload without assuming the exact
# internal record representation.
# ============================================================

GESTURE_FIELD_CANDIDATES = (
    "gestures",
    "gesture_list",
    "actions",
    "trajectory",
    "session",
    "events",
)


def looks_like_gesture(
    value,
):
    if not isinstance(
        value,
        list,
    ):
        return False

    if len(value) == 0:
        return False

    first = value[0]

    if not isinstance(
        first,
        list,
    ):
        return False

    # Empty gesture is allowed.
    if len(first) == 0:
        return True

    event = first[0]

    return (
        hasattr(
            event,
            "timestamp_us",
        )
        and
        hasattr(
            event,
            "x",
        )
        and
        hasattr(
            event,
            "y",
        )
    )


def get_gestures(
    record,
):
    # ------------------------------
    # Dict
    # ------------------------------

    if isinstance(
        record,
        dict,
    ):

        for key in (
            GESTURE_FIELD_CANDIDATES
        ):

            if (
                key in record
                and
                looks_like_gesture(
                    record[key]
                )
            ):

                return record[key]

        for key, value in (
            record.items()
        ):

            if looks_like_gesture(
                value
            ):

                return value

        raise RuntimeError(
            "Could not locate gestures in dict record. "
            f"Keys={list(record.keys())}"
        )

    # ------------------------------
    # Named tuple / object fields
    # ------------------------------

    for key in (
        GESTURE_FIELD_CANDIDATES
    ):

        if hasattr(
            record,
            key,
        ):

            value = getattr(
                record,
                key,
            )

            if looks_like_gesture(
                value
            ):

                return value

    # ------------------------------
    # Tuple/list record
    # ------------------------------

    if isinstance(
        record,
        (
            tuple,
            list,
        ),
    ):

        for value in record:

            if looks_like_gesture(
                value
            ):

                return value

    raise RuntimeError(
        "Could not locate gesture payload in record "
        f"type={type(record)} repr={repr(record)[:500]}"
    )


def replace_gestures(
    record,
    gestures,
):
    # ------------------------------
    # Dict
    # ------------------------------

    if isinstance(
        record,
        dict,
    ):

        result = copy.copy(
            record
        )

        for key in (
            GESTURE_FIELD_CANDIDATES
        ):

            if (
                key in result
                and
                looks_like_gesture(
                    result[key]
                )
            ):

                result[key] = gestures

                return result

        for key, value in list(
            result.items()
        ):

            if looks_like_gesture(
                value
            ):

                result[key] = gestures

                return result

        raise RuntimeError(
            "Could not replace gestures in dict record."
        )

    # ------------------------------
    # NamedTuple
    # ------------------------------

    if (
        isinstance(
            record,
            tuple,
        )
        and
        hasattr(
            record,
            "_fields",
        )
        and
        hasattr(
            record,
            "_replace",
        )
    ):

        for key in (
            record._fields
        ):

            value = getattr(
                record,
                key,
            )

            if looks_like_gesture(
                value
            ):

                return record._replace(
                    **{
                        key:
                            gestures
                    }
                )

    # ------------------------------
    # Plain tuple
    # ------------------------------

    if isinstance(
        record,
        tuple,
    ):

        values = list(
            record
        )

        for index, value in enumerate(
            values
        ):

            if looks_like_gesture(
                value
            ):

                values[index] = (
                    gestures
                )

                return tuple(
                    values
                )

    # ------------------------------
    # Mutable object
    # ------------------------------

    cloned = copy.copy(
        record
    )

    for key in (
        GESTURE_FIELD_CANDIDATES
    ):

        if hasattr(
            cloned,
            key,
        ):

            value = getattr(
                cloned,
                key,
            )

            if looks_like_gesture(
                value
            ):

                try:

                    setattr(
                        cloned,
                        key,
                        gestures,
                    )

                    return cloned

                except Exception:

                    pass

    raise RuntimeError(
        "Could not replace gesture payload in record "
        f"type={type(record)}"
    )


# ============================================================
# REPRODUCE FROZEN LONG-TAP RESULT
#
# Before attacking anything, verify that this runner really
# reaches the same frozen defense.
# ============================================================

print(
    "\n"
    + "=" * 100
)

print(
    "FROZEN LONG-TAP SANITY CHECK"
)

print(
    "=" * 100
)


baseline_row, baseline_detail = (
    score_condition(
        "Long Tap Author",
        long_records,
        policy="released_executor_rule",
    )
)


if "final_detect" not in (
    baseline_detail.columns
):

    raise RuntimeError(
        "score_condition detail does not contain final_detect."
    )


baseline_detection = float(
    baseline_detail[
        "final_detect"
    ].mean()
)


print(
    "sessions:",
    len(
        baseline_detail
    )
)

print(
    "frozen detection:",
    baseline_detection
)

print(
    "expected:",
    0.7655310621242485,
)


if not np.isclose(
    baseline_detection,
    0.7655310621242485,
    atol=1e-12,
):

    raise RuntimeError(
        "Frozen Long-Tap sanity check failed. "
        "Do not run the attack."
    )


print(
    "FROZEN SANITY: PASS"
)


# ============================================================
# ELIGIBLE SESSIONS
#
# Attack only Long-Tap sessions that the frozen defense
# currently detects.
# ============================================================

detected_mask = (
    baseline_detail[
        "final_detect"
    ]
    .astype(bool)
    .to_numpy()
)


eligible_indices = (
    np.flatnonzero(
        detected_mask
    )
)


print(
    "\nCurrently detected Long-Tap sessions:",
    len(
        eligible_indices
    ),
    "/",
    len(
        long_records
    ),
)


expected_detected = int(
    round(
        499
        * 0.7655310621242485
    )
)


print(
    "Expected detected count:",
    expected_detected
)


# ============================================================
# RANDOM SESSION SELECTION
# ============================================================

rng = np.random.default_rng(
    SEED
)


selected_indices = np.asarray(eligible_indices, dtype=int)


print(
    "\nSelected record indices:",
    selected_indices.tolist()
)


# ============================================================
# ATTACK
# ============================================================

rows = []


for attack_number, record_index in enumerate(
    selected_indices,
    start=1,
):

    record_index = int(
        record_index
    )

    original_record = (
        long_records[
            record_index
        ]
    )

    original_gestures = (
        get_gestures(
            original_record
        )
    )


    # ========================================================
    # LABEL-ONLY ORACLE
    #
    # This is the ONLY function given to the attacker.
    #
    # It returns:
    #
    #   True  -> detected as agent
    #   False -> escaped
    #
    # No component scores leave this boundary.
    # ========================================================

    query_counter = {
        "count": 0
    }


    def oracle(
        candidate_gestures,
    ):
        query_counter[
            "count"
        ] += 1

        candidate_record = (
            replace_gestures(
                original_record,
                candidate_gestures,
            )
        )

        _, detail = (
            score_condition(
                "Long Tap Author",
                [
                    candidate_record
                ],
                policy=(
                    "released_executor_rule"
                ),
            )
        )

        if len(detail) != 1:

            raise RuntimeError(
                "Oracle expected exactly one session result."
            )

        verdict = bool(
            detail[
                "final_detect"
            ].iloc[0]
        )

        return verdict


    # Starting-point validity is checked once inside attack_session().

    row_meta = (
        baseline_detail
        .iloc[
            record_index
        ]
    )


    participant = (
        row_meta[
            "participant"
        ]
        if
        "participant"
        in
        baseline_detail.columns
        else
        "unknown"
    )


    session_id = (
        row_meta[
            "session_id"
        ]
        if
        "session_id"
        in
        baseline_detail.columns
        else
        str(
            record_index
        )
    )


    print(
        "\n"
        + "=" * 100
    )

    print(
        f"ATTACK {attack_number}/{len(selected_indices)}"
    )

    print(
        "record index:",
        record_index
    )

    print(
        "participant:",
        participant
    )

    print(
        "session:",
        session_id
    )

    print(
        "actions:",
        len(
            original_gestures
        )
    )

    print(
        "budget:",
        QUERY_BUDGET
    )


    result, adversarial = (
        attack_session(
            original_gestures,
            oracle,
            budget=(
                QUERY_BUDGET
            ),
            refine_queries=(
                REFINE_QUERIES
            ),
            seed=(
                SEED
                + attack_number
            ),
            strategy="adaptive",
        )
    )


    print(
        "\nescaped:",
        result.escaped
    )

    print(
        "queries_to_first_escape:",
        result.queries_to_first_escape
    )

    print(
        "best_family:",
        result.best_family
    )

    print(
        "spatial_rms_px:",
        result.spatial_rms_px
    )

    print(
        "endpoint_shift_px:",
        result.endpoint_shift_px
    )

    print(
        "duration_cost:",
        result.duration_cost
    )

    print(
        "gap_cost:",
        result.gap_cost
    )

    print(
        "actual oracle calls including baseline check:",
        query_counter[
            "count"
        ]
    )


    attack_row = {
        "record_index":
            record_index,

        "participant":
            participant,

        "session_id":
            session_id,

        "n_actions":
            len(
                original_gestures
            ),

        "query_budget":
            QUERY_BUDGET,

        **result.to_dict(),
    }


    rows.append(
        attack_row
    )


# ============================================================
# RESULTS
# ============================================================

results = pd.DataFrame(
    rows
)


OUTPUT = (
    DEFENSE_ROOT
    / "results"
    / f"blackbox_adaptive_b{QUERY_BUDGET}_s{SEED}.csv"
)


OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)


results.to_csv(
    OUTPUT,
    index=False,
)


print(
    "\n"
    + "=" * 100
)

print(
    "BLACK-BOX ADAPTIVE ATTACK — POPULATION SUMMARY"
)

print(
    "=" * 100
)


print(
    results[
        [
            "participant",
            "session_id",
            "escaped",
            "queries_to_first_escape",
            "best_family",
            "spatial_rms_px",
            "endpoint_shift_px",
            "duration_cost",
            "gap_cost",
        ]
    ].to_string(
        index=False
    )
)


print(
    "\nAttack success:",
    int(
        results[
            "escaped"
        ].sum()
    ),
    "/",
    len(
        results
    ),
)


print(
    "ASR:",
    float(
        results[
            "escaped"
        ].mean()
    ),
)


successful = (
    results[
        results[
            "escaped"
        ]
    ]
)


if len(
    successful
):

    print(
        "Median queries to first escape:",
        float(
            successful[
                "queries_to_first_escape"
            ].median()
        ),
    )

    print(
        "Median spatial RMS:",
        float(
            successful[
                "spatial_rms_px"
            ].median()
        ),
    )

    print(
        "Maximum endpoint shift:",
        float(
            successful[
                "endpoint_shift_px"
            ].max()
        ),
    )


print(
    "\nSaved:",
    OUTPUT
)

print(
    "\nPOPULATION RUN COMPLETE"
)