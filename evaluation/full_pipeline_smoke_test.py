"""
End-to-end smoke test for the complete cross-scale pipeline.

This test does not attempt to reproduce paper metrics.

Its purpose is to verify that the independently validated modules can be
assembled into one working inference path:

    raw gestures
      -> session features
      -> source heads
      -> Human-reference score
      -> fake-action artifact
      -> frozen decision layer
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.features.session_features import (
    SESSION_FEATURES,
)

from src.pipeline import (
    CrossScalePipeline,
)


SEED = 20260826


@dataclass
class Event:
    x: float
    y: float
    timestamp_us: float


rng = np.random.default_rng(
    SEED
)


# ============================================================
# Synthetic clean feature data
# ============================================================

n_human = 100
n_agent = 100


human_array = rng.normal(
    0,
    1,
    size=(
        n_human,
        len(
            SESSION_FEATURES
        ),
    ),
)


agent_array = rng.normal(
    0,
    1,
    size=(
        n_agent,
        len(
            SESSION_FEATURES
        ),
    ),
)


# Give the synthetic agent class a modest source shift.
for feature in [
    "direction_resultant",
    "duration_displacement_corr",
    "log_gap_mean",
    "direction_similarity_mean",
]:

    idx = SESSION_FEATURES.index(
        feature
    )

    agent_array[
        :,
        idx
    ] += 1.0


clean_sessions = pd.DataFrame(
    np.vstack(
        [
            human_array,
            agent_array,
        ]
    ),
    columns=SESSION_FEATURES,
)


labels = np.concatenate(
    [
        np.zeros(
            n_human,
            dtype=int,
        ),
        np.ones(
            n_agent,
            dtype=int,
        ),
    ]
)


participants = np.asarray(
    [
        f"human_{i % 5}"
        for i in range(
            n_human
        )
    ]
    +
    [
        f"agent_{i % 5}"
        for i in range(
            n_agent
        )
    ]
)


# ============================================================
# Human raw actions for fake-artifact calibration
# ============================================================

def make_human_gesture(
    i,
):

    x0 = float(
        100
        +
        (
            i % 20
        )
        *
        20
    )

    y0 = float(
        200
        +
        (
            i % 15
        )
        *
        15
    )

    dx = float(
        40
        +
        (
            i % 7
        )
        *
        8
    )

    dy = float(
        20
        +
        (
            i % 5
        )
        *
        6
    )

    t0 = float(
        i
        *
        1_000_000
    )

    return [
        Event(
            x0,
            y0,
            t0,
        ),
        Event(
            x0 + dx / 2,
            y0 + dy / 2,
            t0 + 50_000,
        ),
        Event(
            x0 + dx,
            y0 + dy,
            t0 + 100_000,
        ),
    ]


human_raw_gestures = [
    make_human_gesture(
        i
    )
    for i in range(
        300
    )
]


# ============================================================
# Fit complete pipeline
# ============================================================

pipeline = CrossScalePipeline.fit(
    clean_sessions=
        clean_sessions,

    labels=
        labels,

    participants=
        participants,

    human_raw_gestures=
        human_raw_gestures,
)


# ============================================================
# Insufficient-context branch
# ============================================================

short_session = [
    make_human_gesture(
        1001
    ),
    make_human_gesture(
        1002
    ),
    make_human_gesture(
        1003
    ),
]


short_result = (
    pipeline.predict_session(
        short_session
    )
)


assert short_result.detected is None

assert (
    short_result.status
    ==
    "insufficient_context"
)

assert (
    short_result.href_consulted
    is False
)


# ============================================================
# Normal >=4-action inference path
# ============================================================

full_session = [
    make_human_gesture(
        2001
    ),
    make_human_gesture(
        2002
    ),
    make_human_gesture(
        2003
    ),
    make_human_gesture(
        2004
    ),
    make_human_gesture(
        2005
    ),
]


full_result = (
    pipeline.predict_session(
        full_session
    )
)


assert full_result.status == "ok"

assert isinstance(
    full_result.detected,
    bool,
)

assert isinstance(
    full_result.base_detected,
    bool,
)

assert isinstance(
    full_result.session_detected,
    bool,
)

assert isinstance(
    full_result.strict_detected,
    bool,
)

assert isinstance(
    full_result.fake_detected,
    bool,
)

assert np.isfinite(
    full_result.session_score
)

assert np.isfinite(
    full_result.strict_score
)

assert np.isfinite(
    full_result.human_reference_score
)


print(
    "=" * 72
)

print(
    "FULL PIPELINE SMOKE TEST"
)

print(
    "=" * 72
)


print(
    "Short-prefix status:",
    short_result.status,
)

print(
    "Short-prefix detected:",
    short_result.detected,
)


print()

print(
    "Full-session status:",
    full_result.status,
)

print(
    "Session score:",
    full_result.session_score,
)

print(
    "Strict score:",
    full_result.strict_score,
)

print(
    "Fake artifact:",
    full_result.fake_detected,
)

print(
    "Base detected:",
    full_result.base_detected,
)

print(
    "HREF consulted:",
    full_result.href_consulted,
)

print(
    "Human-reference score:",
    full_result.human_reference_score,
)

print(
    "Final detected:",
    full_result.detected,
)


print()

print(
    "PASS: validated defense components assemble into a working "
    "end-to-end inference pipeline."
)
