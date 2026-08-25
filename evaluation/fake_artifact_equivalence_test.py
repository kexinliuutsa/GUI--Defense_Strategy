"""
Equivalence test for the cleaned fake-action artifact head.

Compares the historical validated formulas against
src.heads.fake_artifact.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.heads.fake_artifact import (
    artifact_count,
    calibrate_fake_artifact,
    gesture_metrics,
)


SEED = 20260826
TARGET_FPR = 0.005


@dataclass
class Event:
    x: float
    y: float
    timestamp_us: float


# ---------------------------------------------------------------------
# Historical implementation
# ---------------------------------------------------------------------

def historical_metrics(g):

    if g is None or len(g) == 0:
        return None

    pts = np.asarray(
        [
            [float(e.x), float(e.y)]
            for e in g
        ],
        dtype=float,
    )

    ts = np.asarray(
        [
            float(e.timestamp_us)
            for e in g
        ],
        dtype=float,
    )

    start = pts[0]
    end = pts[-1]

    displacement = float(
        np.linalg.norm(
            end - start
        )
    )

    if len(pts) >= 2:

        seg = np.linalg.norm(
            np.diff(
                pts,
                axis=0,
            ),
            axis=1,
        )

        path_length = float(
            seg.sum()
        )

    else:

        path_length = 0.0


    ratio = (
        displacement
        /
        path_length
        if path_length > 1e-12
        else 1.0
    )


    duration = (
        float(
            ts[-1] - ts[0]
        )
        if len(ts) >= 2
        else 0.0
    )


    return {
        "n_points":
            len(g),

        "duration_us":
            duration,

        "displacement":
            displacement,

        "path_length":
            path_length,

        "endpoint_path_ratio":
            ratio,

        "exact_zero_displacement":
            displacement <= 1e-6,

        "near_zero_displacement":
            displacement <= 1.0,
    }


# ---------------------------------------------------------------------
# Synthetic gestures
# ---------------------------------------------------------------------

rng = np.random.default_rng(
    SEED
)


def make_gesture():

    n = int(
        rng.integers(
            1,
            15,
        )
    )

    start = rng.uniform(
        [0, 0],
        [1080, 2400],
    )

    mode = int(
        rng.integers(
            0,
            4,
        )
    )


    if mode == 0:
        # Stationary action.
        end = start.copy()

    elif mode == 1:
        # Closed-loop movement:
        # move away, then return near start.
        end = (
            start
            +
            rng.normal(
                0,
                0.2,
                size=2,
            )
        )

    else:
        end = (
            start
            +
            rng.normal(
                0,
                250,
                size=2,
            )
        )


    if n == 1:

        points = np.asarray(
            [start],
            dtype=float,
        )

    else:

        alpha = np.linspace(
            0,
            1,
            n,
        )[:, None]

        points = (
            start[None, :]
            +
            alpha
            *
            (
                end - start
            )[None, :]
        )


        if (
            mode == 1
            and n >= 3
        ):

            midpoint = n // 2

            points[
                midpoint
            ] += rng.normal(
                0,
                300,
                size=2,
            )


        points[
            0
        ] = start

        points[
            -1
        ] = end


    t0 = float(
        rng.integers(
            0,
            10_000_000,
        )
    )

    duration = float(
        rng.integers(
            0,
            1_000_000,
        )
    )


    timestamps = (
        np.linspace(
            t0,
            t0 + duration,
            n,
        )
        if n > 1
        else np.asarray(
            [t0]
        )
    )


    return [
        Event(
            float(points[i, 0]),
            float(points[i, 1]),
            float(timestamps[i]),
        )
        for i in range(n)
    ]


human_gestures = [
    make_gesture()
    for _ in range(
        3000
    )
]


test_gestures = [
    make_gesture()
    for _ in range(
        1000
    )
]


# ---------------------------------------------------------------------
# Metric equivalence
# ---------------------------------------------------------------------

max_metric_diff = 0.0


for gesture in test_gestures:

    a = historical_metrics(
        gesture
    )

    b = gesture_metrics(
        gesture
    )


    for key in [
        "duration_us",
        "displacement",
        "path_length",
        "endpoint_path_ratio",
    ]:

        diff = abs(
            float(a[key])
            -
            float(b[key])
        )

        max_metric_diff = max(
            max_metric_diff,
            diff,
        )


    assert (
        a[
            "n_points"
        ]
        ==
        b[
            "n_points"
        ]
    )

    assert (
        a[
            "exact_zero_displacement"
        ]
        ==
        b[
            "exact_zero_displacement"
        ]
    )

    assert (
        a[
            "near_zero_displacement"
        ]
        ==
        b[
            "near_zero_displacement"
        ]
    )


# ---------------------------------------------------------------------
# Historical calibration
# ---------------------------------------------------------------------

historical_rows = [
    historical_metrics(
        g
    )
    for g in human_gestures
]


positive_lengths = np.asarray(
    [
        x[
            "path_length"
        ]
        for x in historical_rows
        if x[
            "path_length"
        ] > 0
    ]
)


historical_floor = float(
    np.quantile(
        positive_lengths,
        0.25,
    )
)


historical_ratios = np.asarray(
    [
        x[
            "endpoint_path_ratio"
        ]
        for x in historical_rows
        if x[
            "path_length"
        ]
        >=
        historical_floor
    ]
)


historical_ratio_threshold = float(
    np.quantile(
        historical_ratios,
        TARGET_FPR,
        method="lower",
    )
)


clean = calibrate_fake_artifact(
    human_gestures,
    target_human_fpr=
        TARGET_FPR,
    minimum_session_artifacts=
        2,
)


floor_diff = abs(
    historical_floor
    -
    clean.movement_floor
)


ratio_diff = abs(
    historical_ratio_threshold
    -
    clean.ratio_threshold
)


# ---------------------------------------------------------------------
# Session decision equivalence
# ---------------------------------------------------------------------

historical_flags = []


for g in test_gestures:

    m = historical_metrics(
        g
    )

    historical_flags.append(
        (
            m[
                "path_length"
            ]
            >=
            historical_floor
        )
        and
        (
            m[
                "endpoint_path_ratio"
            ]
            <=
            historical_ratio_threshold
        )
    )


historical_count = int(
    np.sum(
        historical_flags
    )
)


clean_count = artifact_count(
    test_gestures,
    clean,
)


print(
    "=" * 72
)

print(
    "FAKE ARTIFACT EQUIVALENCE"
)

print(
    "=" * 72
)

print(
    "Maximum action-metric difference:",
    max_metric_diff,
)

print(
    "Movement-floor difference:",
    floor_diff,
)

print(
    "Ratio-threshold difference:",
    ratio_diff,
)

print(
    "Historical artifact count:",
    historical_count,
)

print(
    "Clean artifact count:",
    clean_count,
)


assert max_metric_diff <= 1e-12
assert floor_diff <= 1e-12
assert ratio_diff <= 1e-12
assert historical_count == clean_count


print()

print(
    "PASS: fake-action artifact implementation is numerically "
    "equivalent to the historical implementation within atol=1e-12."
)
