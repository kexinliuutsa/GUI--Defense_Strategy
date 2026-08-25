"""
Closed-loop fake-action artifact head.

This module implements the specialized artifact detector used by the
cross-scale defense.

The detector operates on raw actions BEFORE swipe filtering.

Calibration uses Human actions only:

1. Compute path length for every Human action.
2. Define meaningful movement using the 25th percentile of positive
   Human path lengths.
3. Among Human moving actions, compute the low-tail threshold of
   endpoint-displacement / path-length ratio.
4. Flag an action when it both:
       - contains meaningful movement, and
       - returns unusually close to its starting point.
5. Flag a session when at least two such artifacts are observed.

The action-level target Human false-positive rate is 0.5%.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------
# Gesture metrics
# ---------------------------------------------------------------------

def gesture_metrics(
    gesture,
) -> Optional[dict[str, float]]:
    """
    Extract the validated raw-action metrics.

    No keep_swipe filtering is applied.
    """

    if (
        gesture is None
        or len(gesture) == 0
    ):
        return None

    pts = np.asarray(
        [
            [
                float(e.x),
                float(e.y),
            ]
            for e in gesture
        ],
        dtype=float,
    )

    ts = np.asarray(
        [
            float(
                e.timestamp_us
            )
            for e in gesture
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


    endpoint_path_ratio = (
        displacement
        /
        path_length
        if path_length > 1e-12
        else 1.0
    )


    duration = (
        float(
            ts[-1]
            -
            ts[0]
        )
        if len(ts) >= 2
        else 0.0
    )


    return {
        "n_points":
            int(
                len(gesture)
            ),

        "duration_us":
            duration,

        "displacement":
            displacement,

        "path_length":
            path_length,

        "endpoint_path_ratio":
            float(
                endpoint_path_ratio
            ),

        "exact_zero_displacement":
            bool(
                displacement
                <=
                1e-6
            ),

        "near_zero_displacement":
            bool(
                displacement
                <=
                1.0
            ),
    }


# ---------------------------------------------------------------------
# Human-only calibration
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class FakeArtifactCalibration:
    """
    Frozen Human-derived thresholds for the fake-action artifact head.
    """

    movement_floor: float

    ratio_threshold: float

    action_target_human_fpr: float = 0.005

    minimum_session_artifacts: int = 2


def calibrate_fake_artifact(
    human_gestures: Iterable,
    *,
    target_human_fpr: float = 0.005,
    minimum_session_artifacts: int = 2,
) -> FakeArtifactCalibration:
    """
    Calibrate artifact thresholds using Human raw actions only.
    """

    metrics = []

    for gesture in human_gestures:

        m = gesture_metrics(
            gesture
        )

        if m is not None:
            metrics.append(
                m
            )


    if len(metrics) == 0:
        raise ValueError(
            "No valid Human gestures were provided."
        )


    positive_human_lengths = np.asarray(
        [
            m[
                "path_length"
            ]
            for m in metrics
            if m[
                "path_length"
            ] > 0
        ],
        dtype=float,
    )


    if len(
        positive_human_lengths
    ) == 0:

        raise RuntimeError(
            "No positive Human path lengths found."
        )


    # Historical implementation uses the 25th percentile.
    movement_floor = float(
        np.quantile(
            positive_human_lengths,
            0.25,
        )
    )


    human_moving_ratios = np.asarray(
        [
            m[
                "endpoint_path_ratio"
            ]
            for m in metrics
            if m[
                "path_length"
            ] >= movement_floor
        ],
        dtype=float,
    )


    if len(
        human_moving_ratios
    ) == 0:

        raise RuntimeError(
            "No Human moving actions remain after movement calibration."
        )


    ratio_threshold = float(
        np.quantile(
            human_moving_ratios,
            target_human_fpr,
            method="lower",
        )
    )


    return FakeArtifactCalibration(
        movement_floor=
            movement_floor,

        ratio_threshold=
            ratio_threshold,

        action_target_human_fpr=
            float(
                target_human_fpr
            ),

        minimum_session_artifacts=
            int(
                minimum_session_artifacts
            ),
    )


# ---------------------------------------------------------------------
# Action-level artifact decision
# ---------------------------------------------------------------------

def is_fake_artifact(
    gesture,
    calibration: FakeArtifactCalibration,
) -> bool:
    """
    Return whether one raw action matches the closed-loop artifact.
    """

    metrics = gesture_metrics(
        gesture
    )

    if metrics is None:
        return False


    moving_action = (
        metrics[
            "path_length"
        ]
        >=
        calibration.movement_floor
    )


    low_endpoint_ratio = (
        metrics[
            "endpoint_path_ratio"
        ]
        <=
        calibration.ratio_threshold
    )


    return bool(
        moving_action
        and
        low_endpoint_ratio
    )


def artifact_flags(
    gestures: Iterable,
    calibration: FakeArtifactCalibration,
) -> np.ndarray:
    """
    Return one Boolean artifact decision per raw action.
    """

    return np.asarray(
        [
            is_fake_artifact(
                gesture,
                calibration,
            )
            for gesture in gestures
        ],
        dtype=bool,
    )


# ---------------------------------------------------------------------
# Session-level head
# ---------------------------------------------------------------------

def artifact_count(
    gestures: Iterable,
    calibration: FakeArtifactCalibration,
) -> int:
    """
    Count closed-loop action artifacts in one session.
    """

    return int(
        artifact_flags(
            gestures,
            calibration,
        ).sum()
    )


def fake_artifact_detect(
    gestures: Iterable,
    calibration: FakeArtifactCalibration,
) -> bool:
    """
    Frozen session-level fake-action decision.

    A session is flagged when at least two action artifacts are observed
    by default.
    """

    count = artifact_count(
        gestures,
        calibration,
    )

    return bool(
        count
        >=
        calibration.minimum_session_artifacts
    )


def detect_from_artifact_count(
    count: int | Sequence[int] | np.ndarray,
    *,
    minimum_session_artifacts: int = 2,
):
    """
    Apply the frozen session rule to already-computed artifact counts.
    """

    counts = np.asarray(
        count,
        dtype=int,
    )

    result = (
        counts
        >=
        int(
            minimum_session_artifacts
        )
    )

    if result.ndim == 0:
        return bool(
            result
        )

    return result
