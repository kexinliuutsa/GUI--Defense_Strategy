"""
Frozen 17-dimensional Strict Cross-Action representation.

This module selects the cross-action subset from the validated 54D
session representation.

The strict representation intentionally excludes:

- action count,
- point-count statistics,
- total session span,
- per-action absolute duration statistics,
- displacement distributions,
- path-length distributions.

It retains:

- gap structure,
- direction structure,
- coordinate reuse,
- and cross-action coupling.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Optional

import numpy as np
import pandas as pd

from .session_features import (
    extract_session_features,
)


# ---------------------------------------------------------------------
# Frozen Strict Cross-Action schema
# ---------------------------------------------------------------------

STRICT_CROSS_ACTION_FEATURES = [
    "direction_resultant",

    "start_unique_cell_ratio",
    "end_unique_cell_ratio",

    "negative_gap_rate",

    "duration_displacement_corr",

    "log_gap_mean",
    "log_gap_std",
    "log_gap_median",
    "log_gap_q10",
    "log_gap_q90",
    "log_gap_cv",

    "direction_similarity_mean",
    "direction_similarity_std",
    "direction_similarity_median",
    "direction_similarity_q10",
    "direction_similarity_q90",
    "direction_similarity_cv",
]


STRICT_CROSS_ACTION_DIM = len(
    STRICT_CROSS_ACTION_FEATURES
)


if STRICT_CROSS_ACTION_DIM != 17:

    raise RuntimeError(
        "Frozen Strict Cross-Action representation "
        "must contain exactly 17 features."
    )


# ---------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------

def validate_cross_action_features(
    feature_names: Iterable[str],
) -> None:

    available = set(
        feature_names
    )

    missing = [
        feature
        for feature
        in STRICT_CROSS_ACTION_FEATURES
        if feature not in available
    ]

    if missing:

        raise ValueError(
            "Missing Strict Cross-Action features: "
            +
            ", ".join(
                missing
            )
        )


# ---------------------------------------------------------------------
# Select Strict subset from one session dictionary
# ---------------------------------------------------------------------

def select_cross_action_features(
    session_features: Mapping[
        str,
        float,
    ],
) -> dict[str, float]:

    validate_cross_action_features(
        session_features.keys()
    )

    result = {}

    for feature in (
        STRICT_CROSS_ACTION_FEATURES
    ):

        try:

            value = float(
                session_features[
                    feature
                ]
            )

        except (
            TypeError,
            ValueError,
        ):

            value = 0.0


        if not np.isfinite(
            value
        ):

            value = 0.0


        result[
            feature
        ] = value


    return result


# ---------------------------------------------------------------------
# Direct gesture interface
# ---------------------------------------------------------------------

def extract_cross_action_features(
    gestures,
) -> Optional[dict[str, float]]:
    """
    Extract the frozen 17D Strict Cross-Action representation directly
    from a session's gestures.
    """

    session = (
        extract_session_features(
            gestures
        )
    )

    if session is None:
        return None

    return (
        select_cross_action_features(
            session
        )
    )


# ---------------------------------------------------------------------
# Vector interface
# ---------------------------------------------------------------------

def cross_action_vector(
    features: Mapping[
        str,
        float,
    ],
) -> np.ndarray:

    selected = (
        select_cross_action_features(
            features
        )
    )

    vector = np.asarray(
        [
            selected[
                feature
            ]
            for feature
            in STRICT_CROSS_ACTION_FEATURES
        ],
        dtype=float,
    )

    if vector.shape != (
        STRICT_CROSS_ACTION_DIM,
    ):

        raise RuntimeError(
            "Unexpected Strict Cross-Action vector shape: "
            f"{vector.shape}"
        )

    return vector


def extract_cross_action_vector(
    gestures,
) -> Optional[np.ndarray]:
    """
    Convenience interface:

        gestures -> validated session features -> frozen 17D vector
    """

    features = (
        extract_cross_action_features(
            gestures
        )
    )

    if features is None:
        return None

    return (
        cross_action_vector(
            features
        )
    )


# ---------------------------------------------------------------------
# DataFrame interface
# ---------------------------------------------------------------------

def select_cross_action_dataframe(
    df: pd.DataFrame,
) -> pd.DataFrame:

    validate_cross_action_features(
        df.columns
    )

    return (
        df[
            STRICT_CROSS_ACTION_FEATURES
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .fillna(
            0.0
        )
        .copy()
    )


def cross_action_matrix(
    df: pd.DataFrame,
) -> np.ndarray:

    selected = (
        select_cross_action_dataframe(
            df
        )
    )

    matrix = selected.to_numpy(
        dtype=float
    )

    if (
        matrix.ndim != 2
        or
        matrix.shape[1]
        != STRICT_CROSS_ACTION_DIM
    ):

        raise RuntimeError(
            "Unexpected Strict Cross-Action matrix shape: "
            f"{matrix.shape}"
        )

    return matrix
