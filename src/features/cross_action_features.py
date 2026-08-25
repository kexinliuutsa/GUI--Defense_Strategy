"""
Frozen feature schema for the Strict Cross-Action Head.

The Strict Cross-Action representation intentionally excludes most
within-action geometry and simple session bookkeeping features.

It focuses on relationships across actions, including:

- direction organization,
- spatial reuse,
- inter-action timing,
- and cross-action coupling.

Important:
    This module freezes the feature schema used by the final defense.
    Exact feature computation is kept separate so that the validated
    experimental implementation can be ported without changing formulas.
"""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Frozen 17-dimensional representation
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
        "Frozen Strict Cross-Action representation must contain 17 features."
    )


# ---------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------

def validate_cross_action_features(
    feature_names: Iterable[str],
) -> None:
    """
    Verify that all frozen Strict Cross-Action features are available.

    Parameters
    ----------
    feature_names:
        Names available in an upstream feature representation.

    Raises
    ------
    ValueError
        If one or more required features are missing.
    """

    available = set(
        feature_names
    )

    missing = [
        feature
        for feature in STRICT_CROSS_ACTION_FEATURES
        if feature not in available
    ]

    if missing:
        raise ValueError(
            "Missing Strict Cross-Action features: "
            + ", ".join(missing)
        )


# ---------------------------------------------------------------------
# DataFrame interface
# ---------------------------------------------------------------------

def select_cross_action_dataframe(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Select the frozen 17D Strict Cross-Action representation.

    Non-finite values follow the evaluation pipeline convention:
    infinities are converted to NaN and missing values are filled with 0.

    The column order is always identical to the frozen model schema.
    """

    validate_cross_action_features(
        df.columns
    )

    result = (
        df[
            STRICT_CROSS_ACTION_FEATURES
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0.0)
        .copy()
    )

    return result


def cross_action_matrix(
    df: pd.DataFrame,
) -> np.ndarray:
    """
    Convert a feature DataFrame into the frozen model input matrix.

    Returns
    -------
    np.ndarray
        Shape: (n_samples, 17)
    """

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


# ---------------------------------------------------------------------
# Single-record interface
# ---------------------------------------------------------------------

def cross_action_vector(
    features: Mapping[str, float],
) -> np.ndarray:
    """
    Convert one precomputed feature dictionary into the frozen 17D vector.
    """

    validate_cross_action_features(
        features.keys()
    )

    values = []

    for feature in STRICT_CROSS_ACTION_FEATURES:

        value = features[
            feature
        ]

        try:
            value = float(
                value
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

        values.append(
            value
        )

    result = np.asarray(
        values,
        dtype=float,
    )

    if result.shape != (
        STRICT_CROSS_ACTION_DIM,
    ):
        raise RuntimeError(
            "Unexpected Strict Cross-Action vector shape: "
            f"{result.shape}"
        )

    return result
