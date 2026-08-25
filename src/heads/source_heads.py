"""
Learned source-classification heads for the cross-scale defense.

Two logistic-regression heads are trained on clean Human vs. Raw-agent
sessions:

1. No-Bookkeeping Session Head (41D)
2. Strict Cross-Action Head (17D)

This module produces continuous source scores only.

Thresholding and final defense composition are handled separately in
src/defense.py.

No humanization condition is used for fitting these heads.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..features.session_features import (
    SESSION_FEATURES,
)

from ..features.cross_action_features import (
    STRICT_CROSS_ACTION_FEATURES,
)


# ---------------------------------------------------------------------
# Frozen feature definitions
# ---------------------------------------------------------------------

BOOKKEEPING_FEATURES = {
    "log_n_actions",
    "log_session_span",
    "two_point_rate",
    "multi_point_rate",
    "zero_displacement_rate",
    "duration_points_corr",
    "distance_points_corr",
}


NO_BOOKKEEPING_FEATURES = [
    feature
    for feature in SESSION_FEATURES
    if (
        feature not in BOOKKEEPING_FEATURES
        and
        not feature.startswith("log_points_")
    )
]


NO_BOOKKEEPING_DIM = len(
    NO_BOOKKEEPING_FEATURES
)


STRICT_CROSS_ACTION_DIM = len(
    STRICT_CROSS_ACTION_FEATURES
)


if NO_BOOKKEEPING_DIM != 41:
    raise RuntimeError(
        "Frozen No-Bookkeeping representation must contain 41 features, "
        f"found {NO_BOOKKEEPING_DIM}."
    )


if STRICT_CROSS_ACTION_DIM != 17:
    raise RuntimeError(
        "Frozen Strict Cross-Action representation must contain 17 features, "
        f"found {STRICT_CROSS_ACTION_DIM}."
    )


# ---------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------

def make_logistic_head() -> Pipeline:
    """
    Construct the exact learned-head architecture used by the validated
    experiments.
    """

    return Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=42,
                ),
            ),
        ]
    )


# ---------------------------------------------------------------------
# Feature validation
# ---------------------------------------------------------------------

def validate_feature_frame(
    frame: pd.DataFrame,
    feature_names: Sequence[str],
) -> None:

    missing = [
        feature
        for feature in feature_names
        if feature not in frame.columns
    ]

    if missing:
        raise ValueError(
            "Missing source-head features: "
            + ", ".join(missing)
        )


def feature_matrix(
    frame: pd.DataFrame,
    feature_names: Sequence[str],
) -> np.ndarray:
    """
    Convert a session feature frame into a clean numeric model matrix.

    This follows the validated evaluation convention:
    non-finite values are replaced by zero.
    """

    validate_feature_frame(
        frame,
        feature_names,
    )

    return (
        frame[
            list(feature_names)
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
        .to_numpy(
            dtype=float
        )
    )


# ---------------------------------------------------------------------
# Learned source heads
# ---------------------------------------------------------------------

@dataclass
class SourceHeadModels:
    """
    Pair of learned source-classification heads.

    The object stores fitted models only. Operating thresholds belong to
    the frozen decision layer, not to the learned-head implementation.
    """

    session_model: Pipeline
    strict_model: Pipeline

    @classmethod
    def fit(
        cls,
        clean_sessions: pd.DataFrame,
        labels: Sequence[int],
    ) -> "SourceHeadModels":
        """
        Fit both heads using clean Human / Raw-agent sessions.

        Parameters
        ----------
        clean_sessions:
            Validated 54D session feature representation.

        labels:
            Binary labels:
                0 = Human
                1 = Raw agent
        """

        labels = np.asarray(
            labels,
            dtype=int,
        )

        if len(clean_sessions) != len(labels):
            raise ValueError(
                "clean_sessions and labels must contain the same number "
                "of samples."
            )

        if set(
            np.unique(labels)
        ) != {0, 1}:
            raise ValueError(
                "Source-head training requires both Human (0) "
                "and Agent (1) samples."
            )

        X_session = feature_matrix(
            clean_sessions,
            NO_BOOKKEEPING_FEATURES,
        )

        X_strict = feature_matrix(
            clean_sessions,
            STRICT_CROSS_ACTION_FEATURES,
        )

        session_model = (
            make_logistic_head()
        )

        strict_model = (
            make_logistic_head()
        )

        session_model.fit(
            X_session,
            labels,
        )

        strict_model.fit(
            X_strict,
            labels,
        )

        return cls(
            session_model=session_model,
            strict_model=strict_model,
        )

    def score(
        self,
        sessions: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Return continuous Agent-probability scores from both heads.

        No threshold is applied here.
        """

        X_session = feature_matrix(
            sessions,
            NO_BOOKKEEPING_FEATURES,
        )

        X_strict = feature_matrix(
            sessions,
            STRICT_CROSS_ACTION_FEATURES,
        )

        session_score = (
            self.session_model
            .predict_proba(
                X_session
            )[:, 1]
        )

        strict_score = (
            self.strict_model
            .predict_proba(
                X_strict
            )[:, 1]
        )

        return pd.DataFrame(
            {
                "session_score":
                    session_score,

                "strict_score":
                    strict_score,
            },
            index=sessions.index,
        )
