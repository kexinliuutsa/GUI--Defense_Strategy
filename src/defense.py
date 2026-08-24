"""
Final cross-scale behavioral defense.

This module contains the clean implementation of the final defense used in
the repository. It intentionally contains no experiment IDs, dataset paths,
or evaluation-specific code.

Defense:
    No-bookkeeping session head
            OR
    Strict cross-action head
            OR
    Fake-action artifact head

The two learned heads are logistic-regression classifiers trained on clean
Human vs. Raw-agent sessions. The fake-action head is a lightweight rule
that operates on a session-level artifact count.

Expected session-feature input
------------------------------
A pandas DataFrame with one row per session and the behavioral features
produced by the repository's session feature extractor.

The no-bookkeeping head uses all numeric session features except metadata
and obvious bookkeeping/count features.

The strict cross-action head uses only relations across actions, such as
inter-action gaps, direction consistency, coordinate reuse, and
duration-displacement coupling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------

DEFAULT_METADATA_COLUMNS = {
    "group",
    "participant",
    "session_id",
    "label",
    "label_agent",
    "family",
    "p_session_agent",
    "session_structure_detect",
    "matched_41c",
    "matched_to_41c",
}


BOOKKEEPING_FEATURES = {
    "log_n_actions",
    "log_session_span",
    "two_point_rate",
    "multi_point_rate",
    "zero_displacement_rate",
    "duration_points_corr",
    "distance_points_corr",
}


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


def no_bookkeeping_feature_names(
    frame: pd.DataFrame,
    metadata_columns: Iterable[str] = DEFAULT_METADATA_COLUMNS,
) -> list[str]:
    """
    Return the session features used by the no-bookkeeping head.

    In addition to the explicitly listed bookkeeping features, every feature
    beginning with ``log_points_`` is removed.
    """
    metadata = set(metadata_columns)

    output = []
    for column in frame.columns:
        if column in metadata:
            continue
        if column in BOOKKEEPING_FEATURES:
            continue
        if column.startswith("log_points_"):
            continue
        if not pd.api.types.is_numeric_dtype(frame[column]):
            continue
        output.append(column)

    return output


def strict_cross_action_feature_names(frame: pd.DataFrame) -> list[str]:
    """
    Return the strict cross-action feature set available in ``frame``.

    Missing features are not silently synthesized. If the complete final
    representation is required, call ``validate_strict_cross_action_features``.
    """
    return [
        feature
        for feature in STRICT_CROSS_ACTION_FEATURES
        if feature in frame.columns
    ]


def validate_strict_cross_action_features(frame: pd.DataFrame) -> None:
    """Raise an informative error if any final strict-cross feature is missing."""
    missing = [
        feature
        for feature in STRICT_CROSS_ACTION_FEATURES
        if feature not in frame.columns
    ]
    if missing:
        raise ValueError(
            "Missing strict cross-action features: "
            + ", ".join(missing)
        )


# ---------------------------------------------------------------------------
# Model / calibration helpers
# ---------------------------------------------------------------------------

def make_logistic_head() -> Pipeline:
    """Construct one learned defense head."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
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


def calibrate_threshold_from_humans(
    scores: Sequence[float],
    labels: Sequence[int],
    target_human_fpr: float = 0.01,
) -> float:
    """
    Calibrate a decision threshold using human scores only.

    Parameters
    ----------
    scores:
        Agent-probability scores from a calibration split.
    labels:
        Binary labels where 0 = Human and 1 = Agent.
    target_human_fpr:
        Desired false-positive operating point.

    Notes
    -----
    For publication-quality evaluation, ``scores`` should be out-of-fold or
    otherwise held out from the model that produced them.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)

    human_scores = scores[labels == 0]
    if len(human_scores) == 0:
        raise ValueError("No human calibration samples were provided.")

    return float(
        np.quantile(
            human_scores,
            1.0 - target_human_fpr,
            method="higher",
        )
    )


def fake_action_detect(
    artifact_count: Sequence[int] | np.ndarray | pd.Series,
    minimum_artifacts: int = 2,
) -> np.ndarray:
    """
    Specialized fake-action decision.

    A session is flagged when it contains at least ``minimum_artifacts``
    closed-loop fake-action artifacts.
    """
    counts = np.asarray(artifact_count, dtype=int)
    return counts >= int(minimum_artifacts)


# ---------------------------------------------------------------------------
# Final defense
# ---------------------------------------------------------------------------

@dataclass
class CrossScaleDefense:
    """
    Final Stage1-free defense.

    Final decision:
        no_bookkeeping
        OR strict_cross_action
        OR fake_action_artifact
    """

    no_bookkeeping_model: Pipeline
    strict_cross_model: Pipeline
    no_bookkeeping_threshold: float
    strict_cross_threshold: float
    no_bookkeeping_features: list[str]
    strict_cross_features: list[str]
    minimum_fake_artifacts: int = 2

    @classmethod
    def fit(
        cls,
        clean_sessions: pd.DataFrame,
        labels: Sequence[int],
        no_bookkeeping_threshold: float,
        strict_cross_threshold: float,
        *,
        metadata_columns: Iterable[str] = DEFAULT_METADATA_COLUMNS,
        minimum_fake_artifacts: int = 2,
    ) -> "CrossScaleDefense":
        """
        Fit both learned heads on clean Human/Raw-agent sessions.

        Thresholds are passed in explicitly so that model fitting is separated
        from threshold calibration. This avoids accidentally calibrating on
        the same evaluation data.
        """
        labels = np.asarray(labels, dtype=int)

        if len(clean_sessions) != len(labels):
            raise ValueError(
                "clean_sessions and labels must contain the same number of rows."
            )

        nb_features = no_bookkeeping_feature_names(
            clean_sessions,
            metadata_columns=metadata_columns,
        )

        validate_strict_cross_action_features(clean_sessions)
        strict_features = list(STRICT_CROSS_ACTION_FEATURES)

        if not nb_features:
            raise ValueError("No no-bookkeeping features were found.")

        nb_model = make_logistic_head()
        strict_model = make_logistic_head()

        nb_model.fit(
            clean_sessions[nb_features].to_numpy(dtype=float),
            labels,
        )

        strict_model.fit(
            clean_sessions[strict_features].to_numpy(dtype=float),
            labels,
        )

        return cls(
            no_bookkeeping_model=nb_model,
            strict_cross_model=strict_model,
            no_bookkeeping_threshold=float(no_bookkeeping_threshold),
            strict_cross_threshold=float(strict_cross_threshold),
            no_bookkeeping_features=nb_features,
            strict_cross_features=strict_features,
            minimum_fake_artifacts=int(minimum_fake_artifacts),
        )

    def score_components(
        self,
        sessions: pd.DataFrame,
    ) -> pd.DataFrame:
        """Return probability scores and binary decisions for both learned heads."""
        missing_nb = [
            f for f in self.no_bookkeeping_features
            if f not in sessions.columns
        ]
        missing_strict = [
            f for f in self.strict_cross_features
            if f not in sessions.columns
        ]

        if missing_nb:
            raise ValueError(
                "Missing no-bookkeeping features: "
                + ", ".join(missing_nb)
            )
        if missing_strict:
            raise ValueError(
                "Missing strict cross-action features: "
                + ", ".join(missing_strict)
            )

        nb_score = self.no_bookkeeping_model.predict_proba(
            sessions[self.no_bookkeeping_features].to_numpy(dtype=float)
        )[:, 1]

        strict_score = self.strict_cross_model.predict_proba(
            sessions[self.strict_cross_features].to_numpy(dtype=float)
        )[:, 1]

        return pd.DataFrame(
            {
                "no_bookkeeping_score": nb_score,
                "strict_cross_score": strict_score,
                "no_bookkeeping_detect": (
                    nb_score >= self.no_bookkeeping_threshold
                ),
                "strict_cross_detect": (
                    strict_score >= self.strict_cross_threshold
                ),
            },
            index=sessions.index,
        )

    def predict(
        self,
        sessions: pd.DataFrame,
        artifact_count: Sequence[int] | np.ndarray | pd.Series,
    ) -> pd.DataFrame:
        """
        Apply the complete defense.

        Parameters
        ----------
        sessions:
            One row per session with extracted behavioral features.
        artifact_count:
            Number of detected closed-loop fake-action artifacts per session.

        Returns
        -------
        pandas.DataFrame
            Component scores, component decisions, and ``final_detect``.
        """
        if len(sessions) != len(artifact_count):
            raise ValueError(
                "sessions and artifact_count must have the same length."
            )

        result = self.score_components(sessions)

        result["fake_action_detect"] = fake_action_detect(
            artifact_count,
            minimum_artifacts=self.minimum_fake_artifacts,
        )

        result["final_detect"] = (
            result["no_bookkeeping_detect"]
            | result["strict_cross_detect"]
            | result["fake_action_detect"]
        )

        return result


# ---------------------------------------------------------------------------
# Minimal direct-use helper
# ---------------------------------------------------------------------------

def combine_defense_heads(
    no_bookkeeping_detect: Sequence[bool],
    strict_cross_detect: Sequence[bool],
    fake_action_detected: Sequence[bool],
) -> np.ndarray:
    """
    Combine already-computed component decisions using the final OR rule.

    This is the core decision rule of the defense.
    """
    nb = np.asarray(no_bookkeeping_detect, dtype=bool)
    strict = np.asarray(strict_cross_detect, dtype=bool)
    fake = np.asarray(fake_action_detected, dtype=bool)

    if not (len(nb) == len(strict) == len(fake)):
        raise ValueError("All component decision arrays must have equal length.")

    return nb | strict | fake
