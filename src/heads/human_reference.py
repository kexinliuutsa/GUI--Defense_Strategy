"""
Human-reference support verification head.

This module implements the Human-only reference model used by the
cross-scale defense.

For each representation:

1. Fit StandardScaler using Human sessions only.
2. Transform Human sessions into standardized space.
3. Compute the Human centroid.
4. Score a sample by RMS standardized distance from that centroid.

Calibration is participant-disjoint:

- Human OOF scores are produced by reference models that exclude the
  held-out participant.
- Each representation is normalized by its own Human OOF threshold.
- The normalized NB and Strict scores are max-fused.
- The final fused threshold is calibrated again using Human OOF scores.

Humanization / attack conditions are not used for reference fitting or
threshold calibration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from .source_heads import (
    NO_BOOKKEEPING_FEATURES,
)

from ..features.cross_action_features import (
    STRICT_CROSS_ACTION_FEATURES,
)


# ---------------------------------------------------------------------
# Matrix helper
# ---------------------------------------------------------------------

def matrix_from_df(
    df: pd.DataFrame,
    features: Sequence[str],
) -> np.ndarray:
    """
    Convert selected DataFrame columns to the numeric matrix used by
    the validated Human-reference implementation.
    """

    missing = [
        feature
        for feature in features
        if feature not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing Human-reference features: "
            + ", ".join(missing)
        )

    return (
        df[
            list(features)
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
        .to_numpy(float)
    )


# ---------------------------------------------------------------------
# Single Human-reference representation
# ---------------------------------------------------------------------

class HumanReference:
    """
    Human-only support reference in one behavioral representation.
    """

    def __init__(
        self,
        features,
        name,
    ):
        self.features = list(
            features
        )

        self.name = name

        self.scaler = StandardScaler()

        self.centroid = None


    def fit_matrix(
        self,
        X,
    ):
        """
        Fit StandardScaler and Human centroid.
        """

        X = np.asarray(
            X,
            dtype=float,
        )

        if (
            X.ndim != 2
            or len(X) == 0
        ):
            raise ValueError(
                f"{self.name}: X must be a non-empty 2D matrix."
            )

        Z = self.scaler.fit_transform(
            X
        )

        self.centroid = np.mean(
            Z,
            axis=0,
        )

        return self


    def score_matrix(
        self,
        X,
    ):
        """
        RMS standardized distance from the fitted Human centroid.
        """

        if self.centroid is None:
            raise RuntimeError(
                f"{self.name}: reference model has not been fitted."
            )

        X = np.asarray(
            X,
            dtype=float,
        )

        Z = self.scaler.transform(
            X
        )

        d = (
            Z
            -
            self.centroid
        )

        return np.sqrt(
            np.mean(
                d ** 2,
                axis=1,
            )
        )


    def fit(
        self,
        df: pd.DataFrame,
    ):
        """
        Convenience DataFrame interface.
        """

        X = matrix_from_df(
            df,
            self.features,
        )

        return self.fit_matrix(
            X
        )


    def score(
        self,
        df: pd.DataFrame,
    ) -> np.ndarray:
        """
        Convenience DataFrame interface.
        """

        X = matrix_from_df(
            df,
            self.features,
        )

        return self.score_matrix(
            X
        )


# ---------------------------------------------------------------------
# Participant-disjoint Human OOF calibration
# ---------------------------------------------------------------------

def human_oof_scores(
    X,
    groups,
    features,
    name,
    target_human_fpr: float,
):
    """
    Produce participant-disjoint Human-only OOF distance scores.

    Every Human sample is scored by a Human reference fitted without
    that participant.
    """

    X = np.asarray(
        X,
        dtype=float,
    )

    groups = np.asarray(
        groups
    )

    if len(X) != len(groups):
        raise ValueError(
            f"{name}: X and groups must contain the same number "
            "of samples."
        )

    unique_groups = np.unique(
        groups
    )

    n_splits = min(
        5,
        len(
            unique_groups
        ),
    )

    if n_splits < 2:
        raise ValueError(
            f"{name}: at least two Human participant groups "
            "are required for OOF calibration."
        )

    cv = GroupKFold(
        n_splits=n_splits
    )

    oof = np.full(
        len(X),
        np.nan,
        dtype=float,
    )


    for (
        train_idx,
        test_idx,
    ) in cv.split(
        X,
        groups=groups,
    ):

        ref = HumanReference(
            features,
            name,
        )

        ref.fit_matrix(
            X[
                train_idx
            ]
        )

        oof[
            test_idx
        ] = ref.score_matrix(
            X[
                test_idx
            ]
        )


    if not np.all(
        np.isfinite(
            oof
        )
    ):
        raise RuntimeError(
            f"{name}: incomplete Human OOF scores."
        )


    threshold = float(
        np.quantile(
            oof,
            1.0
            -
            target_human_fpr,
            method="higher",
        )
    )


    detect = (
        oof
        >=
        threshold
    )


    return (
        oof,
        threshold,
        detect,
    )


# ---------------------------------------------------------------------
# Complete two-view Human-reference head
# ---------------------------------------------------------------------

@dataclass
class HumanReferenceHead:
    """
    Frozen two-view Human-reference support verifier.

    It contains:

    - NB Human reference
    - Strict Human reference
    - per-view Human OOF normalization thresholds
    - fused Human OOF decision threshold
    """

    nb_reference: HumanReference

    strict_reference: HumanReference

    nb_normalization_threshold: float

    strict_normalization_threshold: float

    fused_threshold: float


    @classmethod
    def fit(
        cls,
        human_sessions: pd.DataFrame,
        participants: Sequence,
        *,
        target_human_fpr: float = 0.01,
    ) -> "HumanReferenceHead":
        """
        Calibrate and fit the complete Human-reference head.

        Calibration uses Human data only.

        `participants` is used solely to create participant-disjoint
        OOF Human reference scores.
        """

        participants = np.asarray(
            participants
        )

        if (
            len(human_sessions)
            !=
            len(participants)
        ):
            raise ValueError(
                "human_sessions and participants must contain "
                "the same number of samples."
            )


        # -------------------------------------------------------------
        # Representations
        # -------------------------------------------------------------

        H_NB = matrix_from_df(
            human_sessions,
            NO_BOOKKEEPING_FEATURES,
        )

        H_STRICT = matrix_from_df(
            human_sessions,
            STRICT_CROSS_ACTION_FEATURES,
        )


        # -------------------------------------------------------------
        # Participant-disjoint Human OOF calibration
        # -------------------------------------------------------------

        (
            nb_href_oof,
            nb_threshold,
            _,
        ) = human_oof_scores(
            H_NB,
            participants,
            NO_BOOKKEEPING_FEATURES,
            "NB_HREF",
            target_human_fpr,
        )


        (
            strict_href_oof,
            strict_threshold,
            _,
        ) = human_oof_scores(
            H_STRICT,
            participants,
            STRICT_CROSS_ACTION_FEATURES,
            "STRICT_HREF",
            target_human_fpr,
        )


        # -------------------------------------------------------------
        # Normalize each view by its own Human anomaly boundary
        # -------------------------------------------------------------

        nb_ratio_oof = (
            nb_href_oof
            /
            max(
                nb_threshold,
                1e-12,
            )
        )


        strict_ratio_oof = (
            strict_href_oof
            /
            max(
                strict_threshold,
                1e-12,
            )
        )


        # -------------------------------------------------------------
        # Max fusion
        # -------------------------------------------------------------

        fused_href_oof = np.maximum(
            nb_ratio_oof,
            strict_ratio_oof,
        )


        # -------------------------------------------------------------
        # Final fused Human-only threshold
        # -------------------------------------------------------------

        fused_threshold = float(
            np.quantile(
                fused_href_oof,
                1.0
                -
                target_human_fpr,
                method="higher",
            )
        )


        # -------------------------------------------------------------
        # Production references fitted on all Human sessions
        # -------------------------------------------------------------

        nb_reference = HumanReference(
            NO_BOOKKEEPING_FEATURES,
            "NB_HREF",
        )

        nb_reference.fit_matrix(
            H_NB
        )


        strict_reference = HumanReference(
            STRICT_CROSS_ACTION_FEATURES,
            "STRICT_HREF",
        )

        strict_reference.fit_matrix(
            H_STRICT
        )


        return cls(
            nb_reference=
                nb_reference,

            strict_reference=
                strict_reference,

            nb_normalization_threshold=
                float(
                    nb_threshold
                ),

            strict_normalization_threshold=
                float(
                    strict_threshold
                ),

            fused_threshold=
                float(
                    fused_threshold
                ),
        )


    def score(
        self,
        sessions: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Score sessions using the fitted Human references.

        Returns raw RMS distance, normalized distance, max-fused
        Human-reference score, and fused decision.
        """

        X_nb = matrix_from_df(
            sessions,
            NO_BOOKKEEPING_FEATURES,
        )

        X_strict = matrix_from_df(
            sessions,
            STRICT_CROSS_ACTION_FEATURES,
        )


        nb_score = (
            self.nb_reference
            .score_matrix(
                X_nb
            )
        )


        strict_score = (
            self.strict_reference
            .score_matrix(
                X_strict
            )
        )


        nb_ratio = (
            nb_score
            /
            max(
                self.nb_normalization_threshold,
                1e-12,
            )
        )


        strict_ratio = (
            strict_score
            /
            max(
                self.strict_normalization_threshold,
                1e-12,
            )
        )


        fused_score = np.maximum(
            nb_ratio,
            strict_ratio,
        )


        detected = (
            fused_score
            >=
            self.fused_threshold
        )


        return pd.DataFrame(
            {
                "nb_href_score":
                    nb_score,

                "strict_href_score":
                    strict_score,

                "nb_href_ratio":
                    nb_ratio,

                "strict_href_ratio":
                    strict_ratio,

                "human_reference_score":
                    fused_score,

                "human_reference_detect":
                    detected,
            },
            index=sessions.index,
        )
