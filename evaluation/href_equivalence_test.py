"""
Exact equivalence test for the Human-reference calibration pipeline.

Compares:

1. Literal historical EXP47 calibration logic
2. Clean src.heads.human_reference implementation

The test covers:

- participant-disjoint Human-only OOF scoring
- per-view OOF thresholds
- per-view normalization
- max fusion
- fused Human OOF threshold
- final references fitted on all Human data
- scoring of unseen sessions
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from src.features.session_features import SESSION_FEATURES

from src.heads.source_heads import (
    NO_BOOKKEEPING_FEATURES,
)

from src.features.cross_action_features import (
    STRICT_CROSS_ACTION_FEATURES,
)

from src.heads.human_reference import (
    HumanReferenceHead,
)


TARGET_HUMAN_FPR = 0.01
SEED = 20260826


# ============================================================
# Historical implementation written literally
# ============================================================

class HistoricalHumanReference:

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


def historical_oof_scores(
    X,
    groups,
    features,
    name,
):

    unique_groups = np.unique(
        groups
    )

    n_splits = min(
        5,
        len(
            unique_groups
        ),
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

        ref = HistoricalHumanReference(
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
            f"{name}: incomplete Human OOF scores"
        )


    threshold = float(
        np.quantile(
            oof,
            1.0
            -
            TARGET_HUMAN_FPR,
            method="higher",
        )
    )


    return (
        oof,
        threshold,
    )


# ============================================================
# Synthetic Human data
# ============================================================

rng = np.random.default_rng(
    SEED
)

n_human = 370
n_test = 83


# Produce structured rather than completely independent features,
# so correlations / heavy tails are also exercised.
human_array = rng.normal(
    size=(
        n_human,
        len(
            SESSION_FEATURES
        ),
    )
)

human_array[:, 0] += rng.normal(
    0,
    2,
    size=n_human,
)

human_array[:, -1] *= rng.lognormal(
    mean=0,
    sigma=1,
    size=n_human,
)


human = pd.DataFrame(
    human_array,
    columns=SESSION_FEATURES,
)


# Seven participant groups, similar to the Human grouping structure
# used in the evaluation.
participants = np.asarray(
    [
        f"user{(i % 7) + 1}"
        for i in range(
            n_human
        )
    ]
)


test = pd.DataFrame(
    rng.normal(
        size=(
            n_test,
            len(
                SESSION_FEATURES
            ),
        )
    ),
    columns=SESSION_FEATURES,
)


# ============================================================
# Matrices
# ============================================================

H_NB = (
    human[
        NO_BOOKKEEPING_FEATURES
    ]
    .to_numpy(float)
)

H_STRICT = (
    human[
        STRICT_CROSS_ACTION_FEATURES
    ]
    .to_numpy(float)
)


# ============================================================
# Historical OOF calibration
# ============================================================

(
    historical_nb_oof,
    historical_nb_threshold,
) = historical_oof_scores(
    H_NB,
    participants,
    NO_BOOKKEEPING_FEATURES,
    "NB_HREF",
)


(
    historical_strict_oof,
    historical_strict_threshold,
) = historical_oof_scores(
    H_STRICT,
    participants,
    STRICT_CROSS_ACTION_FEATURES,
    "STRICT_HREF",
)


historical_nb_ratio = (
    historical_nb_oof
    /
    max(
        historical_nb_threshold,
        1e-12,
    )
)


historical_strict_ratio = (
    historical_strict_oof
    /
    max(
        historical_strict_threshold,
        1e-12,
    )
)


historical_fused_oof = np.maximum(
    historical_nb_ratio,
    historical_strict_ratio,
)


historical_fused_threshold = float(
    np.quantile(
        historical_fused_oof,
        1.0
        -
        TARGET_HUMAN_FPR,
        method="higher",
    )
)


# ============================================================
# Clean implementation
# ============================================================

clean = HumanReferenceHead.fit(
    human,
    participants,
    target_human_fpr=
        TARGET_HUMAN_FPR,
)


# ============================================================
# Threshold equivalence
# ============================================================

print(
    "=" * 72
)

print(
    "HREF CALIBRATION EQUIVALENCE"
)

print(
    "=" * 72
)


print(
    "Historical NB threshold:",
    historical_nb_threshold,
)

print(
    "Clean NB threshold:",
    clean.nb_normalization_threshold,
)


print(
    "\nHistorical Strict threshold:",
    historical_strict_threshold,
)

print(
    "Clean Strict threshold:",
    clean.strict_normalization_threshold,
)


print(
    "\nHistorical fused threshold:",
    historical_fused_threshold,
)

print(
    "Clean fused threshold:",
    clean.fused_threshold,
)


nb_threshold_diff = abs(
    historical_nb_threshold
    -
    clean.nb_normalization_threshold
)

strict_threshold_diff = abs(
    historical_strict_threshold
    -
    clean.strict_normalization_threshold
)

fused_threshold_diff = abs(
    historical_fused_threshold
    -
    clean.fused_threshold
)


print(
    "\nNB threshold absolute difference:",
    nb_threshold_diff,
)

print(
    "Strict threshold absolute difference:",
    strict_threshold_diff,
)

print(
    "Fused threshold absolute difference:",
    fused_threshold_diff,
)


assert nb_threshold_diff == 0.0
assert strict_threshold_diff == 0.0
assert fused_threshold_diff == 0.0


# ============================================================
# Final-reference scoring equivalence
# ============================================================

historical_nb_final = HistoricalHumanReference(
    NO_BOOKKEEPING_FEATURES,
    "NB_HREF",
)

historical_nb_final.fit_matrix(
    H_NB
)


historical_strict_final = HistoricalHumanReference(
    STRICT_CROSS_ACTION_FEATURES,
    "STRICT_HREF",
)

historical_strict_final.fit_matrix(
    H_STRICT
)


X_TEST_NB = (
    test[
        NO_BOOKKEEPING_FEATURES
    ]
    .to_numpy(float)
)


X_TEST_STRICT = (
    test[
        STRICT_CROSS_ACTION_FEATURES
    ]
    .to_numpy(float)
)


expected_nb_score = (
    historical_nb_final
    .score_matrix(
        X_TEST_NB
    )
)


expected_strict_score = (
    historical_strict_final
    .score_matrix(
        X_TEST_STRICT
    )
)


expected_fused_score = np.maximum(
    expected_nb_score
    /
    max(
        historical_nb_threshold,
        1e-12,
    ),

    expected_strict_score
    /
    max(
        historical_strict_threshold,
        1e-12,
    ),
)


actual = clean.score(
    test
)


nb_score_diff = np.max(
    np.abs(
        expected_nb_score
        -
        actual[
            "nb_href_score"
        ].to_numpy()
    )
)


strict_score_diff = np.max(
    np.abs(
        expected_strict_score
        -
        actual[
            "strict_href_score"
        ].to_numpy()
    )
)


fused_score_diff = np.max(
    np.abs(
        expected_fused_score
        -
        actual[
            "human_reference_score"
        ].to_numpy()
    )
)


print(
    "\nFinal NB score max difference:",
    nb_score_diff,
)

print(
    "Final Strict score max difference:",
    strict_score_diff,
)

print(
    "Final fused score max difference:",
    fused_score_diff,
)


TOL = 1e-12

assert nb_score_diff <= TOL
assert strict_score_diff <= TOL
assert fused_score_diff <= TOL


print()

print(
    "PASS: Human-reference calibration and scoring pipeline "
    "is exactly equivalent to the historical implementation."
)
