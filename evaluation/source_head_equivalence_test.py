"""
Equivalence test for the cleaned learned source heads.

Compares the historical implementation against:

    src.heads.source_heads.SourceHeadModels

The test covers:

- No-Bookkeeping feature selection and ordering
- Strict Cross-Action feature selection
- StandardScaler parameters
- LogisticRegression parameters
- predicted Agent probabilities
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features.session_features import (
    SESSION_FEATURES,
)

from src.features.cross_action_features import (
    STRICT_CROSS_ACTION_FEATURES,
)

from src.heads.source_heads import (
    NO_BOOKKEEPING_FEATURES,
    SourceHeadModels,
)


SEED = 20260826
TOL = 1e-12


# ============================================================
# Historical feature definitions
# ============================================================

HISTORICAL_BOOKKEEPING_FEATURES = {
    "log_n_actions",
    "log_session_span",
    "two_point_rate",
    "multi_point_rate",
    "zero_displacement_rate",
    "duration_points_corr",
    "distance_points_corr",
}


def historical_no_bookkeeping_feature_names(
    frame: pd.DataFrame,
) -> list[str]:

    output = []

    for column in frame.columns:

        if column in HISTORICAL_BOOKKEEPING_FEATURES:
            continue

        if column.startswith(
            "log_points_"
        ):
            continue

        if not pd.api.types.is_numeric_dtype(
            frame[column]
        ):
            continue

        output.append(
            column
        )

    return output


# ============================================================
# Historical model
# ============================================================

def historical_make_logistic_head():

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


# ============================================================
# Synthetic clean Human / Raw data
# ============================================================

rng = np.random.default_rng(
    SEED
)

n_human = 370
n_agent = 499
n_train = n_human + n_agent
n_test = 211


# Start from a common behavioral distribution.
X = rng.normal(
    size=(
        n_train,
        len(
            SESSION_FEATURES
        ),
    )
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


# Add moderate source-conditioned structure across different
# feature subsets so both heads learn non-trivial boundaries.
agent_mask = (
    labels == 1
)


for feature in [
    "direction_resultant",
    "duration_displacement_corr",
    "log_gap_mean",
    "direction_similarity_mean",
]:

    idx = SESSION_FEATURES.index(
        feature
    )

    X[
        agent_mask,
        idx
    ] += 0.8


for feature in [
    "log_duration_mean",
    "log_displacement_q90",
    "endpoint_ratio_std",
]:

    idx = SESSION_FEATURES.index(
        feature
    )

    X[
        agent_mask,
        idx
    ] -= 0.5


train = pd.DataFrame(
    X,
    columns=SESSION_FEATURES,
)


# Independent test set.
X_test = rng.normal(
    size=(
        n_test,
        len(
            SESSION_FEATURES
        ),
    )
)


test = pd.DataFrame(
    X_test,
    columns=SESSION_FEATURES,
)


# ============================================================
# 1. Feature-schema equivalence
# ============================================================

historical_nb_features = (
    historical_no_bookkeeping_feature_names(
        train
    )
)


print(
    "=" * 72
)

print(
    "SOURCE HEAD EQUIVALENCE"
)

print(
    "=" * 72
)


print(
    "Historical NB dimensions:",
    len(
        historical_nb_features
    ),
)

print(
    "Clean NB dimensions:",
    len(
        NO_BOOKKEEPING_FEATURES
    ),
)

print(
    "Strict dimensions:",
    len(
        STRICT_CROSS_ACTION_FEATURES
    ),
)


if historical_nb_features != NO_BOOKKEEPING_FEATURES:

    print(
        "\nHistorical NB features:"
    )

    print(
        historical_nb_features
    )

    print(
        "\nClean NB features:"
    )

    print(
        NO_BOOKKEEPING_FEATURES
    )

    raise AssertionError(
        "No-Bookkeeping feature order differs."
    )


assert len(
    historical_nb_features
) == 41

assert len(
    STRICT_CROSS_ACTION_FEATURES
) == 17


print(
    "Feature selection/order: PASS"
)


# ============================================================
# 2. Historical models
# ============================================================

historical_nb = (
    historical_make_logistic_head()
)

historical_strict = (
    historical_make_logistic_head()
)


historical_nb.fit(
    train[
        historical_nb_features
    ].to_numpy(
        dtype=float
    ),
    labels,
)


historical_strict.fit(
    train[
        STRICT_CROSS_ACTION_FEATURES
    ].to_numpy(
        dtype=float
    ),
    labels,
)


# ============================================================
# 3. Clean models
# ============================================================

clean = SourceHeadModels.fit(
    train,
    labels,
)


# ============================================================
# 4. Compare scaler parameters
# ============================================================

historical_nb_scaler = (
    historical_nb.named_steps[
        "scaler"
    ]
)

clean_nb_scaler = (
    clean.session_model.named_steps[
        "scaler"
    ]
)


historical_strict_scaler = (
    historical_strict.named_steps[
        "scaler"
    ]
)

clean_strict_scaler = (
    clean.strict_model.named_steps[
        "scaler"
    ]
)


nb_scaler_mean_diff = np.max(
    np.abs(
        historical_nb_scaler.mean_
        -
        clean_nb_scaler.mean_
    )
)


nb_scaler_scale_diff = np.max(
    np.abs(
        historical_nb_scaler.scale_
        -
        clean_nb_scaler.scale_
    )
)


strict_scaler_mean_diff = np.max(
    np.abs(
        historical_strict_scaler.mean_
        -
        clean_strict_scaler.mean_
    )
)


strict_scaler_scale_diff = np.max(
    np.abs(
        historical_strict_scaler.scale_
        -
        clean_strict_scaler.scale_
    )
)


# ============================================================
# 5. Compare LR parameters
# ============================================================

historical_nb_lr = (
    historical_nb.named_steps[
        "classifier"
    ]
)

clean_nb_lr = (
    clean.session_model.named_steps[
        "classifier"
    ]
)


historical_strict_lr = (
    historical_strict.named_steps[
        "classifier"
    ]
)

clean_strict_lr = (
    clean.strict_model.named_steps[
        "classifier"
    ]
)


nb_coef_diff = np.max(
    np.abs(
        historical_nb_lr.coef_
        -
        clean_nb_lr.coef_
    )
)


nb_intercept_diff = np.max(
    np.abs(
        historical_nb_lr.intercept_
        -
        clean_nb_lr.intercept_
    )
)


strict_coef_diff = np.max(
    np.abs(
        historical_strict_lr.coef_
        -
        clean_strict_lr.coef_
    )
)


strict_intercept_diff = np.max(
    np.abs(
        historical_strict_lr.intercept_
        -
        clean_strict_lr.intercept_
    )
)


# ============================================================
# 6. Compare test-set scores
# ============================================================

expected_nb_score = (
    historical_nb.predict_proba(
        test[
            historical_nb_features
        ].to_numpy(
            dtype=float
        )
    )[:, 1]
)


expected_strict_score = (
    historical_strict.predict_proba(
        test[
            STRICT_CROSS_ACTION_FEATURES
        ].to_numpy(
            dtype=float
        )
    )[:, 1]
)


actual = clean.score(
    test
)


actual_nb_score = (
    actual[
        "session_score"
    ].to_numpy()
)


actual_strict_score = (
    actual[
        "strict_score"
    ].to_numpy()
)


nb_score_diff = np.max(
    np.abs(
        expected_nb_score
        -
        actual_nb_score
    )
)


strict_score_diff = np.max(
    np.abs(
        expected_strict_score
        -
        actual_strict_score
    )
)


# ============================================================
# Report
# ============================================================

print()

print(
    "NB scaler mean max diff:",
    nb_scaler_mean_diff,
)

print(
    "NB scaler scale max diff:",
    nb_scaler_scale_diff,
)

print(
    "Strict scaler mean max diff:",
    strict_scaler_mean_diff,
)

print(
    "Strict scaler scale max diff:",
    strict_scaler_scale_diff,
)


print()

print(
    "NB LR coefficient max diff:",
    nb_coef_diff,
)

print(
    "NB LR intercept max diff:",
    nb_intercept_diff,
)

print(
    "Strict LR coefficient max diff:",
    strict_coef_diff,
)

print(
    "Strict LR intercept max diff:",
    strict_intercept_diff,
)


print()

print(
    "NB probability max diff:",
    nb_score_diff,
)

print(
    "Strict probability max diff:",
    strict_score_diff,
)


# ============================================================
# Assertions
# ============================================================

for value in [
    nb_scaler_mean_diff,
    nb_scaler_scale_diff,
    strict_scaler_mean_diff,
    strict_scaler_scale_diff,
    nb_coef_diff,
    nb_intercept_diff,
    strict_coef_diff,
    strict_intercept_diff,
    nb_score_diff,
    strict_score_diff,
]:

    assert value <= TOL


print()

print(
    "PASS: learned source-head feature selection, fitted parameters, "
    "and probability scores are numerically equivalent to the "
    "historical implementation within atol=1e-12."
)
