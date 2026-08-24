from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path.cwd()

OUT = (
    ROOT
    / "experiments/41l_fixed_threshold_generator_audit/results"
)
OUT.mkdir(parents=True, exist_ok=True)

print("=" * 115)
print("EXP41L — FIXED-THRESHOLD GENERATOR TRANSFER AUDIT")
print("=" * 115)


# ============================================================
# 1. LOAD EXP41K HELD-OUT SCORES
# ============================================================

pred = pd.read_csv(
    ROOT
    / "experiments/41k_leave_one_generator_out/results"
    / "heldout_predictions.csv"
)

thresholds = pd.read_csv(
    ROOT
    / "experiments/41k_leave_one_generator_out/results"
    / "fold_thresholds.csv"
)


# ============================================================
# 2. INSPECT THRESHOLD VARIATION
# ============================================================

print("\n" + "=" * 115)
print("LOO THRESHOLDS")
print("=" * 115)

print(
    thresholds.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)


# ============================================================
# 3. GLOBAL THRESHOLD
#
# Use median of thresholds produced across LOO training folds.
#
# IMPORTANT:
# This threshold is determined entirely from clean training
# experiments and does NOT use attack scores.
# ============================================================

GLOBAL_NB = float(
    thresholds[
        "nb_threshold"
    ].median()
)

GLOBAL_STRICT = float(
    thresholds[
        "strict_threshold"
    ].median()
)

print("\nGlobal fixed thresholds:")
print("NB:", GLOBAL_NB)
print("Strict:", GLOBAL_STRICT)


# ============================================================
# 4. APPLY SAME THRESHOLD TO EVERY HELD-OUT GENERATOR
# ============================================================

pred[
    "nb_fixed"
] = (
    pred[
        "p_nb"
    ]
    >=
    GLOBAL_NB
)

pred[
    "strict_fixed"
] = (
    pred[
        "p_strict"
    ]
    >=
    GLOBAL_STRICT
)

pred[
    "candidate_fixed"
] = (
    pred[
        "nb_fixed"
    ]
    |
    pred[
        "strict_fixed"
    ]
    |
    pred[
        "fake_action"
    ].astype(bool)
)


# ============================================================
# 5. SCORE DISTRIBUTIONS
# ============================================================

score_summary = (
    pred.groupby(
        [
            "heldout_family",
            "group",
        ],
        as_index=False,
    )
    .agg(
        n_sessions=(
            "session_id",
            "size",
        ),

        nb_mean=(
            "p_nb",
            "mean",
        ),

        nb_median=(
            "p_nb",
            "median",
        ),

        nb_q10=(
            "p_nb",
            lambda x:
                np.quantile(
                    x,
                    0.10,
                ),
        ),

        nb_q90=(
            "p_nb",
            lambda x:
                np.quantile(
                    x,
                    0.90,
                ),
        ),

        strict_mean=(
            "p_strict",
            "mean",
        ),

        strict_median=(
            "p_strict",
            "median",
        ),

        strict_q10=(
            "p_strict",
            lambda x:
                np.quantile(
                    x,
                    0.10,
                ),
        ),

        strict_q90=(
            "p_strict",
            lambda x:
                np.quantile(
                    x,
                    0.90,
                ),
        ),
    )
)


print("\n" + "=" * 115)
print("HELD-OUT SCORE DISTRIBUTIONS")
print("=" * 115)

print(
    score_summary.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.6f}",
    )
)


# ============================================================
# 6. FIXED-THRESHOLD RESULTS
# ============================================================

rows = []

ORDER = [
    "Raw",
    "Only Swipe Humanized",
    "Rot Tap Humanized",
    "Fake Rot Tap Humanized",
]


for family in [
    "GPT4o",
    "UI-TARS",
    "Claude",
    "CPM",
    "AutoGLM",
]:

    f = pred[
        pred[
            "heldout_family"
        ]
        ==
        family
    ]

    for group in ORDER:

        d = f[
            f[
                "group"
            ]
            ==
            group
        ]

        if len(d) == 0:
            continue

        rows.append({
            "heldout_family":
                family,

            "group":
                group,

            "n_sessions":
                len(d),

            "nb_fixed_detection":
                d[
                    "nb_fixed"
                ].mean(),

            "strict_fixed_detection":
                d[
                    "strict_fixed"
                ].mean(),

            "fake_action_detection":
                d[
                    "fake_action"
                ].astype(bool).mean(),

            "candidate_fixed_detection":
                d[
                    "candidate_fixed"
                ].mean(),
        })


result = pd.DataFrame(
    rows
)


print("\n" + "=" * 115)
print("FIXED-THRESHOLD LEAVE-ONE-GENERATOR-OUT")
print("=" * 115)

print(
    result.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.6f}",
    )
)


# ============================================================
# 7. AGGREGATE
# ============================================================

aggregate_rows = []

for group in [
    "Only Swipe Humanized",
    "Rot Tap Humanized",
    "Fake Rot Tap Humanized",
]:

    d = result[
        result[
            "group"
        ]
        ==
        group
    ]

    if len(d) == 0:
        continue

    weights = d[
        "n_sessions"
    ]

    aggregate_rows.append({
        "attack":
            group,

        "n_generators":
            d[
                "heldout_family"
            ].nunique(),

        "n_sessions":
            int(
                weights.sum()
            ),

        "nb_fixed":
            np.average(
                d[
                    "nb_fixed_detection"
                ],
                weights=weights,
            ),

        "strict_fixed":
            np.average(
                d[
                    "strict_fixed_detection"
                ],
                weights=weights,
            ),

        "candidate_fixed":
            np.average(
                d[
                    "candidate_fixed_detection"
                ],
                weights=weights,
            ),

        "worst_generator":
            d.loc[
                d[
                    "candidate_fixed_detection"
                ].idxmin(),
                "heldout_family",
            ],

        "worst_detection":
            d[
                "candidate_fixed_detection"
            ].min(),
    })


aggregate = pd.DataFrame(
    aggregate_rows
)


print("\n" + "=" * 115)
print("FIXED-THRESHOLD AGGREGATE")
print("=" * 115)

print(
    aggregate.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.6f}",
    )
)


# ============================================================
# 8. COMPARE ADAPTIVE PER-FOLD THRESHOLD VS GLOBAL FIXED
# ============================================================

old = pd.read_csv(
    ROOT
    / "experiments/41k_leave_one_generator_out/results"
    / "leave_one_generator_out_results.csv"
)


compare = old.merge(
    result,
    left_on=[
        "heldout_family",
        "test_group",
    ],
    right_on=[
        "heldout_family",
        "group",
    ],
    how="inner",
)


compare[
    "delta_fixed_minus_fold"
] = (
    compare[
        "candidate_fixed_detection"
    ]
    -
    compare[
        "candidate_detection"
    ]
)


print("\n" + "=" * 115)
print("PER-FOLD THRESHOLD VS GLOBAL FIXED THRESHOLD")
print("=" * 115)

print(
    compare[
        [
            "heldout_family",
            "group",
            "n_sessions_y",
            "candidate_detection",
            "candidate_fixed_detection",
            "delta_fixed_minus_fold",
        ]
    ].to_string(
        index=False,
        float_format=lambda x:
            f"{x:.6f}",
    )
)


# ============================================================
# SAVE
# ============================================================

pred.to_csv(
    OUT /
    "fixed_threshold_predictions.csv",
    index=False,
)

score_summary.to_csv(
    OUT /
    "score_distribution_summary.csv",
    index=False,
)

result.to_csv(
    OUT /
    "fixed_threshold_results.csv",
    index=False,
)

aggregate.to_csv(
    OUT /
    "fixed_threshold_aggregate.csv",
    index=False,
)

compare.to_csv(
    OUT /
    "threshold_comparison.csv",
    index=False,
)


print("\nSaved:", OUT)
print("EXP41L STATUS: COMPLETE")
