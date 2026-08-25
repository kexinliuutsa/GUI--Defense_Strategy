"""
Unseen-generator validation for the frozen cross-scale defense.

Protocol:
- Leave one Raw-agent generator family entirely out of source-head
  training and calibration.
- Fit Human-reference models using Human data only.
- Evaluate the held-out generator under Long-Tap humanization.
- Do not tune thresholds using held-out attack data.

This script reproduces the unseen-generator robustness analysis reported
in results/unseen_generator_results.csv.
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

warnings.filterwarnings("ignore")

ROOT = Path.cwd()

OUT = (
    ROOT
    / "experiments/41k_leave_one_generator_out/results"
)
OUT.mkdir(parents=True, exist_ok=True)

print("=" * 115)
print("EXP41K — LEAVE-ONE-AGENT-FAMILY-OUT DEFENSE")
print("=" * 115)


# ============================================================
# 1. LOAD SESSION FEATURES
# ============================================================

p = (
    ROOT
    / "experiments/41f_session_structure_defense/results"
    / "session_features_and_scores.csv"
)

df = pd.read_csv(p)

df["session_id"] = (
    df["session_id"]
    .astype(str)
)


# ============================================================
# 2. MATCH EXACT EXP41C SESSION POPULATION
# ============================================================

p41c = (
    ROOT
    / "experiments/41c_original_humanization_defense/results"
    / "all_swipe_scores.csv"
)

s41c = pd.read_csv(p41c)

s41c["session_id"] = (
    s41c["session_id"]
    .astype(str)
)

KEYS = [
    "group",
    "participant",
    "session_id",
]

matched = (
    s41c[
        KEYS
    ]
    .drop_duplicates()
)

matched["matched_41c"] = True

df = df.merge(
    matched,
    on=KEYS,
    how="left",
)

df["matched_41c"] = (
    df["matched_41c"]
    .fillna(False)
    .astype(bool)
)

# Use same benchmark population as 41C/H/J
df = (
    df[
        df["matched_41c"]
    ]
    .copy()
    .reset_index(drop=True)
)


# ============================================================
# 3. CANONICAL AGENT FAMILY
# ============================================================

def get_family(name):

    s = str(name).lower()

    if (
        "ui-tars" in s
        or
        "ui_tars" in s
        or
        "uitars" in s
    ):
        return "UI-TARS"

    if (
        "gpt4o" in s
        or
        "gpt_4o" in s
        or
        "gpt-4o" in s
    ):
        return "GPT4o"

    if "claude" in s:
        return "Claude"

    if "cpm" in s:
        return "CPM"

    if "autoglm" in s:
        return "AutoGLM"

    if str(name).startswith("user"):
        return "Human"

    return "Unknown"


df["family"] = (
    df["participant"]
    .map(get_family)
)


print("\nFamily mapping:")

print(
    df[
        [
            "group",
            "participant",
            "family",
        ]
    ]
    .drop_duplicates()
    .sort_values(
        [
            "family",
            "group",
        ]
    )
    .to_string(index=False)
)


if (
    df[
        "family"
    ]
    .eq("Unknown")
    .any()
):
    print("\nWARNING: Unknown family exists!")

    print(
        df.loc[
            df.family == "Unknown",
            [
                "group",
                "participant",
            ]
        ]
        .drop_duplicates()
        .to_string(index=False)
    )


# ============================================================
# 4. FEATURE SETS — EXACT EXP41G DEFINITIONS
# ============================================================

NON_FEATURE = {
    "group",
    "participant",
    "session_id",
    "p_session_agent",
    "session_structure_detect",
    "matched_41c",
    "family",
}

ALL_FEATURES = [
    c
    for c in df.columns
    if c not in NON_FEATURE
]


BOOKKEEPING = {
    "log_n_actions",
    "log_session_span",
    "two_point_rate",
    "multi_point_rate",
    "zero_displacement_rate",
    "duration_points_corr",
    "distance_points_corr",
}

for c in ALL_FEATURES:

    if c.startswith(
        "log_points_"
    ):
        BOOKKEEPING.add(c)


NO_BOOKKEEPING = [
    c
    for c in ALL_FEATURES
    if c not in BOOKKEEPING
]


STRICT_CROSS = [
    c
    for c in [
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
    if c in ALL_FEATURES
]


print("\nFeature dimensions:")
print("No-bookkeeping:", len(NO_BOOKKEEPING))
print("Strict-cross:", len(STRICT_CROSS))


# ============================================================
# 5. LOAD FAKE-ACTION HEAD
#
# This head is human-calibrated and does not train on agent
# families, so using its frozen EXP41D decision does not leak
# held-out generator information.
# ============================================================

fake_path = (
    ROOT
    / "experiments/41d_fake_action_head/results"
    / "session_level_details.csv"
)

fake = pd.read_csv(fake_path)

fake["session_id"] = (
    fake["session_id"]
    .astype(str)
)

if "two_or_more" in fake.columns:

    fake["fake_action"] = (
        fake["two_or_more"]
        .astype(bool)
    )

else:

    fake["fake_action"] = (
        fake["artifact_count"]
        >=
        2
    )

fake = fake[
    KEYS
    +
    [
        "fake_action",
    ]
]


df = df.merge(
    fake,
    on=KEYS,
    how="left",
)

df["fake_action"] = (
    df["fake_action"]
    .fillna(False)
    .astype(bool)
)


# ============================================================
# 6. MODEL
# ============================================================

def make_model():

    return make_pipeline(
        StandardScaler(),

        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=5000,
            random_state=42,
        ),
    )


TARGET_FPR = 0.01


# ============================================================
# 7. OOF CALIBRATION ON TRAINING GENERATORS ONLY
#
# Humanized data NEVER enters training/calibration.
# Held-out Raw generator NEVER enters training/calibration.
# ============================================================

def train_and_calibrate(
    train_clean,
    features,
):

    X = (
        train_clean[
            features
        ]
        .to_numpy(float)
    )

    y = (
        train_clean[
            "group"
        ]
        .eq("Raw")
        .astype(int)
        .to_numpy()
    )

    groups = (
        train_clean[
            "participant"
        ]
        .astype(str)
        .to_numpy()
    )

    # After one agent family is removed,
    # four raw-agent participants remain.
    cv = StratifiedGroupKFold(
        n_splits=4,
        shuffle=True,
        random_state=42,
    )

    oof = np.full(
        len(train_clean),
        np.nan,
    )


    for fold, (
        tr,
        te,
    ) in enumerate(
        cv.split(
            X,
            y,
            groups,
        ),
        1,
    ):

        model = make_model()

        model.fit(
            X[tr],
            y[tr],
        )

        oof[te] = (
            model
            .predict_proba(
                X[te]
            )[:, 1]
        )


    if np.isnan(oof).any():
        raise RuntimeError(
            "Incomplete OOF calibration."
        )


    human_scores = oof[
        y == 0
    ]

    threshold = float(
        np.quantile(
            human_scores,
            1.0 - TARGET_FPR,
            method="higher",
        )
    )


    oof_pred = (
        oof
        >=
        threshold
    )

    oof_human_fpr = float(
        oof_pred[
            y == 0
        ].mean()
    )

    oof_agent_recall = float(
        oof_pred[
            y == 1
        ].mean()
    )


    final_model = make_model()

    final_model.fit(
        X,
        y,
    )


    return (
        final_model,
        threshold,
        oof_human_fpr,
        oof_agent_recall,
    )


# ============================================================
# 8. LEAVE ONE GENERATOR OUT
# ============================================================

FAMILIES = [
    "GPT4o",
    "UI-TARS",
    "Claude",
    "CPM",
    "AutoGLM",
]


result_rows = []
threshold_rows = []
prediction_rows = []


for heldout in FAMILIES:

    print("\n" + "=" * 115)
    print("HELD-OUT GENERATOR:", heldout)
    print("=" * 115)


    # --------------------------------------------------------
    # TRAIN:
    # Humans + Raw agents EXCEPT held-out generator
    # --------------------------------------------------------

    train_clean = df[
        (
            df["group"] == "Human"
        )
        |
        (
            (
                df["group"] == "Raw"
            )
            &
            (
                df["family"] != heldout
            )
        )
    ].copy()


    # No humanized data can enter training.
    if (
        ~train_clean[
            "group"
        ]
        .isin(
            [
                "Human",
                "Raw",
            ]
        )
    ).any():

        raise RuntimeError(
            "Humanized attack leaked into training!"
        )


    print(
        "Train raw families:",
        sorted(
            train_clean.loc[
                train_clean.group == "Raw",
                "family",
            ].unique()
        )
    )


    # --------------------------------------------------------
    # Train two session heads
    # --------------------------------------------------------

    (
        nb_model,
        nb_threshold,
        nb_oof_fpr,
        nb_oof_recall,
    ) = train_and_calibrate(
        train_clean,
        NO_BOOKKEEPING,
    )


    (
        strict_model,
        strict_threshold,
        strict_oof_fpr,
        strict_oof_recall,
    ) = train_and_calibrate(
        train_clean,
        STRICT_CROSS,
    )


    print(
        "NB threshold:",
        round(nb_threshold, 6),
        "train-OOF human FPR:",
        round(nb_oof_fpr, 6),
        "train-OOF agent recall:",
        round(nb_oof_recall, 6),
    )

    print(
        "Strict threshold:",
        round(strict_threshold, 6),
        "train-OOF human FPR:",
        round(strict_oof_fpr, 6),
        "train-OOF agent recall:",
        round(strict_oof_recall, 6),
    )


    threshold_rows.append({
        "heldout_family":
            heldout,

        "nb_threshold":
            nb_threshold,

        "nb_train_oof_human_fpr":
            nb_oof_fpr,

        "nb_train_oof_agent_recall":
            nb_oof_recall,

        "strict_threshold":
            strict_threshold,

        "strict_train_oof_human_fpr":
            strict_oof_fpr,

        "strict_train_oof_agent_recall":
            strict_oof_recall,
    })


    # --------------------------------------------------------
    # TEST:
    #
    # held-out Raw generator
    # +
    # every available humanized variant from same generator
    # --------------------------------------------------------

    test = df[
        df["family"]
        ==
        heldout
    ].copy()


    if len(test) == 0:

        print(
            "No sessions for",
            heldout
        )

        continue


    test[
        "p_nb"
    ] = (
        nb_model
        .predict_proba(
            test[
                NO_BOOKKEEPING
            ]
            .to_numpy(float)
        )[:, 1]
    )


    test[
        "p_strict"
    ] = (
        strict_model
        .predict_proba(
            test[
                STRICT_CROSS
            ]
            .to_numpy(float)
        )[:, 1]
    )


    test[
        "nb_detect"
    ] = (
        test[
            "p_nb"
        ]
        >=
        nb_threshold
    )


    test[
        "strict_detect"
    ] = (
        test[
            "p_strict"
        ]
        >=
        strict_threshold
    )


    test[
        "candidate"
    ] = (
        test[
            "nb_detect"
        ]
        |
        test[
            "strict_detect"
        ]
        |
        test[
            "fake_action"
        ]
    )


    # --------------------------------------------------------
    # Save individual predictions
    # --------------------------------------------------------

    for r in test.itertuples(
        index=False
    ):

        prediction_rows.append({
            "heldout_family":
                heldout,

            "group":
                r.group,

            "participant":
                r.participant,

            "session_id":
                r.session_id,

            "p_nb":
                float(r.p_nb),

            "p_strict":
                float(r.p_strict),

            "nb_detect":
                bool(r.nb_detect),

            "strict_detect":
                bool(r.strict_detect),

            "fake_action":
                bool(r.fake_action),

            "candidate":
                bool(r.candidate),
        })


    # --------------------------------------------------------
    # Group summary
    # --------------------------------------------------------

    group_order = [
        "Raw",
        "Only Swipe Humanized",
        "Rot Tap Humanized",
        "Fake Rot Tap Humanized",
    ]


    for group in group_order:

        g = test[
            test["group"]
            ==
            group
        ]

        if len(g) == 0:
            continue


        result_rows.append({
            "heldout_family":
                heldout,

            "test_group":
                group,

            "n_sessions":
                len(g),

            "nb_detection":
                float(
                    g[
                        "nb_detect"
                    ].mean()
                ),

            "strict_detection":
                float(
                    g[
                        "strict_detect"
                    ].mean()
                ),

            "fake_action_detection":
                float(
                    g[
                        "fake_action"
                    ].mean()
                ),

            "candidate_detection":
                float(
                    g[
                        "candidate"
                    ].mean()
                ),

            "candidate_evasion":
                (
                    float(
                        1.0
                        -
                        g[
                            "candidate"
                        ].mean()
                    )
                    if group != "Raw"
                    else np.nan
                ),
        })


# ============================================================
# 9. RESULTS
# ============================================================

results = pd.DataFrame(
    result_rows
)

thresholds = pd.DataFrame(
    threshold_rows
)

predictions = pd.DataFrame(
    prediction_rows
)


print("\n" + "=" * 115)
print("LEAVE-ONE-GENERATOR-OUT RESULTS")
print("=" * 115)

print(
    results.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.6f}",
    )
)


# ============================================================
# 10. HELD-OUT RAW GENERATOR GENERALIZATION
# ============================================================

raw_results = results[
    results[
        "test_group"
    ]
    ==
    "Raw"
].copy()


print("\n" + "=" * 115)
print("HELD-OUT RAW GENERATOR RECALL")
print("=" * 115)

print(
    raw_results[
        [
            "heldout_family",
            "n_sessions",
            "nb_detection",
            "strict_detection",
            "candidate_detection",
        ]
    ].to_string(
        index=False,
        float_format=lambda x:
            f"{x:.6f}",
    )
)


# ============================================================
# 11. HELD-OUT HUMANIZATION
# ============================================================

attack = results[
    results[
        "test_group"
    ]
    !=
    "Raw"
].copy()


print("\n" + "=" * 115)
print("UNSEEN GENERATOR + UNSEEN HUMANIZATION")
print("=" * 115)

print(
    attack.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.6f}",
    )
)


# ============================================================
# 12. AGGREGATE BY ATTACK FAMILY
# Weighted by number of sessions.
# ============================================================

aggregate_rows = []


for attack_name in [
    "Only Swipe Humanized",
    "Rot Tap Humanized",
    "Fake Rot Tap Humanized",
]:

    x = attack[
        attack[
            "test_group"
        ]
        ==
        attack_name
    ]

    if len(x) == 0:
        continue


    total_n = int(
        x[
            "n_sessions"
        ].sum()
    )


    def weighted(col):

        return float(
            np.average(
                x[col],
                weights=x[
                    "n_sessions"
                ],
            )
        )


    aggregate_rows.append({
        "attack":
            attack_name,

        "n_generators":
            x[
                "heldout_family"
            ].nunique(),

        "n_sessions":
            total_n,

        "nb_detection":
            weighted(
                "nb_detection"
            ),

        "strict_detection":
            weighted(
                "strict_detection"
            ),

        "fake_action_detection":
            weighted(
                "fake_action_detection"
            ),

        "candidate_detection":
            weighted(
                "candidate_detection"
            ),

        "candidate_evasion":
            1.0
            -
            weighted(
                "candidate_detection"
            ),
    })


aggregate = pd.DataFrame(
    aggregate_rows
)


print("\n" + "=" * 115)
print("AGGREGATE HELD-OUT-GENERATOR ATTACK DETECTION")
print("=" * 115)

print(
    aggregate.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.6f}",
    )
)


# ============================================================
# 13. MINIMUM GENERATOR PERFORMANCE
#
# Important reviewer-safe statistic:
# don't only report weighted average.
# ============================================================

worst_rows = []


for attack_name in [
    "Only Swipe Humanized",
    "Rot Tap Humanized",
    "Fake Rot Tap Humanized",
]:

    x = attack[
        attack[
            "test_group"
        ]
        ==
        attack_name
    ]

    if len(x) == 0:
        continue


    worst = x.loc[
        x[
            "candidate_detection"
        ].idxmin()
    ]


    worst_rows.append({
        "attack":
            attack_name,

        "worst_generator":
            worst[
                "heldout_family"
            ],

        "n_sessions":
            int(
                worst[
                    "n_sessions"
                ]
            ),

        "worst_candidate_detection":
            float(
                worst[
                    "candidate_detection"
                ]
            ),
    })


worst_table = pd.DataFrame(
    worst_rows
)


print("\n" + "=" * 115)
print("WORST HELD-OUT GENERATOR")
print("=" * 115)

print(
    worst_table.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.6f}",
    )
)


# ============================================================
# SAVE
# ============================================================

results.to_csv(
    OUT /
    "leave_one_generator_out_results.csv",
    index=False,
)

thresholds.to_csv(
    OUT /
    "fold_thresholds.csv",
    index=False,
)

predictions.to_csv(
    OUT /
    "heldout_predictions.csv",
    index=False,
)

aggregate.to_csv(
    OUT /
    "aggregate_attack_detection.csv",
    index=False,
)

worst_table.to_csv(
    OUT /
    "worst_generator_detection.csv",
    index=False,
)


print("\nIMPORTANT:")
print(
    "This experiment tests unseen-generator transfer."
)

print(
    "Do NOT use its train-OOF human FPR as the final clean-FPR headline."
)

print(
    "Final clean FPR remains EXP41J participant-disjoint candidate = 3/280."
)

print("\nSaved:", OUT)
print("EXP41K STATUS: COMPLETE")
