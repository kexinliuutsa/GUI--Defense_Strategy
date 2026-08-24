from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path.cwd()

OUT = (
    ROOT
    / "experiments/41j_stage1free_clean_audit/results"
)
OUT.mkdir(parents=True, exist_ok=True)

print("=" * 110)
print("EXP41J — STAGE1-FREE JOINT OOF CLEAN AUDIT")
print("=" * 110)

KEYS = [
    "group",
    "participant",
    "session_id",
]


# ============================================================
# 1. EXP41I TRUE HELD-OUT SESSION PREDICTIONS
#
# no_bookkeeping and strict_cross here are already generated
# under participant-disjoint outer folds.
# ============================================================

oof_path = (
    ROOT
    / "experiments/41i_joint_oof_clean_audit/results"
    / "joint_oof_session_predictions.csv"
)

pred = pd.read_csv(oof_path)

pred["session_id"] = (
    pred["session_id"].astype(str)
)

pred = pred[
    pred["group"].isin(
        [
            "Human",
            "Raw",
        ]
    )
].copy()

pred["matched_41c"] = (
    pred["matched_41c"].astype(bool)
)

pred["no_bookkeeping"] = (
    pred["no_bookkeeping"].astype(bool)
)

pred["strict_cross"] = (
    pred["strict_cross"].astype(bool)
)


# Only use exact EXP41C-matched population
matched = (
    pred[
        pred["matched_41c"]
    ]
    .copy()
    .reset_index(drop=True)
)


print("\nMatched clean sessions:")

print(
    matched["group"]
    .value_counts()
    .to_string()
)


# ============================================================
# 2. LOAD RAW ACTION METRICS FROM EXP41D
#
# IMPORTANT:
# Ignore EXP41D's precomputed fake_action_detect column.
#
# We recompute thresholds independently inside each outer fold.
# ============================================================

action_path = (
    ROOT
    / "experiments/41d_fake_action_head/results"
    / "raw_action_artifacts.csv"
)

actions = pd.read_csv(action_path)

actions["session_id"] = (
    actions["session_id"].astype(str)
)

actions = actions[
    actions["group"].isin(
        [
            "Human",
            "Raw",
        ]
    )
].copy()


required = {
    "group",
    "participant",
    "session_id",
    "path_length",
    "endpoint_path_ratio",
}

missing = (
    required
    -
    set(actions.columns)
)

if missing:
    raise RuntimeError(
        f"Missing EXP41D columns: {missing}"
    )


# ============================================================
# 3. PARTICIPANT-DISJOINT FAKE-ACTION HEAD
#
# For each EXP41I outer fold:
#
# - held-out participants never calibrate artifact thresholds
# - movement floor from TRAIN humans only
# - endpoint-ratio threshold from TRAIN humans only
# - session decision: >= 2 artifacts
# ============================================================

TARGET_ACTION_FPR = 0.005
SESSION_ARTIFACT_COUNT = 2

fake_rows = []
threshold_rows = []


for fold in sorted(
    matched["outer_fold"].unique()
):

    test_sessions = (
        matched[
            matched["outer_fold"]
            ==
            fold
        ]
        .copy()
    )

    test_participants = set(
        test_sessions[
            "participant"
        ].astype(str)
    )

    # ------------------------------------------
    # Training humans = everyone except held-out
    # participants from this outer fold.
    # ------------------------------------------

    train_human_actions = actions[
        (actions["group"] == "Human")
        &
        (~actions["participant"]
            .astype(str)
            .isin(test_participants))
    ].copy()


    positive_lengths = (
        train_human_actions.loc[
            train_human_actions[
                "path_length"
            ]
            >
            0,
            "path_length",
        ]
    )


    if len(positive_lengths) == 0:
        raise RuntimeError(
            f"Fold {fold}: no positive human path lengths."
        )


    movement_floor = float(
        np.quantile(
            positive_lengths,
            0.25,
        )
    )


    moving_human = (
        train_human_actions[
            train_human_actions[
                "path_length"
            ]
            >=
            movement_floor
        ]
    )


    ratio_threshold = float(
        np.quantile(
            moving_human[
                "endpoint_path_ratio"
            ],
            TARGET_ACTION_FPR,
            method="lower",
        )
    )


    threshold_rows.append({
        "outer_fold":
            int(fold),

        "movement_floor":
            movement_floor,

        "ratio_threshold":
            ratio_threshold,

        "n_train_human_actions":
            len(train_human_actions),

        "n_train_moving_human_actions":
            len(moving_human),
    })


    # ------------------------------------------
    # Score held-out participants
    # ------------------------------------------

    test_actions = actions[
        actions[
            "participant"
        ]
        .astype(str)
        .isin(
            test_participants
        )
    ].copy()


    test_actions[
        "fake_artifact"
    ] = (
        (
            test_actions[
                "path_length"
            ]
            >=
            movement_floor
        )
        &
        (
            test_actions[
                "endpoint_path_ratio"
            ]
            <=
            ratio_threshold
        )
    )


    session_fake = (
        test_actions
        .groupby(
            KEYS,
            as_index=False,
        )
        .agg(
            n_raw_actions=(
                "session_id",
                "size",
            ),

            artifact_count=(
                "fake_artifact",
                "sum",
            ),
        )
    )


    session_fake[
        "fake_action_oof"
    ] = (
        session_fake[
            "artifact_count"
        ]
        >=
        SESSION_ARTIFACT_COUNT
    )


    # Restrict to exact matched test sessions
    out = (
        test_sessions[
            [
                "outer_fold",
                *KEYS,
            ]
        ]
        .merge(
            session_fake,
            on=KEYS,
            how="left",
        )
    )


    out[
        "artifact_count"
    ] = (
        out[
            "artifact_count"
        ]
        .fillna(0)
        .astype(int)
    )

    out[
        "n_raw_actions"
    ] = (
        out[
            "n_raw_actions"
        ]
        .fillna(0)
        .astype(int)
    )

    out[
        "fake_action_oof"
    ] = (
        out[
            "fake_action_oof"
        ]
        .fillna(False)
        .astype(bool)
    )


    fake_rows.append(
        out
    )


fake_oof = pd.concat(
    fake_rows,
    ignore_index=True,
)


# ============================================================
# 4. MERGE THREE TRUE HELD-OUT HEADS
# ============================================================

joint = matched.merge(
    fake_oof[
        [
            "outer_fold",
            *KEYS,
            "artifact_count",
            "n_raw_actions",
            "fake_action_oof",
        ]
    ],
    on=[
        "outer_fold",
        *KEYS,
    ],
    how="left",
    validate="one_to_one",
)


joint[
    "fake_action_oof"
] = (
    joint[
        "fake_action_oof"
    ]
    .fillna(False)
    .astype(bool)
)


# ============================================================
# 5. STAGE1-FREE CANDIDATE
# ============================================================

joint[
    "nb_strict"
] = (
    joint[
        "no_bookkeeping"
    ]
    |
    joint[
        "strict_cross"
    ]
)


joint[
    "candidate"
] = (
    joint[
        "no_bookkeeping"
    ]
    |
    joint[
        "strict_cross"
    ]
    |
    joint[
        "fake_action_oof"
    ]
)


joint[
    "strict_incremental_after_nb"
] = (
    ~joint[
        "no_bookkeeping"
    ]
    &
    joint[
        "strict_cross"
    ]
)


joint[
    "fake_incremental_after_nb_strict"
] = (
    ~joint[
        "nb_strict"
    ]
    &
    joint[
        "fake_action_oof"
    ]
)


# ============================================================
# 6. WILSON CONFIDENCE INTERVAL
# ============================================================

def wilson(k, n, z=1.96):

    if n == 0:
        return np.nan, np.nan

    p = (
        k / n
    )

    denom = (
        1.0
        +
        z * z / n
    )

    center = (
        p
        +
        z * z
        /
        (2.0 * n)
    ) / denom

    half = (
        z
        *
        np.sqrt(
            (
                p * (1.0 - p)
                +
                z * z
                /
                (4.0 * n)
            )
            /
            n
        )
        /
        denom
    )

    return (
        max(
            0.0,
            center - half
        ),
        min(
            1.0,
            center + half
        ),
    )


# ============================================================
# 7. CLEAN OOF SUMMARY
# ============================================================

heads = [
    "no_bookkeeping",
    "strict_cross",
    "fake_action_oof",
    "nb_strict",
    "candidate",
]

rows = []


for head in heads:

    human = joint[
        joint["group"]
        ==
        "Human"
    ]

    raw = joint[
        joint["group"]
        ==
        "Raw"
    ]


    fp = int(
        human[
            head
        ].sum()
    )

    tp = int(
        raw[
            head
        ].sum()
    )


    fpr_low, fpr_high = wilson(
        fp,
        len(human),
    )

    recall_low, recall_high = wilson(
        tp,
        len(raw),
    )


    rows.append({
        "head":
            head,

        "human_fp":
            fp,

        "human_n":
            len(human),

        "human_fpr":
            float(
                human[
                    head
                ].mean()
            ),

        "human_fpr_ci_low":
            fpr_low,

        "human_fpr_ci_high":
            fpr_high,

        "raw_tp":
            tp,

        "raw_n":
            len(raw),

        "raw_recall":
            float(
                raw[
                    head
                ].mean()
            ),

        "raw_recall_ci_low":
            recall_low,

        "raw_recall_ci_high":
            recall_high,
    })


summary = pd.DataFrame(
    rows
)


print("\n" + "=" * 110)
print("STAGE1-FREE JOINT PARTICIPANT-DISJOINT OOF")
print("=" * 110)

print(
    summary.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.6f}",
    )
)


# ============================================================
# 8. INCREMENTAL ANALYSIS
# ============================================================

for group in [
    "Human",
    "Raw",
]:

    d = joint[
        joint[
            "group"
        ]
        ==
        group
    ]

    print(
        "\n",
        group,
        "incremental:",
    )

    print(
        "No-bookkeeping:",
        round(
            d[
                "no_bookkeeping"
            ].mean(),
            6,
        )
    )

    print(
        "+ Strict new:",
        round(
            d[
                "strict_incremental_after_nb"
            ].mean(),
            6,
        )
    )

    print(
        "+ Fake new:",
        round(
            d[
                "fake_incremental_after_nb_strict"
            ].mean(),
            6,
        )
    )

    print(
        "Final candidate:",
        round(
            d[
                "candidate"
            ].mean(),
            6,
        )
    )


# ============================================================
# 9. HUMAN PER-PARTICIPANT
# ============================================================

human_participant = (
    joint[
        joint[
            "group"
        ]
        ==
        "Human"
    ]
    .groupby(
        "participant",
        as_index=False,
    )
    .agg(
        n_sessions=(
            "session_id",
            "size",
        ),

        nb_fpr=(
            "no_bookkeeping",
            "mean",
        ),

        strict_fpr=(
            "strict_cross",
            "mean",
        ),

        fake_fpr=(
            "fake_action_oof",
            "mean",
        ),

        candidate_fpr=(
            "candidate",
            "mean",
        ),
    )
)


print("\n" + "=" * 110)
print("HUMAN HELD-OUT FPR BY PARTICIPANT")
print("=" * 110)

print(
    human_participant.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.6f}",
    )
)


# ============================================================
# 10. RAW PER-PARTICIPANT
# ============================================================

raw_participant = (
    joint[
        joint[
            "group"
        ]
        ==
        "Raw"
    ]
    .groupby(
        "participant",
        as_index=False,
    )
    .agg(
        n_sessions=(
            "session_id",
            "size",
        ),

        nb_recall=(
            "no_bookkeeping",
            "mean",
        ),

        strict_recall=(
            "strict_cross",
            "mean",
        ),

        fake_recall=(
            "fake_action_oof",
            "mean",
        ),

        candidate_recall=(
            "candidate",
            "mean",
        ),
    )
)


print("\n" + "=" * 110)
print("RAW AGENT HELD-OUT RECALL BY PARTICIPANT")
print("=" * 110)

print(
    raw_participant.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.6f}",
    )
)


# ============================================================
# 11. BRING IN UNSEEN AUTHOR ATTACK PERFORMANCE
#
# Attack side remains EXP41H:
# thresholds were derived from clean humans / clean agents,
# attacks themselves never trained the heads.
# ============================================================

attack_path = (
    ROOT
    / "experiments/41h_final_defense_assembly/results"
    / "combined_session_predictions.csv"
)

attack_df = pd.read_csv(
    attack_path
)


for c in [
    "no_bookkeeping",
    "strict_cross",
    "fake_action",
]:

    attack_df[c] = (
        attack_df[c]
        .astype(bool)
    )


attack_df[
    "stage1free_candidate"
] = (
    attack_df[
        "no_bookkeeping"
    ]
    |
    attack_df[
        "strict_cross"
    ]
    |
    attack_df[
        "fake_action"
    ]
)


ATTACK_GROUPS = [
    "Only Swipe Humanized",
    "Rot Tap Humanized",
    "Fake Rot Tap Humanized",
]


attack_rows = []


for group in ATTACK_GROUPS:

    d = attack_df[
        attack_df[
            "group"
        ]
        ==
        group
    ]

    attack_rows.append({
        "group":
            group,

        "n_sessions":
            len(d),

        "no_bookkeeping":
            d[
                "no_bookkeeping"
            ].mean(),

        "strict_cross":
            d[
                "strict_cross"
            ].mean(),

        "fake_action":
            d[
                "fake_action"
            ].mean(),

        "candidate_detection":
            d[
                "stage1free_candidate"
            ].mean(),

        "candidate_evasion":
            1.0
            -
            d[
                "stage1free_candidate"
            ].mean(),
    })


attack_summary = pd.DataFrame(
    attack_rows
)


print("\n" + "=" * 110)
print("UNSEEN AUTHOR ATTACK — STAGE1-FREE CANDIDATE")
print("=" * 110)

print(
    attack_summary.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.6f}",
    )
)


# ============================================================
# SAVE
# ============================================================

joint.to_csv(
    OUT /
    "stage1free_oof_predictions.csv",
    index=False,
)

summary.to_csv(
    OUT /
    "stage1free_clean_oof_summary.csv",
    index=False,
)

human_participant.to_csv(
    OUT /
    "human_participant_fpr.csv",
    index=False,
)

raw_participant.to_csv(
    OUT /
    "raw_participant_recall.csv",
    index=False,
)

pd.DataFrame(
    threshold_rows
).to_csv(
    OUT /
    "fake_action_outer_fold_thresholds.csv",
    index=False,
)

attack_summary.to_csv(
    OUT /
    "stage1free_attack_summary.csv",
    index=False,
)


print("\nSaved:", OUT)
print("EXP41J STATUS: COMPLETE")
