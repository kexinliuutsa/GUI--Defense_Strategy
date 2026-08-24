from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))

OUT = ROOT / "experiments/41f_session_structure_defense/results"
OUT.mkdir(parents=True, exist_ok=True)

from analysis.lib.gesture_log_reader_utils import (
    ranged_modified_generator_with_session_timestamp,
)

print("=" * 110)
print("EXP41F — CROSS-GESTURE SESSION STRUCTURE DEFENSE")
print("=" * 110)


# ============================================================
# AUTHOR GROUPS
# ============================================================

table = pd.read_excel(
    ROOT / "Formated_Data_Renamed.xlsx",
    dtype=str,
)

cols = list(table.columns)

from operator import itemgetter

GROUPS = {
    "Human":
        list(itemgetter(
            2, 3, 4, 33, 34, 35, 36
        )(cols)),

    "Raw":
        list(itemgetter(
            12, 14, 16, 19, 37
        )(cols)),

    "Only Swipe Humanized":
        list(itemgetter(
            23, 25
        )(cols)),

    "Rot Tap Humanized":
        list(itemgetter(
            5, 7, 9, 20, 39
        )(cols)),

    "Fake Rot Tap Humanized":
        list(itemgetter(
            28, 30, 41
        )(cols)),
}


# ============================================================
# HELPERS
# ============================================================

def safe_stats(x, prefix):

    x = np.asarray(x, dtype=float)

    x = x[
        np.isfinite(x)
    ]

    if len(x) == 0:

        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_median": 0.0,
            f"{prefix}_q10": 0.0,
            f"{prefix}_q90": 0.0,
            f"{prefix}_cv": 0.0,
        }

    mean = float(
        np.mean(x)
    )

    std = float(
        np.std(x)
    )

    return {
        f"{prefix}_mean":
            mean,

        f"{prefix}_std":
            std,

        f"{prefix}_median":
            float(np.median(x)),

        f"{prefix}_q10":
            float(np.quantile(x, 0.10)),

        f"{prefix}_q90":
            float(np.quantile(x, 0.90)),

        f"{prefix}_cv":
            float(
                std /
                (abs(mean) + 1e-9)
            ),
    }


def correlation(a, b):

    a = np.asarray(
        a,
        dtype=float,
    )

    b = np.asarray(
        b,
        dtype=float,
    )

    good = (
        np.isfinite(a)
        &
        np.isfinite(b)
    )

    a = a[good]
    b = b[good]

    if len(a) < 3:
        return 0.0

    if (
        np.std(a) <= 1e-12
        or
        np.std(b) <= 1e-12
    ):
        return 0.0

    return float(
        np.corrcoef(
            a,
            b,
        )[0, 1]
    )


def session_features(
    gestures,
):

    durations = []
    n_points = []
    displacement = []
    path_length = []
    endpoint_ratio = []
    starts = []
    ends = []
    start_times = []
    end_times = []
    directions = []

    for g in gestures:

        if g is None or len(g) == 0:
            continue

        pts = np.asarray(
            [
                [float(e.x), float(e.y)]
                for e in g
            ],
            dtype=float,
        )

        ts = np.asarray(
            [
                float(e.timestamp_us)
                for e in g
            ],
            dtype=float,
        )

        start = pts[0]
        end = pts[-1]

        disp_vec = (
            end -
            start
        )

        disp = float(
            np.linalg.norm(
                disp_vec
            )
        )

        if len(pts) >= 2:

            plen = float(
                np.linalg.norm(
                    np.diff(
                        pts,
                        axis=0,
                    ),
                    axis=1,
                ).sum()
            )

        else:

            plen = 0.0

        dur = float(
            max(
                0.0,
                ts[-1] - ts[0],
            )
        )

        ratio = (
            disp /
            plen
            if plen > 1e-12
            else 1.0
        )

        durations.append(
            dur
        )

        n_points.append(
            len(g)
        )

        displacement.append(
            disp
        )

        path_length.append(
            plen
        )

        endpoint_ratio.append(
            ratio
        )

        starts.append(
            start
        )

        ends.append(
            end
        )

        start_times.append(
            float(ts[0])
        )

        end_times.append(
            float(ts[-1])
        )

        if disp > 1e-6:

            directions.append(
                disp_vec /
                disp
            )


    if len(durations) == 0:
        return None


    durations = np.asarray(
        durations,
        dtype=float,
    )

    n_points = np.asarray(
        n_points,
        dtype=float,
    )

    displacement = np.asarray(
        displacement,
        dtype=float,
    )

    path_length = np.asarray(
        path_length,
        dtype=float,
    )

    endpoint_ratio = np.asarray(
        endpoint_ratio,
        dtype=float,
    )

    start_times = np.asarray(
        start_times,
        dtype=float,
    )

    end_times = np.asarray(
        end_times,
        dtype=float,
    )


    # --------------------------------------------------------
    # Cross-action gaps
    # --------------------------------------------------------

    gaps = []

    negative_gap = []

    for i in range(
        1,
        len(start_times),
    ):

        raw_gap = (
            start_times[i]
            -
            end_times[i - 1]
        )

        negative_gap.append(
            float(
                raw_gap < 0
            )
        )

        gaps.append(
            max(
                0.0,
                raw_gap,
            )
        )


    # --------------------------------------------------------
    # Direction concentration
    #
    # 0 = directions spread around
    # 1 = almost same direction
    # --------------------------------------------------------

    if len(directions) > 0:

        direction_array = np.asarray(
            directions,
            dtype=float,
        )

        mean_vec = np.mean(
            direction_array,
            axis=0,
        )

        direction_resultant = float(
            np.linalg.norm(
                mean_vec
            )
        )

    else:

        direction_resultant = 0.0


    # Consecutive directional similarity
    direction_similarity = []

    if len(directions) >= 2:

        d = np.asarray(
            directions
        )

        for i in range(
            len(d) - 1
        ):

            direction_similarity.append(
                float(
                    np.clip(
                        np.dot(
                            d[i],
                            d[i + 1],
                        ),
                        -1.0,
                        1.0,
                    )
                )
            )


    # --------------------------------------------------------
    # Coordinate reuse
    #
    # Coarsen to 20-pixel cells.
    # --------------------------------------------------------

    def unique_cell_ratio(
        coords,
        cell=20.0,
    ):

        if len(coords) == 0:
            return 0.0

        arr = np.asarray(
            coords,
            dtype=float,
        )

        q = np.round(
            arr /
            cell
        ).astype(int)

        unique = len(
            {
                tuple(x)
                for x in q
            }
        )

        return float(
            unique /
            len(q)
        )


    # --------------------------------------------------------
    # Session span
    # --------------------------------------------------------

    session_span = float(
        max(
            0.0,
            np.max(end_times)
            -
            np.min(start_times),
        )
    )


    feat = {
        "log_n_actions":
            float(
                np.log1p(
                    len(durations)
                )
            ),

        "log_session_span":
            float(
                np.log1p(
                    session_span
                )
            ),

        "zero_displacement_rate":
            float(
                np.mean(
                    displacement
                    <=
                    1.0
                )
            ),

        "two_point_rate":
            float(
                np.mean(
                    n_points
                    <=
                    2
                )
            ),

        "multi_point_rate":
            float(
                np.mean(
                    n_points
                    >
                    2
                )
            ),

        "direction_resultant":
            direction_resultant,

        "start_unique_cell_ratio":
            unique_cell_ratio(
                starts
            ),

        "end_unique_cell_ratio":
            unique_cell_ratio(
                ends
            ),

        "negative_gap_rate":
            (
                float(
                    np.mean(
                        negative_gap
                    )
                )
                if len(negative_gap)
                else 0.0
            ),

        "duration_displacement_corr":
            correlation(
                np.log1p(
                    durations
                ),
                np.log1p(
                    displacement
                ),
            ),

        "duration_points_corr":
            correlation(
                np.log1p(
                    durations
                ),
                np.log1p(
                    n_points
                ),
            ),

        "distance_points_corr":
            correlation(
                np.log1p(
                    displacement
                ),
                np.log1p(
                    n_points
                ),
            ),
    }


    feat.update(
        safe_stats(
            np.log1p(
                durations
            ),
            "log_duration",
        )
    )

    feat.update(
        safe_stats(
            np.log1p(
                n_points
            ),
            "log_points",
        )
    )

    feat.update(
        safe_stats(
            np.log1p(
                displacement
            ),
            "log_displacement",
        )
    )

    feat.update(
        safe_stats(
            np.log1p(
                path_length
            ),
            "log_path_length",
        )
    )

    feat.update(
        safe_stats(
            endpoint_ratio,
            "endpoint_ratio",
        )
    )

    feat.update(
        safe_stats(
            np.log1p(
                gaps
            ),
            "log_gap",
        )
    )

    feat.update(
        safe_stats(
            direction_similarity,
            "direction_similarity",
        )
    )


    return {
        k:
            float(
                np.nan_to_num(
                    v,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
            )
        for k, v
        in feat.items()
    }


# ============================================================
# LOAD RAW SESSIONS
#
# IMPORTANT:
# no keep_swipe.
# Session head observes action composition and cross-action
# structure before gesture filtering.
# ============================================================

rows = []


print("\nLoading sessions...")


for group, participants in GROUPS.items():

    for participant in participants:

        sessions = (
            ranged_modified_generator_with_session_timestamp(
                formated_data_timestamps=table,
                participants=[participant],
                filtering_and_modification_function=None,
                index_range=None,
            )
        )

        count = 0

        for session_id, gestures in sessions:

            feat = session_features(
                gestures
            )

            if feat is None:
                continue

            rows.append({
                "group":
                    group,

                "participant":
                    participant,

                "session_id":
                    str(session_id),

                **feat,
            })

            count += 1


        print(
            f"{group:25s}",
            f"{participant:45s}",
            f"sessions={count}"
        )


df = pd.DataFrame(
    rows
)


META = {
    "group",
    "participant",
    "session_id",
}

FEATURES = [
    c
    for c in df.columns
    if c not in META
]


print(
    "\nTotal sessions:",
    len(df),
)

print(
    "Session feature count:",
    len(FEATURES),
)


# ============================================================
# CLEAN TRAINING
#
# Only Human + Raw.
# Humanized groups are completely unseen.
# ============================================================

clean = df[
    df.group.isin(
        [
            "Human",
            "Raw",
        ]
    )
].copy()


X = clean[
    FEATURES
].to_numpy(
    float
)

y = (
    clean.group
    .eq("Raw")
    .astype(int)
    .to_numpy()
)

groups = (
    clean.participant
    .astype(str)
    .to_numpy()
)


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


# ============================================================
# PARTICIPANT-DISJOINT OOF
# ============================================================

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

oof = np.full(
    len(clean),
    np.nan,
)


for fold, (
    train_idx,
    test_idx,
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
        X[train_idx],
        y[train_idx],
    )

    oof[
        test_idx
    ] = (
        model
        .predict_proba(
            X[test_idx]
        )[:, 1]
    )


valid = ~np.isnan(
    oof
)


auc = roc_auc_score(
    y[valid],
    oof[valid],
)


default_pred = (
    oof
    >=
    0.5
)


default_recall = float(
    default_pred[
        (y == 1)
        &
        valid
    ].mean()
)

default_fpr = float(
    default_pred[
        (y == 0)
        &
        valid
    ].mean()
)


# ============================================================
# CALIBRATE SESSION HEAD
#
# 1% OOF human session FPR budget
# ============================================================

TARGET_SESSION_FPR = 0.01


human_scores = oof[
    (y == 0)
    &
    valid
]


threshold = float(
    np.quantile(
        human_scores,
        1.0
        -
        TARGET_SESSION_FPR,
        method="higher",
    )
)


cal_pred = (
    oof
    >=
    threshold
)


cal_recall = float(
    cal_pred[
        (y == 1)
        &
        valid
    ].mean()
)


cal_fpr = float(
    cal_pred[
        (y == 0)
        &
        valid
    ].mean()
)


print("\n" + "=" * 110)
print("CLEAN PARTICIPANT-DISJOINT SESSION HEAD")
print("=" * 110)

print(
    "AUC:",
    round(
        auc,
        6,
    )
)

print(
    "Default recall:",
    round(
        default_recall,
        6,
    )
)

print(
    "Default human FPR:",
    round(
        default_fpr,
        6,
    )
)

print(
    "Calibrated threshold:",
    round(
        threshold,
        6,
    )
)

print(
    "Calibrated recall:",
    round(
        cal_recall,
        6,
    )
)

print(
    "Calibrated human FPR:",
    round(
        cal_fpr,
        6,
    )
)


# ============================================================
# FINAL MODEL
# ============================================================

final_model = make_model()

final_model.fit(
    X,
    y,
)


df[
    "p_session_agent"
] = (
    final_model
    .predict_proba(
        df[
            FEATURES
        ].to_numpy(float)
    )[:, 1]
)


df[
    "session_structure_detect"
] = (
    df[
        "p_session_agent"
    ]
    >=
    threshold
)


# ============================================================
# STAGE-1 SESSION DETECTION FROM EXP41C
# ============================================================

stage1_path = (
    ROOT
    / "experiments/41c_original_humanization_defense/results"
    / "all_swipe_scores.csv"
)


stage1_swipes = pd.read_csv(
    stage1_path
)


stage1_swipes[
    "session_id"
] = (
    stage1_swipes[
        "session_id"
    ].astype(str)
)


stage1_session = (
    stage1_swipes
    .groupby(
        [
            "group",
            "participant",
            "session_id",
        ],
        as_index=False,
    )
    .agg(
        stage1_detect=(
            "late_or_calibrated_detect",
            "max",
        )
    )
)


# ============================================================
# FAKE-ACTION SPECIALIZED HEAD FROM EXP41D
# ============================================================

fake_path = (
    ROOT
    / "experiments/41d_fake_action_head/results"
    / "session_level_details.csv"
)


if fake_path.exists():

    fake = pd.read_csv(
        fake_path
    )

    fake[
        "session_id"
    ] = (
        fake[
            "session_id"
        ].astype(str)
    )

    if (
        "two_or_more"
        in fake.columns
    ):

        fake[
            "fake_action_detect"
        ] = (
            fake[
                "two_or_more"
            ].astype(bool)
        )

    else:

        fake[
            "fake_action_detect"
        ] = (
            fake[
                "artifact_count"
            ]
            >=
            2
        )


    fake = fake[
        [
            "group",
            "participant",
            "session_id",
            "fake_action_detect",
        ]
    ]


else:

    print(
        "\nWARNING: EXP41D session details not found."
    )

    fake = pd.DataFrame(
        columns=[
            "group",
            "participant",
            "session_id",
            "fake_action_detect",
        ]
    )


# ============================================================
# MERGE
# ============================================================

df[
    "session_id"
] = (
    df[
        "session_id"
    ].astype(str)
)


merged = df.merge(
    stage1_session,
    on=[
        "group",
        "participant",
        "session_id",
    ],
    how="left",
)


merged[
    "stage1_detect"
] = (
    merged[
        "stage1_detect"
    ]
    .fillna(False)
    .astype(bool)
)


merged = merged.merge(
    fake,
    on=[
        "group",
        "participant",
        "session_id",
    ],
    how="left",
)


merged[
    "fake_action_detect"
] = (
    merged[
        "fake_action_detect"
    ]
    .fillna(False)
    .astype(bool)
)


# ============================================================
# DEFENSE RULES
# ============================================================

merged[
    "session_incremental"
] = (
    ~merged[
        "stage1_detect"
    ]
    &
    merged[
        "session_structure_detect"
    ]
)


merged[
    "stage1_plus_session"
] = (
    merged[
        "stage1_detect"
    ]
    |
    merged[
        "session_structure_detect"
    ]
)


merged[
    "full_defense"
] = (
    merged[
        "stage1_detect"
    ]
    |
    merged[
        "session_structure_detect"
    ]
    |
    merged[
        "fake_action_detect"
    ]
)


# ============================================================
# MAIN TABLE
# ============================================================

ORDER = [
    "Human",
    "Raw",
    "Only Swipe Humanized",
    "Rot Tap Humanized",
    "Fake Rot Tap Humanized",
]


result_rows = []


for group in ORDER:

    d = merged[
        merged.group
        ==
        group
    ]

    if len(d) == 0:
        continue


    stage1_missed = d[
        ~d.stage1_detect
    ]


    if len(
        stage1_missed
    ):

        session_catch_among_missed = float(
            stage1_missed[
                "session_structure_detect"
            ].mean()
        )

    else:

        session_catch_among_missed = np.nan


    result_rows.append({
        "group":
            group,

        "n_sessions":
            len(d),

        "stage1_detection":
            float(
                d[
                    "stage1_detect"
                ].mean()
            ),

        "session_structure_detection":
            float(
                d[
                    "session_structure_detect"
                ].mean()
            ),

        "session_catch_among_stage1_misses":
            session_catch_among_missed,

        "stage1_plus_session":
            float(
                d[
                    "stage1_plus_session"
                ].mean()
            ),

        "fake_action_detection":
            float(
                d[
                    "fake_action_detect"
                ].mean()
            ),

        "full_defense_detection":
            float(
                d[
                    "full_defense"
                ].mean()
            ),

        "full_defense_evasion":
            (
                float(
                    1.0
                    -
                    d[
                        "full_defense"
                    ].mean()
                )
                if group != "Human"
                else np.nan
            ),
    })


result = pd.DataFrame(
    result_rows
)


print("\n" + "=" * 110)
print("UNSEEN AUTHOR HUMANIZATION — SESSION STRUCTURE DEFENSE")
print("=" * 110)

print(
    result.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.6f}",
    )
)


print("\nIMPORTANT:")
print(
    "The Human row in the final fitted table is not the final "
    "clean FPR estimate."
)

print(
    "Use the participant-disjoint OOF session-head FPR above "
    "for clean session-head evaluation."
)


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUT /
    "session_features_and_scores.csv",
    index=False,
)

merged.to_csv(
    OUT /
    "combined_session_scores.csv",
    index=False,
)

result.to_csv(
    OUT /
    "session_structure_defense_summary.csv",
    index=False,
)

pd.DataFrame([
    {
        "oof_auc":
            auc,

        "default_recall":
            default_recall,

        "default_human_fpr":
            default_fpr,

        "calibrated_threshold":
            threshold,

        "calibrated_recall":
            cal_recall,

        "calibrated_human_fpr":
            cal_fpr,

        "target_human_fpr":
            TARGET_SESSION_FPR,
    }
]).to_csv(
    OUT /
    "session_head_clean_audit.csv",
    index=False,
)


print("\nSaved:", OUT)
print("EXP41F STATUS: COMPLETE")
