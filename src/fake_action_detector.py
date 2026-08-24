from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))

OUT = ROOT / "experiments/41d_fake_action_head/results"
OUT.mkdir(parents=True, exist_ok=True)

from analysis.lib.gesture_log_reader_utils import (
    ranged_modified_generator_with_session_timestamp,
)

print("=" * 110)
print("EXP41D — RAW-ACTION FAKE-ACTION ARTIFACT HEAD")
print("=" * 110)

# ============================================================
# Author groups
# ============================================================

excel = ROOT / "Formated_Data_Renamed.xlsx"

table = pd.read_excel(
    excel,
    header=0,
    index_col=None,
    dtype=str,
)

cols = list(table.columns)

from operator import itemgetter

GROUPS = {
    "Human": list(itemgetter(
        2, 3, 4, 33, 34, 35, 36
    )(cols)),

    "Raw": list(itemgetter(
        12, 14, 16, 19, 37
    )(cols)),

    "Only Swipe Humanized": list(itemgetter(
        23, 25
    )(cols)),

    "Rot Tap Humanized": list(itemgetter(
        5, 7, 9, 20, 39
    )(cols)),

    "Fake Rot Tap Humanized": list(itemgetter(
        28, 30, 41
    )(cols)),
}

# ============================================================
# Raw gesture metrics
#
# IMPORTANT:
# no keep_swipe here.
# We intentionally inspect actions BEFORE swipe filtering.
# ============================================================

def gesture_metrics(g):

    if g is None or len(g) == 0:
        return None

    pts = np.asarray(
        [[float(e.x), float(e.y)] for e in g],
        dtype=float,
    )

    ts = np.asarray(
        [float(e.timestamp_us) for e in g],
        dtype=float,
    )

    start = pts[0]
    end = pts[-1]

    displacement = float(
        np.linalg.norm(end - start)
    )

    if len(pts) >= 2:
        seg = np.linalg.norm(
            np.diff(pts, axis=0),
            axis=1,
        )
        path_length = float(seg.sum())
    else:
        path_length = 0.0

    ratio = (
        displacement / path_length
        if path_length > 1e-12
        else 1.0
    )

    duration = (
        float(ts[-1] - ts[0])
        if len(ts) >= 2
        else 0.0
    )

    return {
        "n_points": len(g),
        "duration_us": duration,
        "displacement": displacement,
        "path_length": path_length,
        "endpoint_path_ratio": ratio,

        # Useful descriptive bits only.
        "exact_zero_displacement": displacement <= 1e-6,
        "near_zero_displacement": displacement <= 1.0,
    }


# ============================================================
# Load ALL raw gestures
# ============================================================

rows = []

for group, participants in GROUPS.items():

    print("\nLoading:", group)

    for participant in participants:

        sessions = ranged_modified_generator_with_session_timestamp(
            formated_data_timestamps=table,
            participants=[participant],

            # IMPORTANT:
            # inspect raw actions rather than keep_swipe output
            filtering_and_modification_function=None,

            index_range=None,
        )

        count = 0

        for session_id, gestures in sessions:

            for j, gesture in enumerate(gestures):

                m = gesture_metrics(gesture)

                if m is None:
                    continue

                rows.append({
                    "group": group,
                    "participant": participant,
                    "session_id": str(session_id),
                    "gesture_index": int(j),
                    **m,
                })

                count += 1

        print(
            f"  {participant:45s}",
            f"actions={count}"
        )


df = pd.DataFrame(rows)

print("\nTotal raw actions:", len(df))

print("\nGroup counts:")
print(df.group.value_counts())


# ============================================================
# Human-calibrated closed-loop artifact
#
# Fake action signature we care about:
#
#     meaningful movement
#          +
#     endpoint returns close to start
#
# This avoids flagging ordinary stationary taps simply because
# displacement is zero.
# ============================================================

human = df[df.group == "Human"].copy()

# ------------------------------------------------------------
# First define "meaningful movement".
#
# Use human data, rather than an arbitrary 10/20 pixel number.
# Median positive human path length gives a conservative
# moving-action scale.
# ------------------------------------------------------------

positive_human_lengths = human.loc[
    human.path_length > 0,
    "path_length"
]

if len(positive_human_lengths) == 0:
    raise RuntimeError(
        "No positive human path lengths found."
    )

movement_floor = float(
    np.quantile(
        positive_human_lengths,
        0.25,
    )
)

print("\nHuman-derived movement floor:")
print(movement_floor)


# ------------------------------------------------------------
# Among human moving actions, find the low tail of
# endpoint/path ratio.
#
# Smaller ratio:
#
# long path + tiny endpoint displacement
#
# exactly the closed-loop artifact we are testing.
# ------------------------------------------------------------

human_moving = human[
    human.path_length >= movement_floor
].copy()

TARGET_FPR = 0.005

ratio_threshold = float(
    np.quantile(
        human_moving.endpoint_path_ratio,
        TARGET_FPR,
        method="lower",
    )
)

print("\nHuman-calibrated closed-loop threshold:")
print("ratio <=", ratio_threshold)
print("path length >=", movement_floor)


# ============================================================
# Artifact rules
# ============================================================

df["moving_action"] = (
    df.path_length >= movement_floor
)

df["low_endpoint_ratio"] = (
    df.endpoint_path_ratio <= ratio_threshold
)

# Main action-artifact head.
df["fake_action_detect"] = (
    df["moving_action"]
    &
    df["low_endpoint_ratio"]
)

# Descriptive stricter variants.
df["closed_exact"] = (
    df["moving_action"]
    &
    df["exact_zero_displacement"]
)

df["closed_near_zero"] = (
    df["moving_action"]
    &
    df["near_zero_displacement"]
)


# ============================================================
# Action-level summary
# ============================================================

ORDER = [
    "Human",
    "Raw",
    "Only Swipe Humanized",
    "Rot Tap Humanized",
    "Fake Rot Tap Humanized",
]

summary = (
    df.groupby("group", as_index=False)
      .agg(
          n_actions=("gesture_index", "size"),

          moving_rate=("moving_action", "mean"),

          exact_closed_rate=("closed_exact", "mean"),

          near_closed_rate=("closed_near_zero", "mean"),

          fake_action_detection=(
              "fake_action_detect",
              "mean"
          ),

          median_path_length=(
              "path_length",
              "median"
          ),

          median_endpoint_ratio=(
              "endpoint_path_ratio",
              "median"
          ),

          median_n_points=(
              "n_points",
              "median"
          ),
      )
)

summary["order"] = summary["group"].map(
    {x: i for i, x in enumerate(ORDER)}
)

summary = (
    summary
    .sort_values("order")
    .drop(columns="order")
)


print("\n" + "=" * 110)
print("ACTION-LEVEL FAKE-ACTION ARTIFACT")
print("=" * 110)

print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)


# ============================================================
# Session-level head
#
# For a real defense, action-level FPR accumulates over long
# sessions. Therefore report:
#
# 1. ANY artifact in session
# 2. >=2 artifacts
# 3. artifact fraction
#
# This lets us avoid claiming ANY is automatically best.
# ============================================================

session = (
    df.groupby(
        [
            "group",
            "participant",
            "session_id",
        ],
        as_index=False,
    )
    .agg(
        n_actions=("gesture_index", "size"),
        artifact_count=("fake_action_detect", "sum"),
    )
)

session["artifact_fraction"] = (
    session.artifact_count /
    session.n_actions
)

session["any_artifact"] = (
    session.artifact_count >= 1
)

session["two_or_more"] = (
    session.artifact_count >= 2
)


session_summary = (
    session.groupby(
        "group",
        as_index=False,
    )
    .agg(
        n_sessions=("session_id", "size"),

        any_artifact_detection=(
            "any_artifact",
            "mean"
        ),

        two_artifact_detection=(
            "two_or_more",
            "mean"
        ),

        mean_artifact_fraction=(
            "artifact_fraction",
            "mean"
        ),

        median_artifact_count=(
            "artifact_count",
            "median"
        ),
    )
)

session_summary["order"] = (
    session_summary["group"].map(
        {x: i for i, x in enumerate(ORDER)}
    )
)

session_summary = (
    session_summary
    .sort_values("order")
    .drop(columns="order")
)


print("\n" + "=" * 110)
print("SESSION-LEVEL FAKE-ACTION ARTIFACT")
print("=" * 110)

print(
    session_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)


# ============================================================
# Participant-level audit
#
# Important because Fake Rot Tap has many more raw actions.
# We need to make sure the result is not caused by only one
# weird agent implementation.
# ============================================================

participant_summary = (
    df.groupby(
        [
            "group",
            "participant",
        ],
        as_index=False,
    )
    .agg(
        n_actions=("gesture_index", "size"),

        fake_action_detection=(
            "fake_action_detect",
            "mean"
        ),

        exact_closed_rate=(
            "closed_exact",
            "mean"
        ),

        median_endpoint_ratio=(
            "endpoint_path_ratio",
            "median"
        ),
    )
)


print("\n" + "=" * 110)
print("PER-PARTICIPANT ARTIFACT AUDIT")
print("=" * 110)

print(
    participant_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)


# ============================================================
# Save
# ============================================================

df.to_csv(
    OUT / "raw_action_artifacts.csv",
    index=False,
)

summary.to_csv(
    OUT / "action_level_summary.csv",
    index=False,
)

session.to_csv(
    OUT / "session_level_details.csv",
    index=False,
)

session_summary.to_csv(
    OUT / "session_level_summary.csv",
    index=False,
)

participant_summary.to_csv(
    OUT / "participant_summary.csv",
    index=False,
)

pd.DataFrame([
    {
        "movement_floor": movement_floor,
        "ratio_threshold": ratio_threshold,
        "target_human_action_fpr": TARGET_FPR,
    }
]).to_csv(
    OUT / "thresholds.csv",
    index=False,
)


print("\nSaved:", OUT)
print("EXP41D STATUS: COMPLETE")
