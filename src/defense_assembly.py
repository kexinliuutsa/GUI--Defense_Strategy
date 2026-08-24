from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path.cwd()

OUT = (
    ROOT
    / "experiments/41h_final_defense_assembly/results"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

print("=" * 110)
print("EXP41H — FINAL DEFENSE ASSEMBLY / COMPLEMENTARITY AUDIT")
print("=" * 110)

KEYS = [
    "group",
    "participant",
    "session_id",
]


# ============================================================
# 1. STAGE-1
# ============================================================

s1 = pd.read_csv(
    ROOT
    / "experiments/41c_original_humanization_defense/results"
    / "all_swipe_scores.csv"
)

s1["session_id"] = (
    s1["session_id"].astype(str)
)

s1 = (
    s1.groupby(
        KEYS,
        as_index=False,
    )
    .agg(
        stage1=(
            "late_or_calibrated_detect",
            "max",
        )
    )
)


# ============================================================
# 2. SESSION-HEAD PREDICTIONS
# ============================================================

p = pd.read_csv(
    ROOT
    / "experiments/41g_session_shortcut_audit/results"
    / "session_predictions.csv"
)

p["session_id"] = (
    p["session_id"].astype(str)
)

wanted = {
    "FULL_54":
        "full54",

    "NO_BOOKKEEPING":
        "no_bookkeeping",

    "STRICT_CROSS_ACTION":
        "strict_cross",

    "GAP_ONLY":
        "gap_only",

    "DIRECTION_ONLY":
        "direction_only",
}


wide = None


for variant, new_name in wanted.items():

    d = (
        p[
            p.variant == variant
        ][
            KEYS
            +
            [
                "detect",
            ]
        ]
        .rename(
            columns={
                "detect":
                    new_name
            }
        )
    )

    d[new_name] = (
        d[new_name]
        .astype(bool)
    )

    if wide is None:

        wide = d

    else:

        wide = wide.merge(
            d,
            on=KEYS,
            how="inner",
            validate="one_to_one",
        )


# ============================================================
# 3. FAKE ACTION HEAD
# ============================================================

fake = pd.read_csv(
    ROOT
    / "experiments/41d_fake_action_head/results"
    / "session_level_details.csv"
)

fake["session_id"] = (
    fake["session_id"].astype(str)
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


# ============================================================
# 4. MATCH EXACTLY TO EXP41C SESSION POPULATION
# ============================================================

d = s1.merge(
    wide,
    on=KEYS,
    how="left",
)

d = d.merge(
    fake,
    on=KEYS,
    how="left",
)


for c in [
    "full54",
    "no_bookkeeping",
    "strict_cross",
    "gap_only",
    "direction_only",
    "fake_action",
]:

    d[c] = (
        d[c]
        .fillna(False)
        .astype(bool)
    )


d["stage1"] = (
    d["stage1"]
    .astype(bool)
)


print("\nMatched session counts:")

print(
    d.group
    .value_counts()
    .to_string()
)


# ============================================================
# 5. DEFENSE COMBINATIONS
# ============================================================

d["s1_plus_nb"] = (
    d.stage1
    |
    d.no_bookkeeping
)


d["s1_plus_strict"] = (
    d.stage1
    |
    d.strict_cross
)


d["nb_plus_strict"] = (
    d.no_bookkeeping
    |
    d.strict_cross
)


d["s1_nb_strict"] = (
    d.stage1
    |
    d.no_bookkeeping
    |
    d.strict_cross
)


d["final_behavioral"] = (
    d.stage1
    |
    d.no_bookkeeping
    |
    d.strict_cross
    |
    d.fake_action
)


# ============================================================
# 6. INCREMENTAL CATCH
# ============================================================

d["nb_incremental_over_s1"] = (
    (~d.stage1)
    &
    d.no_bookkeeping
)


d["strict_incremental_over_s1"] = (
    (~d.stage1)
    &
    d.strict_cross
)


d["strict_incremental_over_s1_nb"] = (
    (~d.s1_plus_nb)
    &
    d.strict_cross
)


d["fake_incremental_over_behavior"] = (
    (~d.s1_nb_strict)
    &
    d.fake_action
)


# ============================================================
# 7. SUMMARY
# ============================================================

ORDER = [
    "Human",
    "Raw",
    "Only Swipe Humanized",
    "Rot Tap Humanized",
    "Fake Rot Tap Humanized",
]


rows = []


for group in ORDER:

    x = d[
        d.group == group
    ]

    if len(x) == 0:
        continue


    stage1_miss = x[
        ~x.stage1
    ]

    s1_nb_miss = x[
        ~x.s1_plus_nb
    ]


    rows.append({
        "group":
            group,

        "n_sessions":
            len(x),

        "stage1":
            x.stage1.mean(),

        "no_bookkeeping":
            x.no_bookkeeping.mean(),

        "strict_cross":
            x.strict_cross.mean(),

        "fake_action":
            x.fake_action.mean(),

        "stage1_plus_nb":
            x.s1_plus_nb.mean(),

        "stage1_plus_strict":
            x.s1_plus_strict.mean(),

        "nb_plus_strict":
            x.nb_plus_strict.mean(),

        "stage1_nb_strict":
            x.s1_nb_strict.mean(),

        "final_behavioral":
            x.final_behavioral.mean(),

        "nb_catch_among_stage1_misses":
            (
                stage1_miss.no_bookkeeping.mean()
                if len(stage1_miss)
                else np.nan
            ),

        "strict_catch_among_stage1_misses":
            (
                stage1_miss.strict_cross.mean()
                if len(stage1_miss)
                else np.nan
            ),

        "strict_catch_after_s1_nb":
            (
                s1_nb_miss.strict_cross.mean()
                if len(s1_nb_miss)
                else np.nan
            ),

        "final_evasion":
            (
                1.0
                -
                x.final_behavioral.mean()
                if group != "Human"
                else np.nan
            ),
    })


summary = pd.DataFrame(
    rows
)


print("\n" + "=" * 110)
print("FINAL DEFENSE COMPLEMENTARITY")
print("=" * 110)

print(
    summary.to_string(
        index=False,
        float_format=lambda z:
            f"{z:.6f}",
    )
)


# ============================================================
# 8. ROT TAP PER-PARTICIPANT
# ============================================================

rot = d[
    d.group
    ==
    "Rot Tap Humanized"
]


rot_summary = (
    rot.groupby(
        "participant",
        as_index=False,
    )
    .agg(
        n_sessions=(
            "session_id",
            "size"
        ),

        stage1=(
            "stage1",
            "mean"
        ),

        no_bookkeeping=(
            "no_bookkeeping",
            "mean"
        ),

        strict_cross=(
            "strict_cross",
            "mean"
        ),

        stage1_nb_strict=(
            "s1_nb_strict",
            "mean"
        ),

        final_behavioral=(
            "final_behavioral",
            "mean"
        ),
    )
)


print("\n" + "=" * 110)
print("ROT TAP — PER PARTICIPANT")
print("=" * 110)

print(
    rot_summary.to_string(
        index=False,
        float_format=lambda z:
            f"{z:.6f}",
    )
)


# ============================================================
# 9. OVERLAP MATRIX FOR ROT TAP
# ============================================================

heads = [
    "stage1",
    "no_bookkeeping",
    "strict_cross",
    "fake_action",
]


overlap = pd.DataFrame(
    index=heads,
    columns=heads,
    dtype=float,
)


for a in heads:

    for b in heads:

        overlap.loc[
            a,
            b
        ] = float(
            np.mean(
                rot[a]
                &
                rot[b]
            )
        )


print("\n" + "=" * 110)
print("ROT TAP — DETECTION OVERLAP")
print("=" * 110)

print(
    overlap.to_string(
        float_format=lambda z:
            f"{z:.6f}",
    )
)


# ============================================================
# SAVE
# ============================================================

d.to_csv(
    OUT /
    "combined_session_predictions.csv",
    index=False,
)

summary.to_csv(
    OUT /
    "final_defense_summary.csv",
    index=False,
)

rot_summary.to_csv(
    OUT /
    "rot_tap_participant_summary.csv",
    index=False,
)

overlap.to_csv(
    OUT /
    "rot_tap_overlap.csv"
)


print("\nIMPORTANT:")
print(
    "Attack detection rates are valid unseen-attack evaluations."
)

print(
    "Human rates in this table are NOT a valid combined clean FPR, "
    "because the final session models were fitted on clean humans."
)

print(
    "A separate joint OOF clean audit is required before reporting "
    "the final defense FPR."
)

print("\nSaved:", OUT)
print("EXP41H STATUS: COMPLETE")
