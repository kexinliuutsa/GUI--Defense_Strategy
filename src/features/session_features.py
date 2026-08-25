"""
Validated 54-dimensional session-level behavioral feature extractor.

This implementation is ported from the experimental feature extractor
used to train and evaluate the frozen cross-scale defense.

Important design choices are preserved exactly:

- all actions in a session are retained;
- duration is measured in the original timestamp unit;
- negative inter-action gaps are recorded separately;
- gap magnitudes are clipped at zero before log transformation;
- spatial reuse is computed using 20-pixel cells;
- direction features use endpoint displacement directions;
- population standard deviation (NumPy default, ddof=0) is used;
- coefficient of variation uses abs(mean) + 1e-9;
- correlation requires at least three observations;
- all final non-finite feature values are mapped to zero.

No gesture-level keep_swipe filtering is applied here.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import numpy as np


# ---------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------

def safe_stats(
    x: Iterable[float],
    prefix: str,
) -> Dict[str, float]:
    """
    Compute the six frozen summary statistics used by the session model.

    Returns:
        mean, std, median, q10, q90, coefficient of variation.
    """

    x = np.asarray(
        x,
        dtype=float,
    )

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
            float(
                np.median(x)
            ),

        f"{prefix}_q10":
            float(
                np.quantile(
                    x,
                    0.10,
                )
            ),

        f"{prefix}_q90":
            float(
                np.quantile(
                    x,
                    0.90,
                )
            ),

        f"{prefix}_cv":
            float(
                std
                /
                (
                    abs(mean)
                    +
                    1e-9
                )
            ),
    }


def correlation(
    a: Iterable[float],
    b: Iterable[float],
) -> float:
    """
    Frozen correlation helper.

    Correlation is defined only when at least three finite paired
    observations exist and both variables have non-negligible variance.
    """

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

    a = a[
        good
    ]

    b = b[
        good
    ]

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


def unique_cell_ratio(
    coords: Iterable,
    cell: float = 20.0,
) -> float:
    """
    Fraction of distinct spatial cells visited by action endpoints.

    Coordinates are quantized to 20-pixel cells using np.round,
    matching the validated experimental implementation.
    """

    coords = list(
        coords
    )

    if len(coords) == 0:
        return 0.0

    arr = np.asarray(
        coords,
        dtype=float,
    )

    q = np.round(
        arr
        /
        cell
    ).astype(int)

    unique = len(
        {
            tuple(x)
            for x in q
        }
    )

    return float(
        unique
        /
        len(q)
    )


# ---------------------------------------------------------------------
# Frozen 54D feature names
# ---------------------------------------------------------------------

BASE_SESSION_FEATURES = [
    "log_n_actions",
    "log_session_span",
    "zero_displacement_rate",
    "two_point_rate",
    "multi_point_rate",
    "direction_resultant",
    "start_unique_cell_ratio",
    "end_unique_cell_ratio",
    "negative_gap_rate",
    "duration_displacement_corr",
    "duration_points_corr",
    "distance_points_corr",
]


STAT_PREFIXES = [
    "log_duration",
    "log_points",
    "log_displacement",
    "log_path_length",
    "endpoint_ratio",
    "log_gap",
    "direction_similarity",
]


STAT_SUFFIXES = [
    "mean",
    "std",
    "median",
    "q10",
    "q90",
    "cv",
]


SESSION_FEATURES = (
    BASE_SESSION_FEATURES
    +
    [
        f"{prefix}_{suffix}"
        for prefix in STAT_PREFIXES
        for suffix in STAT_SUFFIXES
    ]
)


SESSION_FEATURE_DIM = len(
    SESSION_FEATURES
)


if SESSION_FEATURE_DIM != 54:
    raise RuntimeError(
        "Frozen session representation must contain 54 features."
    )


# ---------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------

def extract_session_features(
    gestures: Iterable,
) -> Optional[Dict[str, float]]:
    """
    Extract the validated 54D session representation.

    Parameters
    ----------
    gestures:
        Iterable of actions. Each action must be iterable and contain
        events with:

            event.x
            event.y
            event.timestamp_us

    Returns
    -------
    dict or None
        54 session features, or None when the session contains no valid
        actions.

    Notes
    -----
    No keep_swipe filtering is performed. The session representation
    intentionally observes the complete action composition.
    """

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


    # -----------------------------------------------------------------
    # Per-action quantities
    # -----------------------------------------------------------------

    for g in gestures:

        if (
            g is None
            or
            len(g) == 0
        ):
            continue

        pts = np.asarray(
            [
                [
                    float(e.x),
                    float(e.y),
                ]
                for e in g
            ],
            dtype=float,
        )

        ts = np.asarray(
            [
                float(
                    e.timestamp_us
                )
                for e in g
            ],
            dtype=float,
        )

        start = pts[
            0
        ]

        end = pts[
            -1
        ]

        disp_vec = (
            end
            -
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
                ts[-1]
                -
                ts[0],
            )
        )

        ratio = (
            disp
            /
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
            float(
                ts[0]
            )
        )

        end_times.append(
            float(
                ts[-1]
            )
        )

        # Zero-displacement actions are excluded from directional
        # concentration statistics.
        if disp > 1e-6:

            directions.append(
                disp_vec
                /
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


    # -----------------------------------------------------------------
    # Cross-action gaps
    # -----------------------------------------------------------------

    gaps = []

    negative_gap = []


    for i in range(
        1,
        len(start_times),
    ):

        raw_gap = (
            start_times[i]
            -
            end_times[
                i - 1
            ]
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


    # -----------------------------------------------------------------
    # Direction concentration
    # -----------------------------------------------------------------

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


    # -----------------------------------------------------------------
    # Consecutive directional similarity
    # -----------------------------------------------------------------

    direction_similarity = []


    if len(directions) >= 2:

        d = np.asarray(
            directions,
            dtype=float,
        )

        for i in range(
            len(d) - 1
        ):

            direction_similarity.append(
                float(
                    np.clip(
                        np.dot(
                            d[i],
                            d[
                                i + 1
                            ],
                        ),
                        -1.0,
                        1.0,
                    )
                )
            )


    # -----------------------------------------------------------------
    # Session span
    # -----------------------------------------------------------------

    session_span = float(
        max(
            0.0,
            np.max(
                end_times
            )
            -
            np.min(
                start_times
            ),
        )
    )


    # -----------------------------------------------------------------
    # Base session / cross-action statistics
    # -----------------------------------------------------------------

    feat = {
        "log_n_actions":
            float(
                np.log1p(
                    len(
                        durations
                    )
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
                if len(
                    negative_gap
                )
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


    # -----------------------------------------------------------------
    # Distribution summaries
    # -----------------------------------------------------------------

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


    # -----------------------------------------------------------------
    # Frozen numerical cleanup
    # -----------------------------------------------------------------

    result = {
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


    # Guard against accidental schema drift.
    if set(
        result.keys()
    ) != set(
        SESSION_FEATURES
    ):

        missing = sorted(
            set(
                SESSION_FEATURES
            )
            -
            set(
                result.keys()
            )
        )

        extra = sorted(
            set(
                result.keys()
            )
            -
            set(
                SESSION_FEATURES
            )
        )

        raise RuntimeError(
            "Session feature schema mismatch. "
            f"Missing={missing}, Extra={extra}"
        )


    return result
