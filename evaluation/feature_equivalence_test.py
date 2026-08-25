"""
Feature-equivalence test for the cleaned repository implementation.

This test compares:

    original validated implementation
        experiments/41f_session_structure_defense/session_structure_defense.py

against:

    src.features.session_features.extract_session_features
    src.features.cross_action_features.extract_cross_action_features

The original functions are extracted with Python AST so the historical
experiment script is NOT executed.

No model training or threshold tuning occurs in this test.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path

import numpy as np


from src.features.session_features import (
    SESSION_FEATURES,
    extract_session_features,
)

from src.features.cross_action_features import (
    STRICT_CROSS_ACTION_FEATURES,
    extract_cross_action_features,
)


# ============================================================
# Synthetic event representation
# ============================================================

@dataclass
class Event:
    x: float
    y: float
    timestamp_us: float


# ============================================================
# Extract original validated functions without executing 41f
# ============================================================

def load_original_extractor(
    ahb_root: Path,
):

    source_path = (
        ahb_root
        / "experiments"
        / "41f_session_structure_defense"
        / "session_structure_defense.py"
    )

    if not source_path.exists():
        raise FileNotFoundError(
            f"Cannot find original extractor: {source_path}"
        )

    source = source_path.read_text(
        encoding="utf-8"
    )

    tree = ast.parse(
        source,
        filename=str(source_path),
    )

    required = {
        "safe_stats",
        "correlation",
        "session_features",
    }

    selected_nodes = []

    for node in tree.body:

        if (
            isinstance(node, ast.FunctionDef)
            and node.name in required
        ):
            selected_nodes.append(
                node
            )

    found = {
        node.name
        for node in selected_nodes
    }

    missing = required - found

    if missing:
        raise RuntimeError(
            "Could not recover original functions: "
            + ", ".join(
                sorted(missing)
            )
        )

    module = ast.Module(
        body=selected_nodes,
        type_ignores=[],
    )

    ast.fix_missing_locations(
        module
    )

    namespace = {
        "np": np,
    }

    exec(
        compile(
            module,
            filename=str(source_path),
            mode="exec",
        ),
        namespace,
    )

    return namespace[
        "session_features"
    ]


# ============================================================
# Synthetic-session generator
# ============================================================

def make_random_session(
    rng: np.random.Generator,
):
    """
    Generate sessions covering:

    - taps,
    - swipes,
    - zero-displacement actions,
    - single-point actions,
    - overlapping actions / negative gaps,
    - repeated coordinates,
    - variable point counts,
    - variable action durations.
    """

    n_actions = int(
        rng.integers(
            1,
            21,
        )
    )

    gestures = []

    current_time = float(
        rng.integers(
            0,
            1_000_000,
        )
    )

    repeated_anchor = np.array(
        [
            float(
                rng.uniform(
                    0,
                    1080,
                )
            ),
            float(
                rng.uniform(
                    0,
                    2400,
                )
            ),
        ]
    )

    for action_idx in range(
        n_actions
    ):

        # ----------------------------------------------------
        # Number of points
        # ----------------------------------------------------

        mode = float(
            rng.random()
        )

        if mode < 0.15:

            n_points = 1

        elif mode < 0.40:

            n_points = 2

        else:

            n_points = int(
                rng.integers(
                    3,
                    13,
                )
            )

        # ----------------------------------------------------
        # Spatial start
        #
        # Sometimes deliberately reuse a previous cell.
        # ----------------------------------------------------

        if (
            action_idx > 0
            and rng.random() < 0.30
        ):

            start = (
                repeated_anchor
                +
                rng.normal(
                    0,
                    4,
                    size=2,
                )
            )

        else:

            start = np.array(
                [
                    rng.uniform(
                        0,
                        1080,
                    ),
                    rng.uniform(
                        0,
                        2400,
                    ),
                ],
                dtype=float,
            )

        # ----------------------------------------------------
        # Motion family
        # ----------------------------------------------------

        motion_type = int(
            rng.integers(
                0,
                5,
            )
        )

        if motion_type == 0:
            # Exact zero-displacement action.
            end = start.copy()

        elif motion_type == 1:
            # Very small motion.
            end = (
                start
                +
                rng.normal(
                    0,
                    0.2,
                    size=2,
                )
            )

        else:
            # Normal swipe-like displacement.
            end = (
                start
                +
                rng.normal(
                    0,
                    250,
                    size=2,
                )
            )

        if n_points == 1:

            points = np.asarray(
                [
                    start
                ],
                dtype=float,
            )

        elif n_points == 2:

            points = np.asarray(
                [
                    start,
                    end,
                ],
                dtype=float,
            )

        else:

            alpha = np.linspace(
                0,
                1,
                n_points,
            )[:, None]

            points = (
                start[None, :]
                +
                alpha
                *
                (
                    end
                    -
                    start
                )[None, :]
            )

            # Curved/noisy path.
            points += rng.normal(
                0,
                12,
                size=points.shape,
            )

            points[
                0
            ] = start

            points[
                -1
            ] = end

        # ----------------------------------------------------
        # Timing
        # ----------------------------------------------------

        duration = float(
            rng.integers(
                0,
                1_000_000,
            )
        )

        if n_points == 1:

            timestamps = np.asarray(
                [
                    current_time
                ],
                dtype=float,
            )

        else:

            timestamps = np.linspace(
                current_time,
                current_time
                +
                duration,
                n_points,
            )

        gesture = [
            Event(
                x=float(
                    points[i, 0]
                ),
                y=float(
                    points[i, 1]
                ),
                timestamp_us=float(
                    timestamps[i]
                ),
            )
            for i in range(
                n_points
            )
        ]

        gestures.append(
            gesture
        )

        repeated_anchor = (
            points[
                -1
            ].copy()
        )

        # ----------------------------------------------------
        # Inter-action gap
        #
        # Sometimes intentionally negative to test
        # negative_gap_rate.
        # ----------------------------------------------------

        gap_mode = float(
            rng.random()
        )

        if gap_mode < 0.15:

            gap = -float(
                rng.integers(
                    1,
                    100_000,
                )
            )

        elif gap_mode < 0.30:

            gap = 0.0

        else:

            gap = float(
                rng.integers(
                    1,
                    2_000_000,
                )
            )

        current_time = (
            float(
                timestamps[
                    -1
                ]
            )
            +
            gap
        )

    return gestures


# ============================================================
# Hand-crafted edge cases
# ============================================================

def edge_case_sessions():

    cases = []

    # --------------------------------------------------------
    # Single-point tap
    # --------------------------------------------------------

    cases.append(
        [
            [
                Event(
                    100,
                    200,
                    1000,
                )
            ]
        ]
    )

    # --------------------------------------------------------
    # Two-point stationary action
    # --------------------------------------------------------

    cases.append(
        [
            [
                Event(
                    100,
                    100,
                    1000,
                ),
                Event(
                    100,
                    100,
                    2000,
                ),
            ]
        ]
    )

    # --------------------------------------------------------
    # Perfect straight swipe
    # --------------------------------------------------------

    cases.append(
        [
            [
                Event(
                    0,
                    0,
                    0,
                ),
                Event(
                    50,
                    0,
                    500,
                ),
                Event(
                    100,
                    0,
                    1000,
                ),
            ]
        ]
    )

    # --------------------------------------------------------
    # Two identical directions
    # --------------------------------------------------------

    cases.append(
        [
            [
                Event(
                    0,
                    0,
                    0,
                ),
                Event(
                    100,
                    0,
                    1000,
                ),
            ],
            [
                Event(
                    100,
                    100,
                    2000,
                ),
                Event(
                    200,
                    100,
                    3000,
                ),
            ],
        ]
    )

    # --------------------------------------------------------
    # Opposite directions -> mean similarity / CV edge case
    # --------------------------------------------------------

    cases.append(
        [
            [
                Event(
                    0,
                    0,
                    0,
                ),
                Event(
                    100,
                    0,
                    1000,
                ),
            ],
            [
                Event(
                    200,
                    0,
                    2000,
                ),
                Event(
                    100,
                    0,
                    3000,
                ),
            ],
            [
                Event(
                    0,
                    0,
                    4000,
                ),
                Event(
                    100,
                    0,
                    5000,
                ),
            ],
        ]
    )

    # --------------------------------------------------------
    # Negative inter-action gap
    # --------------------------------------------------------

    cases.append(
        [
            [
                Event(
                    0,
                    0,
                    0,
                ),
                Event(
                    100,
                    0,
                    10_000,
                ),
            ],
            [
                Event(
                    50,
                    50,
                    5_000,
                ),
                Event(
                    60,
                    60,
                    15_000,
                ),
            ],
        ]
    )

    # --------------------------------------------------------
    # Reused spatial cells
    # --------------------------------------------------------

    cases.append(
        [
            [
                Event(
                    100,
                    100,
                    0,
                ),
                Event(
                    200,
                    100,
                    1000,
                ),
            ],
            [
                Event(
                    102,
                    101,
                    2000,
                ),
                Event(
                    201,
                    99,
                    3000,
                ),
            ],
            [
                Event(
                    98,
                    103,
                    4000,
                ),
                Event(
                    198,
                    102,
                    5000,
                ),
            ],
        ]
    )

    return cases


# ============================================================
# Compare one session
# ============================================================

def compare_session(
    original_extractor,
    gestures,
    session_name: str,
):

    original = original_extractor(
        gestures
    )

    cleaned = extract_session_features(
        gestures
    )

    if (
        original is None
        or cleaned is None
    ):

        if (
            original is None
            and cleaned is None
        ):
            return {
                "session":
                    session_name,

                "max_54d_diff":
                    0.0,

                "max_17d_diff":
                    0.0,
            }

        raise AssertionError(
            f"{session_name}: None mismatch."
        )

    # --------------------------------------------------------
    # Schema check
    # --------------------------------------------------------

    original_keys = set(
        original.keys()
    )

    cleaned_keys = set(
        cleaned.keys()
    )

    expected_keys = set(
        SESSION_FEATURES
    )

    if original_keys != expected_keys:

        raise AssertionError(
            f"{session_name}: original schema mismatch.\n"
            f"Missing={sorted(expected_keys-original_keys)}\n"
            f"Extra={sorted(original_keys-expected_keys)}"
        )

    if cleaned_keys != expected_keys:

        raise AssertionError(
            f"{session_name}: cleaned schema mismatch."
        )

    # --------------------------------------------------------
    # Full 54D comparison
    # --------------------------------------------------------

    max_54d_diff = 0.0

    worst_54d_feature = None

    for feature in SESSION_FEATURES:

        a = float(
            original[
                feature
            ]
        )

        b = float(
            cleaned[
                feature
            ]
        )

        diff = abs(
            a - b
        )

        if diff > max_54d_diff:

            max_54d_diff = diff
            worst_54d_feature = feature

        if not np.isclose(
            a,
            b,
            rtol=0.0,
            atol=1e-12,
        ):

            raise AssertionError(
                f"{session_name}: mismatch in {feature}: "
                f"original={a}, cleaned={b}, diff={diff}"
            )

    # --------------------------------------------------------
    # Strict 17D comparison
    # --------------------------------------------------------

    strict = extract_cross_action_features(
        gestures
    )

    if strict is None:

        raise AssertionError(
            f"{session_name}: Strict extractor returned None."
        )

    max_17d_diff = 0.0

    worst_17d_feature = None

    for feature in (
        STRICT_CROSS_ACTION_FEATURES
    ):

        a = float(
            original[
                feature
            ]
        )

        b = float(
            strict[
                feature
            ]
        )

        diff = abs(
            a - b
        )

        if diff > max_17d_diff:

            max_17d_diff = diff
            worst_17d_feature = feature

        if not np.isclose(
            a,
            b,
            rtol=0.0,
            atol=1e-12,
        ):

            raise AssertionError(
                f"{session_name}: Strict mismatch in {feature}: "
                f"original={a}, cleaned={b}, diff={diff}"
            )

    return {
        "session":
            session_name,

        "max_54d_diff":
            max_54d_diff,

        "worst_54d_feature":
            worst_54d_feature,

        "max_17d_diff":
            max_17d_diff,

        "worst_17d_feature":
            worst_17d_feature,
    }


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ahb-root",
        required=True,
        help=(
            "Path to the original "
            "Passing-the-Turing-Test-on-Screen-Agent-Humanization-Benchmark "
            "repository."
        ),
    )

    parser.add_argument(
        "--random-sessions",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260826,
    )

    args = parser.parse_args()

    ahb_root = Path(
        args.ahb_root
    ).expanduser().resolve()

    print(
        "=" * 72
    )

    print(
        "FEATURE EQUIVALENCE TEST"
    )

    print(
        "=" * 72
    )

    print(
        "AHB root:",
        ahb_root,
    )

    print(
        "Random sessions:",
        args.random_sessions,
    )

    print(
        "Seed:",
        args.seed,
    )

    original_extractor = (
        load_original_extractor(
            ahb_root
        )
    )

    rng = np.random.default_rng(
        args.seed
    )

    results = []

    # --------------------------------------------------------
    # Hand-written edge cases
    # --------------------------------------------------------

    edges = edge_case_sessions()

    for i, gestures in enumerate(
        edges,
        1,
    ):

        results.append(
            compare_session(
                original_extractor,
                gestures,
                f"edge_{i}",
            )
        )

    # --------------------------------------------------------
    # Randomized coverage
    # --------------------------------------------------------

    for i in range(
        args.random_sessions
    ):

        gestures = (
            make_random_session(
                rng
            )
        )

        results.append(
            compare_session(
                original_extractor,
                gestures,
                f"random_{i + 1}",
            )
        )

    max_54d = max(
        r[
            "max_54d_diff"
        ]
        for r in results
    )

    max_17d = max(
        r[
            "max_17d_diff"
        ]
        for r in results
    )

    print()

    print(
        "=" * 72
    )

    print(
        "RESULT"
    )

    print(
        "=" * 72
    )

    print(
        "Sessions tested:",
        len(results),
    )

    print(
        "Session feature dimensions:",
        len(
            SESSION_FEATURES
        ),
    )

    print(
        "Strict feature dimensions:",
        len(
            STRICT_CROSS_ACTION_FEATURES
        ),
    )

    print(
        "Maximum 54D absolute difference:",
        max_54d,
    )

    print(
        "Maximum 17D absolute difference:",
        max_17d,
    )

    print()

    print(
        "PASS: cleaned feature implementation is equivalent "
        "to the validated historical implementation."
    )


if __name__ == "__main__":
    main()
