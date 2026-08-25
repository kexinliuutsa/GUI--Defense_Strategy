"""
Frozen cross-scale decision logic for humanized GUI-agent detection.

The final defense combines four complementary signals:

1. Session Distribution Head
2. Strict Cross-Action Head
3. Fake-Action Artifact Head
4. Conditional Human-Reference Head

Important:
    This module implements the frozen decision rule only.
    Feature extraction and model scoring are implemented separately.

Primary operating point:
    - Session Distribution threshold: 0.8703146248159933
    - Strict Cross-Action threshold: 0.9768090920254805
    - Human-Reference threshold: 1.1488039406123078
    - Minimum observed actions: 4

The Human-reference head is consulted only when all three base heads miss.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------
# Default frozen configuration
# ---------------------------------------------------------------------

DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "frozen_defense.json"
)


# ---------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class DefenseScores:
    """
    Scores produced by the individual defense heads.

    Parameters
    ----------
    session_score:
        Probability / score from the Session Distribution Head.

    strict_score:
        Probability / score from the Strict Cross-Action Head.

    fake_artifact:
        Boolean output from the specialized Fake-Action Artifact Head.

    human_reference_score:
        Normalized Human-reference support-distance score.
    """

    session_score: float
    strict_score: float
    fake_artifact: bool
    human_reference_score: float


@dataclass(frozen=True)
class DefenseDecision:
    """
    Final output of the frozen cross-scale defense.
    """

    detected: Optional[bool]
    status: str

    base_detected: Optional[bool]
    href_consulted: bool
    href_detected: Optional[bool]

    session_detected: Optional[bool]
    strict_detected: Optional[bool]
    fake_detected: Optional[bool]

    n_actions: int

    session_score: float
    strict_score: float
    human_reference_score: float


# ---------------------------------------------------------------------
# Frozen defense
# ---------------------------------------------------------------------

class CrossScaleDefense:
    """
    Frozen decision layer for the cross-scale behavioral defense.

    The base detector is:

        Session Distribution
            OR
        Strict Cross-Action
            OR
        Fake-Action Artifact

    If the base detector misses, the Human-reference score is checked:

        Final =
            Base
            OR
            Conditional Human Reference

    No attack-specific threshold adjustment is performed here.
    """

    def __init__(
        self,
        session_threshold: float,
        strict_threshold: float,
        human_reference_threshold: float,
        minimum_observed_actions: int = 4,
    ):
        self.session_threshold = float(session_threshold)
        self.strict_threshold = float(strict_threshold)
        self.human_reference_threshold = float(
            human_reference_threshold
        )
        self.minimum_observed_actions = int(
            minimum_observed_actions
        )

    # -----------------------------------------------------------------
    # Construction from frozen JSON configuration
    # -----------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
    ) -> "CrossScaleDefense":

        config_path = Path(config_path)

        with config_path.open(
            "r",
            encoding="utf-8",
        ) as f:
            config = json.load(f)

        thresholds = config["thresholds"]

        operating_point = config[
            "primary_operating_point"
        ]

        return cls(
            session_threshold=thresholds[
                "session_distribution_head"
            ],
            strict_threshold=thresholds[
                "strict_cross_action_head"
            ],
            human_reference_threshold=thresholds[
                "human_reference_head"
            ],
            minimum_observed_actions=operating_point[
                "minimum_observed_actions"
            ],
        )

    # -----------------------------------------------------------------
    # Main frozen decision rule
    # -----------------------------------------------------------------

    def predict_from_scores(
        self,
        scores: DefenseScores,
        n_actions: int,
    ) -> DefenseDecision:
        """
        Apply the frozen defense to already-computed component scores.

        The primary rolling detector requires at least four observed
        actions because its cross-action representation requires context.

        For shorter prefixes, the method returns:

            detected = None
            status = "insufficient_context"

        rather than silently classifying the prefix as Human or Agent.
        """

        n_actions = int(n_actions)

        # -------------------------------------------------------------
        # Primary deployment requirement
        # -------------------------------------------------------------

        if n_actions < self.minimum_observed_actions:

            return DefenseDecision(
                detected=None,
                status="insufficient_context",

                base_detected=None,
                href_consulted=False,
                href_detected=None,

                session_detected=None,
                strict_detected=None,
                fake_detected=None,

                n_actions=n_actions,

                session_score=float(
                    scores.session_score
                ),

                strict_score=float(
                    scores.strict_score
                ),

                human_reference_score=float(
                    scores.human_reference_score
                ),
            )

        # -------------------------------------------------------------
        # Base heads
        # -------------------------------------------------------------

        session_detected = (
            scores.session_score
            >=
            self.session_threshold
        )

        strict_detected = (
            scores.strict_score
            >=
            self.strict_threshold
        )

        fake_detected = bool(
            scores.fake_artifact
        )

        base_detected = (
            session_detected
            or
            strict_detected
            or
            fake_detected
        )

        # -------------------------------------------------------------
        # Conditional Human-reference check
        # -------------------------------------------------------------

        href_consulted = not base_detected

        href_detected = (
            href_consulted
            and
            (
                scores.human_reference_score
                >=
                self.human_reference_threshold
            )
        )

        final_detected = (
            base_detected
            or
            href_detected
        )

        return DefenseDecision(
            detected=bool(
                final_detected
            ),

            status="ok",

            base_detected=bool(
                base_detected
            ),

            href_consulted=bool(
                href_consulted
            ),

            href_detected=bool(
                href_detected
            ),

            session_detected=bool(
                session_detected
            ),

            strict_detected=bool(
                strict_detected
            ),

            fake_detected=bool(
                fake_detected
            ),

            n_actions=n_actions,

            session_score=float(
                scores.session_score
            ),

            strict_score=float(
                scores.strict_score
            ),

            human_reference_score=float(
                scores.human_reference_score
            ),
        )
