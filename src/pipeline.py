"""
End-to-end orchestration for the cross-scale behavioral defense.

This module connects the independently validated components:

    raw gestures
        |
        +-- validated 54D session representation
        |       |
        |       +-- 41D Session Distribution source head
        |       |
        |       +-- 17D Strict Cross-Action source head
        |       |
        |       +-- Human-reference support verifier
        |
        +-- raw-action Fake-Action Artifact head
        |
        +-- frozen decision layer

The final decision rule itself remains in src.defense.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .defense import (
    CrossScaleDefense,
    DefenseDecision,
    DefenseScores,
    DEFAULT_CONFIG_PATH,
)

from .features.session_features import (
    extract_session_features,
)

from .heads.source_heads import (
    SourceHeadModels,
)

from .heads.human_reference import (
    HumanReferenceHead,
)

from .heads.fake_artifact import (
    FakeArtifactCalibration,
    calibrate_fake_artifact,
    fake_artifact_detect,
)


@dataclass
class CrossScalePipeline:
    """
    Complete fitted cross-scale defense.

    Learned / calibrated components
    -------------------------------
    source_heads:
        Clean Human-vs-Raw source classifiers.

    human_reference:
        Human-only support verifier.

    fake_artifact_calibration:
        Human-derived closed-loop artifact thresholds.

    decision_layer:
        Frozen operating thresholds and conditional routing rule.
    """

    source_heads: SourceHeadModels

    human_reference: HumanReferenceHead

    fake_artifact_calibration: FakeArtifactCalibration

    decision_layer: CrossScaleDefense


    # -----------------------------------------------------------------
    # Fit from clean calibration/training data
    # -----------------------------------------------------------------

    @classmethod
    def fit(
        cls,
        clean_sessions: pd.DataFrame,
        labels: Sequence[int],
        participants: Sequence,
        human_raw_gestures: Iterable,
        *,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
        href_target_human_fpr: float = 0.01,
        fake_action_target_human_fpr: float = 0.005,
        minimum_fake_artifacts: int = 2,
    ) -> "CrossScalePipeline":
        """
        Fit/calibrate all non-decision components.

        Parameters
        ----------
        clean_sessions:
            Validated 54D session features for clean Human and Raw-agent
            sessions.

        labels:
            0 = Human
            1 = Raw agent

        participants:
            Participant/generator grouping corresponding to clean_sessions.
            Human participant labels are used for participant-disjoint
            Human-reference calibration.

        human_raw_gestures:
            Raw Human actions used to calibrate the specialized
            fake-action artifact head.

        Notes
        -----
        Humanization conditions are not required for fitting or threshold
        calibration here.
        """

        labels = np.asarray(
            labels,
            dtype=int,
        )

        participants = np.asarray(
            participants
        )


        if len(clean_sessions) != len(labels):
            raise ValueError(
                "clean_sessions and labels must have equal length."
            )


        if len(clean_sessions) != len(participants):
            raise ValueError(
                "clean_sessions and participants must have equal length."
            )


        human_mask = (
            labels == 0
        )


        if not np.any(
            human_mask
        ):
            raise ValueError(
                "At least one Human session is required."
            )


        # -------------------------------------------------------------
        # Clean Human / Raw source classifiers
        # -------------------------------------------------------------

        source_heads = (
            SourceHeadModels.fit(
                clean_sessions,
                labels,
            )
        )


        # -------------------------------------------------------------
        # Human-only support verifier
        # -------------------------------------------------------------

        human_sessions = (
            clean_sessions.loc[
                human_mask
            ]
            .copy()
        )


        human_participants = (
            participants[
                human_mask
            ]
        )


        human_reference = (
            HumanReferenceHead.fit(
                human_sessions,
                human_participants,
                target_human_fpr=
                    href_target_human_fpr,
            )
        )


        # -------------------------------------------------------------
        # Human-only fake-action calibration
        # -------------------------------------------------------------

        fake_calibration = (
            calibrate_fake_artifact(
                human_raw_gestures,
                target_human_fpr=
                    fake_action_target_human_fpr,
                minimum_session_artifacts=
                    minimum_fake_artifacts,
            )
        )


        # -------------------------------------------------------------
        # Frozen operating point
        # -------------------------------------------------------------

        decision_layer = (
            CrossScaleDefense.from_config(
                config_path
            )
        )


        return cls(
            source_heads=
                source_heads,

            human_reference=
                human_reference,

            fake_artifact_calibration=
                fake_calibration,

            decision_layer=
                decision_layer,
        )


    # -----------------------------------------------------------------
    # End-to-end session inference
    # -----------------------------------------------------------------

    def predict_session(
        self,
        gestures: Iterable,
        *,
        n_actions: int | None = None,
    ) -> DefenseDecision:
        """
        Run the complete frozen defense on one session or rolling prefix.

        Parameters
        ----------
        gestures:
            Raw actions. No keep_swipe filtering should be applied before
            calling this method.

        n_actions:
            Optional externally supplied observed-action count.

            This is useful when reproducing a specific rolling-prefix
            denominator. If omitted, non-empty raw gestures are counted.
        """

        gestures = list(
            gestures
        )


        observed_actions = sum(
            1
            for gesture in gestures
            if (
                gesture is not None
                and
                len(gesture) > 0
            )
        )


        if n_actions is None:
            n_actions = (
                observed_actions
            )


        n_actions = int(
            n_actions
        )


        # -------------------------------------------------------------
        # Preserve the deployment context requirement.
        #
        # Do not run the learned heads and then silently classify a
        # prefix shorter than the frozen minimum context.
        # -------------------------------------------------------------

        if (
            n_actions
            <
            self.decision_layer.minimum_observed_actions
        ):

            return (
                self.decision_layer
                .predict_from_scores(
                    DefenseScores(
                        session_score=
                            float("nan"),

                        strict_score=
                            float("nan"),

                        fake_artifact=
                            False,

                        human_reference_score=
                            float("nan"),
                    ),
                    n_actions=
                        n_actions,
                )
            )


        # -------------------------------------------------------------
        # Validated session representation
        # -------------------------------------------------------------

        feature_dict = (
            extract_session_features(
                gestures
            )
        )


        if feature_dict is None:
            raise ValueError(
                "No valid actions were available for feature extraction."
            )


        session_frame = pd.DataFrame(
            [
                feature_dict
            ]
        )


        # -------------------------------------------------------------
        # Learned source heads
        # -------------------------------------------------------------

        source_scores = (
            self.source_heads
            .score(
                session_frame
            )
            .iloc[0]
        )


        # -------------------------------------------------------------
        # Human-reference support score
        # -------------------------------------------------------------

        href_scores = (
            self.human_reference
            .score(
                session_frame
            )
            .iloc[0]
        )


        # -------------------------------------------------------------
        # Specialized raw-action artifact
        # -------------------------------------------------------------

        fake_detected = (
            fake_artifact_detect(
                gestures,
                self.fake_artifact_calibration,
            )
        )


        # -------------------------------------------------------------
        # Frozen decision layer
        # -------------------------------------------------------------

        scores = DefenseScores(
            session_score=
                float(
                    source_scores[
                        "session_score"
                    ]
                ),

            strict_score=
                float(
                    source_scores[
                        "strict_score"
                    ]
                ),

            fake_artifact=
                bool(
                    fake_detected
                ),

            human_reference_score=
                float(
                    href_scores[
                        "human_reference_score"
                    ]
                ),
        )


        return (
            self.decision_layer
            .predict_from_scores(
                scores,
                n_actions=
                    n_actions,
            )
        )
