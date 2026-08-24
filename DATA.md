# Data

This repository contains the implementation and evaluation code for the cross-scale behavioral defense.

## Dataset

The experiments are based on the Agent Humanization Benchmark (AHB), which contains human and GUI-agent interaction trajectories collected from multiple agent systems.

The defense uses session-level behavioral information derived from the original trajectories, including:

- action duration statistics,
- inter-action gap statistics,
- movement and displacement statistics,
- direction consistency,
- start/end coordinate reuse,
- cross-action correlations,
- fake-action closed-loop patterns.

## Data Organization

The evaluation distinguishes the following groups:

- Human
- Raw Agent
- Only-Swipe Humanized
- Rot-Tap Humanized
- Fake-Rot-Tap Humanized

The agent families evaluated include:

- GPT-4o-based agent
- UI-TARS
- Claude-based agent
- CPM GUI Agent
- AutoGLM

Not every humanization strategy is available for every agent family.

## Session-Level Evaluation

The final evaluation uses sessions rather than individual gestures.

For the matched clean evaluation:

- Human sessions: 280
- Raw-agent sessions: 184

For unseen humanization evaluation:

- Only-Swipe Humanized: 57 sessions
- Rot-Tap Humanized: 158 sessions
- Fake-Rot-Tap Humanized: 291 sessions

## Data Availability

The original benchmark data are not redistributed in this repository.

Users should obtain the AHB dataset from the original benchmark repository and follow its data preparation procedure.

The scripts in this repository assume that the original trajectory data have already been downloaded and formatted according to the benchmark.

## Derived Features

The final defense uses two learned session representations.

### No-Bookkeeping Session Representation

This representation removes simple bookkeeping features such as:

- number of actions,
- total session span,
- point-count statistics,
- two-point action rate,
- multi-point action rate,
- zero-displacement action rate.

The remaining features capture behavioral distributions within a session.

### Strict Cross-Action Representation

This representation focuses specifically on relationships across actions, including:

- inter-action gap statistics,
- direction consistency,
- start/end coordinate reuse,
- duration-displacement coupling,
- consecutive-action similarity.

## Fake-Action Artifact

Fake-action-based humanization may generate closed-loop actions that travel a substantial distance but return close to their starting location.

The specialized fake-action detector operates on these action-level artifacts and aggregates them at the session level.

## Reproducibility Note

The summary CSV files in `results/` contain the final reported evaluation results.

The raw AHB trajectories and intermediate experiment files are not included in this repository.
