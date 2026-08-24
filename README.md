# Cross-Scale Behavioral Defense for Humanized GUI Agents

This repository contains a behavioral defense developed from our analysis of GUI-agent humanization attacks.

## Motivation

We initially studied whether humanized GUI-agent trajectories could evade behavioral detectors operating on individual gestures.

A natural first defense was to combine multiple gesture-level views, such as handcrafted motion features, normalized spatial paths, and timing information. This helped against simple perturbations, but it was still insufficient against stronger humanization strategies.

Further analysis showed that the remaining machine-detectable signal did not disappear. Instead, it appeared at different behavioral scales depending on the humanization strategy.

For example:

- Some attacks successfully humanized individual swipe trajectories, while the organization of actions within a session still differed from human behavior.
- Some attacks suppressed session-level distributional differences but retained structure in the relationships between consecutive actions.
- Fake-action-based humanization introduced a separate closed-loop action artifact.

This led us to build the defense from these complementary residual signals rather than relying on a single human-vs-agent fingerprint.

## Final Defense

The final defense contains three complementary components:

### 1. Session Distribution Head

This component models behavioral statistics across an entire session.

We intentionally remove simple bookkeeping features such as:

- number of actions,
- total session duration,
- point-count statistics,
- two-point / multi-point action ratios.

The remaining representation captures behavioral properties such as action-duration distributions, inter-action timing variability, spatial reuse, and movement statistics.

### 2. Strict Cross-Action Head

This component focuses only on relationships between actions rather than the internal shape of individual gestures.

Examples include:

- inter-action gap statistics,
- direction consistency,
- start/end location reuse,
- duration-displacement coupling,
- consecutive-action similarity.

This representation is particularly useful when individual gestures have already been humanized.

### 3. Fake-Action Artifact Head

Some humanization methods generate additional fake actions.

We found that these actions can introduce a closed-loop pattern in which an action travels a substantial distance but returns close to its starting position.

A lightweight specialized detector is used to capture this artifact.

## Decision Rule

The three components are combined conservatively:

```text
Session Distribution
        OR
Strict Cross-Action
        OR
Fake-Action Artifact
        ↓
     Agent
