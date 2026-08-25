# Cross-Scale Behavioral Defense for Humanized GUI Agents

This repository contains a cross-scale behavioral defense for detecting humanized mobile GUI-agent behavior.

Existing behavioral detectors often operate on individual actions, making them vulnerable to humanization strategies that directly modify the targeted representation. In our experiments, suppressing one behavioral representation does not necessarily eliminate all source-conditioned structure. Detectable structure can remain at other behavioral scales, such as session-level distributions and relationships across actions.

Our defense combines multiple complementary behavioral views rather than relying on a single human-vs-agent fingerprint.

---

## Defense Architecture

The frozen defense contains four components.

### 1. Session Distribution Head

Models behavioral statistics across a session while excluding simple bookkeeping shortcuts such as action count, total session span, and point-count statistics.

The remaining representation captures properties such as inter-action timing, action-duration distributions, spatial reuse, and behavioral variability.

### 2. Strict Cross-Action Head

Focuses on relationships between actions rather than the internal geometry of individual gestures.

Examples include:

- inter-action gap statistics,
- direction consistency,
- start/end location reuse,
- duration-displacement coupling,
- consecutive-action similarity.

This head is designed to remain useful when individual swipe trajectories have already been humanized.

### 3. Fake-Action Artifact Head

Some fake-action strategies introduce closed-loop actions that travel a substantial distance but end close to their starting point.

We include a lightweight specialized detector for this artifact.

This component is treated as attack-family-specific rather than as a universal agent signature.

### 4. Conditional Human-Reference Head

A source classifier may be crossed without the resulting behavior actually entering the Human distribution.

The Human-reference head therefore checks whether a base-detector miss still lies outside Human behavioral support.

It is consulted only when the three base heads do not already detect the session.

```text
Session Distribution
        OR
Strict Cross-Action
        OR
Fake-Action Artifact
        OR
Conditional Human-Reference
        ↓
Final Detection
