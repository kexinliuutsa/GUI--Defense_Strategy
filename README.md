# Cross-Scale Behavioral Defense for Humanized GUI Agents

This repository contains a cross-scale behavioral defense for detecting
humanized mobile GUI-agent behavior.

The project builds on the Agent Humanization Benchmark (AHB), which shows
that modifying swipe trajectories, interaction intervals, and tap durations
can substantially weaken behavioral detectors operating on individual actions.

Our central observation is that suppressing one behavioral representation does
not necessarily eliminate all source-conditioned structure. Under the evaluated
humanization methods, detectable structure can remain in other representations
or at other behavioral scales.

Rather than relying on a single human-vs-agent behavioral fingerprint, the
defense combines complementary session-level and cross-action signals.

---

## Motivation

Behavioral GUI-agent detectors are often evaluated on individual gestures.

This creates a natural attack surface: if an attacker knows which representation
is being measured, it can directly humanize that representation.

For example:

- B-spline perturbation modifies the geometry of individual swipe trajectories.
- History Matching replaces or adjusts swipe behavior using empirical human
  trajectory statistics.
- Long Tap modifies action duration.
- Fake-action methods alter interaction timing by injecting additional actions.

Our experiments show that these transformations can substantially reduce the
accuracy of detectors operating on the directly targeted representation.

However, under the evaluated attacks, source-conditioned structure often remains
visible at other behavioral scales.

Examples include:

- session-level behavioral distributions,
- relationships between consecutive actions,
- spatial reuse across a session,
- temporal organization across actions,
- and specialized artifacts introduced by fake-action generation.

This motivates a cross-scale defense.

---

# Frozen Defense

The final frozen defense contains four complementary components.

## 1. Session Distribution Head

The first head models behavioral statistics across a session.

To reduce dependence on trivial bookkeeping shortcuts, we exclude features such
as:

- number of actions,
- total session span,
- point-count statistics,
- and simple two-point / multi-point action ratios.

The remaining representation captures behavioral distributions including:

- inter-action timing,
- action-duration statistics,
- spatial reuse,
- movement statistics,
- and session-level behavioral variability.

---

## 2. Strict Cross-Action Head

The second head intentionally avoids relying on the internal geometry of
individual gestures.

Instead, it models relationships across actions.

The frozen strict representation includes statistics derived from:

- inter-action gaps,
- direction consistency,
- start-location reuse,
- end-location reuse,
- duration-displacement coupling,
- and consecutive-action similarity.

This head is designed to remain useful when individual swipe trajectories have
already been humanized.

---

## 3. Fake-Action Artifact Head

Some fake-action humanization strategies introduce a distinctive closed-loop
action pattern.

A fake action can travel a substantial path while ending close to its starting
position.

We therefore include a lightweight specialized detector for this artifact.

This component is intentionally treated as attack-family-specific rather than
as a universal agent signature.

---

## 4. Conditional Human-Reference Head

A source classifier can fail in two different ways.

An agent trajectory may move toward the Human distribution, but it may also
overshoot the source-classification boundary while remaining far outside the
actual Human support.

To detect this second case, we include a Human-only support model.

The Human-reference head is evaluated only when the three base heads do not
already classify the behavior as agent-generated.

The final decision rule is:

```text
Session Distribution
        OR
Strict Cross-Action
        OR
Fake-Action Artifact
        |
        v
    Base Decision
        |
        | if base misses
        v
Conditional Human-Reference Check
        |
        v
   Final Decision
     Agent
