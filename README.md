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
```
The frozen configuration is stored in:`configs/frozen_defense.json`

## Main Results

### Session-Level Detection

At the frozen operating point:

| Condition | Detection Rate |
|---|---:|
| Raw agents | 100.0% |
| B-spline | 100.0% |
| History Matching | 100.0% |
| Long Tap | 76.6% |

**Primary Human false-positive rate: 2.43%.**

The defense is trained and calibrated using clean Human and Raw-agent data only. Humanization conditions are evaluated after the operating point is frozen.

---

## Comparison with AHB Table 1

We additionally evaluate the frozen defense using the behavioral units and task clusters used in the Agent Humanization Benchmark (AHB).

The complete comparison is available at:

`results/ahb_table1_comparison.csv`

One representative result is **Long Tap**, which directly targets tap-duration-based detection.

| Domain | AHB Tap Accuracy | Ours |
|---|---:|---:|
| Social Media | 63.4% | **93.3%** |
| Shopping | 61.3% | **90.8%** |
| Video Streaming | 61.9% | **93.6%** |
| Trip Planning | 62.6% | **91.9%** |
| Office & Learning | 62.6% | **95.3%** |

These results suggest that modifying tap duration can substantially weaken the targeted tap detector while leaving source-conditioned structure detectable at other behavioral scales.

For Swipe units, our reported accuracy uses an AHB-style 30% stratified reporting denominator. The cross-scale detector itself remains frozen and is not retrained on each humanization condition.

---

## Swipe Humanization

B-spline and History Matching modify the intra-swipe representation targeted by the original action-level detector.

The frozen cross-scale decision is highly stable:

| Humanization | Final-Decision Flip Rate |
|---|---:|
| B-spline | 0.0000% |
| History Matching | 0.0263% |

We interpret this as **decision robustness** rather than claiming representation invariance.

Detailed score-stability diagnostics are provided in:

`results/href_stability_summary.csv`

---

## Full-Denominator Audit

The primary rolling detector begins after four observed actions because cross-action context is required.

We also evaluate every prefix, including actions 1–3, to verify that the result is not caused by excluding difficult early actions.

Under this full-denominator evaluation, Long-Tap tap-level accuracy remains high:

| Domain | Ours, All Prefixes |
|---|---:|
| Social Media | 87.4% |
| Shopping | 85.3% |
| Video Streaming | 88.4% |
| Trip Planning | 86.5% |
| Office & Learning | 90.5% |

Detailed results are available in:

`results/full_denominator_table.csv`

Short prefixes produce substantially higher Human false-positive rates, supporting the four-action requirement as an operational context requirement rather than an attack-specific filtering rule.

---

## Evaluation Protocol

The main evaluation follows these constraints:

- Source heads are trained using clean Human and Raw-agent data only.
- Human-reference models use Human data only.
- Humanization conditions do not retrain the final detector.
- Primary thresholds are frozen before attack evaluation.
- Human rolling false-positive rates use participant-disjoint out-of-fold predictions.
- AHB task clusters are recovered from the original participant/session task provenance.
- The primary rolling detector starts after four observed actions.

The detailed protocol is documented in:

`docs/evaluation_protocol.md`

---

## Repository Structure

```text
GUI--Defense_Strategy/
├── README.md
├── requirements.txt
│
├── configs/
│   └── frozen_defense.json
│
├── src/
│   ├── defense.py
│   ├── features/
│   │   ├── session_features.py
│   │   ├── cross_action_features.py
│   │   └── fake_action_features.py
│   └── heads/
│       ├── source_heads.py
│       ├── human_reference.py
│       └── fake_artifact.py
│
├── evaluation/
│   ├── component_evaluation.py
│   ├── unseen_generator_validation.py
│   ├── full_denominator_audit.py
│   └── href_stability_audit.py
│
├── results/
│   ├── README.md
│   ├── ahb_table1_comparison.csv
│   ├── primary_session_results.csv
│   ├── unseen_generator_results.csv
│   ├── full_denominator_table.csv
│   └── href_stability_summary.csv
│
└── docs/
    ├── methodology.md
    ├── evaluation_protocol.md
    └── limitations.md
```

---

## Scope and Limitations

The results should not be interpreted as evidence of an immutable behavioral fingerprint.

The current evidence supports a narrower claim:

> Under the evaluated humanization mechanisms, suppressing one behavioral representation does not necessarily eliminate all source-conditioned structure. Detectable structure can remain at other behavioral scales.

Important limitations include:

- B-spline, History Matching, and Long Tap are predefined humanization mechanisms rather than fully adaptive attacks against the final defense.
- Performance is generator-dependent.
- The primary rolling detector requires four observed actions before producing its operational decision.
- Fake Interval currently relies on a coordinate-materialized offline replay proxy and is therefore excluded from the exact author-component comparison.
- Rare heavy-tailed Human-reference score changes exist, although the largest observed cases do not affect the final conditional decision.
- The current results demonstrate attack-transfer barriers under the evaluated threat model, not permanent robustness against a fully adaptive attacker.

---

## Reproducibility

The repository separates clean training/calibration from humanization evaluation and stores the frozen operating parameters explicitly.

Evaluation scripts reproduce:

- session-level detection,
- AHB-compatible component evaluation,
- unseen-generator validation,
- full-denominator rolling evaluation,
- Human-reference stability diagnostics.

## Status

The defense architecture and primary operating point are frozen.

Current work focuses on reproducibility packaging and paper artifacts rather than further detector tuning.
