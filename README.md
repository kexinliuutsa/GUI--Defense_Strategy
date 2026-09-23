# Cross-Scale Behavioral Defense for Humanized GUI Agents

This repository contains a frozen V1+V2+V3 behavioral defense for detecting humanized mobile GUI-agent behavior.

## Defense Architecture

The current public defense replaces the previous four-component cross-scale implementation with a hardened three-layer frozen defense.

The final decision is:

```text
Final Detection = V1 OR V2 OR V3
V1. Source / Human-Reference Behavioral Ensemble

V1 corresponds to the original cross-scale behavioral defense. It combines:

Session Distribution Head
Strict Cross-Action Head
Fake-Action Artifact Head
Conditional Human-Reference Head

This layer captures session-level distributions, cross-action relationships, fake-action artifacts, and whether a base-detector miss still lies outside Human behavioral support.

V2. Temporal-Shape Head

V2 is a temporal-shape detector. It extracts gesture-level timestamp-rhythm features and applies a frozen temporal policy.

The required temporal policy artifact is stored at:

results/defense_v2_temporal_policy.joblib

The temporal feature implementation is stored at:

src/heads/temporal_shape.py
V3. Task-Cluster Action-Interval Head

V3 is a task-cluster-specific action-interval head. It computes inter-gesture intervals from each gesture's start and end timestamps and checks whether the interval violation rate exceeds the frozen threshold.

V3 is mainly a long-gap interval detector rather than a short-gap detector.

The task-cluster lookup data is stored at:

results/ahb_cheap_layer/session_predictions.csv
Repository Structure
GUI--Defense_Strategy/
├── README.md
├── USAGE.md
├── DATA.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── defense.py
│   └── heads/
│       ├── __init__.py
│       └── temporal_shape.py
├── evaluation/
│   └── run_blackbox_attack.py
└── results/
    ├── defense_v2_temporal_policy.joblib
    └── ahb_cheap_layer/
        └── session_predictions.csv
Public Entry Point

The main public defense implementation is:

src/defense.py

It exposes the hardened frozen V1+V2+V3 defense.

Status

The public repository currently contains the frozen V1+V2+V3 defense implementation and required frozen artifacts.
