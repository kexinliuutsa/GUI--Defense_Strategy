# Usage

This repository contains the final implementation and evaluation scripts for the cross-scale behavioral defense.

## Source files

- `src/session_structure_defense.py`  
  Extracts session-level behavioral features and implements the session-structure detector.

- `src/fake_action_detector.py`  
  Detects closed-loop fake-action artifacts.

- `src/defense_assembly.py`  
  Combines the complementary defense heads.

## Evaluation

- `evaluation/clean_oof_evaluation.py`  
  Participant-disjoint clean evaluation.

- `evaluation/leave_one_generator_out.py`  
  Leave-one-agent-family-out generalization evaluation.

- `evaluation/fixed_threshold_audit.py`  
  Fixed-threshold audit for cross-generator transfer.

## Final decision

The final defense combines:

```text
No-bookkeeping session head
        OR
Strict cross-action head
        OR
Fake-action artifact head
