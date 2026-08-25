# Results

This directory contains the frozen evaluation results used in the main analysis.

## AHB Table 1 Comparison

`ahb_table1_comparison.csv` compares the frozen cross-scale defense with the detection accuracies reported in AHB Table 1.

The `ahb_*` columns are values reported by the AHB paper.

The `ours_*` columns are produced by the frozen cross-scale defense.

### Protocol Notes

- AHB SVM/XGBoost results follow the original AHB classifier protocol.
- Our source heads are trained using clean Human and Raw-agent data only.
- Humanization conditions do not retrain the final detector.
- `ours_swipe_acc` uses an AHB-style 30% stratified reporting denominator.
- `ours_interval_acc` and `ours_tap_acc` use all eligible units after the four-action context requirement.
- The statistical units are aligned where possible, but the classifier training protocols are not identical.

Fake Interval is excluded from the exact component comparison because the current offline implementation uses a coordinate-materialized replay proxy.

## Full-Denominator Audit

[`full_denominator_table.md`](./full_denominator_table.md) evaluates the frozen
rolling defense on every behavioral prefix, including actions 1--3.

This audit tests whether the primary `prefix >= 4` results depend on excluding
early actions.

No thresholds are modified based on this evaluation.
