# Full-Denominator Audit

The primary rolling detector starts after four observed actions because its
cross-action features require behavioral context.

To verify that the primary results are not caused by excluding difficult early
actions, we additionally evaluate **all prefixes**, including actions 1--3.

No early actions are removed in this audit.

The machine-readable results are available in
[`full_denominator_table.csv`](./full_denominator_table.csv).

---

## Long-Tap Results

Long Tap is the most challenging evaluated humanization mechanism for the final
defense.

When all prefixes are included:

| Domain | Swipe Accuracy | Interval Accuracy | Tap Accuracy |
|---|---:|---:|---:|
| Social Media | 89.02% | 89.48% | **87.36%** |
| Shopping | 93.25% | 88.83% | **85.31%** |
| Video Streaming | 88.26% | 90.10% | **88.42%** |
| Trip Planning | 93.31% | 89.64% | **86.54%** |
| Office & Learning | 97.50% | 93.70% | **90.54%** |

Thus, even without the four-action context restriction, Long-Tap tap-level
accuracy remains between **85.31% and 90.54%** across the five task domains.

---

## Comparison with the Primary Rolling Setting

The primary evaluation uses prefixes with at least four observed actions.

Including prefixes 1--3 lowers accuracy, particularly for Tap units:

| Domain | Long-Tap Tap Acc. (`prefix >= 4`) | All Prefixes | Change |
|---|---:|---:|---:|
| Social Media | 93.30% | 87.36% | -5.94 pp |
| Shopping | 90.79% | 85.31% | -5.48 pp |
| Video Streaming | 93.62% | 88.42% | -5.20 pp |
| Trip Planning | 91.86% | 86.54% | -5.32 pp |
| Office & Learning | 95.27% | 90.54% | -4.73 pp |

The degradation is therefore systematic but moderate.

---

## Human False Positives

The main cost of evaluating very short prefixes is not only lower agent
detection accuracy, but substantially higher Human false positives.

Under participant-disjoint Human out-of-fold evaluation:

| Unit | Human FPR |
|---|---:|
| Swipe | 6.28% |
| Interval | 6.51% |
| Tap | 12.48% |
| All prefixes | 10.51% |

For comparison, the primary full-session Human false-positive operating point
is **2.43%**.

This indicates that prefixes containing only one to three actions provide
insufficient context for reliable cross-action discrimination.

---

## Interpretation

The four-action requirement is therefore treated as an **operational context
requirement**, not as an attack-specific filtering rule.

The full-denominator audit shows that:

1. Long-Tap robustness does not disappear when early actions are restored.
2. Early prefixes substantially increase Human false positives.
3. The primary `prefix >= 4` setting provides a more meaningful operating point
   for a detector based on cross-action behavior.

The full-denominator results are reported as a robustness and fairness audit;
they are not used to retune the frozen defense.
