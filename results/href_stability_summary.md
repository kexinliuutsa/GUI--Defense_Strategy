# Human-Reference Stability Audit

This audit examines whether B-spline and History Matching cause numerical
instability in the frozen Human-reference component.

The machine-readable summary is available in
[`href_stability_summary.csv`](./href_stability_summary.csv).

---

## Final-Decision Stability

Across 7,596 evaluated prefixes per condition:

| Condition | Final-Decision Flip Rate |
|---|---:|
| B-spline | **0.0000%** |
| History Matching | **0.0263%** |

The final cross-scale decision is therefore essentially unchanged under the
evaluated swipe-humanization transformations.

---

## Continuous HREF Scores

Continuous Human-reference scores are not perfectly invariant.

| Condition | Median `|ΔHREF|` | 95th Percentile | Maximum |
|---|---:|---:|---:|
| B-spline | 0.0000 | 0.2391 | 3,967,678.43 |
| History Matching | 0.0000 | 0.0301 | 0.7271 |

The large B-spline maximum is caused by a small number of heavy-tailed
cross-action statistics rather than widespread changes across the population.

---

## Source of the B-spline Extremes

The largest B-spline HREF shifts are dominated by
`direction_similarity_cv`.

In the audited extreme cases, the corresponding Raw value can become extremely
large, while B-spline transforms it to a much smaller but still out-of-support
value.

This produces a large difference in standardized Human-reference distance.

The behavior is therefore better described as a heavy-tailed support-distance
effect than as evidence that B-spline generally destabilizes the detector.

---

## Operational Relevance

Most importantly, the extreme B-spline HREF values do not determine the final
classification.

Among the top 20 B-spline HREF outliers:

| HREF Delta Threshold | Extreme Cases | Cases Entering Conditional HREF |
|---|---:|---:|
| > 10 | 20 | **0** |
| > 100 | 20 | **0** |
| > 1,000 | 20 | **0** |
| > 100,000 | 20 | **0** |
| > 1,000,000 | 20 | **0** |

These sessions are already detected by the base source heads before the
conditional Human-reference check is reached.

---

## Interpretation

We therefore do **not** claim that the internal representation is invariant to
B-spline or History Matching.

The supported claim is narrower:

> Swipe humanization leaves the frozen cross-scale detection decision
> essentially unchanged, even though some continuous Human-reference scores
> can exhibit heavy-tailed shifts.

This audit is diagnostic only. No feature, threshold, or model component is
modified based on these results.
