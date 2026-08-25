# AHB Table 1 Comparison

This page presents a human-readable comparison between the detection accuracies
reported in AHB Table 1 and our frozen cross-scale defense.

The complete machine-readable results are available in
[`ahb_table1_comparison.csv`](./ahb_table1_comparison.csv).

> **Protocol note:** The statistical units are aligned where possible, but the
> classifier training protocols are not identical. Our defense is trained on
> clean Human + Raw-agent data and remains frozen during humanization evaluation.

---

## Long Tap

Long Tap directly modifies tap duration, so the most relevant comparison is
AHB's tap detector versus our cross-scale detector evaluated on Tap units.

| Domain | AHB Tap Acc. | Ours Tap Acc. | Difference |
|---|---:|---:|---:|
| Social Media | 63.41% | **93.30%** | **+29.89 pp** |
| Shopping | 61.33% | **90.79%** | **+29.46 pp** |
| Video Streaming | 61.86% | **93.62%** | **+31.76 pp** |
| Trip Planning | 62.64% | **91.86%** | **+29.22 pp** |
| Office & Learning | 62.63% | **95.27%** | **+32.64 pp** |

---

## B-spline

B-spline modifies intra-swipe trajectory geometry.

| Domain | AHB SVM | AHB XGB | Ours Swipe |
|---|---:|---:|---:|
| Social Media | 96.33% | 98.17% | **98.04%** |
| Shopping | 97.74% | 99.25% | **97.19%** |
| Video Streaming | 96.50% | 98.50% | **93.42%** |
| Trip Planning | 97.26% | 99.09% | **100.00%** |
| Office & Learning | 97.39% | 99.13% | **98.70%** |

The important observation here is not uniform superiority over every AHB
classifier. Instead, B-spline has almost no effect on the frozen cross-scale
decision relative to Raw-agent behavior.

---

## History Matching

History Matching also targets swipe behavior using human trajectory statistics.

| Domain | AHB SVM | AHB XGB | Ours Swipe |
|---|---:|---:|---:|
| Social Media | 96.33% | 94.50% | **98.04%** |
| Shopping | 93.23% | 99.25% | **97.19%** |
| Video Streaming | 93.00% | 98.50% | **93.42%** |
| Trip Planning | 89.95% | 98.63% | **100.00%** |
| Office & Learning | 93.91% | 97.39% | **98.70%** |

As with B-spline, the key result is stability across behavioral scales rather
than uniform dominance over every action-level classifier.

---

## Raw Agents

Raw-agent results provide the clean reference condition.

| Domain | AHB SVM | AHB XGB | Ours Swipe | Ours Interval | Ours Tap |
|---|---:|---:|---:|---:|---:|
| Social Media | 98.17% | 100.00% | 98.04% | 99.35% | 99.94% |
| Shopping | 98.87% | 100.00% | 97.19% | 98.87% | 99.45% |
| Video Streaming | 98.50% | 100.00% | 93.42% | 97.55% | 99.66% |
| Trip Planning | 98.17% | 99.54% | 100.00% | 99.71% | 99.90% |
| Office & Learning | 98.26% | 100.00% | 98.70% | 99.41% | 99.70% |

---

## Interpretation

The comparison should not be read as a claim that our detector uniformly
outperforms every AHB detector in every setting.

The main result is a **transfer barrier across behavioral scales**:

- swipe-targeted humanization has little effect on the frozen cross-scale
  decision;
- Long Tap substantially weakens AHB's duration-based tap detector while our
  cross-scale detector retains much higher detection accuracy;
- the defense is not retrained for individual humanization conditions.

Fake Interval is excluded from this exact comparison because the current
offline implementation uses a coordinate-materialized replay proxy.
