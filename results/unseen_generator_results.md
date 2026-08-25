# Unseen-Generator Validation

We evaluate whether the frozen defense transfers to agent generators that are
excluded from source-head training and calibration.

For each fold, one Raw-agent generator family is held out entirely.

## Aggregate Long-Tap Results

| Metric | Base Defense | Final Defense | Gain |
|---|---:|---:|---:|
| Micro Average | 63.33% | **75.75%** | **+12.42 pp** |
| Macro Average | 63.00% | **74.28%** | **+11.28 pp** |

The Human-reference extension therefore improves Long-Tap detection even when
the evaluated generator is excluded from source-head training.

## Held-Out Generator Results

| Held-Out Generator | Final Long-Tap Detection |
|---|---:|
| Claude | **100.00%** |
| GPT-4o | **100.00%** |
| CPM | 63.16% |
| AutoGLM | 56.84% |
| UI-TARS | 51.40% |

The result is clearly generator-dependent. We therefore do not claim uniform
robustness across agent families.

Instead, this experiment shows that the Human-reference recovery observed in
the main evaluation is not limited to generators used during source-head
training.
