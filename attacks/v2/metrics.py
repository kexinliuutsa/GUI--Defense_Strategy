from __future__ import annotations

"""
Cost adapter for Attack V2.

We reuse legacy perturbation_metrics() so that
Attack V2 evaluates perturbation size using the
same cost definition as the original attack.
"""

from attacks.blackbox_adaptive_attack import perturbation_metrics


DEFAULT_COST_KEY = "search_cost"


def compute_metrics(original_session, candidate_session):
    metrics = perturbation_metrics(
        original_session,
        candidate_session,
    )

    if not isinstance(metrics, dict):
        raise TypeError("perturbation_metrics() must return a dict.")

    return metrics


def cost_fn(
    original_session,
    candidate_session,
    cost_key=DEFAULT_COST_KEY,
):
    metrics = compute_metrics(
        original_session,
        candidate_session,
    )

    if cost_key not in metrics:
        raise KeyError(
            f"Missing cost key {cost_key!r}. "
            f"Available keys: {sorted(metrics.keys())}"
        )

    return float(metrics[cost_key])


def compact_metrics(original_session, candidate_session):
    metrics = compute_metrics(
        original_session,
        candidate_session,
    )

    out = {}

    for k, v in metrics.items():
        try:
            out[k] = float(v)
        except Exception:
            out[k] = v

    return out
