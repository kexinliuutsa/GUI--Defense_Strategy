from __future__ import annotations

import numpy as np

from attacks.v2.history import QueryHistory
from attacks.v2.history import QueryRecord
from attacks.v2.result import AttackResult


def _query(oracle, session):
    verdict = oracle(session)

    if isinstance(verdict, np.bool_):
        verdict = bool(verdict)

    if not isinstance(verdict, bool):
        raise TypeError(
            "Oracle must return bool only: True=detected, False=escaped."
        )

    return verdict


class AttackRunner:
    """
    Unified decision-only black-box attack runner.

    Original trajectory detection is treated as
    an eligibility check and is not counted toward
    the adversarial query budget.
    """

    def __init__(
        self,
        optimizer,
        oracle,
        mutate,
        cost_fn,
        budget=100,
        seed=0,
        require_initial_detected=True,
    ):
        self.optimizer = optimizer
        self.oracle = oracle
        self.mutate = mutate
        self.cost_fn = cost_fn
        self.budget = int(budget)
        self.rng = np.random.default_rng(seed)
        self.require_initial_detected = bool(require_initial_detected)

    def run(self, session):
        initial_detected = _query(
            self.oracle,
            session,
        )

        if self.require_initial_detected and not initial_detected:
            raise ValueError(
                "Input session already escapes. "
                "Use only sessions detected by the frozen defense."
            )

        history = QueryHistory()

        best_latent = None
        best_cost = None

        self.optimizer.begin(
            session,
            self.budget,
        )

        for query in range(1, self.budget + 1):
            latent = self.optimizer.ask()

            candidate = self.mutate(
                session,
                latent,
                self.rng,
            )

            detected = _query(
                self.oracle,
                candidate,
            )

            escaped = not detected

            cost = float(
                self.cost_fn(
                    session,
                    candidate,
                )
            )

            stage = (
                "search"
                if history.first_escape_query is None
                else "refine"
            )

            history.append(
                QueryRecord(
                    query=query,
                    latent=latent,
                    detected=detected,
                    escaped=escaped,
                    cost=cost,
                    stage=stage,
                )
            )

            self.optimizer.tell(
                latent,
                escaped,
                cost,
            )

            if escaped:
                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    best_latent = latent

        self.optimizer.end(history)

        return AttackResult(
            initial_detected=initial_detected,
            escaped=history.escaped,
            queries_used=history.queries_used,
            queries_to_first_escape=history.first_escape_query,
            best_cost=best_cost,
            best_latent=best_latent,
            history=history,
        )
