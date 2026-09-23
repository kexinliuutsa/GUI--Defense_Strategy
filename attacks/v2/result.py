from __future__ import annotations

from dataclasses import dataclass

from attacks.v2.search_space import LatentVector
from attacks.v2.history import QueryHistory


@dataclass
class AttackResult:
    initial_detected: bool
    escaped: bool
    queries_used: int
    queries_to_first_escape: int | None
    best_cost: float | None
    best_latent: LatentVector | None
    history: QueryHistory

    def to_dict(self):
        if self.best_latent is None:
            best_latent = None
        else:
            best_latent = self.best_latent.to_dict()

        return {
            "initial_detected": bool(self.initial_detected),
            "escaped": bool(self.escaped),
            "queries_used": int(self.queries_used),
            "queries_to_first_escape": self.queries_to_first_escape,
            "best_cost": self.best_cost,
            "best_latent": best_latent,
            "history": self.history.to_list(),
        }
