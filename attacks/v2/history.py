from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field

from attacks.v2.search_space import LatentVector


@dataclass
class QueryRecord:
    """
    One adversarial oracle query.

    Query index is 1-based and counts candidate
    trajectories only. The original trajectory
    eligibility check is not counted.
    """

    query: int
    latent: LatentVector
    detected: bool
    escaped: bool
    cost: float
    stage: str = "search"


@dataclass
class QueryHistory:
    records: list[QueryRecord] = field(default_factory=list)

    def append(self, record: QueryRecord):
        self.records.append(record)

    @property
    def queries_used(self):
        return len(self.records)

    @property
    def first_escape_query(self):
        for r in self.records:
            if r.escaped:
                return r.query
        return None

    @property
    def escaped(self):
        return any(r.escaped for r in self.records)

    def best_escape(self):
        escaped_records = [
            r
            for r in self.records
            if r.escaped
        ]

        if not escaped_records:
            return None

        return min(
            escaped_records,
            key=lambda r: r.cost,
        )

    def escape_at(self, budget):
        return any(
            r.query <= budget and r.escaped
            for r in self.records
        )

    def best_cost_at(self, budget):
        candidates = [
            r.cost
            for r in self.records
            if r.query <= budget and r.escaped
        ]

        if not candidates:
            return None

        return float(min(candidates))

    def to_list(self):
        out = []

        for r in self.records:
            out.append(
                {
                    "query": int(r.query),
                    "latent": r.latent.to_dict(),
                    "detected": bool(r.detected),
                    "escaped": bool(r.escaped),
                    "cost": float(r.cost),
                    "stage": r.stage,
                }
            )

        return out
