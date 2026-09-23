from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from attacks.v2.search_space import LatentVector
from attacks.v2.optimizers.base import Optimizer
from attacks.v2.optimizers.llm_batch import BatchOllamaLLMOptimizer


@dataclass
class EscapePoint:
    cost: float
    latent: np.ndarray


class LLMBatchHybridOptimizer(Optimizer):
    """
    Strict black-box LLM + local refinement.

    The LLM does not know the detector and only receives:
    - previous z
    - escaped / detected
    - cost

    Strategy:
    - Before first escape: use LLM-batch proposals for discovery.
    - After first escape: mostly refine locally around escaped candidates.
    - Keep some LLM/random restarts for diversity.
    """

    def __init__(
        self,
        model: str = "qwen3:8b",
        seed: int | None = None,
        candidate_count: int = 8,
        max_history: int = 40,
        timeout: int = 120,
        temperature: float = 0.8,
        p_llm_after_escape: float = 0.20,
        p_random_after_escape: float = 0.10,
        sigma0: float = 0.16,
        sigma_min: float = 0.025,
        elite_k: int = 8,
    ):
        self.model = model
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.llm = BatchOllamaLLMOptimizer(
            model=model,
            seed=seed,
            timeout=timeout,
            temperature=temperature,
            max_history=max_history,
            candidate_count=candidate_count,
            min_distance=0.08,
            fallback_to_random=True,
        )

        self.p_llm_after_escape = float(p_llm_after_escape)
        self.p_random_after_escape = float(p_random_after_escape)
        self.sigma0 = float(sigma0)
        self.sigma_min = float(sigma_min)
        self.elite_k = int(elite_k)

        self.escapes: list[EscapePoint] = []
        self.query_count = 0

    def begin(self, session: Any, budget: int):
        self.llm.begin(session, budget)
        self.escapes = []
        self.query_count = 0

    def _random_latent(self) -> LatentVector:
        return LatentVector.from_numpy(self.rng.random(8))

    def _local_refine(self) -> LatentVector:
        elites = sorted(self.escapes, key=lambda x: x.cost)[: self.elite_k]

        weights = np.array(
            [1.0 / (1e-6 + e.cost) for e in elites],
            dtype=float,
        )
        weights = weights / weights.sum()

        center = elites[int(self.rng.choice(len(elites), p=weights))].latent

        sigma = max(
            self.sigma_min,
            self.sigma0 / np.sqrt(1.0 + 0.20 * len(self.escapes)),
        )

        z = center + self.rng.normal(0.0, sigma, size=8)
        z = np.clip(z, 0.0, 1.0)

        return LatentVector.from_numpy(z)

    def ask(self) -> LatentVector:
        self.query_count += 1

        # Before first escape, let LLM-batch discover.
        if not self.escapes:
            return self.llm.ask()

        u = self.rng.random()

        # Occasionally ask LLM again to propose new global regions.
        if u < self.p_llm_after_escape:
            return self.llm.ask()

        # Occasionally restart randomly.
        if u < self.p_llm_after_escape + self.p_random_after_escape:
            return self._random_latent()

        # Most of the time refine locally around successful evasions.
        return self._local_refine()

    def tell(self, latent: LatentVector, escaped: bool, cost: float):
        self.llm.tell(latent, escaped, cost)

        if escaped:
            self.escapes.append(
                EscapePoint(
                    cost=float(cost),
                    latent=latent.to_numpy().astype(float),
                )
            )

    def end(self, history):
        self.llm.end(history)
