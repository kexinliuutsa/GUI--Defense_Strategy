from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from attacks.v2.search_space import LatentVector
from attacks.v2.optimizers.base import Optimizer


@dataclass
class EscapePoint:
    cost: float
    latent: np.ndarray


class HybridOptimizer(Optimizer):
    """
    Black-box hybrid search.

    Design:
    - Before the first escape, behave mostly like random search.
    - Bias random samples toward temporal mutation because previous black-box
      smoke results show escaped queries almost always need temporal changes.
    - After the first escape, refine locally around escaped latent vectors.
    - Keep random restarts to preserve discovery ability.

    It uses only:
    - escaped / detected binary verdict
    - attacker-side perturbation cost
    """

    def __init__(
        self,
        seed: int | None = None,
        warmup: int = 10,
        p_random_before_escape: float = 1.0,
        p_random_after_escape: float = 0.30,
        sigma0: float = 0.18,
        sigma_min: float = 0.035,
        elite_k: int = 5,
        temporal_bias: float = 0.85,
        spatial_bias: float = 0.65,
    ):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.warmup = int(warmup)
        self.p_random_before_escape = float(p_random_before_escape)
        self.p_random_after_escape = float(p_random_after_escape)
        self.sigma0 = float(sigma0)
        self.sigma_min = float(sigma_min)
        self.elite_k = int(elite_k)

        self.temporal_bias = float(temporal_bias)
        self.spatial_bias = float(spatial_bias)

        self.query_count = 0
        self.escapes: list[EscapePoint] = []
        self.last_latent: LatentVector | None = None

    def begin(self, session: Any, budget: int):
        self.query_count = 0
        self.escapes = []
        self.last_latent = None

    def _biased_random(self) -> LatentVector:
        z = self.rng.random(8)

        # The gates are still sampled, but biased toward useful subspaces.
        if self.rng.random() < self.temporal_bias:
            z[7] = 0.5 + 0.5 * self.rng.random()
        else:
            z[7] = 0.5 * self.rng.random()

        if self.rng.random() < self.spatial_bias:
            z[6] = 0.5 + 0.5 * self.rng.random()
        else:
            z[6] = 0.5 * self.rng.random()

        return LatentVector.from_numpy(z)

    def _local_refine(self) -> LatentVector:
        elites = sorted(self.escapes, key=lambda x: x.cost)[: self.elite_k]

        # Prefer lower-cost escapes but keep diversity.
        weights = np.array([1.0 / (1e-6 + e.cost) for e in elites], dtype=float)
        weights = weights / weights.sum()

        idx = int(self.rng.choice(len(elites), p=weights))
        center = elites[idx].latent

        sigma = max(
            self.sigma_min,
            self.sigma0 / np.sqrt(1.0 + 0.25 * len(self.escapes)),
        )

        z = center + self.rng.normal(loc=0.0, scale=sigma, size=center.shape)
        z = np.clip(z, 0.0, 1.0)

        # Preserve the empirical useful gate unless local noise strongly flips it.
        if self.rng.random() < self.temporal_bias:
            z[7] = max(z[7], 0.55)

        return LatentVector.from_numpy(z)

    def ask(self) -> LatentVector:
        self.query_count += 1

        if self.query_count <= self.warmup:
            latent = self._biased_random()
            self.last_latent = latent
            return latent

        if not self.escapes:
            if self.rng.random() < self.p_random_before_escape:
                latent = self._biased_random()
            else:
                latent = LatentVector.from_numpy(self.rng.random(8))

            self.last_latent = latent
            return latent

        if self.rng.random() < self.p_random_after_escape:
            latent = self._biased_random()
        else:
            latent = self._local_refine()

        self.last_latent = latent
        return latent

    def tell(self, latent: LatentVector, escaped: bool, cost: float):
        if escaped:
            self.escapes.append(
                EscapePoint(
                    cost=float(cost),
                    latent=latent.to_numpy().astype(float),
                )
            )

    def end(self, history):
        pass
