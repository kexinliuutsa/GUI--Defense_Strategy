from __future__ import annotations

import numpy as np

from attacks.v2.optimizers.base import Optimizer
from attacks.v2.search_space import LatentVector
from attacks.v2.search_space import LATENT_KEYS


class RandomOptimizer(Optimizer):
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def begin(self, session, budget):
        return

    def ask(self):
        return LatentVector(
            self.rng.uniform(
                0.0,
                1.0,
                size=len(LATENT_KEYS),
            )
        )

    def tell(self, latent, escaped, cost):
        return

    def end(self, history):
        return
