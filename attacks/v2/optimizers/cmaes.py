from __future__ import annotations

"""
CMA-ES optimizer for Attack V2.

The runner is sequential, while CMA-ES is
population-based. This wrapper buffers one
generation internally.
"""

import numpy as np

from attacks.v2.optimizers.base import Optimizer
from attacks.v2.search_space import LatentVector
from attacks.v2.search_space import LATENT_KEYS


class CMAESOptimizer(Optimizer):
    def __init__(
        self,
        seed=0,
        sigma0=0.25,
        popsize=None,
        x0=None,
        failure_penalty=1_000_000.0,
    ):
        self.seed = int(seed)
        self.sigma0 = float(sigma0)
        self.popsize = popsize
        self.x0 = x0
        self.failure_penalty = float(failure_penalty)
        self.dim = len(LATENT_KEYS)

        self.rng = np.random.default_rng(self.seed)

        self.es = None
        self._pending = []
        self._solutions = []
        self._objectives = []
        self._last_solution = None

    def begin(self, session, budget):
        try:
            import cma
        except Exception as e:
            raise ImportError(
                "CMAESOptimizer requires cma. Install it with: pip install cma"
            ) from e

        if self.x0 is None:
            x0 = np.full(
                self.dim,
                0.5,
                dtype=float,
            )
        else:
            x0 = np.asarray(
                self.x0,
                dtype=float,
            )

            if x0.shape != (self.dim,):
                raise ValueError(
                    f"x0 must have shape {(self.dim,)}, got {x0.shape}"
                )

        opts = {
            "bounds": [
                np.zeros(self.dim),
                np.ones(self.dim),
            ],
            "seed": self.seed,
            "verbose": -9,
        }

        if self.popsize is not None:
            opts["popsize"] = int(self.popsize)

        self.es = cma.CMAEvolutionStrategy(
            x0,
            self.sigma0,
            opts,
        )

        self._pending = []
        self._solutions = []
        self._objectives = []
        self._last_solution = None

    def _ensure_population(self):
        if self.es is None:
            raise RuntimeError("CMAESOptimizer.begin() must be called before ask().")

        if not self._pending:
            self._pending = list(self.es.ask())

    def ask(self):
        self._ensure_population()

        x = np.asarray(
            self._pending.pop(0),
            dtype=float,
        )

        x = np.clip(
            x,
            0.0,
            1.0,
        )

        self._last_solution = x.copy()

        return LatentVector.from_numpy(
            x,
            clip=True,
        )

    def tell(self, latent, escaped, cost):
        if self.es is None:
            raise RuntimeError("CMAESOptimizer.begin() must be called before tell().")

        if self._last_solution is None:
            raise RuntimeError("CMAESOptimizer.tell() called before ask().")

        objective = float(cost) if escaped else self.failure_penalty

        self._solutions.append(self._last_solution.copy())
        self._objectives.append(float(objective))

        self._last_solution = None

        if not self._pending:
            self.es.tell(
                self._solutions,
                self._objectives,
            )

            self._solutions = []
            self._objectives = []

    def end(self, history):
        return
