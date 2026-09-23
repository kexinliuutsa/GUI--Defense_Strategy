from __future__ import annotations

"""
TPE optimizer for Attack V2.

Objective:
    escaped candidate -> objective = cost
    detected candidate -> objective = failure_penalty

Thus any escape is better than any non-escape,
and cost is optimized only among escaped candidates.
"""

import numpy as np

from attacks.v2.optimizers.base import Optimizer
from attacks.v2.search_space import LatentVector
from attacks.v2.search_space import LATENT_KEYS
from attacks.v2.search_space import LATENT_SPACE


class TPEOptimizer(Optimizer):
    def __init__(
        self,
        seed=0,
        n_startup_trials=10,
        failure_penalty=1_000_000.0,
        multivariate=True,
        group=True,
    ):
        self.seed = int(seed)
        self.n_startup_trials = int(n_startup_trials)
        self.failure_penalty = float(failure_penalty)
        self.multivariate = bool(multivariate)
        self.group = bool(group)

        self.study = None
        self._last_trial = None
        self._last_latent = None
        self.rng = np.random.default_rng(self.seed)

    def begin(self, session, budget):
        try:
            import optuna
        except Exception as e:
            raise ImportError(
                "TPEOptimizer requires optuna. Install it with: pip install optuna"
            ) from e

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        sampler = optuna.samplers.TPESampler(
            seed=self.seed,
            n_startup_trials=self.n_startup_trials,
            multivariate=self.multivariate,
            group=self.group,
        )

        self.study = optuna.create_study(
            direction="minimize",
            sampler=sampler,
        )

        self._last_trial = None
        self._last_latent = None

    def ask(self):
        if self.study is None:
            raise RuntimeError("TPEOptimizer.begin() must be called before ask().")

        trial = self.study.ask()

        values = []

        for key in LATENT_KEYS:
            low, high = LATENT_SPACE[key]
            value = trial.suggest_float(key, low, high)
            values.append(value)

        latent = LatentVector.from_numpy(
            values,
            clip=True,
        )

        self._last_trial = trial
        self._last_latent = latent

        return latent

    def tell(self, latent, escaped, cost):
        if self.study is None:
            raise RuntimeError("TPEOptimizer.begin() must be called before tell().")

        if self._last_trial is None:
            raise RuntimeError("TPEOptimizer.tell() called before ask().")

        if self._last_latent is not None:
            if not np.allclose(
                latent.to_numpy(),
                self._last_latent.to_numpy(),
                rtol=0.0,
                atol=1e-12,
            ):
                raise RuntimeError(
                    "TPEOptimizer received a latent vector different "
                    "from the most recent ask()."
                )

        objective = float(cost) if escaped else self.failure_penalty

        self.study.tell(
            self._last_trial,
            objective,
        )

        self._last_trial = None
        self._last_latent = None

    def end(self, history):
        return

    def trials_dataframe(self):
        if self.study is None:
            return None
        return self.study.trials_dataframe()
