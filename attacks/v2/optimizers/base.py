from __future__ import annotations

from abc import ABC
from abc import abstractmethod

from attacks.v2.search_space import LatentVector
from attacks.v2.history import QueryHistory


class Optimizer(ABC):
    """
    Unified black-box optimizer interface.

    Optimizers only manipulate latent vector z.
    They never see detector scores, margins,
    thresholds, features, task-cluster diagnostics,
    or gradients.
    """

    def begin(self, session, budget):
        pass

    @abstractmethod
    def ask(self) -> LatentVector:
        raise NotImplementedError

    @abstractmethod
    def tell(
        self,
        latent: LatentVector,
        escaped: bool,
        cost: float,
    ):
        raise NotImplementedError

    def end(self, history: QueryHistory):
        pass
