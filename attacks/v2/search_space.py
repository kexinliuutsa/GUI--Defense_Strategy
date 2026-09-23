from __future__ import annotations

from dataclasses import dataclass

import numpy as np


LATENT_KEYS = [
    "z_spatial",
    "z_frequency",
    "z_duration",
    "z_gap",
    "z_jitter",
    "z_heterogeneity",
    "z_use_spatial",
    "z_use_temporal",
]


LATENT_SPACE = {
    "z_spatial": (0.0, 1.0),
    "z_frequency": (0.0, 1.0),
    "z_duration": (0.0, 1.0),
    "z_gap": (0.0, 1.0),
    "z_jitter": (0.0, 1.0),
    "z_heterogeneity": (0.0, 1.0),
    "z_use_spatial": (0.0, 1.0),
    "z_use_temporal": (0.0, 1.0),
}


@dataclass(frozen=True)
class LatentVector:
    """
    NumPy-backed latent vector z in [0,1]^8.

    Optimizers manipulate only this latent vector.
    The mapping from z to concrete trajectory parameters
    is handled in parameterization.py.
    """

    values: np.ndarray

    def __post_init__(self):
        v = np.asarray(self.values, dtype=float)

        if v.shape != (len(LATENT_KEYS),):
            raise ValueError(
                f"Expected {len(LATENT_KEYS)} dimensions, got {v.shape}"
            )

        object.__setattr__(self, "values", v)

    def to_numpy(self):
        return self.values.copy()

    @classmethod
    def from_numpy(cls, x, clip=False):
        latent = cls(np.asarray(x, dtype=float))

        if clip:
            return latent.clip()

        return latent

    @classmethod
    def from_dict(cls, d, clip=False):
        x = [float(d[k]) for k in LATENT_KEYS]
        return cls.from_numpy(x, clip=clip)

    @classmethod
    def zeros(cls):
        return cls(np.zeros(len(LATENT_KEYS), dtype=float))

    @classmethod
    def midpoint(cls):
        return cls(np.full(len(LATENT_KEYS), 0.5, dtype=float))

    def to_dict(self):
        return {
            k: float(v)
            for k, v in zip(LATENT_KEYS, self.values)
        }

    @property
    def z_spatial(self):
        return float(self.values[0])

    @property
    def z_frequency(self):
        return float(self.values[1])

    @property
    def z_duration(self):
        return float(self.values[2])

    @property
    def z_gap(self):
        return float(self.values[3])

    @property
    def z_jitter(self):
        return float(self.values[4])

    @property
    def z_heterogeneity(self):
        return float(self.values[5])

    @property
    def z_use_spatial(self):
        return float(self.values[6])

    @property
    def z_use_temporal(self):
        return float(self.values[7])

    def clip(self):
        x = self.values.copy()

        for i, key in enumerate(LATENT_KEYS):
            lo, hi = LATENT_SPACE[key]
            x[i] = np.clip(x[i], lo, hi)

        return LatentVector(x)

    def distance(self, other):
        return float(np.linalg.norm(self.values - other.values))
