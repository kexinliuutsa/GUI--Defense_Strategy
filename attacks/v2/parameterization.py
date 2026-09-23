from __future__ import annotations

from dataclasses import dataclass

from attacks.v2.search_space import LatentVector


@dataclass(frozen=True)
class TrajectoryParameters:
    use_spatial: bool
    use_temporal: bool

    spatial_amp_px: float
    spatial_freq: int

    duration_scale: float
    gap_scale: float
    timing_jitter: float

    heterogeneity: float


def _linear(z, low, high):
    return low + z * (high - low)


def parameterize(z: LatentVector) -> TrajectoryParameters:
    """
    Fixed mapping phi(z) -> trajectory parameters theta.
    """

    use_spatial = z.z_use_spatial >= 0.5
    use_temporal = z.z_use_temporal >= 0.5

    spatial_amp = _linear(
        z.z_spatial,
        0.0,
        10.0,
    )

    duration = _linear(
        z.z_duration,
        0.55,
        1.80,
    )

    gap = _linear(
        z.z_gap,
        0.55,
        1.80,
    )

    jitter = _linear(
        z.z_jitter,
        0.0,
        0.35,
    )

    heterogeneity = _linear(
        z.z_heterogeneity,
        0.0,
        0.55,
    )

    freqs = [
        1,
        2,
        4,
        6,
        8,
        12,
        16,
    ]

    idx = int(
        min(
            len(freqs) - 1,
            z.z_frequency * len(freqs),
        )
    )

    return TrajectoryParameters(
        use_spatial=bool(use_spatial),
        use_temporal=bool(use_temporal),
        spatial_amp_px=float(spatial_amp),
        spatial_freq=int(freqs[idx]),
        duration_scale=float(duration),
        gap_scale=float(gap),
        timing_jitter=float(jitter),
        heterogeneity=float(heterogeneity),
    )
