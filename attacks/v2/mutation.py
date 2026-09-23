from __future__ import annotations

"""
Mutation adapter for Attack V2.

Attack V2 optimizers propose only latent vector z.

This module performs:

    z
        ↓
    parameterization(z)
        ↓
    legacy MutationSpec
        ↓
    legacy mutate_session()

The trajectory mutation executor itself is not
reimplemented here.
"""

from attacks.blackbox_adaptive_attack import MutationSpec
from attacks.blackbox_adaptive_attack import (
    mutate_session as legacy_mutate_session,
)

from attacks.v2.search_space import LatentVector
from attacks.v2.parameterization import parameterize


def latent_to_spec(latent: LatentVector):
    params = parameterize(latent)

    if params.use_spatial:
        spatial_amp_px = params.spatial_amp_px
        spatial_freq = params.spatial_freq
    else:
        spatial_amp_px = 0.0
        spatial_freq = 1

    if params.use_temporal:
        duration_scale = params.duration_scale
        gap_scale = params.gap_scale
        timing_jitter = params.timing_jitter
        heterogeneity = params.heterogeneity
    else:
        duration_scale = 1.0
        gap_scale = 1.0
        timing_jitter = 0.0
        heterogeneity = 0.0

    return MutationSpec(
        family="v2_unified",
        spatial_amp_px=float(spatial_amp_px),
        spatial_freq=int(spatial_freq),
        duration_scale=float(duration_scale),
        gap_scale=float(gap_scale),
        timing_jitter=float(timing_jitter),
        heterogeneity=float(heterogeneity),
    )


def mutate(session, latent: LatentVector, rng):
    spec = latent_to_spec(latent)

    return legacy_mutate_session(
        session,
        spec,
        rng,
    )
