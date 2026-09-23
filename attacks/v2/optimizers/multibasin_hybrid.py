from __future__ import annotations

from typing import Any
import numpy as np

from attacks.v2.search_space import LatentVector
from attacks.v2.optimizers.base import Optimizer


class MultiBasinHybridOptimizer(Optimizer):
    """
    Multi-basin explore-then-refine black-box attack.

    Difference from normal Hybrid:
    - Normal Hybrid may refine immediately after the first escape.
    - Multi-basin Hybrid reserves an early exploration stage even after escape,
      so it can discover multiple escape basins before local refinement.

    It only uses:
    - latent parameters z
    - binary outcome escaped/detected
    - perturbation cost

    It does NOT use:
    - detector architecture
    - detector features
    - thresholds
    - V1/V2/V3 layer feedback
    """

    def __init__(
        self,
        seed: int | None = None,
        explore_fraction: float = 0.60,
        basin_threshold: float = 0.38,
        top_k_basins: int = 3,
        candidate_pool: int = 128,
        random_restart_prob: float = 0.10,
    ):
        self.rng = np.random.default_rng(seed)
        self.seed = seed

        self.explore_fraction = float(explore_fraction)
        self.basin_threshold = float(basin_threshold)
        self.top_k_basins = int(top_k_basins)
        self.candidate_pool = int(candidate_pool)
        self.random_restart_prob = float(random_restart_prob)

        self.budget = None
        self.explore_budget = None
        self.query_count = 0
        self.records = []
        self.escapes = []

    def begin(self, session: Any, budget: int):
        self.budget = int(budget)
        self.explore_budget = max(2, int(round(self.budget * self.explore_fraction)))
        self.query_count = 0
        self.records = []
        self.escapes = []

    def ask(self) -> LatentVector:
        self.query_count += 1

        # Stage 1: forced broad exploration.
        if self.query_count <= self.explore_budget:
            z = self._explore_candidate()
            return LatentVector.from_numpy(z)

        # If no escape found, continue broad exploration.
        if not self.escapes:
            z = self._explore_candidate()
            return LatentVector.from_numpy(z)

        # Stage 2: multi-basin local refinement.
        if self.rng.random() < self.random_restart_prob:
            z = self._explore_candidate()
        else:
            z = self._refine_candidate()

        return LatentVector.from_numpy(z)

    def tell(self, latent: LatentVector, escaped: bool, cost: float):
        z = latent.to_numpy().astype(float)
        rec = {
            "x": z.tolist(),
            "escaped": bool(escaped),
            "cost": float(cost),
            "query": int(self.query_count),
        }
        self.records.append(rec)

        if escaped:
            self.escapes.append(rec)

    def end(self, history):
        pass

    # ------------------------------------------------------------------
    # Stage 1: broad exploration
    # ------------------------------------------------------------------

    def _cost_proxy(self, z: np.ndarray) -> float:
        """
        Cheap attack-side cost proxy.
        This uses only mutation-space semantics, not detector internals.
        """
        z = np.clip(np.asarray(z, dtype=float), 0.0, 1.0)

        # z layout:
        # 0 spatial, 1 frequency, 2 duration, 3 gap,
        # 4 jitter, 5 heterogeneity, 6 use_spatial, 7 use_temporal
        spatial = z[0] * (1.0 if z[6] >= 0.5 else 0.20)
        timing = abs(z[2] - 0.36) + abs(z[3] - 0.36)
        jitter = z[4]
        hetero = z[5]
        temporal_switch = 1.0 if z[7] >= 0.5 else 0.0

        return float(
            0.30 * spatial
            + 0.25 * timing
            + 0.20 * jitter
            + 0.20 * hetero
            + 0.05 * temporal_switch
        )

    def _min_dist_to_records(self, z: np.ndarray) -> float:
        if not self.records:
            return 1.0
        X = np.array([r["x"] for r in self.records], dtype=float)
        return float(np.min(np.linalg.norm(X - z, axis=1)))

    def _explore_candidate(self) -> np.ndarray:
        """
        Maximin-style exploration:
        sample many random candidates and choose one that is far from previous trials,
        while avoiding extremely high proxy cost.
        """
        candidates = []

        # Mostly random.
        for _ in range(self.candidate_pool):
            candidates.append(self.rng.random(8))

        # Add a few structured corners to cover switch combinations.
        for use_spatial in [0.25, 0.75]:
            for use_temporal in [0.25, 0.75]:
                z = self.rng.random(8)
                z[6] = use_spatial
                z[7] = use_temporal
                candidates.append(z)

        best_z = None
        best_score = -1e18

        for z in candidates:
            z = np.clip(np.asarray(z, dtype=float), 0.0, 1.0)
            diversity = self._min_dist_to_records(z)
            cost_penalty = self._cost_proxy(z)

            # Exploration prefers diversity, but avoids very costly regions.
            score = 0.85 * diversity - 0.25 * cost_penalty + 0.02 * self.rng.random()

            if score > best_score:
                best_score = score
                best_z = z

        return np.clip(best_z, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Stage 2: multi-basin refinement
    # ------------------------------------------------------------------

    def _cluster_escapes(self):
        """
        Greedy clustering of escaped points into basins.
        Each basin is represented by low-cost escaped points nearby.
        """
        escapes = sorted(self.escapes, key=lambda r: float(r["cost"]))

        basins = []

        for r in escapes:
            z = np.array(r["x"], dtype=float)

            assigned = False
            for basin in basins:
                center = basin["center"]
                if np.linalg.norm(z - center) <= self.basin_threshold:
                    basin["points"].append(r)
                    X = np.array([p["x"] for p in basin["points"]], dtype=float)
                    basin["center"] = X.mean(axis=0)
                    assigned = True
                    break

            if not assigned:
                basins.append({
                    "center": z.copy(),
                    "points": [r],
                })

        scored = []
        for basin in basins:
            costs = np.array([float(p["cost"]) for p in basin["points"]], dtype=float)
            count = len(basin["points"])
            med_cost = float(np.median(costs))

            # Prefer basins with more successes and lower cost.
            score = count + 1.0 / (1.0 + med_cost)
            scored.append((score, basin))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [b for _, b in scored[: self.top_k_basins]]

    def _refine_candidate(self) -> np.ndarray:
        basins = self._cluster_escapes()

        if not basins:
            return self._explore_candidate()

        # Round-robin among top basins to avoid collapsing into only one basin.
        idx = (self.query_count - self.explore_budget - 1) % len(basins)
        basin = basins[idx]
        center = np.array(basin["center"], dtype=float)

        # Later queries use smaller local noise.
        progress = self.query_count / max(1, self.budget)
        if progress < 0.75:
            sigma = 0.16
        else:
            sigma = 0.08

        candidates = []
        for _ in range(self.candidate_pool):
            z = center + self.rng.normal(0.0, sigma, size=8)

            # Occasionally only perturb a small block of dimensions.
            if self.rng.random() < 0.50:
                z2 = center.copy()
                dims = self.rng.choice(8, size=self.rng.choice([1, 2, 3]), replace=False)
                z2[dims] = z[dims]
                z = z2

            candidates.append(np.clip(z, 0.0, 1.0))

        # Pick the candidate that is close to basin but not identical, and lower proxy cost.
        best_z = None
        best_score = -1e18

        failures = [r for r in self.records if not r["escaped"]]
        X_fail = np.array([r["x"] for r in failures], dtype=float) if failures else None

        for z in candidates:
            dist_center = float(np.linalg.norm(z - center))
            cost_penalty = self._cost_proxy(z)

            if X_fail is not None and len(X_fail) > 0:
                dist_fail = float(np.min(np.linalg.norm(X_fail - z, axis=1)))
            else:
                dist_fail = 1.0

            score = (
                -0.35 * dist_center
                + 0.25 * dist_fail
                - 0.35 * cost_penalty
                + 0.02 * self.rng.random()
            )

            if score > best_score:
                best_score = score
                best_z = z

        return np.clip(best_z, 0.0, 1.0)
