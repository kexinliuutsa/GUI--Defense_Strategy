from __future__ import annotations

import json
import os
import re
import urllib.request
from typing import Any, ClassVar

import numpy as np

from attacks.v2.search_space import LatentVector
from attacks.v2.optimizers.base import Optimizer


KEYS = [
    "z_spatial",
    "z_frequency",
    "z_duration",
    "z_gap",
    "z_jitter",
    "z_heterogeneity",
    "z_use_spatial",
    "z_use_temporal",
]


def _json_from_text(text: str):
    try:
        return json.loads(text)
    except Exception:
        pass

    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass

    return None


def _clip_arr(x):
    try:
        arr = np.array(x, dtype=float).reshape(-1)
        if arr.shape[0] != 8:
            return None
        return np.clip(arr, 0.0, 1.0)
    except Exception:
        return None


class DeepSeekNextGenOptimizer(Optimizer):
    """
    Next-generation DeepSeek API black-box attack optimizers.

    All modes only use:
    - previous latent vector
    - success/failure
    - perturbation cost

    They do NOT use:
    - detector architecture
    - V1/V2/V3
    - thresholds
    - detector features
    - layer-level feedback
    """

    GLOBAL_ARCHIVE: ClassVar[list[dict]] = []

    def __init__(
        self,
        mode: str,
        model: str | None = None,
        seed: int | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: int = 120,
        temperature: float = 0.8,
        candidate_count: int = 8,
        max_history: int = 40,
    ):
        self.mode = mode
        self.model = model or os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
        self.base_url = (
            base_url or os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        ).rstrip("/")
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")

        if not self.api_key:
            raise RuntimeError("Missing DEEPSEEK_API_KEY")

        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.timeout = int(timeout)
        self.temperature = float(temperature)
        self.candidate_count = int(candidate_count)
        self.max_history = int(max_history)

        self.records = []
        self.escapes = []
        self.pending = []
        self.query_count = 0
        self.budget = None

        # For segment-square / latent-block search.
        self.current_center = None
        self.square_block = 4

    # ---------------------------------------------------------------------
    # Lifecycle
    # ---------------------------------------------------------------------

    def begin(self, session: Any, budget: int):
        self.records = []
        self.escapes = []
        self.pending = []
        self.query_count = 0
        self.budget = int(budget)

        self.current_center = self.rng.random(8)
        self.square_block = 4

        # Transfer archive warm start.
        if self.mode == "transfer_archive":
            self._seed_from_global_archive()

    def ask(self) -> LatentVector:
        self.query_count += 1

        if self.pending:
            return self.pending.pop(0)

        if self.mode == "cem":
            self._refill_cem()

        elif self.mode == "bo":
            self._refill_bo()

        elif self.mode == "segment_square":
            self._refill_segment_square()

        elif self.mode == "transfer_archive":
            self._refill_transfer_archive()

        elif self.mode == "multifidelity":
            self._refill_multifidelity()

        else:
            raise ValueError(f"Unknown next-gen mode: {self.mode}")

        if not self.pending:
            self.pending = [self._random_latent() for _ in range(self.candidate_count)]

        return self.pending.pop(0)

    def tell(self, latent: LatentVector, escaped: bool, cost: float):
        z = latent.to_numpy().astype(float)

        rec = {
            "x": z.tolist(),
            "success": bool(escaped),
            "cost": float(cost),
            "query": int(self.query_count),
        }

        self.records.append(rec)

        if escaped:
            self.escapes.append(rec)
            self.current_center = z.copy()

            DeepSeekNextGenOptimizer.GLOBAL_ARCHIVE.append(
                {
                    "x": z.tolist(),
                    "cost": float(cost),
                    "mode": self.mode,
                }
            )

            # Keep archive bounded.
            DeepSeekNextGenOptimizer.GLOBAL_ARCHIVE = sorted(
                DeepSeekNextGenOptimizer.GLOBAL_ARCHIVE,
                key=lambda r: float(r["cost"]),
            )[:256]

    def end(self, history):
        pass

    # ---------------------------------------------------------------------
    # Basic helpers
    # ---------------------------------------------------------------------

    def _random_latent(self) -> LatentVector:
        return LatentVector.from_numpy(self.rng.random(8))

    def _as_latent(self, z) -> LatentVector | None:
        arr = _clip_arr(z)
        if arr is None:
            return None
        return LatentVector.from_numpy(arr)

    def _compact_history(self):
        recent = self.records[-self.max_history :]
        out = []
        for i, r in enumerate(recent):
            out.append(
                {
                    "trial": len(self.records) - len(recent) + i + 1,
                    "z": [round(float(v), 4) for v in r["x"]],
                    "success": int(bool(r["success"])),
                    "cost": round(float(r["cost"]), 6),
                }
            )
        return out

    def _best_successes(self, k=8):
        successes = sorted(
            [r for r in self.records if r["success"]],
            key=lambda r: float(r["cost"]),
        )[:k]

        return [
            {
                "z": [round(float(v), 4) for v in r["x"]],
                "cost": round(float(r["cost"]), 6),
            }
            for r in successes
        ]

    def _summary(self):
        successes = [r for r in self.records if r["success"]]
        failures = [r for r in self.records if not r["success"]]

        def mean_x(rows):
            if not rows:
                return None
            return [round(float(v), 4) for v in np.mean([r["x"] for r in rows], axis=0)]

        return {
            "n_trials": len(self.records),
            "n_successes": len(successes),
            "n_failures": len(failures),
            "success_rate_so_far": len(successes) / len(self.records) if self.records else 0.0,
            "mean_success_z": mean_x(successes),
            "mean_failure_z": mean_x(failures),
            "best_successes": self._best_successes(k=5),
        }

    def _cost_proxy(self, z):
        """
        Cheap attack-side cost proxy.
        This uses mutation-space semantics only, not detector internals.
        """
        z = np.clip(np.array(z, dtype=float), 0.0, 1.0)

        neutral_time = 0.36

        spatial = z[0] * (1.0 if z[6] >= 0.5 else 0.15)
        timing = abs(z[2] - neutral_time) + abs(z[3] - neutral_time)
        jitter = z[4]
        hetero = z[5]

        return float(
            0.30 * spatial
            + 0.25 * timing
            + 0.20 * jitter
            + 0.20 * hetero
            + 0.05 * (1.0 if z[7] >= 0.5 else 0.0)
        )

    def _min_dist(self, z, rows):
        if not rows:
            return 1.0
        z = np.array(z, dtype=float)
        return float(min(np.linalg.norm(z - np.array(r["x"], dtype=float)) for r in rows))

    def _select_diverse(self, candidates, k=None, cost_weight=0.25):
        k = k or self.candidate_count

        arrs = []
        for c in candidates:
            if isinstance(c, LatentVector):
                arr = c.to_numpy()
            else:
                arr = _clip_arr(c)
            if arr is not None:
                arrs.append(np.clip(np.array(arr, dtype=float), 0.0, 1.0))

        if not arrs:
            return [self._random_latent() for _ in range(k)]

        successes = [r for r in self.records if r["success"]]
        failures = [r for r in self.records if not r["success"]]

        selected = []

        while arrs and len(selected) < k:
            best_i = None
            best_score = -1e18

            for i, z in enumerate(arrs):
                cost_proxy = self._cost_proxy(z)

                novelty_fail = self._min_dist(z, failures)

                if successes:
                    dist_success = self._min_dist(z, successes)
                    success_affinity = float(np.exp(-2.0 * dist_success))
                else:
                    success_affinity = 0.0

                if selected:
                    diversity = min(float(np.linalg.norm(z - s)) for s in selected)
                else:
                    diversity = 1.0

                score = (
                    0.40 * success_affinity
                    + 0.30 * novelty_fail
                    + 0.30 * diversity
                    - cost_weight * cost_proxy
                    + 0.02 * self.rng.random()
                )

                if score > best_score:
                    best_score = score
                    best_i = i

            selected.append(arrs.pop(best_i))

        while len(selected) < k:
            selected.append(self.rng.random(8))

        return [LatentVector.from_numpy(np.clip(z, 0.0, 1.0)) for z in selected[:k]]

    # ---------------------------------------------------------------------
    # API
    # ---------------------------------------------------------------------

    def _call_api(self, prompt: str) -> str:
        url = f"{self.base_url}/chat/completions"

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "Return only valid JSON. Do not explain.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
            "stream": False,
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            data = json.loads(r.read().decode("utf-8"))

        return data["choices"][0]["message"]["content"]

    def _prompt_candidates(self, n: int, extra_instruction: str = ""):
        return f"""
You are optimizing a strict black-box objective.

You do not know the target system's architecture, features, thresholds, or internal decision process.
You only observe previous candidate vectors, success/failure outcomes, and cost.

The candidate is an 8-dimensional normalized vector with these generic trajectory-mutation parameters:
{json.dumps(KEYS)}

Each value must be in [0, 1].
A successful candidate with lower cost is better.

Recent trials:
{json.dumps(self._compact_history(), ensure_ascii=False)}

Summary:
{json.dumps(self._summary(), ensure_ascii=False)}

Extra instruction:
{extra_instruction}

Generate {n} diverse candidate vectors.
Return ONLY valid JSON:
{{
  "candidates": [
    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
  ]
}}
""".strip()

    def _llm_candidates(self, n=24, extra_instruction=""):
        try:
            text = self._call_api(self._prompt_candidates(n, extra_instruction))
            obj = _json_from_text(text)
            raw = obj.get("candidates", []) if isinstance(obj, dict) else []

            cands = []
            for x in raw:
                arr = _clip_arr(x)
                if arr is not None:
                    cands.append(arr)

            return cands

        except Exception as e:
            print(f"[{self.mode} LLM fallback random]", repr(e))
            return []

    # ---------------------------------------------------------------------
    # Cheap surrogate / acquisition
    # ---------------------------------------------------------------------

    def _kernel_predict(self, z):
        """
        Nonparametric black-box surrogate:
        estimates success probability and expected inverse cost from past trials.
        """
        if len(self.records) < 4:
            return 0.15, 0.50, 0.50

        z = np.array(z, dtype=float)

        X = np.array([r["x"] for r in self.records], dtype=float)
        y = np.array([1.0 if r["success"] else 0.0 for r in self.records], dtype=float)
        c = np.array([float(r["cost"]) for r in self.records], dtype=float)

        d = np.linalg.norm(X - z, axis=1)

        h = 0.45
        w = np.exp(-(d ** 2) / (2 * h * h)) + 1e-9

        p = float((w * y).sum() / w.sum())
        inv_cost = float((w * (1.0 / (1.0 + c))).sum() / w.sum())
        novelty = float(np.min(d))

        # Bernoulli-like uncertainty.
        uncertainty = float(np.sqrt(max(1e-6, p * (1.0 - p))))

        return p, inv_cost, max(novelty, uncertainty)

    def _acquisition_score(self, z, beta=0.25, lam=0.30):
        p, inv_cost, explore = self._kernel_predict(z)
        return (
            0.65 * p
            + 0.20 * inv_cost
            + beta * explore
            - lam * self._cost_proxy(z)
            + 0.02 * self.rng.random()
        )

    def _rank_by_acquisition(self, candidates, k=None, beta=0.25, lam=0.30):
        k = k or self.candidate_count

        arrs = []
        for c in candidates:
            arr = c.to_numpy() if isinstance(c, LatentVector) else _clip_arr(c)
            if arr is not None:
                arrs.append(arr)

        if not arrs:
            arrs = [self.rng.random(8) for _ in range(64)]

        scored = [(self._acquisition_score(z, beta=beta, lam=lam), z) for z in arrs]
        scored.sort(key=lambda x: x[0], reverse=True)

        top = [z for _, z in scored[: min(64, len(scored))]]
        return self._select_diverse(top, k=k, cost_weight=lam)

    # ---------------------------------------------------------------------
    # Mode 1: LLM-CEM
    # ---------------------------------------------------------------------

    def _cem_population(self, n=256):
        pool = []

        successes = sorted(
            [r for r in self.records if r["success"]],
            key=lambda r: float(r["cost"]),
        )

        if successes:
            elites = successes[: min(12, len(successes))]
            X = np.array([r["x"] for r in elites], dtype=float)

            mean = X.mean(axis=0)

            if len(elites) >= 2:
                cov = np.cov(X.T) + np.eye(8) * 0.015
            else:
                cov = np.eye(8) * 0.030

            # Shrink covariance as more escapes are found.
            shrink = max(0.35, 1.0 / np.sqrt(1.0 + 0.15 * len(successes)))
            cov = cov * shrink + np.eye(8) * 0.008

            for _ in range(n):
                try:
                    z = self.rng.multivariate_normal(mean, cov)
                except Exception:
                    z = mean + self.rng.normal(0.0, 0.12, size=8)
                pool.append(np.clip(z, 0.0, 1.0))

        else:
            # No escape yet: combine LLM seed + random.
            pool.extend(
                self._llm_candidates(
                    n=32,
                    extra_instruction=(
                        "No success may be available yet. Generate broad but plausible "
                        "candidates that explore different parameter interactions."
                    ),
                )
            )
            pool.extend([self.rng.random(8) for _ in range(n)])

        # Random restarts.
        pool.extend([self.rng.random(8) for _ in range(32)])

        # Local neighbors around best successes.
        for r in successes[:8]:
            base = np.array(r["x"], dtype=float)
            for _ in range(12):
                pool.append(np.clip(base + self.rng.normal(0.0, 0.10, size=8), 0.0, 1.0))

        return pool

    def _refill_cem(self):
        pool = self._cem_population(n=256)
        self.pending.extend(self._rank_by_acquisition(pool, k=self.candidate_count, beta=0.20, lam=0.25))

    # ---------------------------------------------------------------------
    # Mode 2: LLM + BO-style acquisition
    # ---------------------------------------------------------------------

    def _refill_bo(self):
        pool = []

        pool.extend(
            self._llm_candidates(
                n=48,
                extra_instruction=(
                    "Generate a mix of exploitative candidates near successful regions "
                    "and exploratory candidates far from repeated failures."
                ),
            )
        )

        # Random candidates.
        pool.extend([self.rng.random(8) for _ in range(192)])

        # Success neighbors.
        successes = sorted(
            [r for r in self.records if r["success"]],
            key=lambda r: float(r["cost"]),
        )

        for r in successes[:10]:
            base = np.array(r["x"], dtype=float)
            for sigma in [0.05, 0.10, 0.18]:
                for _ in range(8):
                    pool.append(np.clip(base + self.rng.normal(0.0, sigma, size=8), 0.0, 1.0))

        self.pending.extend(self._rank_by_acquisition(pool, k=self.candidate_count, beta=0.35, lam=0.30))

    # ---------------------------------------------------------------------
    # Mode 3: Segment-Square / latent-block square search
    # ---------------------------------------------------------------------

    def _refill_segment_square(self):
        """
        Square Attack style, but applied to latent blocks.
        Since current executor only accepts global 8D latent z, this does not
        yet mutate per-gesture segments. It tests structured block-wise search.
        """
        pool = []

        # Initialize center using LLM if no history.
        if len(self.records) == 0:
            llm = self._llm_candidates(
                n=16,
                extra_instruction=(
                    "Generate diverse initial candidates. Prefer low-cost candidates "
                    "but keep enough exploration."
                ),
            )
            if llm:
                self.current_center = np.array(llm[0], dtype=float)
                pool.extend(llm)

        # Pick best center: lowest-cost success, otherwise current center.
        successes = sorted(
            [r for r in self.records if r["success"]],
            key=lambda r: float(r["cost"]),
        )
        if successes:
            self.current_center = np.array(successes[0]["x"], dtype=float)

        center = np.array(self.current_center, dtype=float)

        # Block size decays with query count.
        progress = self.query_count / max(1, self.budget or 10)
        if progress < 0.30:
            block = 4
            amp = 0.35
        elif progress < 0.65:
            block = 2
            amp = 0.22
        else:
            block = 1
            amp = 0.12

        for _ in range(96):
            z = center.copy()
            dims = self.rng.choice(8, size=block, replace=False)

            # Square-like replacement: only perturb a block of dimensions.
            for d in dims:
                if self.rng.random() < 0.50:
                    z[d] = self.rng.random()
                else:
                    z[d] = z[d] + self.rng.choice([-1.0, 1.0]) * amp * self.rng.random()

            pool.append(np.clip(z, 0.0, 1.0))

        # A few random restarts.
        pool.extend([self.rng.random(8) for _ in range(24)])

        self.pending.extend(self._rank_by_acquisition(pool, k=self.candidate_count, beta=0.20, lam=0.28))

    # ---------------------------------------------------------------------
    # Mode 4: Cross-session transfer archive
    # ---------------------------------------------------------------------

    def _seed_from_global_archive(self):
        archive = sorted(
            DeepSeekNextGenOptimizer.GLOBAL_ARCHIVE,
            key=lambda r: float(r["cost"]),
        )

        seeds = []
        for r in archive[:16]:
            base = np.array(r["x"], dtype=float)
            seeds.append(base)
            for _ in range(2):
                seeds.append(np.clip(base + self.rng.normal(0.0, 0.08, size=8), 0.0, 1.0))

        if seeds:
            self.pending.extend(self._select_diverse(seeds, k=min(self.candidate_count, len(seeds)), cost_weight=0.35))

    def _refill_transfer_archive(self):
        pool = []

        # Global archive transfer.
        archive = sorted(
            DeepSeekNextGenOptimizer.GLOBAL_ARCHIVE,
            key=lambda r: float(r["cost"]),
        )

        for r in archive[:32]:
            base = np.array(r["x"], dtype=float)
            pool.append(base)
            for sigma in [0.04, 0.08, 0.15]:
                for _ in range(4):
                    pool.append(np.clip(base + self.rng.normal(0.0, sigma, size=8), 0.0, 1.0))

        # Within-session successes.
        for r in sorted(self.escapes, key=lambda r: float(r["cost"]))[:8]:
            base = np.array(r["x"], dtype=float)
            for _ in range(12):
                pool.append(np.clip(base + self.rng.normal(0.0, 0.08, size=8), 0.0, 1.0))

        # LLM restart if archive is not enough.
        if len(pool) < 64:
            pool.extend(
                self._llm_candidates(
                    n=32,
                    extra_instruction=(
                        "Use the previous black-box history to propose candidates. "
                        "Prefer candidates that may transfer across similar tasks."
                    ),
                )
            )

        pool.extend([self.rng.random(8) for _ in range(64)])

        self.pending.extend(self._rank_by_acquisition(pool, k=self.candidate_count, beta=0.18, lam=0.32))

    # ---------------------------------------------------------------------
    # Mode 5: Multi-fidelity search
    # ---------------------------------------------------------------------

    def _refill_multifidelity(self):
        """
        Large cheap candidate pool:
        LLM generates many proposals, then cheap surrogate + cost proxy + diversity
        chooses a small number for expensive defense queries.
        """
        pool = []

        pool.extend(
            self._llm_candidates(
                n=96,
                extra_instruction=(
                    "Generate a large diverse pool. Include both low-cost candidates "
                    "and candidates near previously successful regions."
                ),
            )
        )

        # Cheap synthetic candidates.
        pool.extend([self.rng.random(8) for _ in range(256)])

        # Local around success archive.
        successes = sorted(
            [r for r in self.records if r["success"]],
            key=lambda r: float(r["cost"]),
        )

        for r in successes[:12]:
            base = np.array(r["x"], dtype=float)
            for sigma in [0.03, 0.06, 0.12, 0.20]:
                for _ in range(6):
                    pool.append(np.clip(base + self.rng.normal(0.0, sigma, size=8), 0.0, 1.0))

        # Cost-aware repaired variants.
        repaired = []
        for z in pool[:128]:
            arr = np.array(z, dtype=float)

            r1 = arr.copy()
            r1[0] *= 0.65
            if r1[0] < 0.20:
                r1[6] = min(r1[6], 0.49)
            repaired.append(np.clip(r1, 0.0, 1.0))

            r2 = arr.copy()
            neutral = 0.36
            r2[2] = 0.70 * r2[2] + 0.30 * neutral
            r2[3] = 0.70 * r2[3] + 0.30 * neutral
            repaired.append(np.clip(r2, 0.0, 1.0))

            r3 = arr.copy()
            r3[4] *= 0.75
            r3[5] *= 0.75
            repaired.append(np.clip(r3, 0.0, 1.0))

        pool.extend(repaired)

        self.pending.extend(self._rank_by_acquisition(pool, k=self.candidate_count, beta=0.28, lam=0.38))
