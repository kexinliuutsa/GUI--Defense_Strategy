from __future__ import annotations

import json
import os
import re
import urllib.request
from typing import Any

import numpy as np

from attacks.v2.search_space import LatentVector
from attacks.v2.optimizers.base import Optimizer


SEMANTIC_KEYS = [
    "z_spatial",
    "z_frequency",
    "z_duration",
    "z_gap",
    "z_jitter",
    "z_heterogeneity",
    "z_use_spatial",
    "z_use_temporal",
]

ANON_KEYS = ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8"]


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
    arr = np.array(x, dtype=float).reshape(-1)
    if arr.shape[0] != 8:
        return None
    return np.clip(arr, 0.0, 1.0)


class DeepSeekAPIAttackPackOptimizer(Optimizer):
    """
    8 LLM attack variants for strict black-box evaluation.

    The optimizer never receives:
    - defense architecture
    - V1/V2/V3
    - thresholds
    - detector features
    - layer-level decisions

    Runtime feedback:
    - candidate vector
    - success/failure
    - cost
    """

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
        min_distance: float = 0.06,
        trigger_fail_streak: int = 4,
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
        self.min_distance = float(min_distance)
        self.trigger_fail_streak = int(trigger_fail_streak)

        self.records = []
        self.pending = []
        self.escapes = []
        self.fail_streak = 0
        self.query_count = 0

    def begin(self, session: Any, budget: int):
        self.records = []
        self.pending = []
        self.escapes = []
        self.fail_streak = 0
        self.query_count = 0
        self.budget = int(budget)

    # ---------------------------------------------------------------------
    # Basic helpers
    # ---------------------------------------------------------------------

    def _random_latent(self) -> LatentVector:
        return LatentVector.from_numpy(self.rng.random(8))

    def _as_latent(self, arr) -> LatentVector | None:
        arr = _clip_arr(arr)
        if arr is None:
            return None
        return LatentVector.from_numpy(arr)

    def _record_x(self, r):
        return r["x"]

    def _compact_history(self, anonymous: bool = False):
        recent = self.records[-self.max_history :]
        out = []

        for i, r in enumerate(recent):
            key = "x" if anonymous else "z"
            out.append(
                {
                    "trial": len(self.records) - len(recent) + i + 1,
                    key: [round(float(v), 4) for v in r["x"]],
                    "success": int(bool(r["success"])),
                    "cost": round(float(r["cost"]), 6),
                }
            )

        return out

    def _best_successes(self, k=8, anonymous: bool = False):
        successes = [r for r in self.records if r["success"]]
        successes = sorted(successes, key=lambda r: float(r["cost"]))[:k]

        key = "x" if anonymous else "z"

        return [
            {
                key: [round(float(v), 4) for v in r["x"]],
                "cost": round(float(r["cost"]), 6),
            }
            for r in successes
        ]

    def _summary_stats(self):
        n = len(self.records)
        successes = [r for r in self.records if r["success"]]
        failures = [r for r in self.records if not r["success"]]

        best = None
        if successes:
            b = sorted(successes, key=lambda r: float(r["cost"]))[0]
            best = {
                "z": [round(float(v), 4) for v in b["x"]],
                "cost": round(float(b["cost"]), 6),
            }

        def mean_x(rows):
            if not rows:
                return None
            return [round(float(v), 4) for v in np.mean([r["x"] for r in rows], axis=0)]

        return {
            "n_trials": n,
            "n_successes": len(successes),
            "n_failures": len(failures),
            "success_rate_so_far": len(successes) / n if n else 0.0,
            "fail_streak": self.fail_streak,
            "best_success": best,
            "mean_success_z": mean_x(successes),
            "mean_failure_z": mean_x(failures),
        }

    def _cost_proxy(self, z):
        """
        Cheap attack-side cost proxy.
        This uses only mutation-space semantics, not defense internals.
        """
        z = np.clip(np.array(z, dtype=float), 0.0, 1.0)

        # duration/gap scale is closest to 1.0 around normalized value ~0.36
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

    def _select_diverse(self, candidates, k=None, cost_weight=0.30):
        k = k or self.candidate_count

        valid = []
        for c in candidates:
            if isinstance(c, LatentVector):
                arr = c.to_numpy()
            else:
                arr = _clip_arr(c)

            if arr is None:
                continue

            valid.append(arr)

        if not valid:
            return [self._random_latent() for _ in range(k)]

        successes = [r for r in self.records if r["success"]]
        failures = [r for r in self.records if not r["success"]]

        selected = []

        while valid and len(selected) < k:
            best_i = None
            best_score = -1e18

            for i, z in enumerate(valid):
                cost_proxy = self._cost_proxy(z)

                novelty_from_failures = self._min_dist(z, failures)
                if successes:
                    dist_success = self._min_dist(z, successes)
                    success_affinity = np.exp(-dist_success * 2.0)
                else:
                    success_affinity = 0.0

                if selected:
                    diversity = min(float(np.linalg.norm(z - s)) for s in selected)
                else:
                    diversity = 1.0

                score = (
                    0.40 * success_affinity
                    + 0.35 * novelty_from_failures
                    + 0.35 * diversity
                    - cost_weight * cost_proxy
                    + 0.02 * self.rng.random()
                )

                if score > best_score:
                    best_score = score
                    best_i = i

            selected.append(valid.pop(best_i))

        while len(selected) < k:
            selected.append(self.rng.random(8))

        return [LatentVector.from_numpy(np.clip(z, 0.0, 1.0)) for z in selected[:k]]

    def _local_refine(self):
        if not self.escapes:
            return self._random_latent()

        elites = sorted(self.escapes, key=lambda r: float(r["cost"]))[:8]
        weights = np.array([1.0 / (1e-6 + float(r["cost"])) for r in elites], dtype=float)
        weights = weights / weights.sum()

        base = np.array(elites[int(self.rng.choice(len(elites), p=weights))]["x"], dtype=float)
        sigma = max(0.025, 0.16 / np.sqrt(1.0 + 0.2 * len(self.escapes)))
        z = np.clip(base + self.rng.normal(0.0, sigma, size=8), 0.0, 1.0)
        return LatentVector.from_numpy(z)

    def _repair_candidates(self, candidates):
        """
        Cost-aware repair after LLM proposal.
        Does not use defense internals.
        """
        repaired = []

        for c in candidates:
            arr = c.to_numpy() if isinstance(c, LatentVector) else _clip_arr(c)
            if arr is None:
                continue

            arr = np.array(arr, dtype=float)

            # Original candidate
            repaired.append(arr.copy())

            # Repair 1: reduce spatial cost but preserve temporal tendency.
            r1 = arr.copy()
            r1[0] *= 0.65
            if r1[0] < 0.20:
                r1[6] = min(r1[6], 0.49)
            repaired.append(np.clip(r1, 0.0, 1.0))

            # Repair 2: pull duration/gap slightly toward neutral timing cost.
            r2 = arr.copy()
            neutral = 0.36
            r2[2] = 0.65 * r2[2] + 0.35 * neutral
            r2[3] = 0.65 * r2[3] + 0.35 * neutral
            repaired.append(np.clip(r2, 0.0, 1.0))

            # Repair 3: reduce jitter/heterogeneity.
            r3 = arr.copy()
            r3[4] *= 0.75
            r3[5] *= 0.75
            repaired.append(np.clip(r3, 0.0, 1.0))

        return self._select_diverse(repaired, k=self.candidate_count, cost_weight=0.45)

    # ---------------------------------------------------------------------
    # API call and candidate parsing
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

    def _parse_candidates(self, text: str):
        obj = _json_from_text(text)
        if not obj:
            return []

        raw = obj.get("candidates", [])
        if not isinstance(raw, list):
            return []

        out = []

        for item in raw:
            arr = _clip_arr(item)
            if arr is None:
                continue
            out.append(LatentVector.from_numpy(arr))

        return out

    def _parse_policies(self, text: str):
        obj = _json_from_text(text)
        if not obj:
            return []

        policies = obj.get("policies", [])
        if isinstance(policies, dict):
            policies = [policies]

        candidates = []

        for pol in policies:
            if not isinstance(pol, dict):
                continue

            ranges = pol.get("ranges", {})
            if not isinstance(ranges, dict):
                continue

            for _ in range(4):
                arr = []

                for k in SEMANTIC_KEYS:
                    val = ranges.get(k, [0.0, 1.0])
                    try:
                        lo, hi = float(val[0]), float(val[1])
                        lo, hi = max(0.0, min(lo, hi)), min(1.0, max(lo, hi))
                    except Exception:
                        lo, hi = 0.0, 1.0

                    if hi - lo < 0.04:
                        mid = (lo + hi) / 2
                        lo = max(0.0, mid - 0.02)
                        hi = min(1.0, mid + 0.02)

                    arr.append(self.rng.uniform(lo, hi))

                candidates.append(LatentVector.from_numpy(np.clip(arr, 0.0, 1.0)))

        return candidates

    # ---------------------------------------------------------------------
    # Prompts
    # ---------------------------------------------------------------------

    def _prompt_candidates(self, n: int, anonymous: bool = False, rich: bool = False):
        if anonymous:
            dim_text = (
                "Each input is an 8-dimensional normalized vector "
                "[x1, x2, x3, x4, x5, x6, x7, x8]."
            )
            history = self._compact_history(anonymous=True)
            best = self._best_successes(anonymous=True)
            extra = ""
        else:
            dim_text = (
                "The 8 normalized parameters are: "
                + json.dumps(SEMANTIC_KEYS)
                + ". These are generic trajectory-mutation parameters."
            )
            history = self._compact_history(anonymous=False)
            best = self._best_successes(anonymous=False)
            extra = ""

        if rich:
            extra = "\nAdditional black-box summary:\n" + json.dumps(
                self._summary_stats(), ensure_ascii=False
            )

        return f"""
You are optimizing an unknown black-box objective.

You do not know the target system's internal model, rules, thresholds, features, architecture, or decision layers.
You only observe previous trials, success/failure, and cost.

{dim_text}

Lower cost is better.
A successful low-cost candidate is best.
Balance exploration and exploitation.

Recent trials:
{json.dumps(history, ensure_ascii=False)}

Best successful trials:
{json.dumps(best, ensure_ascii=False)}
{extra}

Generate {n} diverse new candidate vectors.
Each vector must contain exactly 8 numbers in [0, 1].

Return ONLY valid JSON:
{{
  "candidates": [
    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
  ]
}}
""".strip()

    def _prompt_policy(self):
        return f"""
You are designing high-level search policies for an unknown black-box objective.

You do not know the target system's internal model, rules, thresholds, features, architecture, or decision layers.
You only observe previous trials, success/failure, and cost.

The 8 normalized parameters are:
{json.dumps(SEMANTIC_KEYS)}

Recent trials:
{json.dumps(self._compact_history(anonymous=False), ensure_ascii=False)}

Best successful trials:
{json.dumps(self._best_successes(anonymous=False), ensure_ascii=False)}

Task:
Generate 4 diverse high-level search policies.
Each policy should define a promising range [low, high] for each parameter.
Use values between 0 and 1.

Return ONLY valid JSON:
{{
  "policies": [
    {{
      "name": "short_policy_name",
      "ranges": {{
        "z_spatial": [0.0, 1.0],
        "z_frequency": [0.0, 1.0],
        "z_duration": [0.0, 1.0],
        "z_gap": [0.0, 1.0],
        "z_jitter": [0.0, 1.0],
        "z_heterogeneity": [0.0, 1.0],
        "z_use_spatial": [0.0, 1.0],
        "z_use_temporal": [0.0, 1.0]
      }}
    }}
  ]
}}
""".strip()

    # ---------------------------------------------------------------------
    # Refill strategies
    # ---------------------------------------------------------------------

    def _refill_llm_basic(self, n=None, anonymous=False, rich=False, filter_after=False):
        n = n or self.candidate_count

        try:
            text = self._call_api(
                self._prompt_candidates(n=n, anonymous=anonymous, rich=rich)
            )
            cands = self._parse_candidates(text)
        except Exception as e:
            print(f"[{self.mode} fallback random]", repr(e))
            cands = []

        if not cands:
            cands = [self._random_latent() for _ in range(self.candidate_count)]

        if filter_after:
            self.pending.extend(self._select_diverse(cands, k=self.candidate_count))
        else:
            self.pending.extend(cands[: self.candidate_count])

        while len(self.pending) < self.candidate_count:
            self.pending.append(self._random_latent())

    def _refill_policy(self):
        try:
            text = self._call_api(self._prompt_policy())
            cands = self._parse_policies(text)
        except Exception as e:
            print(f"[{self.mode} fallback random]", repr(e))
            cands = []

        if not cands:
            cands = [self._random_latent() for _ in range(self.candidate_count)]

        self.pending.extend(self._select_diverse(cands, k=self.candidate_count))

    def _refill_repair(self):
        try:
            text = self._call_api(self._prompt_candidates(n=16, anonymous=False, rich=True))
            cands = self._parse_candidates(text)
        except Exception as e:
            print(f"[{self.mode} fallback random]", repr(e))
            cands = []

        if not cands:
            cands = [self._random_latent() for _ in range(self.candidate_count)]

        self.pending.extend(self._repair_candidates(cands))

    def _surrogate_rank(self):
        """
        Cheap nonparametric surrogate over previous black-box outcomes.
        No defense internals.
        """
        pool = [self.rng.random(8) for _ in range(256)]

        # Add local neighbors around successes.
        for r in self.escapes:
            base = np.array(r["x"], dtype=float)
            for _ in range(12):
                pool.append(np.clip(base + self.rng.normal(0.0, 0.10, size=8), 0.0, 1.0))

        if len(self.records) < 6:
            return self._select_diverse(pool, k=self.candidate_count)

        X = np.array([r["x"] for r in self.records], dtype=float)
        y = np.array([1.0 if r["success"] else 0.0 for r in self.records], dtype=float)
        c = np.array([float(r["cost"]) for r in self.records], dtype=float)

        scored = []
        h = 0.45

        for z in pool:
            d = np.linalg.norm(X - z, axis=1)
            w = np.exp(-(d ** 2) / (2 * h * h)) + 1e-9

            p_success = float((w * y).sum() / w.sum())
            inv_cost = float((w * (1.0 / (1.0 + c))).sum() / w.sum())
            novelty = float(np.min(d))

            score = (
                0.65 * p_success
                + 0.20 * inv_cost
                + 0.20 * novelty
                - 0.20 * self._cost_proxy(z)
                + 0.02 * self.rng.random()
            )

            scored.append((score, z))

        scored.sort(key=lambda x: x[0], reverse=True)
        cands = [z for _, z in scored[:64]]
        return self._select_diverse(cands, k=self.candidate_count)

    def _refill_by_mode(self):
        if self.mode == "anon_control":
            self._refill_llm_basic(n=self.candidate_count, anonymous=True, rich=False, filter_after=False)

        elif self.mode == "semantic_batch":
            self._refill_llm_basic(n=self.candidate_count, anonymous=False, rich=False, filter_after=False)

        elif self.mode == "rich_feedback":
            self._refill_llm_basic(n=self.candidate_count, anonymous=False, rich=True, filter_after=False)

        elif self.mode == "filter32":
            self._refill_llm_basic(n=32, anonymous=False, rich=True, filter_after=True)

        elif self.mode == "triggered_local":
            self._refill_llm_basic(n=self.candidate_count, anonymous=False, rich=True, filter_after=True)

        elif self.mode == "repair_filter":
            self._refill_repair()

        elif self.mode == "policy_repr":
            self._refill_policy()

        elif self.mode == "surrogate":
            if len(self.records) < 8 or len(self.escapes) == 0:
                self._refill_llm_basic(n=24, anonymous=False, rich=True, filter_after=True)
            else:
                self.pending.extend(self._surrogate_rank())

        else:
            raise ValueError(f"Unknown LLM attack-pack mode: {self.mode}")

    # ---------------------------------------------------------------------
    # Optimizer API
    # ---------------------------------------------------------------------

    def ask(self) -> LatentVector:
        self.query_count += 1

        # Mode 5: triggered local search.
        # LLM only called before first escape or after local search gets stuck.
        if self.mode == "triggered_local":
            if self.escapes and self.fail_streak < self.trigger_fail_streak:
                return self._local_refine()

            if not self.pending:
                self._refill_by_mode()

            if self.pending:
                return self.pending.pop(0)

            return self._random_latent()

        # Mode 4: hybrid loop.
        # Implemented through semantic_batch name? Keep it as a separate method via factory.
        if self.mode == "hybrid_loop":
            if self.escapes:
                u = self.rng.random()

                # Most queries do local refine after escape.
                if u < 0.70:
                    return self._local_refine()

                # Some random restart.
                if u < 0.85:
                    return self._random_latent()

                # Occasionally ask LLM again.
                if not self.pending:
                    self._refill_llm_basic(n=self.candidate_count, anonymous=False, rich=True, filter_after=True)

                if self.pending:
                    return self.pending.pop(0)

                return self._random_latent()

            if not self.pending:
                self._refill_llm_basic(n=self.candidate_count, anonymous=False, rich=True, filter_after=True)

            if self.pending:
                return self.pending.pop(0)

            return self._random_latent()

        # Other modes.
        if not self.pending:
            self._refill_by_mode()

        if not self.pending:
            return self._random_latent()

        return self.pending.pop(0)

    def tell(self, latent: LatentVector, escaped: bool, cost: float):
        x = latent.to_numpy().astype(float)

        rec = {
            "x": x.tolist(),
            "success": bool(escaped),
            "cost": float(cost),
        }

        self.records.append(rec)

        if escaped:
            self.escapes.append(rec)
            self.fail_streak = 0
        else:
            self.fail_streak += 1

    def end(self, history):
        pass
