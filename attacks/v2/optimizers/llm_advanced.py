from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass, field
from typing import Any

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


def _clip_arr(arr):
    arr = np.array(arr, dtype=float).reshape(-1)
    if arr.shape[0] != 8:
        raise ValueError("latent must have 8 dimensions")
    return np.clip(arr, 0.0, 1.0)


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


class OllamaMixin:
    def _call_ollama(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": self.temperature,
                "num_predict": self.num_predict,
            },
        }

        req = urllib.request.Request(
            f"{self.host}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            out = json.loads(r.read().decode("utf-8"))

        return str(out.get("response", ""))

    def _compact_history(self):
        recent = self.records[-self.max_history :]
        out = []

        for i, r in enumerate(recent):
            out.append(
                {
                    "i": len(self.records) - len(recent) + i + 1,
                    "z": [round(float(x), 4) for x in r["z"]],
                    "escaped": bool(r["escaped"]),
                    "cost": round(float(r["cost"]), 6),
                }
            )

        return out

    def _best_escapes(self, k=8):
        escapes = [r for r in self.records if r["escaped"]]
        escapes = sorted(escapes, key=lambda r: float(r["cost"]))[:k]

        return [
            {
                "z": [round(float(x), 4) for x in r["z"]],
                "cost": round(float(r["cost"]), 6),
            }
            for r in escapes
        ]

    def _random_latent(self):
        return LatentVector.from_numpy(self.rng.random(8))

    def _repair_candidate(self, arr):
        arr = _clip_arr(arr)
        return LatentVector.from_numpy(arr)

    def _record(self, latent: LatentVector, escaped: bool, cost: float):
        self.records.append(
            {
                "z": latent.to_numpy().astype(float).tolist(),
                "escaped": bool(escaped),
                "cost": float(cost),
            }
        )


@dataclass
class EscapePoint:
    cost: float
    latent: np.ndarray


# ======================================================================================
# 1. LLM Region Hybrid
# ======================================================================================

class LLMRegionHybridOptimizer(Optimizer, OllamaMixin):
    """
    LLM proposes search regions instead of exact points.

    Strict black-box:
    - no defense internals
    - no V1/V2/V3 labels
    - only z / escaped / cost history

    Strategy:
    - LLM generates regions.
    - Program samples candidates inside regions.
    - Once escape is found, local refinement around escaped points.
    """

    def __init__(
        self,
        model="qwen3:8b",
        seed=None,
        host="http://127.0.0.1:11434",
        timeout=120,
        temperature=0.8,
        num_predict=1536,
        max_history=40,
        candidate_count=8,
        region_count=4,
        p_llm_after_escape=0.20,
        p_random_after_escape=0.10,
        sigma0=0.16,
        sigma_min=0.025,
        elite_k=8,
    ):
        self.model = model
        self.seed = seed
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature
        self.num_predict = num_predict
        self.max_history = max_history
        self.candidate_count = candidate_count
        self.region_count = region_count

        self.p_llm_after_escape = p_llm_after_escape
        self.p_random_after_escape = p_random_after_escape
        self.sigma0 = sigma0
        self.sigma_min = sigma_min
        self.elite_k = elite_k

        self.rng = np.random.default_rng(seed)
        self.records = []
        self.pending = []
        self.escapes = []

    def begin(self, session: Any, budget: int):
        self.records = []
        self.pending = []
        self.escapes = []
        self.budget = budget

    def _prompt_regions(self):
        return f"""
You are optimizing a strict black-box GUI trajectory mutation attack.

You do not know the detector architecture, thresholds, layers, or features.
You only observe previous latent vectors, escaped/detected outcome, and cost.

Latent vector dimensions:
{json.dumps(KEYS)}

Recent black-box history:
{json.dumps(self._compact_history(), ensure_ascii=False)}

Best escaped candidates:
{json.dumps(self._best_escapes(), ensure_ascii=False)}

Task:
Propose {self.region_count} promising search regions.
Each region gives a range [low, high] for some or all dimensions.
Use values between 0 and 1.
Prefer diverse regions.
If no escape has been found, explore broad but plausible regions.
If escapes exist, focus around lower-cost escape regions.

Return ONLY JSON:
{{
  "regions": [
    {{
      "name": "short_name",
      "weight": 1.0,
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

    def _parse_regions(self, text):
        obj = _json_from_text(text)
        if not obj:
            return []

        regions = obj.get("regions", [])
        if isinstance(regions, dict):
            regions = [regions]

        cleaned = []

        for reg in regions:
            if not isinstance(reg, dict):
                continue

            ranges = reg.get("ranges", {})
            if not isinstance(ranges, dict):
                continue

            clean_ranges = {}
            for k in KEYS:
                val = ranges.get(k, [0.0, 1.0])
                try:
                    lo, hi = float(val[0]), float(val[1])
                    lo, hi = max(0.0, min(lo, hi)), min(1.0, max(lo, hi))
                    if hi - lo < 0.05:
                        mid = (lo + hi) / 2
                        lo = max(0.0, mid - 0.025)
                        hi = min(1.0, mid + 0.025)
                    clean_ranges[k] = [lo, hi]
                except Exception:
                    clean_ranges[k] = [0.0, 1.0]

            cleaned.append(
                {
                    "name": str(reg.get("name", "region")),
                    "weight": max(1e-6, float(reg.get("weight", 1.0))),
                    "ranges": clean_ranges,
                }
            )

        return cleaned[: self.region_count]

    def _sample_from_regions(self, regions):
        if not regions:
            return [self._random_latent() for _ in range(self.candidate_count)]

        weights = np.array([r["weight"] for r in regions], dtype=float)
        weights = weights / weights.sum()

        candidates = []
        for _ in range(self.candidate_count):
            reg = regions[int(self.rng.choice(len(regions), p=weights))]
            arr = []
            for k in KEYS:
                lo, hi = reg["ranges"].get(k, [0.0, 1.0])
                arr.append(self.rng.uniform(lo, hi))
            candidates.append(self._repair_candidate(arr))

        return candidates

    def _refill_llm(self):
        try:
            text = self._call_ollama(self._prompt_regions())
            regions = self._parse_regions(text)
            self.pending.extend(self._sample_from_regions(regions))
        except Exception:
            self.pending.extend([self._random_latent() for _ in range(self.candidate_count)])

    def _local_refine(self):
        elites = sorted(self.escapes, key=lambda x: x.cost)[: self.elite_k]
        weights = np.array([1.0 / (1e-6 + e.cost) for e in elites], dtype=float)
        weights = weights / weights.sum()

        center = elites[int(self.rng.choice(len(elites), p=weights))].latent

        sigma = max(
            self.sigma_min,
            self.sigma0 / np.sqrt(1.0 + 0.20 * len(self.escapes)),
        )

        arr = np.clip(center + self.rng.normal(0.0, sigma, size=8), 0.0, 1.0)
        return LatentVector.from_numpy(arr)

    def ask(self):
        if self.escapes:
            u = self.rng.random()
            if u > self.p_llm_after_escape + self.p_random_after_escape:
                return self._local_refine()
            if u > self.p_llm_after_escape:
                return self._random_latent()

        if not self.pending:
            self._refill_llm()

        if not self.pending:
            return self._random_latent()

        return self.pending.pop(0)

    def tell(self, latent, escaped, cost):
        self._record(latent, escaped, cost)
        if escaped:
            self.escapes.append(EscapePoint(float(cost), latent.to_numpy().astype(float)))

    def end(self, history):
        pass


# ======================================================================================
# 2. LLM Direction Hybrid
# ======================================================================================

class LLMDirectionHybridOptimizer(Optimizer, OllamaMixin):
    """
    LLM outputs direction instead of exact candidates.

    Example:
    - increase z_gap
    - decrease z_jitter
    - keep z_use_temporal high

    Program converts direction into candidate samples.
    """

    def __init__(
        self,
        model="qwen3:8b",
        seed=None,
        host="http://127.0.0.1:11434",
        timeout=120,
        temperature=0.7,
        num_predict=1024,
        max_history=40,
        candidate_count=8,
        sigma=0.08,
    ):
        self.model = model
        self.seed = seed
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature
        self.num_predict = num_predict
        self.max_history = max_history
        self.candidate_count = candidate_count
        self.sigma = sigma

        self.rng = np.random.default_rng(seed)
        self.records = []
        self.pending = []

    def begin(self, session: Any, budget: int):
        self.records = []
        self.pending = []
        self.budget = budget

    def _prompt_direction(self):
        return f"""
You are optimizing a strict black-box GUI trajectory mutation attack.

You do not know the detector internals.
You only observe z, escaped/detected, and cost.

Latent dimensions:
{json.dumps(KEYS)}

Recent history:
{json.dumps(self._compact_history(), ensure_ascii=False)}

Best escaped candidates:
{json.dumps(self._best_escapes(), ensure_ascii=False)}

Task:
Infer a useful search direction.
Each direction value should be between -1 and 1:
- positive means increase this parameter
- negative means decrease this parameter
- zero means keep similar

Return ONLY JSON:
{{
  "base": "best_escape_or_recent",
  "step_size": 0.15,
  "direction": {{
    "z_spatial": 0.0,
    "z_frequency": 0.0,
    "z_duration": 0.0,
    "z_gap": 0.0,
    "z_jitter": 0.0,
    "z_heterogeneity": 0.0,
    "z_use_spatial": 0.0,
    "z_use_temporal": 0.0
  }}
}}
""".strip()

    def _choose_base(self):
        escapes = [r for r in self.records if r["escaped"]]
        if escapes:
            best = sorted(escapes, key=lambda r: float(r["cost"]))[0]
            return np.array(best["z"], dtype=float)

        if self.records:
            # Use a random recent point. No detector knowledge is used.
            recent = self.records[-min(8, len(self.records)) :]
            return np.array(self.rng.choice(recent)["z"], dtype=float)

        return self.rng.random(8)

    def _parse_direction(self, text):
        obj = _json_from_text(text)
        if not obj:
            return None, None

        direction = obj.get("direction", {})
        if not isinstance(direction, dict):
            return None, None

        vec = []
        for k in KEYS:
            try:
                vec.append(float(direction.get(k, 0.0)))
            except Exception:
                vec.append(0.0)

        vec = np.clip(np.array(vec, dtype=float), -1.0, 1.0)

        try:
            step = float(obj.get("step_size", 0.15))
        except Exception:
            step = 0.15

        step = float(np.clip(step, 0.02, 0.35))
        return vec, step

    def _refill_direction(self):
        if len(self.records) < 2:
            self.pending.extend([self._random_latent() for _ in range(self.candidate_count)])
            return

        try:
            text = self._call_ollama(self._prompt_direction())
            direction, step = self._parse_direction(text)
        except Exception:
            direction, step = None, None

        if direction is None:
            self.pending.extend([self._random_latent() for _ in range(self.candidate_count)])
            return

        base = self._choose_base()

        candidates = []
        for i in range(self.candidate_count):
            scale = step * self.rng.uniform(0.5, 1.5)
            noise = self.rng.normal(0.0, self.sigma, size=8)
            arr = np.clip(base + scale * direction + noise, 0.0, 1.0)
            candidates.append(LatentVector.from_numpy(arr))

        self.pending.extend(candidates)

    def ask(self):
        if not self.pending:
            self._refill_direction()

        if not self.pending:
            return self._random_latent()

        return self.pending.pop(0)

    def tell(self, latent, escaped, cost):
        self._record(latent, escaped, cost)

    def end(self, history):
        pass


# ======================================================================================
# 3. LLM Arm Bandit
# ======================================================================================

@dataclass
class Arm:
    name: str
    ranges: dict
    alpha: float = 1.0
    beta: float = 1.0
    pulls: int = 0
    escapes: int = 0
    costs: list = field(default_factory=list)


class LLMArmBanditOptimizer(Optimizer, OllamaMixin):
    """
    LLM dynamically creates search arms.
    Thompson sampling chooses arms.
    Program samples z from selected arm.

    This upgrades old fixed mutation families into adaptive LLM-generated arms.
    """

    def __init__(
        self,
        model="qwen3:8b",
        seed=None,
        host="http://127.0.0.1:11434",
        timeout=120,
        temperature=0.8,
        num_predict=1536,
        max_history=40,
        arm_count=6,
        refresh_every=16,
    ):
        self.model = model
        self.seed = seed
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature
        self.num_predict = num_predict
        self.max_history = max_history
        self.arm_count = arm_count
        self.refresh_every = refresh_every

        self.rng = np.random.default_rng(seed)
        self.records = []
        self.arms = []
        self.last_arm = None
        self.query_count = 0

    def begin(self, session: Any, budget: int):
        self.records = []
        self.arms = []
        self.last_arm = None
        self.query_count = 0
        self.budget = budget

    def _prompt_arms(self):
        current_arms = [
            {
                "name": a.name,
                "pulls": a.pulls,
                "escapes": a.escapes,
                "mean_cost_escaped": float(np.mean(a.costs)) if a.costs else None,
                "ranges": a.ranges,
            }
            for a in self.arms
        ]

        return f"""
You are designing adaptive search arms for a strict black-box GUI trajectory mutation attack.

You do not know the detector.
You only observe previous latent vectors, escaped/detected outcome, and cost.

Latent dimensions:
{json.dumps(KEYS)}

Recent black-box history:
{json.dumps(self._compact_history(), ensure_ascii=False)}

Current arms and performance:
{json.dumps(current_arms, ensure_ascii=False)}

Task:
Create {self.arm_count} diverse search arms.
Each arm is a strategy with ranges over the 8 latent dimensions.
Arms should be meaningfully different.
Some arms may focus on spatial changes, some on temporal changes, some on interactions.

Return ONLY JSON:
{{
  "arms": [
    {{
      "name": "temporal_compression",
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

    def _parse_arms(self, text):
        obj = _json_from_text(text)
        if not obj:
            return []

        raw = obj.get("arms", [])
        if isinstance(raw, dict):
            raw = [raw]

        arms = []

        for item in raw:
            if not isinstance(item, dict):
                continue

            ranges = item.get("ranges", {})
            if not isinstance(ranges, dict):
                continue

            clean = {}
            for k in KEYS:
                val = ranges.get(k, [0.0, 1.0])
                try:
                    lo, hi = float(val[0]), float(val[1])
                    lo, hi = max(0.0, min(lo, hi)), min(1.0, max(lo, hi))
                    if hi - lo < 0.05:
                        mid = (lo + hi) / 2
                        lo = max(0.0, mid - 0.025)
                        hi = min(1.0, mid + 0.025)
                    clean[k] = [lo, hi]
                except Exception:
                    clean[k] = [0.0, 1.0]

            arms.append(Arm(name=str(item.get("name", "arm")), ranges=clean))

        return arms[: self.arm_count]

    def _refresh_arms(self):
        try:
            text = self._call_ollama(self._prompt_arms())
            new_arms = self._parse_arms(text)
        except Exception:
            new_arms = []

        if not new_arms:
            new_arms = [
                Arm(
                    name=f"random_arm_{i}",
                    ranges={k: [0.0, 1.0] for k in KEYS},
                )
                for i in range(self.arm_count)
            ]

        self.arms = new_arms

    def _select_arm(self):
        if not self.arms:
            self._refresh_arms()

        if not self.arms:
            return None

        samples = [
            self.rng.beta(a.alpha, a.beta)
            for a in self.arms
        ]
        idx = int(np.argmax(samples))
        return self.arms[idx]

    def _sample_from_arm(self, arm):
        if arm is None:
            return self._random_latent()

        arr = []
        for k in KEYS:
            lo, hi = arm.ranges.get(k, [0.0, 1.0])
            arr.append(self.rng.uniform(lo, hi))

        return LatentVector.from_numpy(np.clip(np.array(arr, dtype=float), 0.0, 1.0))

    def ask(self):
        self.query_count += 1

        if not self.arms or (
            self.query_count > 1 and self.query_count % self.refresh_every == 1
        ):
            self._refresh_arms()

        arm = self._select_arm()
        self.last_arm = arm
        return self._sample_from_arm(arm)

    def tell(self, latent, escaped, cost):
        self._record(latent, escaped, cost)

        if self.last_arm is not None:
            self.last_arm.pulls += 1

            if escaped:
                self.last_arm.escapes += 1
                self.last_arm.alpha += 1.0
                self.last_arm.costs.append(float(cost))
            else:
                self.last_arm.beta += 1.0

    def end(self, history):
        pass
