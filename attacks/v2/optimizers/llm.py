from __future__ import annotations

import json
import re
import urllib.request
from typing import Any

import numpy as np

from attacks.v2.search_space import LatentVector
from attacks.v2.optimizers.base import Optimizer


class OllamaLLMOptimizer(Optimizer):
    """
    Black-box LLM-guided parameter proposal.

    The LLM only observes:
    - previous latent z
    - escaped / detected result
    - attacker-side perturbation cost

    It does NOT observe:
    - V1/V2/V3 layer decisions
    - thresholds
    - feature values
    - detector architecture
    """

    def __init__(
        self,
        model: str = "qwen3:8b",
        seed: int | None = None,
        host: str = "http://127.0.0.1:11434",
        max_history: int = 30,
        timeout: int = 90,
        temperature: float = 0.6,
        fallback_to_random: bool = True,
        temporal_bias: float = 0.85,
        spatial_bias: float = 0.65,
        cold_start: int = 3,
    ):
        self.model = model
        self.seed = seed
        self.host = host.rstrip("/")
        self.max_history = int(max_history)
        self.timeout = int(timeout)
        self.temperature = float(temperature)
        self.fallback_to_random = bool(fallback_to_random)
        self.temporal_bias = float(temporal_bias)
        self.spatial_bias = float(spatial_bias)
        self.cold_start = int(cold_start)

        self.rng = np.random.default_rng(seed)
        self.records: list[dict[str, Any]] = []
        self.budget = None

    def begin(self, session: Any, budget: int):
        self.records = []
        self.budget = int(budget)

    def _biased_random(self) -> LatentVector:
        z = self.rng.random(8)

        # Empirical prior from our black-box experiments:
        # escaped trajectories almost always require temporal mutation.
        if self.rng.random() < self.temporal_bias:
            z[7] = 0.5 + 0.5 * self.rng.random()
        else:
            z[7] = 0.5 * self.rng.random()

        if self.rng.random() < self.spatial_bias:
            z[6] = 0.5 + 0.5 * self.rng.random()
        else:
            z[6] = 0.5 * self.rng.random()

        return LatentVector.from_numpy(z)

    def _prompt(self) -> str:
        recent = self.records[-self.max_history :]

        compact_history = []
        for r in recent:
            compact_history.append(
                {
                    "z": [round(float(x), 4) for x in r["z"]],
                    "escaped": bool(r["escaped"]),
                    "detected": not bool(r["escaped"]),
                    "cost": round(float(r["cost"]), 6),
                }
            )

        return f"""
You are an attacker optimizing parameters for a black-box GUI trajectory mutation attack.

You do NOT know the detector architecture.
You do NOT know any internal detector features, thresholds, layers, or rules.
You only receive black-box query feedback.

Each query proposes a latent vector z in [0,1]^8.

Dimensions:
0 z_spatial: spatial perturbation amplitude
1 z_frequency: spatial wave frequency
2 z_duration: duration scaling
3 z_gap: inter-action gap scaling
4 z_jitter: timing jitter
5 z_heterogeneity: per-action heterogeneity
6 z_use_spatial: >=0.5 enables spatial mutation
7 z_use_temporal: >=0.5 enables temporal mutation

History from previous black-box queries:
{json.dumps(compact_history, ensure_ascii=False)}

Objective:
1. First find escaped=true.
2. After escape is found, prefer lower cost.
3. Avoid repeating old vectors.
4. Explore interactions among spatial, duration, gap, jitter, and heterogeneity.
5. Return one next candidate only.

Return ONLY valid JSON with this exact schema:
{{"z":[float,float,float,float,float,float,float,float]}}

Every value must be between 0 and 1.
""".strip()

    def _call_ollama(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": 256,
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

    def _parse_response(self, text: str) -> LatentVector:
        # Try strict JSON first.
        try:
            obj = json.loads(text)
            arr = np.array(obj["z"], dtype=float)
            if arr.shape == (8,):
                return LatentVector.from_numpy(np.clip(arr, 0.0, 1.0))
        except Exception:
            pass

        # Try extracting a JSON object from model output.
        m = re.search(r"\{.*?\}", text, flags=re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
                arr = np.array(obj["z"], dtype=float)
                if arr.shape == (8,):
                    return LatentVector.from_numpy(np.clip(arr, 0.0, 1.0))
            except Exception:
                pass

        # Final fallback: first 8 numbers.
        nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
        vals = [float(x) for x in nums[:8]]
        if len(vals) == 8:
            arr = np.array(vals, dtype=float)
            return LatentVector.from_numpy(np.clip(arr, 0.0, 1.0))

        raise ValueError(f"Could not parse LLM response: {text[:500]}")

    def ask(self) -> LatentVector:
        # Cold start: give LLM a few real observations first.
        if len(self.records) < self.cold_start:
            return self._biased_random()

        try:
            response = self._call_ollama(self._prompt())
            return self._parse_response(response)
        except Exception:
            if self.fallback_to_random:
                return self._biased_random()
            raise

    def tell(self, latent: LatentVector, escaped: bool, cost: float):
        self.records.append(
            {
                "z": latent.to_numpy().astype(float).tolist(),
                "escaped": bool(escaped),
                "cost": float(cost),
            }
        )

    def end(self, history):
        pass
