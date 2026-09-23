from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import dataclass
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


@dataclass
class EscapePoint:
    cost: float
    latent: np.ndarray


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


class DeepSeekAPIBatchHybridOptimizer(Optimizer):
    """
    Strict black-box DeepSeek API batch-hybrid optimizer.

    The LLM only sees:
    - previous latent vectors z
    - escaped / detected outcome
    - cost

    It does NOT see:
    - detector architecture
    - V1 / V2 / V3
    - thresholds
    - features
    - layer-level decisions
    """

    def __init__(
        self,
        model: str | None = None,
        seed: int | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: int = 120,
        temperature: float = 0.8,
        candidate_count: int = 8,
        max_history: int = 40,
        p_llm_after_escape: float = 0.20,
        p_random_after_escape: float = 0.10,
        sigma0: float = 0.16,
        sigma_min: float = 0.025,
        elite_k: int = 8,
    ):
        self.model = model or os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
        self.base_url = (base_url or os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")).rstrip("/")
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")

        if not self.api_key:
            raise RuntimeError("Missing DEEPSEEK_API_KEY. Run: export DEEPSEEK_API_KEY='your_key'")

        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.timeout = timeout
        self.temperature = temperature
        self.candidate_count = candidate_count
        self.max_history = max_history

        self.p_llm_after_escape = float(p_llm_after_escape)
        self.p_random_after_escape = float(p_random_after_escape)
        self.sigma0 = float(sigma0)
        self.sigma_min = float(sigma_min)
        self.elite_k = int(elite_k)

        self.records = []
        self.pending = []
        self.escapes: list[EscapePoint] = []

    def begin(self, session: Any, budget: int):
        self.records = []
        self.pending = []
        self.escapes = []
        self.budget = budget

    def _random_latent(self) -> LatentVector:
        return LatentVector.from_numpy(self.rng.random(8))

    def _compact_history(self):
        recent = self.records[-self.max_history :]
        out = []
        for i, r in enumerate(recent):
            out.append({
                "i": len(self.records) - len(recent) + i + 1,
                "z": [round(float(x), 4) for x in r["z"]],
                "escaped": bool(r["escaped"]),
                "cost": round(float(r["cost"]), 6),
            })
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

    def _prompt(self):
        return f"""
You are proposing candidate parameters for a strict black-box GUI trajectory mutation attack.

You do not know the detector architecture, thresholds, layers, or features.
You only observe previous latent vectors, escaped/detected outcome, and cost.

Latent vector dimensions:
{json.dumps(KEYS)}

Recent black-box history:
{json.dumps(self._compact_history(), ensure_ascii=False)}

Best escaped candidates:
{json.dumps(self._best_escapes(), ensure_ascii=False)}

Task:
Generate {self.candidate_count} diverse candidate latent vectors.
Each latent vector must contain exactly 8 numbers in [0, 1].
Prefer candidates likely to escape while keeping perturbation cost low.
Do not explain.

Return ONLY valid JSON in this format:
{{
  "candidates": [
    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
  ]
}}
""".strip()

    def _call_api(self, prompt: str) -> str:
        url = f"{self.base_url}/chat/completions"

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a JSON-only black-box optimization assistant. Return only valid JSON.",
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
        candidates = []

        for x in raw:
            try:
                arr = np.array(x, dtype=float).reshape(-1)
                if arr.shape[0] != 8:
                    continue
                arr = np.clip(arr, 0.0, 1.0)
                candidates.append(LatentVector.from_numpy(arr))
            except Exception:
                continue

        return candidates

    def _refill_llm(self):
        try:
            text = self._call_api(self._prompt())
            cands = self._parse_candidates(text)
            if not cands:
                raise RuntimeError("No valid candidates parsed from API response.")
            self.pending.extend(cands)
        except Exception as e:
            print("[DeepSeek API fallback random]", repr(e))
            self.pending.extend([self._random_latent() for _ in range(self.candidate_count)])

    def _local_refine(self) -> LatentVector:
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

    def ask(self) -> LatentVector:
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

    def tell(self, latent: LatentVector, escaped: bool, cost: float):
        z = latent.to_numpy().astype(float)

        self.records.append({
            "z": z.tolist(),
            "escaped": bool(escaped),
            "cost": float(cost),
        })

        if escaped:
            self.escapes.append(EscapePoint(cost=float(cost), latent=z))

    def end(self, history):
        pass
