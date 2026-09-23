from __future__ import annotations

import json
import os
import re
import urllib.request
from typing import Any

import numpy as np

from attacks.v2.search_space import LatentVector
from attacks.v2.optimizers.base import Optimizer


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


class StrictAnonymousDeepSeekBatchOptimizer(Optimizer):
    """
    Strict anonymous LLM batch optimizer.

    What the LLM sees:
    - x1...x8
    - success / failure
    - cost

    What the LLM does NOT see:
    - GUI
    - trajectory
    - defense
    - detector
    - spatial / temporal names
    - V1 / V2 / V3
    - thresholds
    - features
    - layer-level decisions

    This is a generic black-box optimizer using only query-level feedback.
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
        min_distance: float = 0.06,
    ):
        self.model = model or os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
        self.base_url = (
            base_url or os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        ).rstrip("/")
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")

        if not self.api_key:
            raise RuntimeError(
                "Missing DEEPSEEK_API_KEY. Run: export DEEPSEEK_API_KEY='your_key'"
            )

        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.timeout = int(timeout)
        self.temperature = float(temperature)
        self.candidate_count = int(candidate_count)
        self.max_history = int(max_history)
        self.min_distance = float(min_distance)

        self.records = []
        self.pending = []

    def begin(self, session: Any, budget: int):
        self.records = []
        self.pending = []
        self.budget = int(budget)

    def _random_latent(self) -> LatentVector:
        return LatentVector.from_numpy(self.rng.random(8))

    def _compact_history(self):
        recent = self.records[-self.max_history :]
        out = []

        for i, r in enumerate(recent):
            out.append(
                {
                    "trial": len(self.records) - len(recent) + i + 1,
                    "x": [round(float(v), 4) for v in r["x"]],
                    "success": int(bool(r["success"])),
                    "cost": round(float(r["cost"]), 6),
                }
            )

        return out

    def _best_successes(self, k=8):
        successes = [r for r in self.records if r["success"]]
        successes = sorted(successes, key=lambda r: float(r["cost"]))[:k]

        return [
            {
                "x": [round(float(v), 4) for v in r["x"]],
                "cost": round(float(r["cost"]), 6),
            }
            for r in successes
        ]

    def _prompt(self):
        return f"""
You are optimizing an unknown black-box system.

You do not know what the system is.
You do not know its internal model, rules, thresholds, features, architecture, labels, or domain.
You only observe previous trials.

Each input is an 8-dimensional normalized vector:
[x1, x2, x3, x4, x5, x6, x7, x8]

Each dimension must be between 0 and 1.

For each previous trial:
- success = 1 means the trial achieved the objective
- success = 0 means the trial failed
- lower cost is better

Recent trials:
{json.dumps(self._compact_history(), ensure_ascii=False)}

Best successful trials:
{json.dumps(self._best_successes(), ensure_ascii=False)}

Task:
Generate {self.candidate_count} diverse new candidate vectors.
Prefer candidates that are likely to get success = 1 while keeping cost low.
Balance exploration and exploitation.
Do not explain.

Return ONLY valid JSON:
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
                    "content": (
                        "You are a JSON-only optimizer for an unknown black-box "
                        "system. Return only valid JSON."
                    ),
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

    def _valid_arr(self, x):
        try:
            arr = np.array(x, dtype=float).reshape(-1)
            if arr.shape[0] != 8:
                return None
            return np.clip(arr, 0.0, 1.0)
        except Exception:
            return None

    def _too_close(self, arr, accepted):
        for a in accepted:
            dist = float(np.linalg.norm(arr - a.to_numpy()))
            if dist < self.min_distance:
                return True
        return False

    def _parse_candidates(self, text: str):
        obj = _json_from_text(text)
        if not obj:
            return []

        raw = obj.get("candidates", [])
        if not isinstance(raw, list):
            return []

        accepted = []

        for item in raw:
            arr = self._valid_arr(item)
            if arr is None:
                continue

            cand = LatentVector.from_numpy(arr)

            if self._too_close(arr, accepted):
                continue

            accepted.append(cand)

            if len(accepted) >= self.candidate_count:
                break

        return accepted

    def _refill_llm(self):
        try:
            text = self._call_api(self._prompt())
            candidates = self._parse_candidates(text)

            if not candidates:
                raise RuntimeError("No valid anonymous candidates parsed.")

            self.pending.extend(candidates)

        except Exception as e:
            print("[strict-anon DeepSeek fallback random]", repr(e))
            self.pending.extend(
                [self._random_latent() for _ in range(self.candidate_count)]
            )

        while len(self.pending) < self.candidate_count:
            self.pending.append(self._random_latent())

    def ask(self) -> LatentVector:
        if not self.pending:
            self._refill_llm()

        if not self.pending:
            return self._random_latent()

        return self.pending.pop(0)

    def tell(self, latent: LatentVector, escaped: bool, cost: float):
        x = latent.to_numpy().astype(float)

        self.records.append(
            {
                "x": x.tolist(),
                "success": bool(escaped),
                "cost": float(cost),
            }
        )

    def end(self, history):
        pass
