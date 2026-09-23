#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import importlib
import json
import math
import os
import random
from pathlib import Path
from statistics import median

import numpy as np
import pandas as pd


DEFAULT_DEFENSE_MODULE = "evaluation.frozen_v1v2v3_defense_hardened"


# ======================================================================================
# Basic record/event utilities
# ======================================================================================

def get(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def set_attr(obj, key, value):
    if isinstance(obj, dict):
        obj[key] = value
    else:
        setattr(obj, key, value)


def get_x(e):
    for k in ["x", "X", "screen_x", "clientX", "touch_x", "pos_x"]:
        v = get(e, k, None)
        if v is not None:
            return float(v)
    return 0.0


def get_y(e):
    for k in ["y", "Y", "screen_y", "clientY", "touch_y", "pos_y"]:
        v = get(e, k, None)
        if v is not None:
            return float(v)
    return 0.0


def set_x(e, value):
    for k in ["x", "X", "screen_x", "clientX", "touch_x", "pos_x"]:
        if get(e, k, None) is not None:
            set_attr(e, k, float(value))
            return
    set_attr(e, "x", float(value))


def set_y(e, value):
    for k in ["y", "Y", "screen_y", "clientY", "touch_y", "pos_y"]:
        if get(e, k, None) is not None:
            set_attr(e, k, float(value))
            return
    set_attr(e, "y", float(value))


def get_t(e):
    return float(get(e, "timestamp_us"))


def set_t(e, value):
    set_attr(e, "timestamp_us", int(round(value)))


def valid_gesture(g):
    return isinstance(g, list) and len(g) >= 2


def get_gestures(record):
    return [
        g for g in get(record, "gestures", [])
        if valid_gesture(g)
    ]


def gesture_distance(g):
    if not valid_gesture(g):
        return 0.0
    return math.hypot(
        get_x(g[-1]) - get_x(g[0]),
        get_y(g[-1]) - get_y(g[0]),
    )


def gesture_duration(g):
    if not valid_gesture(g):
        return 0.0
    return max(0.0, get_t(g[-1]) - get_t(g[0]))


def ease_in_out(u):
    return 3 * u * u - 2 * u * u * u


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


# ======================================================================================
# Defense / dataset
# ======================================================================================

def load_defense():
    module_name = os.environ.get("GUI_DEFENSE_MODULE", DEFAULT_DEFENSE_MODULE)
    mod = importlib.import_module(module_name)
    cls = getattr(mod, "FrozenV1V2V3Defense")
    print("defense module:", module_name)
    return cls()


def score_record(defense, record):
    out = defense.score_record(
        record,
        task_cluster=None,
        v1_condition="Long Tap Author",
    )
    final_detect = bool(out.get("final_detect", False))
    return final_detect, out


def load_source_records(defense, source_list):
    inner = getattr(defense, "inner", {})
    if not isinstance(inner, dict) or source_list not in inner:
        raise RuntimeError(f"Cannot find inner.{source_list}")
    return list(inner[source_list])


def select_initially_detected(defense, source_records, n_sessions, max_scan):
    selected = []
    scanned = 0
    skipped = 0

    for rec in source_records:
        if max_scan is not None and scanned >= max_scan:
            break
        scanned += 1

        try:
            detected, _ = score_record(defense, rec)
        except Exception:
            skipped += 1
            continue

        if detected:
            selected.append(rec)

        if len(selected) >= n_sessions:
            break

    print(f"selected initially detected records: {len(selected)} / scanned {scanned}, skipped {skipped}")

    if len(selected) < n_sessions:
        print(f"WARNING: requested {n_sessions}, got {len(selected)}")

    return selected


# ======================================================================================
# Candidate parameterization
# ======================================================================================

OP_MODES_PARAM_ONLY = [
    "none",
    "curve",
]

OP_MODES_CONSERVATIVE = [
    "none",
    "curve",
    "split_swipe",
    "micro_pause",
    "dwell_click",
]

OP_MODES_FULL = [
    "none",
    "curve",
    "split_swipe",
    "micro_pause",
    "dwell_click",
    "micro_tap",
]


FAMILIES = [
    "spatial_low_amp_high_freq",
    "temporal_gap_duration",
    "mixed_balanced",
    "heterogeneous_mixed",
    "operation_split",
    "operation_pause",
    "operation_tap",
]


def sample_candidate(rng, method, family=None):
    if family is None:
        family = rng.choice(FAMILIES)

    if method == "two-stage-param":
        allowed_ops = OP_MODES_PARAM_ONLY
    elif method == "two-stage-op-conservative":
        allowed_ops = OP_MODES_CONSERVATIVE
    elif method == "two-stage-op":
        allowed_ops = OP_MODES_FULL
    else:
        raise ValueError(f"Unknown method: {method}")

    c = {
        "family": family,
        "spatial_amp_px": float(rng.uniform(0.0, 10.0)),
        "frequency": int(rng.choice([1, 2, 4, 6, 8, 12, 16])),
        "duration_scale": float(rng.uniform(0.55, 1.80)),
        "gap_scale": float(rng.uniform(0.55, 1.80)),
        "jitter_frac": float(rng.uniform(0.0, 0.35)),
        "heterogeneity": float(rng.uniform(0.0, 0.55)),
        "use_spatial": bool(rng.random() < 0.70),
        "use_temporal": bool(rng.random() < 0.70),
        "op_mode": str(rng.choice(allowed_ops)),
        "op_prob": float(rng.uniform(0.15, 0.75)),
        "op_strength": float(rng.uniform(0.10, 1.00)),
    }

    if family == "spatial_low_amp_high_freq":
        c.update({
            "spatial_amp_px": float(rng.uniform(0.5, 4.0)),
            "frequency": int(rng.choice([8, 12, 16])),
            "duration_scale": float(rng.uniform(0.85, 1.25)),
            "gap_scale": float(rng.uniform(0.85, 1.25)),
            "jitter_frac": float(rng.uniform(0.02, 0.18)),
            "heterogeneity": float(rng.uniform(0.05, 0.30)),
            "use_spatial": True,
            "use_temporal": bool(rng.random() < 0.45),
            "op_mode": str(rng.choice(["none", "curve"])),
        })

    elif family == "temporal_gap_duration":
        c.update({
            "spatial_amp_px": float(rng.uniform(0.0, 2.0)),
            "frequency": int(rng.choice([1, 2, 4])),
            "duration_scale": float(rng.uniform(0.65, 1.65)),
            "gap_scale": float(rng.uniform(0.65, 1.65)),
            "jitter_frac": float(rng.uniform(0.05, 0.30)),
            "heterogeneity": float(rng.uniform(0.00, 0.25)),
            "use_spatial": bool(rng.random() < 0.30),
            "use_temporal": True,
            "op_mode": "none",
        })

    elif family == "mixed_balanced":
        c.update({
            "spatial_amp_px": float(rng.uniform(1.0, 7.0)),
            "frequency": int(rng.choice([2, 4, 6, 8])),
            "duration_scale": float(rng.uniform(0.75, 1.50)),
            "gap_scale": float(rng.uniform(0.75, 1.50)),
            "jitter_frac": float(rng.uniform(0.04, 0.25)),
            "heterogeneity": float(rng.uniform(0.10, 0.45)),
            "use_spatial": True,
            "use_temporal": True,
            "op_mode": str(rng.choice([x for x in allowed_ops if x != "micro_tap"])),
        })

    elif family == "heterogeneous_mixed":
        c.update({
            "spatial_amp_px": float(rng.uniform(0.5, 9.0)),
            "frequency": int(rng.choice([1, 2, 4, 8, 12])),
            "duration_scale": float(rng.uniform(0.65, 1.75)),
            "gap_scale": float(rng.uniform(0.65, 1.75)),
            "jitter_frac": float(rng.uniform(0.08, 0.35)),
            "heterogeneity": float(rng.uniform(0.35, 0.75)),
            "use_spatial": True,
            "use_temporal": True,
            "op_mode": str(rng.choice(allowed_ops)),
        })

    elif family == "operation_split":
        c.update({
            "spatial_amp_px": float(rng.uniform(0.5, 5.0)),
            "frequency": int(rng.choice([2, 4, 6])),
            "duration_scale": float(rng.uniform(0.80, 1.35)),
            "gap_scale": float(rng.uniform(0.75, 1.40)),
            "jitter_frac": float(rng.uniform(0.02, 0.20)),
            "heterogeneity": float(rng.uniform(0.10, 0.45)),
            "use_spatial": True,
            "use_temporal": True,
            "op_mode": "split_swipe" if "split_swipe" in allowed_ops else "curve",
            "op_prob": float(rng.uniform(0.25, 0.85)),
        })

    elif family == "operation_pause":
        c.update({
            "spatial_amp_px": float(rng.uniform(0.0, 4.0)),
            "frequency": int(rng.choice([1, 2, 4])),
            "duration_scale": float(rng.uniform(0.85, 1.60)),
            "gap_scale": float(rng.uniform(0.80, 1.60)),
            "jitter_frac": float(rng.uniform(0.04, 0.30)),
            "heterogeneity": float(rng.uniform(0.10, 0.55)),
            "use_spatial": bool(rng.random() < 0.50),
            "use_temporal": True,
            "op_mode": "micro_pause" if "micro_pause" in allowed_ops else "none",
            "op_prob": float(rng.uniform(0.25, 0.90)),
        })

    elif family == "operation_tap":
        c.update({
            "spatial_amp_px": float(rng.uniform(0.0, 3.0)),
            "frequency": int(rng.choice([1, 2, 4])),
            "duration_scale": float(rng.uniform(0.85, 1.45)),
            "gap_scale": float(rng.uniform(0.80, 1.55)),
            "jitter_frac": float(rng.uniform(0.04, 0.25)),
            "heterogeneity": float(rng.uniform(0.10, 0.50)),
            "use_spatial": bool(rng.random() < 0.35),
            "use_temporal": True,
            "op_mode": "micro_tap" if "micro_tap" in allowed_ops else "dwell_click" if "dwell_click" in allowed_ops else "none",
            "op_prob": float(rng.uniform(0.15, 0.55)),
        })

    if c["op_mode"] not in allowed_ops:
        c["op_mode"] = "none"

    return c


def perturb_candidate(rng, base, method, scale=0.16):
    c = dict(base)

    def jitter_log(name, lo, hi):
        v = float(c[name])
        factor = math.exp(rng.normal(0, scale))
        c[name] = float(clamp(v * factor, lo, hi))

    def jitter_add(name, lo, hi):
        v = float(c[name])
        c[name] = float(clamp(v + rng.normal(0, scale), lo, hi))

    jitter_log("spatial_amp_px", 0.0, 12.0)
    jitter_log("duration_scale", 0.45, 2.20)
    jitter_log("gap_scale", 0.45, 2.20)
    jitter_add("jitter_frac", 0.0, 0.45)
    jitter_add("heterogeneity", 0.0, 0.85)
    jitter_add("op_prob", 0.05, 0.95)
    jitter_add("op_strength", 0.05, 1.25)

    if rng.random() < 0.25:
        c["frequency"] = int(rng.choice([1, 2, 4, 6, 8, 12, 16]))

    if rng.random() < 0.12:
        c["use_spatial"] = not bool(c["use_spatial"])

    if rng.random() < 0.12:
        c["use_temporal"] = not bool(c["use_temporal"])

    if rng.random() < 0.20:
        if method == "two-stage-param":
            allowed_ops = OP_MODES_PARAM_ONLY
        elif method == "two-stage-op-conservative":
            allowed_ops = OP_MODES_CONSERVATIVE
        else:
            allowed_ops = OP_MODES_FULL
        c["op_mode"] = str(rng.choice(allowed_ops))

    c["family"] = str(c.get("family", "refined")) + "_refined"
    return c


# ======================================================================================
# Spatial mutation
# ======================================================================================

def apply_spatial_curve(g, cand, rng):
    g2 = copy.deepcopy(g)
    n = len(g2)
    if n < 2:
        return g2

    x0, y0 = get_x(g2[0]), get_y(g2[0])
    x1, y1 = get_x(g2[-1]), get_y(g2[-1])

    dx = x1 - x0
    dy = y1 - y0
    dist = max(1e-6, math.hypot(dx, dy))

    nx = -dy / dist
    ny = dx / dist

    amp = float(cand["spatial_amp_px"])
    freq = int(cand["frequency"])
    hetero = float(cand["heterogeneity"])

    local_amp = amp * math.exp(rng.normal(0, hetero * 0.5))
    phase = rng.uniform(0, 2 * math.pi)

    for i, e in enumerate(g2):
        u = i / max(n - 1, 1)

        # Preserve endpoints so task semantics are less likely to change.
        if i == 0 or i == n - 1:
            continue

        envelope = math.sin(math.pi * u)
        wave = math.sin(2 * math.pi * freq * u + phase)
        lateral = local_amp * envelope * wave

        # Small tremor.
        tremor = rng.normal(0, max(0.15, local_amp * 0.08))

        set_x(e, get_x(e) + nx * lateral + tremor)
        set_y(e, get_y(e) + ny * lateral + rng.normal(0, max(0.15, local_amp * 0.08)))

    return g2


# ======================================================================================
# Operation-level mutations
# ======================================================================================

def interpolate_like(template, p0, p1):
    out = copy.deepcopy(template)
    n = len(out)

    for i, e in enumerate(out):
        u = i / max(n - 1, 1)
        u2 = ease_in_out(u)
        x = p0[0] + (p1[0] - p0[0]) * u2
        y = p0[1] + (p1[1] - p0[1]) * u2
        set_x(e, x)
        set_y(e, y)

    return out


def op_split_swipe(g, cand, rng):
    if len(g) < 4 or gesture_distance(g) < 25:
        return [copy.deepcopy(g)]

    n = len(g)
    mid_idx = int(clamp(round(n * rng.uniform(0.40, 0.60)), 1, n - 2))

    x0, y0 = get_x(g[0]), get_y(g[0])
    x1, y1 = get_x(g[-1]), get_y(g[-1])
    dx, dy = x1 - x0, y1 - y0
    dist = max(1e-6, math.hypot(dx, dy))
    nx, ny = -dy / dist, dx / dist

    frac = rng.uniform(0.42, 0.58)
    offset = rng.normal(0, float(cand["op_strength"]) * 0.08 * dist)

    mid = (
        x0 + dx * frac + nx * offset,
        y0 + dy * frac + ny * offset,
    )

    g1_template = g[: mid_idx + 1]
    g2_template = g[mid_idx:]

    g1 = interpolate_like(g1_template, (x0, y0), mid)
    g2 = interpolate_like(g2_template, mid, (x1, y1))

    return [g1, g2]


def op_micro_pause(g, cand, rng):
    if len(g) < 3:
        return [copy.deepcopy(g)]

    g2 = copy.deepcopy(g)
    n_insert = 1 + int(rng.random() < 0.30)
    idx = rng.integers(1, len(g2) - 1)

    base = copy.deepcopy(g2[idx])
    inserts = []

    for _ in range(n_insert):
        e = copy.deepcopy(base)
        set_x(e, get_x(e) + rng.normal(0, 0.4 + cand["op_strength"]))
        set_y(e, get_y(e) + rng.normal(0, 0.4 + cand["op_strength"]))
        inserts.append(e)

    return [g2[: idx + 1] + inserts + g2[idx + 1:]]


def op_dwell_click(g, cand, rng):
    # Mainly useful for tap-like gestures.
    if gesture_distance(g) > 20:
        return [copy.deepcopy(g)]

    g2 = copy.deepcopy(g)
    center = copy.deepcopy(g2[len(g2) // 2])
    extra = []

    n_extra = 1 + int(rng.random() < 0.50)

    for _ in range(n_extra):
        e = copy.deepcopy(center)
        set_x(e, get_x(e) + rng.normal(0, 0.8 + cand["op_strength"]))
        set_y(e, get_y(e) + rng.normal(0, 0.8 + cand["op_strength"]))
        extra.append(e)

    idx = max(1, len(g2) // 2)
    return [g2[:idx] + extra + g2[idx:]]


def op_micro_tap(g, cand, rng):
    # This may be semantically unstable in a real app.
    # For this offline detector test, it represents adding a no-op-like operation.
    g_main = copy.deepcopy(g)

    anchor = copy.deepcopy(g[0] if rng.random() < 0.5 else g[-1])
    x = get_x(anchor)
    y = get_y(anchor)

    e1 = copy.deepcopy(anchor)
    e2 = copy.deepcopy(anchor)

    r = rng.uniform(1.0, 4.0) * float(cand["op_strength"])
    set_x(e1, x + rng.normal(0, r))
    set_y(e1, y + rng.normal(0, r))
    set_x(e2, x + rng.normal(0, r))
    set_y(e2, y + rng.normal(0, r))

    tap = [e1, e2]

    if rng.random() < 0.5:
        return [tap, g_main]
    else:
        return [g_main, tap]


def apply_operation(g, cand, rng):
    mode = str(cand.get("op_mode", "none"))

    if mode == "none" or rng.random() > float(cand.get("op_prob", 0.0)):
        return [copy.deepcopy(g)]

    if mode == "curve":
        return [apply_spatial_curve(g, cand, rng)]

    if mode == "split_swipe":
        return op_split_swipe(g, cand, rng)

    if mode == "micro_pause":
        return op_micro_pause(g, cand, rng)

    if mode == "dwell_click":
        return op_dwell_click(g, cand, rng)

    if mode == "micro_tap":
        return op_micro_tap(g, cand, rng)

    return [copy.deepcopy(g)]


# ======================================================================================
# Temporal mutation / repair
# ======================================================================================

def repair_global_timestamps(gestures, cand, rng):
    if not gestures:
        return gestures

    new_gestures = copy.deepcopy(gestures)

    prev_old_end = None
    prev_new_end = None

    for gi, g in enumerate(new_gestures):
        old_ts = np.array([get_t(e) for e in g], dtype=float)

        if len(old_ts) < 2:
            rel = np.array([0.0])
            base_duration = 50000.0
        else:
            old_start = old_ts[0]
            old_end = old_ts[-1]
            base_duration = max(1000.0, old_end - old_start)

            if old_end > old_start:
                rel = (old_ts - old_start) / (old_end - old_start)
            else:
                rel = np.linspace(0, 1, len(old_ts))

        if prev_old_end is None:
            base_gap = 0.0
            new_start = old_ts[0]
        else:
            base_gap = max(1000.0, old_ts[0] - prev_old_end)
            if bool(cand["use_temporal"]):
                gap = base_gap * float(cand["gap_scale"])
                gap *= math.exp(rng.normal(0, float(cand["heterogeneity"]) * 0.25))
                gap += rng.normal(0, float(cand["jitter_frac"]) * max(base_gap, 10000.0))
                gap = max(1000.0, gap)
            else:
                gap = base_gap

            new_start = prev_new_end + gap

        if bool(cand["use_temporal"]):
            duration = base_duration * float(cand["duration_scale"])
            duration *= math.exp(rng.normal(0, float(cand["heterogeneity"]) * 0.25))
            duration += rng.normal(0, float(cand["jitter_frac"]) * max(base_duration, 10000.0))
            duration = max(1000.0, duration)
        else:
            duration = base_duration

        new_ts = new_start + rel * duration

        # Strictly non-decreasing inside gesture.
        for i in range(1, len(new_ts)):
            if new_ts[i] <= new_ts[i - 1]:
                new_ts[i] = new_ts[i - 1] + 1000.0

        for e, t in zip(g, new_ts):
            set_t(e, t)

        prev_old_end = old_ts[-1]
        prev_new_end = new_ts[-1]

    return new_gestures


def mutate_record(record, cand, rng):
    new = copy.deepcopy(record)
    old_gestures = get_gestures(record)
    new_gestures = []

    for g in old_gestures:
        g2 = copy.deepcopy(g)

        if bool(cand["use_spatial"]):
            g2 = apply_spatial_curve(g2, cand, rng)

        produced = apply_operation(g2, cand, rng)
        new_gestures.extend(produced)

    new_gestures = repair_global_timestamps(new_gestures, cand, rng)

    new["gestures"] = new_gestures
    new["_two_stage_operation_attack"] = copy.deepcopy(cand)

    return new


# ======================================================================================
# Cost estimation
# ======================================================================================

def flatten_events(record, max_events=2000):
    arr = []
    for g in get_gestures(record):
        for e in g:
            arr.append((get_x(e), get_y(e), get_t(e)))
            if len(arr) >= max_events:
                return arr
    return arr


def estimate_cost(original, mutated):
    a = flatten_events(original)
    b = flatten_events(mutated)

    if not a or not b:
        return 999.0

    m = min(len(a), len(b))
    aa = np.array(a[:m], dtype=float)
    bb = np.array(b[:m], dtype=float)

    spatial_rms = float(np.sqrt(np.mean((aa[:, :2] - bb[:, :2]) ** 2)))

    dur_a = max(1.0, a[-1][2] - a[0][2])
    dur_b = max(1.0, b[-1][2] - b[0][2])
    duration_cost = abs(math.log(dur_b / dur_a))

    event_cost = abs(len(b) - len(a)) / max(1, len(a))
    gesture_cost = abs(len(get_gestures(mutated)) - len(get_gestures(original))) / max(1, len(get_gestures(original)))

    # Normalized rough cost. Lower is better.
    return float(
        spatial_rms / 20.0
        + duration_cost
        + 1.5 * event_cost
        + 2.0 * gesture_cost
    )


# ======================================================================================
# Two-stage search
# ======================================================================================

def choose_refinement_base(rng, history, method):
    successes = [h for h in history if h["escaped"]]

    if successes:
        # Prefer low-cost successful basins.
        successes = sorted(successes, key=lambda x: x["cost"])
        top_k = max(1, min(5, len(successes)))
        base = rng.choice(successes[:top_k])
        return base["candidate"]

    # No escape yet: choose promising family/op by empirical success with smoothing.
    by_arm = {}

    for h in history:
        cand = h["candidate"]
        arm = (cand.get("family", "unknown"), cand.get("op_mode", "none"))
        if arm not in by_arm:
            by_arm[arm] = {"n": 0, "s": 0}
        by_arm[arm]["n"] += 1
        by_arm[arm]["s"] += int(h["escaped"])

    if by_arm:
        arms = list(by_arm.keys())
        weights = []
        for arm in arms:
            n = by_arm[arm]["n"]
            s = by_arm[arm]["s"]
            weights.append((s + 1.0) / (n + 2.0))

        weights = np.array(weights, dtype=float)
        weights = weights / weights.sum()

        chosen_idx = rng.choice(len(arms), p=weights)
        family, _op = arms[chosen_idx]
        return sample_candidate(rng, method, family=family)

    return sample_candidate(rng, method)


def attack_one_record(defense, record, budget, seed, method, explore_frac):
    rng = np.random.default_rng(seed)

    n_explore = int(math.ceil(budget * explore_frac))
    n_explore = max(1, min(budget, n_explore))

    history = []
    first_escape_q = None
    best_cost = None
    best_candidate = None
    best_output = None

    for q in range(1, budget + 1):
        if q <= n_explore:
            # Force broad basin coverage in early stage.
            family = FAMILIES[(q - 1) % len(FAMILIES)]
            cand = sample_candidate(rng, method, family=family)
        else:
            if rng.random() < 0.20:
                cand = sample_candidate(rng, method)
            else:
                base = choose_refinement_base(rng, history, method)
                cand = perturb_candidate(rng, base, method, scale=0.14)

        try:
            mutated = mutate_record(record, cand, rng)
            detected, out = score_record(defense, mutated)
            escaped = not detected
            cost = estimate_cost(record, mutated)

            item = {
                "q": q,
                "ok": True,
                "detected": detected,
                "escaped": escaped,
                "cost": cost,
                "candidate": cand,
                "error": "",
            }

            if escaped:
                if first_escape_q is None:
                    first_escape_q = q

                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    best_candidate = cand
                    best_output = out

        except Exception as e:
            item = {
                "q": q,
                "ok": False,
                "detected": None,
                "escaped": False,
                "cost": None,
                "candidate": cand,
                "error": repr(e),
            }

        history.append(item)

    return {
        "escaped": first_escape_q is not None,
        "queries_to_first_escape": first_escape_q,
        "best_cost": best_cost,
        "best_candidate": best_candidate,
        "best_output": best_output,
        "history": history,
    }


# ======================================================================================
# Aggregation
# ======================================================================================

def summarize_session_rows(rows):
    df = pd.DataFrame(rows)

    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    seed_rows = []

    for (method, budget, seed), g in df.groupby(["method", "budget", "seed"]):
        escaped_g = g[g["escaped"] == True]

        seed_rows.append({
            "method": method,
            "budget": int(budget),
            "seed": int(seed),
            "n_total": len(g),
            "n_completed": int(g["ok"].sum()),
            "escaped": int(g["escaped"].sum()),
            "seed_asr": float(g["escaped"].mean()) if len(g) else 0.0,
            "median_q_first": float(escaped_g["queries_to_first_escape"].median()) if len(escaped_g) else None,
            "median_best_cost": float(escaped_g["best_cost"].median()) if len(escaped_g) else None,
        })

    seed_df = pd.DataFrame(seed_rows)

    agg_rows = []

    for (method, budget), g in seed_df.groupby(["method", "budget"]):
        agg_rows.append({
            "method": method,
            "budget": int(budget),
            "n_total": int(g["n_total"].sum()),
            "n_completed": int(g["n_completed"].sum()),
            "escaped": int(g["escaped"].sum()),
            "pooled_asr": float(g["escaped"].sum() / max(1, g["n_total"].sum())),
            "median_of_seed_median_q_first": float(g["median_q_first"].dropna().median()) if g["median_q_first"].notna().any() else None,
            "median_of_seed_median_best_cost": float(g["median_best_cost"].dropna().median()) if g["median_best_cost"].notna().any() else None,
        })

    agg_df = pd.DataFrame(agg_rows).sort_values(["method", "budget"])
    seed_df = seed_df.sort_values(["method", "budget", "seed"])

    return seed_df, agg_df


# ======================================================================================
# Main
# ======================================================================================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--methods",
        default="two-stage-param,two-stage-op-conservative,two-stage-op",
    )
    ap.add_argument(
        "--budgets",
        default="10,30",
    )
    ap.add_argument(
        "--seeds",
        default="20260917",
    )
    ap.add_argument(
        "--n-sessions",
        type=int,
        default=100,
    )
    ap.add_argument(
        "--max-scan",
        type=int,
        default=499,
    )
    ap.add_argument(
        "--source-list",
        default="long_records",
        help="Default: long_records, because this is the stronger existing humanization condition.",
    )
    ap.add_argument(
        "--explore-frac",
        type=float,
        default=0.55,
    )
    ap.add_argument(
        "--output-dir",
        default="results/v2_two_stage_operation_attack",
    )
    ap.add_argument(
        "--save-history",
        choices=["none", "full"],
        default="full",
    )

    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    budgets = [int(x) for x in args.budgets.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    print("=" * 100)
    print("TWO-STAGE + OPERATION-AWARE BLACK-BOX ATTACK")
    print("=" * 100)
    print("methods:", methods)
    print("budgets:", budgets)
    print("seeds:", seeds)
    print("n_sessions:", args.n_sessions)
    print("source_list:", args.source_list)
    print("explore_frac:", args.explore_frac)
    print("output_dir:", out_dir)

    defense = load_defense()
    source_records = load_source_records(defense, args.source_list)
    records = select_initially_detected(
        defense=defense,
        source_records=source_records,
        n_sessions=args.n_sessions,
        max_scan=args.max_scan,
    )

    if not records:
        raise SystemExit("No initially detected records selected.")

    session_rows = []
    history_path = out_dir / "attack_history.jsonl"

    if history_path.exists():
        history_path.unlink()

    for method in methods:
        for budget in budgets:
            for seed in seeds:
                print("\n" + "=" * 100)
                print(f"method={method} budget={budget} seed={seed}")
                print("=" * 100)

                for idx, rec in enumerate(records):
                    rec_seed = seed + idx * 100003 + budget * 9176

                    participant = str(get(rec, "participant", "UNKNOWN"))
                    session_id = str(get(rec, "session_id", "UNKNOWN"))

                    result = attack_one_record(
                        defense=defense,
                        record=rec,
                        budget=budget,
                        seed=rec_seed,
                        method=method,
                        explore_frac=args.explore_frac,
                    )

                    row = {
                        "method": method,
                        "budget": budget,
                        "seed": seed,
                        "record_index": idx,
                        "participant": participant,
                        "session_id": session_id,
                        "ok": True,
                        "escaped": bool(result["escaped"]),
                        "queries_to_first_escape": result["queries_to_first_escape"],
                        "best_cost": result["best_cost"],
                        "best_candidate_json": json.dumps(result["best_candidate"], ensure_ascii=False, sort_keys=True),
                    }

                    session_rows.append(row)

                    if args.save_history == "full":
                        with history_path.open("a") as f:
                            f.write(json.dumps({
                                **row,
                                "history": result["history"],
                            }, ensure_ascii=False, default=str) + "\n")

                    if (idx + 1) % 20 == 0 or idx + 1 == len(records):
                        tmp_seed_df, tmp_agg_df = summarize_session_rows(session_rows)
                        sub = tmp_agg_df[
                            (tmp_agg_df["method"] == method)
                            & (tmp_agg_df["budget"] == budget)
                        ]
                        if len(sub):
                            r = sub.iloc[0].to_dict()
                            print(
                                f"processed {idx + 1}/{len(records)} | "
                                f"current pooled ASR={r['pooled_asr'] * 100:.2f}% "
                                f"escaped={int(r['escaped'])}/{int(r['n_total'])}"
                            )

                session_df = pd.DataFrame(session_rows)
                seed_df, agg_df = summarize_session_rows(session_rows)

                session_df.to_csv(out_dir / "session_results.csv", index=False)
                seed_df.to_csv(out_dir / "seed_summary.csv", index=False)
                agg_df.to_csv(out_dir / "aggregate_by_budget_method.csv", index=False)

                print("\nCurrent aggregate:")
                print(agg_df.to_string(index=False))

    session_df = pd.DataFrame(session_rows)
    seed_df, agg_df = summarize_session_rows(session_rows)

    session_df.to_csv(out_dir / "session_results.csv", index=False)
    seed_df.to_csv(out_dir / "seed_summary.csv", index=False)
    agg_df.to_csv(out_dir / "aggregate_by_budget_method.csv", index=False)

    print("\n" + "=" * 100)
    print("FINAL AGGREGATE")
    print("=" * 100)
    print(agg_df.to_string(index=False))

    print("\nsaved:")
    print(out_dir / "session_results.csv")
    print(out_dir / "seed_summary.csv")
    print(out_dir / "aggregate_by_budget_method.csv")
    if args.save_history == "full":
        print(history_path)


if __name__ == "__main__":
    main()
