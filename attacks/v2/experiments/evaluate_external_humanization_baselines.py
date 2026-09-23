#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.frozen_v1v2v3_defense_hardened import FrozenV1V2V3Defense


OUT = Path("results/frozen_defense_external_humanization")
OUT.mkdir(parents=True, exist_ok=True)


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


def valid_gestures(record):
    gs = get(record, "gestures", [])
    return [g for g in gs if isinstance(g, list) and len(g) >= 2]


def ease_in_out(t):
    return 3 * t * t - 2 * t * t * t


def cubic_bezier(p0, p1, p2, p3, t):
    return (
        ((1 - t) ** 3) * p0
        + 3 * ((1 - t) ** 2) * t * p1
        + 3 * (1 - t) * (t ** 2) * p2
        + (t ** 3) * p3
    )


def rewrite_gesture_path(gesture, xs, ys, ts=None):
    g = copy.deepcopy(gesture)
    n = len(g)

    for i, e in enumerate(g):
        set_x(e, xs[i])
        set_y(e, ys[i])
        if ts is not None:
            set_t(e, ts[i])

    return g


def ghost_cursor_style(gesture, rng):
    """
    GhostCursor-inspired:
    cubic Bezier curve + mild overshoot-like curvature + eased timing.
    """
    n = len(gesture)
    if n < 2:
        return copy.deepcopy(gesture)

    x0, y0 = get_x(gesture[0]), get_y(gesture[0])
    x1, y1 = get_x(gesture[-1]), get_y(gesture[-1])
    t0, t1 = get_t(gesture[0]), get_t(gesture[-1])

    dx, dy = x1 - x0, y1 - y0
    dist = math.hypot(dx, dy)
    if dist < 1e-6:
        return copy.deepcopy(gesture)

    nx, ny = -dy / dist, dx / dist

    curve = rng.normal(0, 0.12 * dist)
    p0 = np.array([x0, y0])
    p3 = np.array([x1, y1])
    p1 = p0 + np.array([dx, dy]) * rng.uniform(0.25, 0.45) + np.array([nx, ny]) * curve
    p2 = p0 + np.array([dx, dy]) * rng.uniform(0.55, 0.80) - np.array([nx, ny]) * curve * rng.uniform(0.4, 1.0)

    us = np.linspace(0, 1, n)
    eased = np.array([ease_in_out(u) for u in us])

    pts = np.array([cubic_bezier(p0, p1, p2, p3, u) for u in eased])

    duration = max(1.0, t1 - t0)
    duration *= rng.uniform(0.9, 1.35)
    ts = t0 + eased * duration

    return rewrite_gesture_path(gesture, pts[:, 0], pts[:, 1], ts)


def human_cursor_style(gesture, rng):
    """
    HumanCursor-inspired:
    variable speed + acceleration/deceleration + curvature + small tremor.
    """
    n = len(gesture)
    if n < 2:
        return copy.deepcopy(gesture)

    x0, y0 = get_x(gesture[0]), get_y(gesture[0])
    x1, y1 = get_x(gesture[-1]), get_y(gesture[-1])
    t0, t1 = get_t(gesture[0]), get_t(gesture[-1])

    dx, dy = x1 - x0, y1 - y0
    dist = math.hypot(dx, dy)
    if dist < 1e-6:
        return copy.deepcopy(gesture)

    nx, ny = -dy / dist, dx / dist
    us = np.linspace(0, 1, n)

    speed_profile = np.array([ease_in_out(u) for u in us])
    lateral = np.sin(us * math.pi) * rng.normal(0, 0.08 * dist)
    tremor = rng.normal(0, max(0.5, 0.008 * dist), size=n)

    xs = x0 + dx * speed_profile + nx * lateral + tremor
    ys = y0 + dy * speed_profile + ny * lateral + rng.normal(0, max(0.5, 0.008 * dist), size=n)

    xs[0], ys[0] = x0, y0
    xs[-1], ys[-1] = x1, y1

    duration = max(1.0, t1 - t0)
    duration *= rng.uniform(0.95, 1.5)
    ts = t0 + speed_profile * duration

    return rewrite_gesture_path(gesture, xs, ys, ts)


def becaptcha_function_style(gesture, rng):
    """
    BeCAPTCHA-Mouse-inspired function-based trajectory:
    polynomial/exponential path family + non-uniform velocity profile.
    """
    n = len(gesture)
    if n < 2:
        return copy.deepcopy(gesture)

    x0, y0 = get_x(gesture[0]), get_y(gesture[0])
    x1, y1 = get_x(gesture[-1]), get_y(gesture[-1])
    t0, t1 = get_t(gesture[0]), get_t(gesture[-1])

    dx, dy = x1 - x0, y1 - y0
    dist = math.hypot(dx, dy)
    if dist < 1e-6:
        return copy.deepcopy(gesture)

    nx, ny = -dy / dist, dx / dist
    us = np.linspace(0, 1, n)

    family = rng.choice(["quadratic", "exponential", "sinusoidal"])
    if family == "quadratic":
        progress = us ** rng.uniform(0.75, 1.35)
    elif family == "exponential":
        a = rng.uniform(1.2, 2.5)
        progress = (np.exp(a * us) - 1) / (np.exp(a) - 1)
    else:
        progress = us + rng.uniform(-0.06, 0.06) * np.sin(2 * math.pi * us)
        progress = np.clip(progress, 0, 1)
        progress = np.maximum.accumulate(progress)
        progress = progress / max(progress[-1], 1e-9)

    lateral_amp = rng.uniform(0.02, 0.10) * dist
    lateral = lateral_amp * np.sin(math.pi * us) * rng.choice([-1, 1])

    xs = x0 + dx * progress + nx * lateral
    ys = y0 + dy * progress + ny * lateral

    xs[0], ys[0] = x0, y0
    xs[-1], ys[-1] = x1, y1

    duration = max(1.0, t1 - t0)
    duration *= rng.uniform(0.85, 1.45)
    ts = t0 + progress * duration

    return rewrite_gesture_path(gesture, xs, ys, ts)


def windmouse_style(gesture, rng):
    """
    WindMouse / motor-control-inspired baseline:
    noisy force-like path with attraction to target.
    """
    n = len(gesture)
    if n < 2:
        return copy.deepcopy(gesture)

    x0, y0 = get_x(gesture[0]), get_y(gesture[0])
    x1, y1 = get_x(gesture[-1]), get_y(gesture[-1])
    t0, t1 = get_t(gesture[0]), get_t(gesture[-1])

    pos = np.array([x0, y0], dtype=float)
    target = np.array([x1, y1], dtype=float)
    start = np.array([x0, y0], dtype=float)

    dist0 = max(1e-6, np.linalg.norm(target - start))
    v = np.zeros(2, dtype=float)
    pts = []

    for i in range(n):
        alpha = i / max(n - 1, 1)

        desired = start + (target - start) * alpha
        to_desired = desired - pos

        wind = rng.normal(0, 0.015 * dist0, size=2)
        gravity = 0.45 * to_desired

        v = 0.65 * v + gravity + wind
        pos = pos + v

        pts.append(pos.copy())

    pts = np.asarray(pts)
    pts[0] = start
    pts[-1] = target

    us = np.linspace(0, 1, n)
    progress = np.array([ease_in_out(u) for u in us])

    duration = max(1.0, t1 - t0)
    duration *= rng.uniform(0.9, 1.6)
    ts = t0 + progress * duration

    return rewrite_gesture_path(gesture, pts[:, 0], pts[:, 1], ts)


def engineering_randomization_style(gesture, rng):
    """
    Common engineering baseline:
    small random offsets + duration/gap-like timing jitter.
    """
    n = len(gesture)
    if n < 2:
        return copy.deepcopy(gesture)

    xs = np.array([get_x(e) for e in gesture], dtype=float)
    ys = np.array([get_y(e) for e in gesture], dtype=float)
    ts = np.array([get_t(e) for e in gesture], dtype=float)

    jitter_px = rng.uniform(1.0, 4.0)
    xs2 = xs + rng.normal(0, jitter_px, size=n)
    ys2 = ys + rng.normal(0, jitter_px, size=n)

    # Preserve endpoints to avoid task failure.
    xs2[0], ys2[0] = xs[0], ys[0]
    xs2[-1], ys2[-1] = xs[-1], ys[-1]

    t0 = ts[0]
    duration = max(1.0, ts[-1] - ts[0])
    duration *= rng.uniform(0.85, 1.4)

    us = np.linspace(0, 1, n)
    ts2 = t0 + us * duration
    ts2 += rng.normal(0, 0.015 * duration, size=n)
    ts2 = np.maximum.accumulate(ts2)
    ts2[0] = t0

    return rewrite_gesture_path(gesture, xs2, ys2, ts2)


BASELINES = {
    "ghost_cursor_style": ghost_cursor_style,
    "human_cursor_style": human_cursor_style,
    "becaptcha_function_style": becaptcha_function_style,
    "windmouse_style": windmouse_style,
    "engineering_randomization": engineering_randomization_style,
}


def apply_baseline(record, method, seed):
    rng = np.random.default_rng(seed)
    new = copy.deepcopy(record)

    gestures = valid_gestures(new)
    new_gestures = []

    for g in gestures:
        new_gestures.append(BASELINES[method](g, rng))

    new["gestures"] = new_gestures
    new["_external_humanization"] = {
        "method": method,
        "seed": seed,
        "note": "experimental adapter for frozen-defense baseline evaluation",
    }

    return new


def load_raw_records(defense):
    # Prefer the frozen defense internal raw_records.
    for space_name in ["inner", "ns"]:
        space = getattr(defense, space_name, {})
        if isinstance(space, dict) and "raw_records" in space:
            return list(space["raw_records"])
    raise RuntimeError("Cannot find raw_records inside frozen defense.")


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sessions", type=int, default=499)
    ap.add_argument("--seeds", default="20260917")
    ap.add_argument("--methods", default=",".join(BASELINES.keys()))
    args = ap.parse_args()

    defense = FrozenV1V2V3Defense()
    raw_records = load_raw_records(defense)[: args.n_sessions]

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]

    rows = []
    detail_rows = []

    print("=" * 100)
    print("External humanization baseline evaluation")
    print("=" * 100)
    print("n raw records:", len(raw_records))
    print("methods:", methods)
    print("seeds:", seeds)

    for method in methods:
        if method not in BASELINES:
            raise ValueError(f"Unknown method: {method}")

        for seed in seeds:
            print("\n" + "-" * 100)
            print("method:", method, "seed:", seed)
            print("-" * 100)

            n = 0
            detected = 0
            errors = 0

            for i, rec in enumerate(raw_records):
                try:
                    mutated = apply_baseline(rec, method, seed + i * 9973)
                    out = defense.score_record(
                        mutated,
                        task_cluster=None,
                        v1_condition="Long Tap Author",
                    )
                    final_detect = bool(out.get("final_detect", False))

                    n += 1
                    detected += int(final_detect)

                    detail_rows.append({
                        "method": method,
                        "seed": seed,
                        "record_index": i,
                        "participant": str(get(rec, "participant", "UNKNOWN")),
                        "session_id": str(get(rec, "session_id", "UNKNOWN")),
                        "ok": True,
                        "final_detect": final_detect,
                        "error": "",
                    })

                except Exception as e:
                    errors += 1
                    detail_rows.append({
                        "method": method,
                        "seed": seed,
                        "record_index": i,
                        "participant": str(get(rec, "participant", "UNKNOWN")),
                        "session_id": str(get(rec, "session_id", "UNKNOWN")),
                        "ok": False,
                        "final_detect": None,
                        "error": repr(e),
                    })

                if (i + 1) % 100 == 0:
                    print("processed", i + 1, "/", len(raw_records))

            detect_rate = detected / n * 100 if n else None
            escape_rate = 100 - detect_rate if detect_rate is not None else None

            rows.append({
                "method": method,
                "seed": seed,
                "n_scored": n,
                "n_errors": errors,
                "detected": detected,
                "detection_rate_%": round(detect_rate, 2) if detect_rate is not None else None,
                "escape_rate_%": round(escape_rate, 2) if escape_rate is not None else None,
            })

            print("summary:", rows[-1])

    summary = pd.DataFrame(rows)
    detail = pd.DataFrame(detail_rows)

    summary_path = OUT / "external_humanization_summary.csv"
    detail_path = OUT / "external_humanization_details.csv"

    summary.to_csv(summary_path, index=False)
    detail.to_csv(detail_path, index=False)

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(summary.to_string(index=False))
    print("\nsaved:", summary_path)
    print("saved:", detail_path)


if __name__ == "__main__":
    main()
