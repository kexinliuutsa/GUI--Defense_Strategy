from __future__ import annotations
"""
Frozen-defense oracle adapter for Attack V2.

This file connects Attack V2 to the frozen
V1+V2+V3 defense.

The attacker-facing oracle returns only:

    True  -> detected as agent
    False -> escaped / human-like

It does not expose:
    - V1/V2/V3 scores
    - margins
    - thresholds
    - task-cluster diagnostics
    - feature values
"""

import os
import importlib
from pathlib import Path


DEFAULT_AHB_ROOT = (
    "/storage/cjh/self-evolving-safety/"
    "Passing-the-Turing-Test-on-Screen-Agent-Humanization-Benchmark-main"
)


def configure_paths(
    ahb_root: str | None = None,
):
    """
    Configure AHB_ROOT before importing the frozen defense.

    frozen_v1v2v3_defense.py reads AHB_ROOT during import,
    so this must happen first.
    """

    if ahb_root is None:
        ahb_root = os.environ.get(
            "AHB_ROOT",
            DEFAULT_AHB_ROOT,
        )

    os.environ["AHB_ROOT"] = str(
        Path(ahb_root).expanduser().resolve()
    )

    return os.environ["AHB_ROOT"]





def load_defense(
    ahb_root: str | None = None,
):
    """
    Load attacker-facing defense.

    Default:
        evaluation.frozen_v1v2v3_defense_hardened

    Override:
        export GUI_DEFENSE_MODULE=evaluation.defense_loop_v1_hardcase_memory
    """
    configure_paths(ahb_root)

    module_name = os.environ.get(
        "GUI_DEFENSE_MODULE",
        "evaluation.frozen_v1v2v3_defense_hardened",
    )

    mod = importlib.import_module(module_name)
    cls = getattr(mod, "FrozenV1V2V3Defense")
    defense = cls()

    try:
        m = defense.manifest()
        print("[oracle] loaded defense module:", module_name)
        print("[oracle] defense version:", m.get("defense_version"))
        print("[oracle] loop defense version:", m.get("loop_defense_version"))
        print("[oracle] loop memory radius:", m.get("loop_memory_radius"))
        print("[oracle] n hardcases:", m.get("n_hardcases"))
    except Exception as e:
        print("[oracle] loaded defense module:", module_name)
        print("[oracle] manifest unavailable:", repr(e))

    return defense


def rec_get(
    record,
    key,
    default=None,
):
    if isinstance(record, dict):
        return record.get(
            key,
            default,
        )

    return getattr(
        record,
        key,
        default,
    )


def participant(
    record,
):
    return str(
        rec_get(
            record,
            "participant",
            "UNKNOWN",
        )
    )


def session_id(
    record,
):
    return str(
        rec_get(
            record,
            "session_id",
            "UNKNOWN",
        )
    )


def extract_session(
    record,
):
    """
    Return the gesture list consumed by Attack V2.
    """

    gestures = rec_get(
        record,
        "gestures",
        None,
    )

    if gestures is None:
        raise ValueError(
            "Record does not contain a 'gestures' field."
        )

    return gestures


def make_record_oracle(
    original_record,
    *,
    defense=None,
    ahb_root: str | None = None,
    task_cluster=None,
    v1_condition="Long Tap Author",
):
    """
    Build a label-only oracle for one original record.

    The returned callable expects candidate_gestures.

    It returns only:
        True  -> detected
        False -> escaped
    """

    if defense is None:
        defense = load_defense(
            ahb_root=ahb_root,
        )

    return defense.make_oracle(
        original_record,
        task_cluster=task_cluster,
        v1_condition=v1_condition,
    )


def load_long_tap_records(
    defense=None,
    ahb_root: str | None = None,
):
    """
    Load Long-Tap records from the frozen V1 prefix.

    This is useful for smoke tests and population runs.
    """

    if defense is None:
        defense = load_defense(
            ahb_root=ahb_root,
        )

    ns = defense.ns
    inner = defense.inner

    records = list(
        ns.get(
            "long_records",
            inner.get(
                "long_records",
                [],
            ),
        )
    )

    return records


def find_detected_record(
    records,
    defense,
    *,
    max_scan=None,
):
    """
    Find one record that is detected by the frozen defense.

    This avoids running attacks on sessions that already escape.
    """

    checked = 0

    for record in records:

        if max_scan is not None and checked >= max_scan:
            break

        checked += 1

        try:
            detected = bool(
                defense.detect_record(record)
            )
        except Exception:
            continue

        if detected:
            return record

    raise RuntimeError(
        "Could not find a detected record in the provided records."
    )


