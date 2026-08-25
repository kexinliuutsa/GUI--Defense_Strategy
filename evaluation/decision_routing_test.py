"""
Deterministic routing test for the frozen decision layer.

Covers:
1. insufficient context
2. base-head detection
3. base miss -> HREF detection
4. base miss -> HREF miss
"""

from src.defense import (
    CrossScaleDefense,
    DefenseScores,
)


def scores(
    session,
    strict,
    fake,
    href,
):
    return DefenseScores(
        session_score=session,
        strict_score=strict,
        fake_artifact=fake,
        human_reference_score=href,
    )


defense = CrossScaleDefense.from_config()


# ------------------------------------------------------------
# 1. Insufficient context
# ------------------------------------------------------------

r = defense.predict_from_scores(
    scores(
        session=1.0,
        strict=1.0,
        fake=True,
        href=10.0,
    ),
    n_actions=3,
)

assert r.detected is None
assert r.status == "insufficient_context"
assert r.href_consulted is False


# ------------------------------------------------------------
# 2. Base head detects -> HREF must NOT be consulted
# ------------------------------------------------------------

r = defense.predict_from_scores(
    scores(
        session=1.0,
        strict=0.0,
        fake=False,
        href=0.0,
    ),
    n_actions=4,
)

assert r.detected is True
assert r.base_detected is True
assert r.href_consulted is False
assert r.href_detected is False


# ------------------------------------------------------------
# 3. Base misses -> HREF detects
# ------------------------------------------------------------

r = defense.predict_from_scores(
    scores(
        session=0.0,
        strict=0.0,
        fake=False,
        href=10.0,
    ),
    n_actions=4,
)

assert r.base_detected is False
assert r.href_consulted is True
assert r.href_detected is True
assert r.detected is True


# ------------------------------------------------------------
# 4. Base misses -> HREF also misses
# ------------------------------------------------------------

r = defense.predict_from_scores(
    scores(
        session=0.0,
        strict=0.0,
        fake=False,
        href=0.0,
    ),
    n_actions=4,
)

assert r.base_detected is False
assert r.href_consulted is True
assert r.href_detected is False
assert r.detected is False


print("=" * 72)
print("DECISION ROUTING TEST")
print("=" * 72)
print("insufficient context       : PASS")
print("base detection             : PASS")
print("base miss -> HREF detection: PASS")
print("base miss -> HREF miss     : PASS")
print()
print("PASS: frozen conditional-routing logic behaves as specified.")
