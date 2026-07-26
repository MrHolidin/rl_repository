"""Adaptive entropy controller: one-sided pressure toward ``entropy_target``.

The property that matters is asymmetry -- the controller may only ever add
exploration pressure relative to the fixed-coefficient baseline. A regression
that let it push the coefficient below the configured floor would silently make
the collapse it exists to prevent *worse*, and would look like a normal run.
"""

from __future__ import annotations

import math

import pytest

from src.agents.ppo_structured_minibg_agent import MiniBGPPOStructuredAgent as Ag


def _ctl(coef=0.01, target=1.2, rate=0.05, cap=0.2, until=0):
    a = Ag.__new__(Ag)
    a.entropy_coef = coef
    a._entropy_coef_base = coef
    a.entropy_target = target
    a.entropy_adapt_rate = rate
    a.entropy_coef_max = cap
    a.entropy_target_until_step = until
    a._trained_steps = 0
    return a


def test_disabled_target_is_a_noop():
    a = _ctl(target=0.0)
    a._adapt_entropy_coef(0.1)
    assert a.entropy_coef == 0.01


def test_below_target_raises_pressure():
    a = _ctl()
    a._adapt_entropy_coef(0.61)  # the value fixed-coef runs settle at
    assert a.entropy_coef > 0.01
    assert a.entropy_coef == pytest.approx(0.01 * math.exp(0.05 * (1.2 - 0.61)))


def test_above_target_relaxes_but_never_below_the_floor():
    a = _ctl()
    for _ in range(500):
        a._adapt_entropy_coef(2.2)  # early training, naturally high entropy
    assert a.entropy_coef == pytest.approx(0.01)


def test_floor_holds_even_after_pressure_was_added():
    a = _ctl()
    for _ in range(50):
        a._adapt_entropy_coef(0.5)
    assert a.entropy_coef > 0.01
    for _ in range(500):
        a._adapt_entropy_coef(2.2)
    assert a.entropy_coef == pytest.approx(0.01)


def test_pressure_is_capped():
    a = _ctl(cap=0.05)
    for _ in range(10_000):
        a._adapt_entropy_coef(0.0)
    assert a.entropy_coef == pytest.approx(0.05)


def test_converges_to_a_fixed_point_at_target():
    a = _ctl()
    a._adapt_entropy_coef(1.2)
    assert a.entropy_coef == pytest.approx(0.01)


def test_repeated_below_target_is_monotone():
    a = _ctl()
    seen = []
    for _ in range(20):
        a._adapt_entropy_coef(0.6)
        seen.append(a.entropy_coef)
    assert all(b > x for x, b in zip(seen, seen[1:]))


# --- step cutoff -----------------------------------------------------------
# Past the cutoff the run must converge on a sharpened policy, so the controller
# has to both stop adapting AND undo the pressure it had already accumulated --
# merely freezing the coefficient would leave the tail of training noisy.


def test_cutoff_snaps_back_to_floor_and_stays_there():
    a = _ctl(until=1_000)
    a._trained_steps = 500
    a._adapt_entropy_coef(0.5)
    assert a.entropy_coef > 0.01  # controller was active before the cutoff

    a._trained_steps = 1_001
    a._adapt_entropy_coef(0.5)  # still far below target, but past the cutoff
    assert a.entropy_coef == pytest.approx(0.01)
    for _ in range(20):
        a._adapt_entropy_coef(0.5)
    assert a.entropy_coef == pytest.approx(0.01)


def test_cutoff_is_inclusive_of_the_boundary_step():
    a = _ctl(until=1_000)
    a._trained_steps = 1_000  # exactly at the cutoff -> still adapting
    a._adapt_entropy_coef(0.5)
    assert a.entropy_coef > 0.01


def test_zero_cutoff_means_no_cutoff():
    a = _ctl(until=0)
    a._trained_steps = 10**9
    a._adapt_entropy_coef(0.5)
    assert a.entropy_coef > 0.01


def test_cutoff_does_not_fire_when_target_is_disabled():
    a = _ctl(target=0.0, until=1_000)
    a._trained_steps = 10**6
    a._adapt_entropy_coef(0.5)
    assert a.entropy_coef == pytest.approx(0.01)
