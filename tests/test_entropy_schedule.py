"""Unit tests for EntropyCoefScheduleCallback."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.training.callbacks.entropy_schedule import EntropyCoefScheduleCallback


def _fake_trainer(coef):
    agent = SimpleNamespace(entropy_coef=coef)
    return SimpleNamespace(agent=agent)


def test_coef_at_endpoints():
    cb = EntropyCoefScheduleCallback(
        start_coef=0.05, end_coef=0.01, schedule_steps=4_000_000
    )
    assert cb._coef_for_step(0) == pytest.approx(0.05)
    assert cb._coef_for_step(4_000_000) == pytest.approx(0.01)
    assert cb._coef_for_step(5_000_000) == pytest.approx(0.01)
    assert cb._coef_for_step(2_000_000) == pytest.approx(0.03)


def test_on_train_begin_sets_start():
    cb = EntropyCoefScheduleCallback(start_coef=0.05, end_coef=0.01, schedule_steps=10000)
    trainer = _fake_trainer(0.02)
    cb.on_train_begin(trainer)
    assert trainer.agent.entropy_coef == pytest.approx(0.05)


def test_on_step_end_fires_on_interval_crossing():
    """Step counts in distributed mode jump by ~rollout_steps; ensure modulo logic
    doesn't lock us out."""
    cb = EntropyCoefScheduleCallback(
        start_coef=0.05, end_coef=0.01, schedule_steps=4_000_000, interval_steps=50_000
    )
    trainer = _fake_trainer(0.05)

    metrics: dict = {}
    # Step jumps by 8096 each round (not divisible by 50_000).
    fired = 0
    for round_idx in range(20):
        step = (round_idx + 1) * 8096
        cb.on_step_end(trainer, step, transition=None, metrics=metrics)
        if "entropy_coef" in metrics:
            fired += 1
            metrics = {}  # reset for next observation
    # 20 rounds × 8096 ≈ 162k steps. With 50k interval, should fire ~3 times.
    assert fired >= 2, f"expected schedule to fire several times in 20 rounds, got {fired}"


def test_coef_monotone_decreasing():
    cb = EntropyCoefScheduleCallback(start_coef=0.05, end_coef=0.01, schedule_steps=4_000_000)
    coefs = [cb._coef_for_step(s) for s in range(0, 4_500_000, 100_000)]
    for a, b in zip(coefs, coefs[1:]):
        assert b <= a + 1e-9


def test_schedule_steps_validation():
    with pytest.raises(ValueError):
        EntropyCoefScheduleCallback(start_coef=0.05, end_coef=0.01, schedule_steps=0)
    with pytest.raises(ValueError):
        EntropyCoefScheduleCallback(
            start_coef=0.05, end_coef=0.01, schedule_steps=1000, interval_steps=0
        )
