"""Two run-shaping knobs: the cumulative pool dump and the two-stage entropy target.

``self_play_frozen.json`` only ever shows the survivors of ``max_frozen_agents``,
so a run reads as "the pool was always these N". ``full_pool.json`` keeps the
evicted rows as they stood when they were dropped.

``entropy_target_until_step`` used to mean "stop pressing after step X"; with
``entropy_target_after`` it means "press to a second target after step X".
"""

from __future__ import annotations

import pytest

from src.training.selfplay.game_record import GameRecord, ParticipantOutcome, SLOT_CURRENT
from src.training.selfplay.league_state import LeagueController


def _league() -> LeagueController:
    lg = LeagueController(rating_kind="trueskill")
    lg.register_meta_slot(SLOT_CURRENT)
    return lg


def _play(lg, sid, n, *, place=8):
    for _ in range(n):
        lg.submit(
            GameRecord(
                (ParticipantOutcome(SLOT_CURRENT, 1), ParticipantOutcome(sid, place))
            )
        )


# --------------------------------------------------------------------------- #
# full_pool
# --------------------------------------------------------------------------- #


def test_evicted_rows_are_kept_out_of_the_live_pool_but_in_the_full_one():
    lg = _league()
    ids = [lg.add_frozen_bytes(b"w", episode=e) for e in (0, 1, 2)]
    # Give every slot enough games to be an eviction candidate, and make the
    # first one clearly the worst so the choice is not a coin flip.
    for sid in ids:
        _play(lg, sid, 120, place=8 if sid == ids[0] else 2)
    lg.evict_worst(2)

    live = [r for r in lg.get_pool_stats_for_status() if r.get("kind") == "frozen"]
    full = lg.get_full_pool_data()
    assert len(live) == 2
    assert full["counts"]["live_frozen"] == 2
    assert full["counts"]["evicted"] == 1
    assert full["counts"]["frozen_ever"] == 3
    # The evicted row carries its rating as of the moment it was dropped.
    ev = full["evicted"][0]
    assert ev["evicted"] is True
    assert ev["kind"] == "frozen"
    assert "mu" in ev and "games" in ev
    assert ev["slot_id"] not in {r["slot_id"] for r in live}


def test_full_pool_is_empty_of_evictions_when_nothing_was_dropped():
    lg = _league()
    lg.add_frozen_bytes(b"w", episode=0)
    full = lg.get_full_pool_data()
    assert full["evicted"] == []
    assert full["counts"]["frozen_ever"] == full["counts"]["live_frozen"] == 1


def test_evictions_accumulate_in_order():
    lg = _league()
    for e in range(4):
        lg.add_frozen_bytes(b"w", episode=e)
        lg.evict_worst(1)
    full = lg.get_full_pool_data()
    assert full["counts"]["evicted"] == 3
    assert [r["evicted_seq"] for r in full["evicted"]] == [0, 1, 2]


# --------------------------------------------------------------------------- #
# entropy target schedule
# --------------------------------------------------------------------------- #


class _Ctl:
    """Just the controller state ``_adapt_entropy_coef`` touches."""

    from src.agents.ppo_structured_minibg_agent import MiniBGPPOStructuredAgent

    _adapt_entropy_coef = MiniBGPPOStructuredAgent._adapt_entropy_coef

    def __init__(self, *, target, until=0, after=0.0, coef=0.02):
        self.entropy_target = target
        self.entropy_target_until_step = until
        self.entropy_target_after = after
        self.entropy_coef = coef
        self._entropy_coef_base = coef
        self.entropy_coef_max = 0.3
        self.entropy_adapt_rate = 0.05
        self._trained_steps = 0


def test_below_target_raises_the_coefficient():
    c = _Ctl(target=1.5)
    c._adapt_entropy_coef(1.0)
    assert c.entropy_coef > 0.02


def test_above_target_falls_back_to_the_floor_never_below():
    c = _Ctl(target=1.5, coef=0.05)
    for _ in range(200):
        c._adapt_entropy_coef(3.0)
    assert c.entropy_coef == pytest.approx(0.05)


def test_after_the_cutoff_the_second_target_takes_over():
    c = _Ctl(target=1.5, until=10, after=1.0)
    c._trained_steps = 11
    # Entropy 1.2 is under the first target but over the second: pressure must
    # come off (down to the floor), not keep pushing toward 1.5.
    before = c.entropy_coef
    c._adapt_entropy_coef(1.2)
    assert c.entropy_coef <= before
    # ...and under the second target it presses again.
    c.entropy_coef = 0.05
    c._adapt_entropy_coef(0.5)
    assert c.entropy_coef > 0.05


def test_without_a_second_target_the_cutoff_still_snaps_back():
    c = _Ctl(target=1.5, until=10, after=0.0, coef=0.02)
    c.entropy_coef = 0.2
    c._trained_steps = 11
    c._adapt_entropy_coef(0.1)
    assert c.entropy_coef == pytest.approx(0.02)


def test_before_the_cutoff_the_first_target_applies():
    c = _Ctl(target=1.5, until=1000, after=1.0)
    c._trained_steps = 10
    c.entropy_coef = 0.02
    c._adapt_entropy_coef(1.2)  # under 1.5 -> still pressing
    assert c.entropy_coef > 0.02
