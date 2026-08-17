"""Lockboxes — the Pirate card that pays out on a timer.

The rule the cards are built around is that a seat holds *one* box: every card
that makes one says "if you already have one, it opens sooner", so a second box
is spent as acceleration rather than kept. Getting that wrong hands a Pirate
seat five boxes and five Golden minions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.bg_catalog.patch_context import PatchContext
from src.bg_core.minion import Minion, Race
from src.bg_lobby.player import PlayerPhase, PlayerState
from src.bg_recruitment import lockbox

PATCH_74257 = "data/bgcore/19_6_0_74257"


@pytest.fixture(scope="module")
def patch():
    return PatchContext.load(Path(PATCH_74257))


def _player(hand_slots: int = 10) -> PlayerState:
    return PlayerState(
        health=40,
        gold=10,
        tavern_tier=1,
        board=[],
        shop=[None] * 6,
        hand=[None] * hand_slots,
        phase=PlayerPhase.SHOP,
        shop_actions_used=0,
    )


def _tick(player, patch, times: int = 1):
    out = None
    for _ in range(times):
        out = lockbox.tick_lockboxes(player, rng=np.random.default_rng(0), patch=patch)
    return out


def _boxes(player):
    return [c for c in player.hand if lockbox.is_lockbox(c)]


def test_a_lockbox_lands_in_hand_with_its_timer_set():
    player = _player()
    assert lockbox.give_lockbox(player) is True
    (box,) = _boxes(player)
    assert box.turns_until_open == lockbox.LOCKBOX_TURNS == 5


def test_the_timer_counts_down_one_turn_at_a_time(patch):
    player = _player()
    lockbox.give_lockbox(player)
    assert _tick(player, patch, 2) is None
    (box,) = _boxes(player)
    assert box.turns_until_open == 3


def test_it_opens_into_a_golden_minion_with_a_tribe(patch):
    player = _player()
    lockbox.give_lockbox(player)
    payout = _tick(player, patch, lockbox.LOCKBOX_TURNS)

    assert isinstance(payout, Minion)
    assert payout.is_golden, "the card promises a Golden minion"
    assert payout.race not in (None, Race.ALL), "and one *with a type*"
    assert not _boxes(player), "the box became the minion"
    assert payout in [c for c in player.hand if c is not None]


def test_a_second_box_accelerates_the_first_instead_of_stacking():
    player = _player()
    lockbox.give_lockbox(player)
    assert lockbox.give_lockbox(player) is False, "no second box was created"

    boxes = _boxes(player)
    assert len(boxes) == 1
    assert boxes[0].turns_until_open == lockbox.LOCKBOX_TURNS - 1


def test_a_trinket_sized_accelerator_moves_it_two_turns():
    player = _player()
    lockbox.give_lockbox(player)
    lockbox.give_lockbox(player, sooner=2)
    (box,) = _boxes(player)
    assert box.turns_until_open == lockbox.LOCKBOX_TURNS - 2


def test_acceleration_cannot_take_the_timer_below_zero(patch):
    player = _player()
    lockbox.give_lockbox(player)
    lockbox.give_lockbox(player, sooner=99)
    (box,) = _boxes(player)
    assert box.turns_until_open == 0

    payout = _tick(player, patch)
    assert isinstance(payout, Minion), "a box at zero opens on the next tick"


def test_a_full_hand_has_no_room_for_a_box():
    player = _player(hand_slots=0)
    assert lockbox.give_lockbox(player) is False
    assert not _boxes(player)


def test_ticking_without_a_box_does_nothing(patch):
    player = _player()
    assert _tick(player, patch) is None
