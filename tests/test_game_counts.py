"""Game-long tallies: the seat counts, every copy recomputes.

The cards read "+X for each ... this game (wherever this is)", so the number
lives on the seat and each card derives its stats from it. That is what keeps
"for each *other*" from needing to know which copies were standing when: there
is one tally and one subtraction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.bg_catalog.cards import make_minion
from src.bg_catalog.patch_context import PatchContext
from src.bg_core.minion import Minion
from src.bg_lobby.player import PlayerPhase, PlayerState, copy_player_state
from src.bg_recruitment.combat_seat import PlayerCombatSeat
from src.bg_recruitment.game_counts import refresh_count_bonuses
from src.bg_recruitment.shop_triggers import ShopTriggers
from tests.minibg_helpers import simulate_battle

PATCH_DIR = Path("data/bgcore/36_2_0_248348")
AUTOMATON = "BG_TTN_401"
KNIGHT = "BG25_008"


@pytest.fixture(scope="module")
def patch() -> PatchContext:
    return PatchContext.load(PATCH_DIR)


@pytest.fixture()
def triggers(patch):
    return ShopTriggers(np.random.default_rng(0), patch=patch)


def _player(patch, board=(), **kw) -> PlayerState:
    base = dict(
        health=30,
        gold=10,
        tavern_tier=2,
        board=list(board),
        shop=[None] * 7,
        hand=[None] * 10,
        phase=PlayerPhase.SHOP,
        shop_actions_used=0,
        ruleset=patch.meta.ruleset,
    )
    base.update(kw)
    return PlayerState(**base)


def _place(triggers, player, minion):
    """Play a minion the way the shop does — which is also a summon."""
    player.board.append(minion)
    triggers.fire_on_place(minion, player, None)
    triggers.fire_shop_friendly_summoned(player, minion)


def _extra(minion) -> tuple:
    return (
        minion.raw_attack - minion.base_attack,
        minion.max_health - minion.base_health,
    )


# --------------------------------------------------------------------------- #
# "for each other ... you've summoned"
# --------------------------------------------------------------------------- #


def test_one_automaton_counts_no_others(patch, triggers):
    player = _player(patch)
    first = make_minion(AUTOMATON, patch=patch)
    _place(triggers, player, first)
    assert _extra(first) == (0, 0)


@pytest.mark.parametrize("copies,each", [(2, (3, 2)), (3, (6, 4)), (4, (9, 6))])
def test_every_copy_counts_the_others_and_not_itself(patch, triggers, copies, each):
    """The count grows linearly, not with its square: one bump per arrival."""
    player = _player(patch)
    placed = []
    for _ in range(copies):
        m = make_minion(AUTOMATON, patch=patch)
        placed.append(m)
        _place(triggers, player, m)
    assert {_extra(m) for m in placed} == {each}


def test_a_copy_in_hand_counts_every_summoned_one(patch, triggers):
    """It has not been summoned, so none of them is *itself*."""
    player = _player(patch)
    for _ in range(2):
        _place(triggers, player, make_minion(AUTOMATON, patch=patch))
    in_hand = make_minion(AUTOMATON, patch=patch)
    player.hand[0] = in_hand
    refresh_count_bonuses(player)
    assert _extra(in_hand) == (6, 4)


def test_the_same_copy_drops_to_the_others_count_once_played(patch, triggers):
    player = _player(patch)
    for _ in range(2):
        _place(triggers, player, make_minion(AUTOMATON, patch=patch))
    latecomer = make_minion(AUTOMATON, patch=patch)
    player.hand[0] = latecomer
    refresh_count_bonuses(player)
    player.hand[0] = None
    _place(triggers, player, latecomer)
    assert _extra(latecomer) == (6, 4)  # three summoned, two others


def test_another_card_does_not_feed_the_tally(patch, triggers):
    player = _player(patch)
    automaton = make_minion(AUTOMATON, patch=patch)
    _place(triggers, player, automaton)
    _place(triggers, player, Minion(card_id="other", base_attack=1, base_health=1, tier=1))
    assert _extra(automaton) == (0, 0)


def test_a_summon_mid_combat_counts_too(patch, triggers):
    """"Summoned" is every arrival, not only the ones played from hand."""
    from src.bg_core.effects import Ability, SummonEffect, Trigger

    player = _player(patch)
    standing = make_minion(AUTOMATON, patch=patch)
    _place(triggers, player, standing)
    summoner = Minion(
        card_id="summoner",
        base_attack=1,
        base_health=1,
        tier=1,
        abilities=(Ability(Trigger.ON_DEATH, SummonEffect(token_id=AUTOMATON, count=1)),),
    )
    seat = PlayerCombatSeat(player)
    simulate_battle(
        [summoner],
        [Minion(card_id="wall", base_attack=20, base_health=40, tier=1)],
        p0_has_initiative=True,
        rng=np.random.default_rng(0),
        patch=patch,
        seats=(seat, PlayerCombatSeat(_player(patch))),
    )
    refresh_count_bonuses(player)
    assert _extra(standing) == (3, 2)


def test_recomputing_twice_changes_nothing(patch, triggers):
    player = _player(patch)
    for _ in range(3):
        _place(triggers, player, make_minion(AUTOMATON, patch=patch))
    before = [_extra(m) for m in player.board]
    for _ in range(5):
        refresh_count_bonuses(player)
    assert [_extra(m) for m in player.board] == before


def test_a_copied_seat_keeps_its_own_tally(patch, triggers):
    player = _player(patch)
    _place(triggers, player, make_minion(AUTOMATON, patch=patch))
    twin = copy_player_state(player)
    _place(triggers, twin, make_minion(AUTOMATON, patch=patch))
    assert player.game_counts["summoned:" + AUTOMATON] == 1
    assert twin.game_counts["summoned:" + AUTOMATON] == 2


# --------------------------------------------------------------------------- #
# "for each ... that died"
# --------------------------------------------------------------------------- #


def test_a_knight_that_dies_pays_the_copies_that_did_not(patch):
    dying = make_minion(KNIGHT, patch=patch)
    in_hand = make_minion(KNIGHT, patch=patch)
    player = _player(patch, [dying])
    player.hand[0] = in_hand
    simulate_battle(
        [dying],
        [Minion(card_id="wall", base_attack=20, base_health=30, tier=1)],
        p0_has_initiative=True,
        rng=np.random.default_rng(0),
        patch=patch,
        seats=(PlayerCombatSeat(player), PlayerCombatSeat(_player(patch))),
    )
    refresh_count_bonuses(player)
    assert _extra(in_hand) == (4, 2)


def test_a_death_of_another_card_is_not_counted(patch):
    in_hand = make_minion(KNIGHT, patch=patch)
    player = _player(patch)
    player.hand[0] = in_hand
    simulate_battle(
        [Minion(card_id="other", base_attack=1, base_health=1, tier=1)],
        [Minion(card_id="wall", base_attack=20, base_health=30, tier=1)],
        p0_has_initiative=True,
        rng=np.random.default_rng(0),
        patch=patch,
        seats=(PlayerCombatSeat(player), PlayerCombatSeat(_player(patch))),
    )
    refresh_count_bonuses(player)
    assert _extra(in_hand) == (0, 0)
