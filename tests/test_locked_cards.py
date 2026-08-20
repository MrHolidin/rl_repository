"""A locked card is held shut: nothing can play it and nothing can see it.

"Lock it in your hand for 1 turn" is a general mechanic rather than one card's
rule, and "does not interact with anything" is only true if *every* place that
walks the hand asks the same question. This walks them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import src.envs.minibg.actions as A
from src.bg_catalog.patch_context import PatchContext
from src.bg_core.effects import BuffHandMinionsEffect
from src.bg_core.minion import Minion, is_locked
from src.bg_lobby.player import PlayerPhase, PlayerState
from src.bg_player_turn.engine import PlayerTurnEngine
from src.bg_recruitment.discover import resolve_discover_pick
from src.bg_recruitment.shop_triggers import ShopTriggers
from src.bg_recruitment.tavern_spells import cast_tavern_spell
from src.bg_recruitment.triples import resolve_triples_loop

PATCH_DIR = Path("data/bgcore/36_2_0_248348")


@pytest.fixture(scope="module")
def patch() -> PatchContext:
    return PatchContext.load(PATCH_DIR)


def _player(patch, **kw) -> PlayerState:
    base = dict(
        health=30, gold=10, tavern_tier=4, board=[], shop=[None] * 7,
        hand=[None] * 10, phase=PlayerPhase.SHOP, shop_actions_used=0,
        ruleset=patch.meta.ruleset,
    )
    base.update(kw)
    return PlayerState(**base)


def _locked(patch, card_id="BG25_008", turns=1):
    card = patch.make_minion(card_id)
    card.locked_turns = turns
    return card


def test_search_through_time_hands_over_a_locked_card(patch):
    player = _player(patch)
    cast_tavern_spell(
        player,
        patch.tavern_spells["BG34_330"],
        rng=np.random.default_rng(0),
        patch=patch,
    )
    resolve_discover_pick(
        player, 0, None, rng=np.random.default_rng(0),
        on_after_placed=lambda _p, _m: None, patch=patch,
    )
    assert is_locked(player.hand[0])
    assert player.hand[0].locked_turns == 1


def test_a_locked_card_cannot_be_played(patch):
    player = _player(patch)
    player.hand[0] = _locked(patch)
    assert int(A.Action.PLAY_HAND_0) not in PlayerTurnEngine().legal_actions(
        player, patch.meta.ruleset
    )


def test_a_locked_card_cannot_be_magnetized(patch):
    player = _player(patch, board=[patch.make_minion("BG_TTN_401")])  # a Mech
    magnetic = _locked(patch, "BG_DEEP_015")  # Magnetic
    player.hand[0] = magnetic
    legal = PlayerTurnEngine().legal_actions(player, patch.meta.ruleset)
    assert not any(
        act >= int(A.Action.MAGNET_HAND_0_BOARD_0)
        and act < int(A.Action.MAGNET_HAND_0_BOARD_0) + A.HAND_SIZE * A.BOARD_SIZE
        for act in legal
    )


def test_a_locked_card_does_not_complete_a_triple(patch):
    player = _player(patch)
    player.hand[0] = _locked(patch, "BG25_008")
    player.hand[1] = patch.make_minion("BG25_008")
    player.hand[2] = patch.make_minion("BG25_008")
    resolve_triples_loop(player, patch=patch)
    assert not any(c is not None and c.is_golden for c in player.hand)


def test_a_locked_card_is_invisible_to_a_hand_buff(patch):
    player = _player(patch)
    held = _locked(patch, "BG25_008")
    twin = patch.make_minion("BG25_008")
    player.hand[0] = held
    player.hand[1] = twin
    ShopTriggers(np.random.default_rng(0), patch=patch).apply_shop_effect(
        player, None, BuffHandMinionsEffect(attack=5, health=5), placed=None
    )
    assert (held.raw_attack, held.max_health) == (
        held.base_attack,
        held.base_health,
    )
    assert twin.raw_attack > twin.base_attack


def test_a_locked_card_cannot_be_summoned_out_of_hand_in_combat(patch):
    """Diremuck Forager reaches into the hand; a held card is not there."""
    from src.bg_recruitment.combat_seat import PlayerCombatSeat

    player = _player(patch)
    player.hand[0] = _locked(patch, "BG25_008")
    seat = PlayerCombatSeat(player, patch=patch)
    assert seat.hand_minions() == ()
    player.hand[0].locked_turns = 0
    assert len(seat.hand_minions()) == 1


def test_a_locked_card_still_takes_up_its_slot(patch):
    """Holding one is the cost of the card, so the slot is not free."""
    from src.bg_recruitment.hand_slots import first_free_hand_slot

    player = _player(patch)
    player.hand[0] = _locked(patch)
    assert first_free_hand_slot(player) == 1


def test_the_lock_counts_down_in_the_seats_own_turns(patch):
    player = _player(patch)
    card = _locked(patch, turns=2)
    player.hand[0] = card
    triggers = ShopTriggers(np.random.default_rng(0), patch=patch)
    triggers.fire_on_turn_start(player)
    assert card.locked_turns == 1 and is_locked(card)
    triggers.fire_on_turn_start(player)
    assert card.locked_turns == 0 and not is_locked(card)
    assert int(A.Action.PLAY_HAND_0) in PlayerTurnEngine().legal_actions(
        player, patch.meta.ruleset
    )
