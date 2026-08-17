"""Activate — pay gold, fire a minion's ability, once per turn.

Seventeen minions in the live pool carry it at 1 or 2 gold, and Blizzard's own
test cards state the rule outright: "Activate: Give a minion +1/+1 (one per
turn)" / "(cost 1 gold)".

It is the only trigger that is a move rather than an event, so these tests call
it the way a player would and check what it costs, what it spends, and when it
comes back.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.bg_catalog.patch_context import PatchContext
from src.bg_core.effects import Ability, BuffMatching, BuffSelf, BuffTarget, Trigger
from src.bg_core.minion import Minion
from src.bg_lobby.player import PlayerPhase, PlayerState
from src.bg_recruitment import activate as bg_activate
from src.bg_recruitment.activate import ActivateNotAllowed
from src.bg_recruitment.shop_triggers import ShopTriggers

PATCH_DIR = "data/bgcore/19_6_0_74257"


@pytest.fixture(scope="module")
def patch():
    return PatchContext.load(Path(PATCH_DIR))


def _activator(card_id: str = "activator", cost: int = 1, effect=None) -> Minion:
    return Minion(
        card_id=card_id,
        base_attack=2,
        base_health=2,
        tier=1,
        abilities=(
            Ability(
                Trigger.ON_ACTIVATE,
                effect or BuffSelf(attack=3, health=3),
                activate_cost=cost,
            ),
        ),
    )


def _plain(card_id: str = "plain") -> Minion:
    return Minion(card_id=card_id, base_attack=1, base_health=1, tier=1)


def _player(board, gold: int = 10, **kw) -> PlayerState:
    base = dict(
        health=40,
        gold=gold,
        tavern_tier=1,
        board=list(board),
        shop=[None] * 6,
        hand=[None] * 10,
        phase=PlayerPhase.SHOP,
        shop_actions_used=0,
    )
    base.update(kw)
    return PlayerState(**base)


def _fire(player, patch, index: int = 0):
    bg_activate.activate_minion(
        player, index, rng=np.random.default_rng(0), patch=patch
    )


# --------------------------------------------------------------------------- #
# Cost and effect
# --------------------------------------------------------------------------- #


def test_activating_costs_gold_and_applies_the_effect(patch):
    player = _player([_activator(cost=2)], gold=5)
    _fire(player, patch)
    assert player.gold == 3
    assert (player.board[0].raw_attack, player.board[0].max_health) == (5, 5)


def test_a_minion_without_activate_has_no_cost():
    assert bg_activate.activate_cost(_plain()) is None
    assert not bg_activate.can_activate(_player([_plain()]), _plain())


def test_activating_a_plain_minion_is_refused(patch):
    player = _player([_plain()])
    with pytest.raises(ActivateNotAllowed, match="no Activate ability"):
        _fire(player, patch)


def test_gold_that_is_short_refuses_and_spends_nothing(patch):
    player = _player([_activator(cost=2)], gold=1)
    assert not bg_activate.can_activate(player, player.board[0])
    with pytest.raises(ActivateNotAllowed, match="costs 2 to activate"):
        _fire(player, patch)
    assert player.gold == 1
    assert player.board[0].raw_attack == 2, "the effect did not sneak through"


# --------------------------------------------------------------------------- #
# Once per turn
# --------------------------------------------------------------------------- #


def test_the_second_activation_this_turn_is_refused(patch):
    player = _player([_activator(cost=1)], gold=10)
    _fire(player, patch)
    assert not bg_activate.can_activate(player, player.board[0])
    with pytest.raises(ActivateNotAllowed, match="already used its Activate"):
        _fire(player, patch)
    assert player.gold == 9, "the refused attempt charged nothing"


def test_the_turn_gives_it_back(patch):
    player = _player([_activator(cost=1)], gold=10)
    _fire(player, patch)
    ShopTriggers(np.random.default_rng(0), patch=patch).fire_on_turn_start(player)
    assert bg_activate.can_activate(player, player.board[0])
    _fire(player, patch)
    assert player.board[0].raw_attack == 8, "+3/+3 twice, one per turn"


def test_one_minion_being_spent_does_not_spend_another(patch):
    player = _player([_activator("first"), _activator("second")], gold=10)
    _fire(player, patch, 0)
    assert player.board[0].activate_used_this_turn
    assert not player.board[1].activate_used_this_turn
    assert bg_activate.can_activate(player, player.board[1])


# --------------------------------------------------------------------------- #
# Where it is allowed
# --------------------------------------------------------------------------- #


def test_activate_is_a_recruit_phase_move(patch):
    player = _player([_activator()], phase=PlayerPhase.DONE)
    assert not bg_activate.can_activate(player, player.board[0])
    with pytest.raises(ActivateNotAllowed, match="recruit-phase move"):
        _fire(player, patch)


def test_an_empty_slot_is_refused(patch):
    player = _player([_activator()])
    with pytest.raises(ActivateNotAllowed, match="no minion at board index"):
        _fire(player, patch, 3)


def test_the_effect_can_reach_the_rest_of_the_warband(patch):
    """"Activate (1): Give another minion +3/+3" — the shop dispatcher resolves it."""
    activator = _activator(
        effect=BuffMatching(target=BuffTarget.ALL_FRIENDLY, attack=1, health=1)
    )
    player = _player([activator, _plain()], gold=5)
    _fire(player, patch)
    assert player.board[1].raw_attack == 2


def test_an_effect_the_dispatcher_cannot_resolve_is_loud(patch):
    """Same contract as Rally: never a silent no-op."""
    from src.bg_core.effects import CleaveOnAttack
    from src.bg_recruitment.shop_triggers import UnhandledShopEffect

    player = _player([_activator(effect=CleaveOnAttack())], gold=5)
    with pytest.raises(UnhandledShopEffect):
        _fire(player, patch)
