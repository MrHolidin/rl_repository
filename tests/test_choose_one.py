"""Choose One — the card offers two effects and the seat takes one.

Eleven live cards print it and four more bend it, so the tests here are about
the *shape* rather than any one card: the choice parks itself like a Discover,
the option that was picked is the one that happens, and a seat holding a
"both effects combined" charge takes the pair and spends the charge.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.bg_catalog.patch_context import PatchContext
from src.bg_core.effects import (
    Ability,
    BuffSelf,
    ChooseOneEffect,
    GainGoldThisTurnEffect,
    Trigger,
)
from src.bg_core.minion import Minion
from src.bg_lobby.player import PendingChoiceKind, PlayerPhase, PlayerState
from src.bg_recruitment import choose_one
from src.bg_recruitment.shop_triggers import ShopTriggers

PATCH_74257 = "data/bgcore/19_6_0_74257"


@pytest.fixture(scope="module")
def patch():
    return PatchContext.load(Path(PATCH_74257))


@pytest.fixture
def triggers(patch):
    return ShopTriggers(np.random.default_rng(0), patch=patch)


def _minion(card_id: str, *abilities, attack: int = 1, health: int = 1) -> Minion:
    return Minion(
        card_id=card_id,
        base_attack=attack,
        base_health=health,
        tier=1,
        abilities=tuple(abilities),
    )


def _player(board=None) -> PlayerState:
    return PlayerState(
        health=40,
        gold=5,
        tavern_tier=1,
        board=list(board or []),
        shop=[None] * 6,
        hand=[None] * 10,
        phase=PlayerPhase.SHOP,
        shop_actions_used=0,
    )


#: "Choose One - gain 3 Gold; or grow by +2/+2" — two effects the dispatcher
#: already knows, so these tests exercise the choice rather than the effects.
_GOLD = GainGoldThisTurnEffect(amount=3)
_GROW = BuffSelf(attack=2, health=2)
_CHOICE = ChooseOneEffect(first=_GOLD, second=_GROW)


def _play(triggers, player, minion):
    player.board.append(minion)
    triggers.fire_on_place(minion, player, None)


def _resolve(triggers, player, index):
    choose_one.resolve_choose_one(
        player,
        index,
        apply_effect=lambda source, effect: triggers.apply_shop_effect(
            player, source if source is not None else player.board[0], effect, None
        ),
    )


def test_playing_the_card_parks_a_choice_instead_of_resolving(triggers):
    player = _player()
    _play(triggers, player, _minion("chooser", Ability(Trigger.ON_PLACE, _CHOICE)))

    pc = player.pending_choice
    assert pc is not None and pc.kind is PendingChoiceKind.CHOOSE_ONE
    assert len(pc.effects) == choose_one.CHOOSE_ONE_OPTIONS == 2
    assert player.gold == 5, "nothing resolved until the seat picks"


def test_the_picked_option_is_the_one_that_happens(triggers):
    player = _player()
    minion = _minion("chooser", Ability(Trigger.ON_PLACE, _CHOICE))
    _play(triggers, player, minion)
    _resolve(triggers, player, 0)

    assert player.gold == 8, "took the gold half"
    assert minion.bonus_attack == 0, "and not the growth half"
    assert player.pending_choice is None


def test_the_other_option_is_the_other_one(triggers):
    player = _player()
    minion = _minion("chooser", Ability(Trigger.ON_PLACE, _CHOICE))
    _play(triggers, player, minion)
    _resolve(triggers, player, 1)

    assert player.gold == 5
    assert (minion.bonus_attack, minion.bonus_health) == (2, 2)


def test_a_combined_charge_takes_both_halves_and_is_spent(triggers):
    """Thorned Trailblazer: one Choose One card each turn has both effects."""
    player = _player()
    choose_one.grant_combined_choose_one(player)
    minion = _minion("chooser", Ability(Trigger.ON_PLACE, _CHOICE))
    _play(triggers, player, minion)
    _resolve(triggers, player, 0)

    assert player.gold == 8
    assert (minion.bonus_attack, minion.bonus_health) == (2, 2)
    assert player.choose_one_combined_charges == 0, "the charge was spent"


def test_without_a_charge_the_next_card_is_back_to_one_option(triggers):
    player = _player()
    choose_one.grant_combined_choose_one(player)
    first = _minion("first", Ability(Trigger.ON_PLACE, _CHOICE))
    _play(triggers, player, first)
    _resolve(triggers, player, 0)

    second = _minion("second", Ability(Trigger.ON_PLACE, _CHOICE))
    _play(triggers, player, second)
    _resolve(triggers, player, 1)

    assert player.gold == 8, "only the first card's gold half"
    assert (second.bonus_attack, second.bonus_health) == (2, 2)
    assert second is not first


def test_an_out_of_range_option_is_rejected(triggers):
    player = _player()
    _play(triggers, player, _minion("chooser", Ability(Trigger.ON_PLACE, _CHOICE)))
    with pytest.raises(ValueError, match="out of range"):
        _resolve(triggers, player, 2)


def test_resolving_without_a_pending_choice_is_rejected(triggers):
    player = _player()
    with pytest.raises(ValueError, match="no Choose One is pending"):
        _resolve(triggers, player, 0)


def test_the_played_listener_sees_the_card(triggers):
    """Turbo Hogrider's shape: a board minion that reacts to a Choose One card."""
    seen = []
    listener = _minion(
        "hogrider", Ability(Trigger.ON_CHOOSE_ONE_PLAYED, BuffSelf(attack=1, health=1))
    )
    player = _player([listener])
    _play(triggers, player, _minion("chooser", Ability(Trigger.ON_PLACE, _CHOICE)))
    choose_one.resolve_choose_one(
        player,
        0,
        apply_effect=lambda source, effect: triggers.apply_shop_effect(
            player, source if source is not None else player.board[0], effect, None
        ),
        fire_played_listeners=lambda p: choose_one.fire_choose_one_played(
            p,
            lambda src, eff: seen.append((src.card_id, eff)),
        ),
    )
    assert [card_id for card_id, _ in seen] == ["hogrider"]
