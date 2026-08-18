"""Standing "this game" bonuses: the seat holds them, the cards catch up.

Thirty-seven cards across the pool say *this game*, and the wording that makes
them one mechanic is "wherever they are": the modifier belongs to the seat, not
to the board it was played on, so a card bought three turns later arrives
already carrying it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.bg_catalog.cards import make_minion
from src.bg_catalog.patch_context import PatchContext
from src.bg_core.effects import ScopeKind
from src.bg_core.minion import Minion, Race
from src.bg_lobby.player import PlayerPhase, PlayerState, copy_player_state
from src.bg_recruitment.shop_triggers import ShopTriggers
from src.bg_recruitment.standing_bonuses import (
    BonusScope,
    raise_standing_bonus,
    settle_standing_bonuses,
    standing_bonus_for,
)

PATCH_DIR = Path("data/bgcore/36_2_0_248348")


@pytest.fixture(scope="module")
def patch() -> PatchContext:
    return PatchContext.load(PATCH_DIR)


@pytest.fixture()
def triggers(patch):
    return ShopTriggers(np.random.default_rng(0), patch=patch)


def _player(patch, board=(), hand=(), shop=(), **kw) -> PlayerState:
    hand_slots = list(hand) + [None] * (10 - len(hand))
    shop_slots = list(shop) + [None] * (7 - len(shop))
    base = dict(
        health=30,
        gold=10,
        tavern_tier=2,
        board=list(board),
        shop=shop_slots,
        hand=hand_slots,
        phase=PlayerPhase.SHOP,
        shop_actions_used=0,
        ruleset=patch.meta.ruleset,
    )
    base.update(kw)
    return PlayerState(**base)


def _undead(card_id="u", atk=1, hp=1) -> Minion:
    return Minion(card_id=card_id, base_attack=atk, base_health=hp, tier=1, race=Race.UNDEAD)


UNDEAD_SCOPE = BonusScope(ScopeKind.TRIBE, Race.UNDEAD)


# --------------------------------------------------------------------------- #
# The substrate
# --------------------------------------------------------------------------- #


def test_a_raised_bonus_reaches_what_the_seat_already_owns(patch):
    on_board, in_hand, in_shop = _undead("a"), _undead("b"), _undead("c")
    player = _player(patch, [on_board], hand=[in_hand], shop=[in_shop])
    raise_standing_bonus(player, UNDEAD_SCOPE, 1, 0)
    assert on_board.raw_attack == in_hand.raw_attack == in_shop.raw_attack == 2


def test_it_reaches_a_card_that_arrives_afterwards(patch):
    """"Wherever they are" is the whole point: the seat carries it, not the board."""
    player = _player(patch)
    raise_standing_bonus(player, UNDEAD_SCOPE, 1, 0)
    latecomer = _undead("late")
    player.board.append(latecomer)
    settle_standing_bonuses(player)
    assert latecomer.raw_attack == 2


def test_a_minion_of_another_tribe_is_untouched(patch):
    beast = Minion(card_id="b", base_attack=1, base_health=1, tier=1, race=Race.BEAST)
    player = _player(patch, [beast])
    raise_standing_bonus(player, UNDEAD_SCOPE, 1, 0)
    assert beast.raw_attack == 1


def test_settling_twice_pays_once(patch):
    """Idempotence is what lets this be called from anywhere instead of hooked
    into all eight places a minion can enter a zone."""
    undead = _undead()
    player = _player(patch, [undead])
    raise_standing_bonus(player, UNDEAD_SCOPE, 1, 0)
    for _ in range(5):
        settle_standing_bonuses(player)
    assert undead.raw_attack == 2


def test_two_raises_stack(patch):
    undead = _undead()
    player = _player(patch, [undead])
    raise_standing_bonus(player, UNDEAD_SCOPE, 1, 0)
    raise_standing_bonus(player, UNDEAD_SCOPE, 2, 1)
    assert (undead.raw_attack, undead.max_health) == (4, 2)


def test_scopes_are_summed_not_shadowed(patch):
    """A Beetle that is also an Undead gets both."""
    beetle = Minion(
        card_id="BG28_603t", base_attack=2, base_health=2, tier=1, race=Race.UNDEAD
    )
    player = _player(patch, [beetle])
    raise_standing_bonus(player, UNDEAD_SCOPE, 1, 0)
    raise_standing_bonus(player, BonusScope(ScopeKind.CARD, "BG28_603t"), 2, 1)
    assert standing_bonus_for(player, beetle) == (3, 1)
    assert (beetle.raw_attack, beetle.max_health) == (5, 3)


def test_a_shop_scope_reaches_the_counter_and_not_the_board(patch):
    on_board = Minion(card_id="a", base_attack=1, base_health=1, tier=1)
    in_shop = Minion(card_id="b", base_attack=1, base_health=1, tier=1)
    player = _player(patch, [on_board], shop=[in_shop])
    raise_standing_bonus(player, BonusScope(ScopeKind.SHOP), 5, 5)
    assert (in_shop.raw_attack, in_shop.max_health) == (6, 6)
    assert (on_board.raw_attack, on_board.max_health) == (1, 1)


def test_a_copied_seat_does_not_share_the_table(patch):
    """Copies run once per action; a shared dict would leak one seat's Undead
    buff into another's."""
    player = _player(patch, [_undead()])
    raise_standing_bonus(player, UNDEAD_SCOPE, 1, 0)
    twin = copy_player_state(player)
    raise_standing_bonus(twin, UNDEAD_SCOPE, 5, 5)
    assert player.standing_bonuses[UNDEAD_SCOPE] == (1, 0)
    assert twin.standing_bonuses[UNDEAD_SCOPE] == (6, 5)


def test_a_seat_with_no_bonuses_settles_to_nothing(patch):
    undead = _undead()
    player = _player(patch, [undead])
    settle_standing_bonuses(player)
    assert (undead.raw_attack, undead.max_health) == (1, 1)


# --------------------------------------------------------------------------- #
# The cards
# --------------------------------------------------------------------------- #


def test_nerubian_deathswarmer_buffs_undead_wherever_they_are(patch, triggers):
    swarmer = make_minion("BG25_011", patch=patch)  # 1/4 Undead
    on_board, in_hand, in_shop = _undead("a"), _undead("b"), _undead("c")
    player = _player(patch, [swarmer, on_board], hand=[in_hand], shop=[in_shop])
    triggers.fire_on_place(swarmer, player, None)
    assert on_board.raw_attack == in_hand.raw_attack == in_shop.raw_attack == 2
    # It is Undead itself, so it takes its own buff.
    assert swarmer.raw_attack == swarmer.base_attack + 1


def test_forest_rover_buffs_beetles_it_never_meets(patch, triggers):
    rover = make_minion("BG31_801", patch=patch)
    player = _player(patch, [rover])
    triggers.fire_on_place(rover, player, None)
    # The Beetle its own deathrattle summons is a later arrival like any other.
    beetle = make_minion("BG28_603t", patch=patch)
    player.board.append(beetle)
    settle_standing_bonuses(player)
    assert (beetle.raw_attack, beetle.max_health) == (4, 3)


def test_forest_rover_still_has_its_deathrattle(patch):
    from src.bg_core.effects import Trigger

    rover = make_minion("BG31_801", patch=patch)
    triggers_on = {a.trigger for a in rover.abilities}
    assert Trigger.ON_PLACE in triggers_on and Trigger.ON_DEATH in triggers_on


# --------------------------------------------------------------------------- #
# Tavern bonuses: filtered, and kept once bought
# --------------------------------------------------------------------------- #


def test_a_tavern_bonus_can_name_one_tribe(patch):
    """"Give Elementals in the Tavern +8/+8 this game" — the others get nothing."""
    elemental = Minion(
        card_id="e", base_attack=1, base_health=1, tier=1, race=Race.ELEMENTAL
    )
    murloc = Minion(card_id="m", base_attack=1, base_health=1, tier=1, race=Race.MURLOC)
    player = _player(patch, shop=[elemental, murloc])
    raise_standing_bonus(player, BonusScope(ScopeKind.SHOP, Race.ELEMENTAL), 8, 8)
    assert (elemental.raw_attack, elemental.max_health) == (9, 9)
    assert (murloc.raw_attack, murloc.max_health) == (1, 1)


def test_a_tavern_bonus_can_cap_the_tier_it_reaches(patch):
    """"Minions in the Tavern from Tier 3 and below +3/+3"."""
    low = Minion(card_id="low", base_attack=1, base_health=1, tier=3)
    high = Minion(card_id="high", base_attack=1, base_health=1, tier=4)
    player = _player(patch, shop=[low, high])
    raise_standing_bonus(player, BonusScope(ScopeKind.SHOP, None, 3), 3, 3)
    assert (low.raw_attack, low.max_health) == (4, 4)
    assert (high.raw_attack, high.max_health) == (1, 1)


def test_a_tavern_bonus_survives_being_bought(patch):
    """The stats are the seat's to keep: buying moves the same card, and the
    bonus is never reclaimed once it has been paid."""
    from src.bg_recruitment.economy import buy_from_shop

    minion = Minion(card_id="m", base_attack=1, base_health=1, tier=1)
    player = _player(patch, shop=[minion], gold=10)
    raise_standing_bonus(player, BonusScope(ScopeKind.SHOP), 5, 5)
    assert (minion.raw_attack, minion.max_health) == (6, 6)

    buy_from_shop(
        player,
        0,
        on_bought=lambda m, p: None,
        on_triples=lambda p: None,
    )
    settle_standing_bonuses(player)
    assert any(c is minion for c in player.hand)
    assert (minion.raw_attack, minion.max_health) == (6, 6)


def test_a_bought_minion_takes_no_further_tavern_bonuses(patch):
    """It keeps what it was given, but it is not on the counter any more."""
    minion = Minion(card_id="m", base_attack=1, base_health=1, tier=1)
    player = _player(patch, board=[minion])
    minion.standing_absorbed = ((BonusScope(ScopeKind.SHOP), 5, 5),)
    minion.bonus_attack = minion.bonus_health = 5
    raise_standing_bonus(player, BonusScope(ScopeKind.SHOP), 5, 5)
    assert (minion.raw_attack, minion.max_health) == (6, 6)


def test_a_kept_tavern_bonus_does_not_block_a_later_tribe_bonus(patch):
    """Absorption is per scope: the tavern buff it kept must not make it look
    already-paid when a different bonus is raised."""
    undead = _undead("u")
    player = _player(patch, shop=[undead])
    raise_standing_bonus(player, BonusScope(ScopeKind.SHOP), 5, 5)
    player.shop[0] = None
    player.board.append(undead)
    raise_standing_bonus(player, UNDEAD_SCOPE, 1, 0)
    assert (undead.raw_attack, undead.max_health) == (7, 6)
