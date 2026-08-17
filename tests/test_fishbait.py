"""Fishbait — the tavern puts up a 0/1 and your Beast kills it for +5/+5.

The exchange is resolved without a combat runtime, which is only sound because
the bait is fixed at 0/1 and cannot gain stats. These tests hold that reading
in place: the attacker never takes damage, the bait always dies, the reward
follows the printing, and — the part that is easy to forget — attacking is
attacking, so the attacker's Rally fires.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.bg_catalog.patch_context import PatchContext
from src.bg_core.effects import Ability, BuffSelf, GainGoldThisTurnEffect, Trigger
from src.bg_core.minion import Minion, Race
from src.bg_lobby.player import PlayerPhase, PlayerState
from src.bg_recruitment import fishbait
from src.bg_recruitment.shop import buff_all_shop_offers
from src.bg_recruitment.shop_triggers import ShopTriggers

PATCH_74257 = "data/bgcore/19_6_0_74257"


@pytest.fixture(scope="module")
def patch():
    return PatchContext.load(Path(PATCH_74257))


@pytest.fixture
def triggers(patch):
    return ShopTriggers(np.random.default_rng(0), patch=patch)


def _minion(card_id, *abilities, race=Race.BEAST, attack=3, health=4) -> Minion:
    return Minion(
        card_id=card_id,
        base_attack=attack,
        base_health=health,
        tier=1,
        race=race,
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


def test_the_bait_is_a_zero_one_beast_that_cannot_grow():
    bait = fishbait.make_fishbait()
    assert (bait.base_attack, bait.base_health) == (0, 1)
    assert bait.race is Race.BEAST
    assert bait.cannot_gain_stats


def test_a_tavern_wide_buff_leaves_the_bait_alone():
    player = _player()
    other = _minion("other", attack=1, health=1)
    player.shop[0] = other
    fishbait.place_fishbait(player, 1)

    buff_all_shop_offers(player, attack=2, health=2)
    assert (other.bonus_attack, other.bonus_health) == (2, 2)
    bait = player.shop[1]
    assert (bait.bonus_attack, bait.bonus_health) == (0, 0)


def test_the_beast_kills_it_and_takes_the_reward():
    wolf = _minion("wolf")
    player = _player([wolf])
    fishbait.place_fishbait(player, 2)

    attacker = fishbait.attack_fishbait(player, 2)
    assert attacker is wolf
    assert (wolf.bonus_attack, wolf.bonus_health) == (5, 5)
    assert player.shop[2] is None, "the bait is gone from the counter"


def test_the_attacker_takes_no_damage():
    """A 0-Attack bait cannot hurt back, which is why no runtime is needed."""
    wolf = _minion("wolf", attack=3, health=4)
    player = _player([wolf])
    fishbait.place_fishbait(player, 0)
    fishbait.attack_fishbait(player, 0)
    assert wolf.damage_taken == 0
    assert wolf.max_health == 4 + 5


def test_the_golden_bait_pays_double():
    wolf = _minion("wolf")
    player = _player([wolf])
    fishbait.place_fishbait(player, 0, golden=True)
    fishbait.attack_fishbait(player, 0)
    assert (wolf.bonus_attack, wolf.bonus_health) == (10, 10)


def test_the_left_most_beast_is_the_one_that_attacks():
    pirate = _minion("pirate", race=Race.PIRATE)
    first_beast = _minion("first")
    second_beast = _minion("second")
    player = _player([pirate, first_beast, second_beast])
    fishbait.place_fishbait(player, 0)

    assert fishbait.attack_fishbait(player, 0) is first_beast
    assert second_beast.bonus_attack == 0


def test_an_amalgam_counts_as_a_beast():
    amalgam = _minion("amalgam", race=Race.ALL)
    player = _player([amalgam])
    fishbait.place_fishbait(player, 0)
    assert fishbait.attack_fishbait(player, 0) is amalgam


def test_with_no_beast_the_bait_stays_on_the_counter():
    player = _player([_minion("pirate", race=Race.PIRATE)])
    fishbait.place_fishbait(player, 0)

    assert fishbait.attack_fishbait(player, 0) is None
    assert player.shop[0] is not None, "nothing killed it, so it is still there"


def test_the_attackers_rally_fires(triggers):
    """Attacking a bait is attacking: "whenever this attacks" applies."""
    wolf = _minion("wolf", Ability(Trigger.ON_ATTACK, GainGoldThisTurnEffect(amount=2)))
    player = _player([wolf])
    fishbait.place_fishbait(player, 0)

    fishbait.attack_fishbait(
        player,
        0,
        fire_rally=lambda attacker: fishbait.fire_tavern_rally(
            player,
            attacker,
            lambda source, effect: triggers.apply_shop_effect(
                player, source, effect, None
            ),
        ),
    )
    assert player.gold == 7, "the Rally paid out"
    assert (wolf.bonus_attack, wolf.bonus_health) == (5, 5), "and the kill still paid"


def test_a_rally_the_tavern_cannot_resolve_is_loud(triggers):
    """A Rally with no tavern meaning must be a decision, not a silent no-op."""
    from src.bg_core.effects import DealDamageRandomEnemyMinion

    wolf = _minion(
        "wolf", Ability(Trigger.ON_ATTACK, DealDamageRandomEnemyMinion(amount=3))
    )
    player = _player([wolf])
    fishbait.place_fishbait(player, 0)

    with pytest.raises(Exception):
        fishbait.attack_fishbait(
            player,
            0,
            fire_rally=lambda attacker: fishbait.fire_tavern_rally(
                player,
                attacker,
                lambda source, effect: triggers.apply_shop_effect(
                    player, source, effect, None
                ),
            ),
        )


def test_attacking_an_empty_slot_is_rejected():
    player = _player([_minion("wolf")])
    with pytest.raises(ValueError, match="does not hold a Fishbait"):
        fishbait.attack_fishbait(player, 3)
