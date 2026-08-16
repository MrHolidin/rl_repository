"""Venomous and Avenge — two keywords the modern pool needs.

Venomous is Poisonous that its kill uses up; Avenge (N) is a friendly-death
counter in front of an effect. Both are combat-only, and both keep their state
on the combat copy, so neither leaves a trace on the board the shop phase sees.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pytest

from tests.minibg_helpers import simulate_battle
from src.bg_combat.battle.state import battle_copy
from src.bg_core.effects import Ability, AvengeEffect, BuffSelf, Keyword, Trigger
from src.bg_core.minion import Minion


def _minion(card_id: str, atk: int, hp: int, **kw) -> Minion:
    return Minion(card_id=card_id, base_attack=atk, base_health=hp, tier=1, **kw)


def _fight(board_0, board_1, seed: int = 0):
    """Run a combat and hand back (damage, deaths, surviving side-0 bodies)."""
    deaths: List[Tuple[int, str]] = []
    survivors_0: List[Minion] = []
    result = simulate_battle(
        board_0,
        board_1,
        p0_has_initiative=True,
        rng=np.random.default_rng(seed),
        death_log=deaths,
        p0_board_out=survivors_0,
    )
    return result, deaths, survivors_0


def _side_1_deaths(deaths):
    return [card_id for side, card_id in deaths if side == 1]


# --------------------------------------------------------------------------- #
# Venomous
# --------------------------------------------------------------------------- #


def test_venomous_destroys_what_it_damages():
    venom = _minion("venom", 1, 10, keywords=frozenset({Keyword.VENOMOUS}))
    fat = _minion("fat", 1, 20)
    result, deaths, _ = _fight([venom], [fat])
    assert "fat" in _side_1_deaths(deaths), "a 1/10 Venomous should kill a 1/20 outright"
    assert result.damage_p1 > 0 and result.damage_p0 == 0, "side 0 won"


def test_venomous_is_used_up_by_its_kill():
    """Two identical fat bodies; only the first one is poisoned.

    Neither enemy can deal damage, so the venomous minion swings freely all
    combat. It still cannot get through the second body: 1 Attack against 500
    health is what is left once the venom is gone.
    """
    venom = _minion("venom", 1, 30, keywords=frozenset({Keyword.VENOMOUS}))
    _result, deaths, _ = _fight([venom], [_minion("a", 0, 500), _minion("b", 0, 500)])
    assert len(_side_1_deaths(deaths)) == 1


def test_poisonous_is_not_used_up():
    """The contrast that gives Venomous its meaning: same board, both bodies die."""
    poison = _minion("poison", 1, 30, keywords=frozenset({Keyword.POISONOUS}))
    _result, deaths, _ = _fight([poison], [_minion("a", 0, 500), _minion("b", 0, 500)])
    assert sorted(_side_1_deaths(deaths)) == ["a", "b"]


def test_divine_shield_does_not_spend_the_venom():
    shielded = _minion(
        "shielded", 0, 4, keywords=frozenset({Keyword.SHIELD}), has_shield=True
    )
    plain = _minion("plain", 0, 20)
    venom = _minion("venom", 1, 30, keywords=frozenset({Keyword.VENOMOUS}))
    _result, deaths, _ = _fight([venom], [shielded, plain])
    killed = _side_1_deaths(deaths)
    # The shield ate the first hit; the venom was still armed for the 20-health body.
    assert "plain" in killed


def test_venom_is_rearmed_next_combat():
    venom = _minion("venom", 1, 10, keywords=frozenset({Keyword.VENOMOUS}))
    first = battle_copy(venom, 1)
    first.venom_spent = True
    second = battle_copy(venom, 2)
    assert second.venom_spent is False
    assert venom.venom_spent is False


# --------------------------------------------------------------------------- #
# Avenge
# --------------------------------------------------------------------------- #


def _avenger(count: int) -> Minion:
    """An Avenge (N) minion that cannot die during the test combat.

    The health is absurd on purpose: the counter is what is under test, and a
    combat that outlives the avenger would only be measuring how long two
    minions take to trade.
    """
    return _minion(
        "avenger",
        1,
        10_000,
        abilities=(
            Ability(
                Trigger.ON_FRIENDLY_MINION_DIED,
                AvengeEffect(count=count, effect=BuffSelf(attack=10, health=0)),
            ),
        ),
    )


def _avenge_fight(count: int, fodder_count: int):
    fodder = [_minion(f"f{i}", 0, 1) for i in range(fodder_count)]
    killer = _minion("killer", 5, 10_000)
    _result, deaths, survivors = _fight([_avenger(count)] + fodder, [killer])
    assert len([c for side, c in deaths if side == 0]) == fodder_count
    return next(m for m in survivors if m.card_id == "avenger")


def test_avenge_fires_on_the_nth_death():
    avenger = _avenge_fight(count=3, fodder_count=3)
    assert avenger.bonus_attack == 10, "three friendly deaths → Avenge (3) once"
    assert avenger.avenge_progress == 0


def test_avenge_does_not_fire_below_the_count():
    avenger = _avenge_fight(count=3, fodder_count=2)
    assert avenger.bonus_attack == 0
    assert avenger.avenge_progress == 2, "progress is kept, the effect is not owed"


def test_avenge_one_fires_on_every_death():
    avenger = _avenge_fight(count=1, fodder_count=2)
    assert avenger.bonus_attack == 20


def test_avenge_rearms_after_firing():
    avenger = _avenge_fight(count=2, fodder_count=4)
    assert avenger.bonus_attack == 20, "four deaths → Avenge (2) twice"
    assert avenger.avenge_progress == 0


def test_avenge_counts_reset_between_combats():
    fresh = battle_copy(_avenger(3), 1)
    fresh.avenge_progress = 2
    again = battle_copy(_avenger(3), 2)
    assert again.avenge_progress == 0


def test_unhandled_death_listener_effect_is_loud():
    """An Avenge wrapping an effect this trigger cannot apply must not pass silently."""
    from src.bg_core.effects import DealHeroDamage

    avenger = _minion(
        "avenger",
        1,
        10_000,
        abilities=(
            Ability(
                Trigger.ON_FRIENDLY_MINION_DIED,
                AvengeEffect(count=1, effect=DealHeroDamage(amount=1)),
            ),
        ),
    )
    with pytest.raises(NotImplementedError, match="DealHeroDamage"):
        _fight([avenger, _minion("f", 0, 1)], [_minion("killer", 5, 10_000)])
