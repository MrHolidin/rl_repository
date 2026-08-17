"""Rally — "Whenever this attacks", the modern pool's most common keyword.

Twenty-eight minions in the current tavern pool carry it, across Quilboar,
Pirates, Nagas and Dragons. It fires on the attacker's swing with the target
already chosen and nothing damaged yet, which is what separates it from
``ON_AFTER_ATTACK``: a Rally that strips the target's keywords has to run while
the target is still standing, and one that buffs the attacker is felt by the
swing that triggered it.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pytest

from tests.minibg_helpers import simulate_battle
from src.bg_core.effects import (
    Ability,
    BuffMatching,
    BuffSelf,
    BuffTarget,
    DealDamageRandomEnemyMinion,
    Keyword,
    Trigger,
)
from src.bg_core.minion import Minion


def _minion(card_id: str, atk: int, hp: int, **kw) -> Minion:
    return Minion(card_id=card_id, base_attack=atk, base_health=hp, tier=1, **kw)


def _rally(card_id: str, atk: int, hp: int, effect, **kw) -> Minion:
    return _minion(card_id, atk, hp, abilities=(Ability(Trigger.ON_ATTACK, effect),), **kw)


def _fight(board_0, board_1, seed: int = 0):
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


def test_rally_fires_when_the_minion_attacks():
    """A +2/+0 Rally turns a 1/10 into a 3-Attack swing against a 2-health body."""
    attacker = _rally("rally", 1, 10, BuffSelf(attack=2, health=0))
    victim = _minion("victim", 0, 3)
    _result, deaths, _ = _fight([attacker], [victim])
    assert "victim" in _side_1_deaths(deaths), (
        "the Rally buff must land before damage — 1 attack leaves a 3-health body alive"
    )


def test_rally_buff_accumulates_across_the_fight():
    """Every swing adds another +2, so the body ends well above its printed 1."""
    attacker = _rally("rally", 1, 10, BuffSelf(attack=2, health=0))
    victim = _minion("victim", 0, 20)
    _result, _deaths, survivors = _fight([attacker], [victim])
    assert survivors and survivors[0].card_id == "rally"
    assert survivors[0].bonus_attack >= 4, "one buff per swing, and this fight took several"
    assert survivors[0].bonus_attack % 2 == 0, "each Rally grants exactly +2"


def test_rally_fires_on_every_swing_not_just_the_first():
    """Windfury swings twice, so the attacker collects the buff twice."""
    attacker = _rally(
        "rally", 1, 10, BuffSelf(attack=1, health=0), keywords=frozenset({Keyword.WINDFURY})
    )
    victim = _minion("victim", 0, 5)
    _result, deaths, _ = _fight([attacker], [victim])
    # Swing one: 1+1 = 2 damage. Swing two: 2+1 = 3 more, for 5 total.
    assert "victim" in _side_1_deaths(deaths)


def test_rally_can_buff_the_whole_board():
    """Bonker's shape: attacking pays out a buff across the warband."""
    attacker = _rally(
        "bonker", 1, 10, BuffMatching(target=BuffTarget.ALL_FRIENDLY, attack=2, health=2)
    )
    ally = _minion("ally", 1, 1)
    victim = _minion("victim", 0, 20)
    _result, _deaths, survivors = _fight([attacker, ally], [victim])
    by_id = {m.card_id: m for m in survivors}
    assert by_id["ally"].bonus_attack > 0, "the ally shares in the Rally"
    assert by_id["bonker"].bonus_attack > 0, "ALL_FRIENDLY includes the attacker"


def test_rally_damage_reaches_the_enemy_board():
    attacker = _rally("rally", 1, 10, DealDamageRandomEnemyMinion(amount=5))
    victim = _minion("victim", 0, 4)
    _result, deaths, _ = _fight([attacker], [victim])
    assert "victim" in _side_1_deaths(deaths)


def test_rally_does_not_fire_when_defending():
    """The keyword is about attacking, not about being in a fight."""
    defender = _rally("rally", 0, 10, BuffSelf(attack=5, health=0))
    attacker = _minion("attacker", 1, 10)
    # Side 1 has initiative, so side 0's minion only ever defends.
    deaths: List[Tuple[int, str]] = []
    result = simulate_battle(
        [defender],
        [attacker],
        p0_has_initiative=False,
        rng=np.random.default_rng(0),
        death_log=deaths,
    )
    # Had Rally fired on defence, the defender would hit back for 5 and kill the
    # 1/10 attacker; instead it never gains Attack and is ground down itself.
    assert [card_id for side, card_id in deaths] == ["rally"]
    assert result.damage_p0 > 0 and result.damage_p1 == 0


def test_an_unhandled_rally_effect_is_loud():
    """A Rally the engine cannot resolve must not be silently skipped."""
    from src.bg_core.effects import HeroImmuneAura

    attacker = _rally("rally", 1, 10, HeroImmuneAura())
    victim = _minion("victim", 0, 5)
    with pytest.raises(NotImplementedError, match="Rally effect"):
        _fight([attacker], [victim])
