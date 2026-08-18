"""What a combat hands its owner, and what stays behind with the copy.

A fight runs on copies: damage, popped shields and spent Venomous die with
them. Some printed effects are meant to escape that — a "permanent" Blood Gem,
a "this game" modifier — and they leave through the seat rather than by writing
to the board behind the engine's back.

Two properties matter and are easy to get backwards: a Gem *without* permanent
must not survive the fight, and a permanent one must land on the owner's real
minion — the one the combat copy was made from, addressed by identity.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.bg_catalog.patch_context import PatchContext
from src.bg_combat.battle.seat import RecordingSeat
from src.bg_core.effects import (
    Ability,
    BloodGemTarget,
    IncreaseBloodGemBonusEffect,
    PlayBloodGemsEffect,
    Trigger,
)
from src.bg_core.minion import Minion, Race
from src.bg_lobby.player import PlayerPhase, PlayerState
from src.bg_recruitment.combat_seat import PlayerCombatSeat
from tests.minibg_helpers import simulate_battle

PATCH_74257 = "data/bgcore/19_6_0_74257"


@pytest.fixture(scope="module")
def patch():
    return PatchContext.load(Path(PATCH_74257))


def _minion(card_id, *abilities, attack=4, health=6, race=Race.QUILBOAR) -> Minion:
    return Minion(
        card_id=card_id,
        base_attack=attack,
        base_health=health,
        tier=1,
        race=race,
        abilities=tuple(abilities),
    )


def _player(board) -> PlayerState:
    return PlayerState(
        health=40,
        gold=5,
        tavern_tier=1,
        board=list(board),
        shop=[None] * 6,
        hand=[None] * 10,
        phase=PlayerPhase.SHOP,
        shop_actions_used=0,
    )


def _fight(player, enemy_board, *, seat=None):
    return simulate_battle(
        player.board,
        enemy_board,
        p0_has_initiative=True,
        rng=np.random.default_rng(0),
        seats=(seat if seat is not None else RecordingSeat(), RecordingSeat()),
    )


def test_a_permanent_gem_lands_on_the_real_minion(patch):
    """Razorfen Vineweaver's shape: "Rally: plays permanent Blood Gems on itself"."""
    boar = _minion(
        "vineweaver",
        Ability(
            Trigger.ON_ATTACK,
            PlayBloodGemsEffect(target=BloodGemTarget.SELF, count=1, permanent=True),
        ),
    )
    player = _player([boar])
    _fight(player, [_minion("dummy", attack=0, health=30)], seat=PlayerCombatSeat(player))

    assert boar.blood_gem_attack > 0, "the Gem reached the board minion"
    assert boar.bonus_attack == boar.blood_gem_attack
    assert boar.blood_gem_attack % 1 == 0


def test_a_plain_gem_dies_with_the_combat_copy(patch):
    """Without "permanent" printed, a Gem played in a fight leaves no trace."""
    boar = _minion(
        "bonker",
        Ability(
            Trigger.ON_ATTACK,
            PlayBloodGemsEffect(target=BloodGemTarget.SELF, count=1),
        ),
    )
    player = _player([boar])
    _fight(player, [_minion("dummy", attack=0, health=30)], seat=PlayerCombatSeat(player))

    assert (boar.bonus_attack, boar.blood_gem_attack) == (0, 0)


def test_a_gem_raised_mid_combat_is_worth_more_when_it_lands(patch):
    """The ordering the seat exists for.

    One Rally raises the seat's Gem value, a later one plays a permanent Gem.
    Collecting the requests and pricing them afterwards would pay the same for
    both; reading the seat at the moment it is played does not.
    """
    raiser = _minion(
        "refiner",
        Ability(Trigger.ON_ATTACK, IncreaseBloodGemBonusEffect(attack=2, health=2)),
    )
    player = _player([raiser])
    seat = PlayerCombatSeat(player)
    _fight(player, [_minion("dummy", attack=0, health=30)], seat=seat)

    assert player.blood_gem_bonus_attack >= 2, "the raise reached the seat"
    assert seat.blood_gem_value() == (
        1 + player.blood_gem_bonus_attack,
        1 + player.blood_gem_bonus_health,
    )


def test_a_seatless_combat_still_runs(patch):
    """Every combat test calls simulate_battle with no player at all."""
    boar = _minion(
        "vineweaver",
        Ability(
            Trigger.ON_ATTACK,
            PlayBloodGemsEffect(target=BloodGemTarget.SELF, count=1, permanent=True),
        ),
    )
    seat = RecordingSeat()
    simulate_battle(
        [boar],
        [_minion("dummy", attack=0, health=30)],
        p0_has_initiative=True,
        rng=np.random.default_rng(0),
        seats=(seat, RecordingSeat()),
    )
    assert seat.permanent_gems, "the request was recorded rather than applied"
    assert boar.bonus_attack == 0, "and nothing was written to the caller's minion"
