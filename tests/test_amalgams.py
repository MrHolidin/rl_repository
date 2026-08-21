"""The Amalgams, and the deathrattles that used to take the engine down.

All three Amalgams say the same sentence as each other or as a spell — "a
friendly minion of each type" — which is now what an empty tribe list means.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pytest

from src.bg_catalog.cards import make_minion
from src.bg_catalog.patch_context import PatchContext
from src.bg_combat.battle.effects import _DEATHRATTLE_HANDLERS
from src.bg_core.effects import Trigger
from src.bg_core.minion import ALL_TRIBES, Minion, Race
from src.bg_lobby.player import PlayerPhase, PlayerState
from src.bg_recruitment.combat_seat import PlayerCombatSeat
from src.bg_recruitment.shop_triggers import ShopTriggers
from src.bg_recruitment.tavern_spells import cast_tavern_spell
from tests.minibg_helpers import simulate_battle

PATCH_DIR = Path("data/bgcore/36_2_0_248348")


@pytest.fixture(scope="module")
def patch() -> PatchContext:
    return PatchContext.load(PATCH_DIR)


@pytest.fixture()
def triggers(patch):
    return ShopTriggers(np.random.default_rng(0), patch=patch)


def _card(patch, card_id):
    return make_minion(card_id, patch=patch)


def _player(patch, board=(), **kw) -> PlayerState:
    base = dict(
        health=30, gold=10, tavern_tier=6, board=list(board), shop=[None] * 7,
        hand=[None] * 10, phase=PlayerPhase.SHOP, shop_actions_used=0,
        ruleset=patch.meta.ruleset,
    )
    base.update(kw)
    return PlayerState(**base)


def _of(card_id, race, atk=1, hp=1) -> Minion:
    return Minion(card_id=card_id, base_attack=atk, base_health=hp, tier=1, race=race)


def _wall(hp=60, atk=0):
    return Minion(card_id="wall", base_attack=atk, base_health=hp, tier=1)


def _fight(board_0, board_1, patch, seats=None):
    survivors: List[Minion] = []
    deaths: List[tuple] = []
    kwargs = {"seats": seats} if seats is not None else {}
    simulate_battle(
        board_0, board_1, p0_has_initiative=True, rng=np.random.default_rng(0),
        patch=patch, p0_board_out=survivors, death_log=deaths, **kwargs,
    )
    return survivors, deaths


def _seats(patch, player):
    return (PlayerCombatSeat(player, patch=patch), PlayerCombatSeat(_player(patch)))


# --------------------------------------------------------------------------- #
# "A friendly minion of each type"
# --------------------------------------------------------------------------- #


def test_all_tribes_leaves_out_the_amalgam_marker(patch):
    assert Race.ALL not in ALL_TRIBES
    assert len(ALL_TRIBES) == len(Race) - 1


def test_misplaced_tea_set_pays_one_of_each_type(patch):
    board = [
        _of("beast", Race.BEAST),
        _of("mech", Race.MECHANICAL),
        _of("naga", Race.NAGA),
        _of("beast2", Race.BEAST),
    ]
    player = _player(patch, board=board)
    cast_tavern_spell(
        player, patch.tavern_spells["BG28_888"], rng=np.random.default_rng(0), patch=patch
    )
    grown = [m for m in board if m.raw_attack > 1]
    # One Beast, one Mech, one Naga — three picks, not four.
    assert len(grown) == 3
    assert all(m.raw_attack == 3 and m.max_health == 3 for m in grown)


def test_motley_phalanx_no_longer_takes_the_engine_down(patch):
    """Its deathrattle had no handler, so a fight it died in raised."""
    phalanx = _card(patch, "BG27_080")
    friend = _of("beast", Race.BEAST, 1, 40)
    survivors, _ = _fight([phalanx, friend], [_wall(hp=1, atk=40)], patch)
    assert next(m for m in survivors if m.card_id == "beast").raw_attack == 3


def test_the_last_one_standing_pays_one_of_each_type_for_keeps(patch):
    player = _player(patch)
    standing = _card(patch, "BG34_320")  # 12/12
    beast = _of("beast", Race.BEAST, 1, 40)
    player.board = [standing, beast]
    _fight([standing, beast], [_wall(hp=60)], patch, seats=_seats(patch, player))
    # Two swings, +12/+12 apiece, and it survived the fight on the seat's board.
    assert beast.raw_attack == 1 + 24
    assert beast.max_health == 40 + 24


def test_the_last_one_standing_pays_nothing_it_cannot_reach(patch):
    player = _player(patch)
    standing = _card(patch, "BG34_320")
    player.board = [standing]
    _fight([standing], [_wall(hp=60)], patch, seats=_seats(patch, player))
    assert standing.raw_attack == 12  # "a friendly minion" is another one


def test_golden_last_one_standing_does_it_twice_not_double(patch):
    (ability,) = patch.triple_merge_golden_abilities("BG34_320")
    assert (ability.effect.attack, ability.effect.health) == (12, 12)
    assert ability.effect.repeats == 2


# --------------------------------------------------------------------------- #
# The Tea Set cards
# --------------------------------------------------------------------------- #


def test_nightmare_par_tea_guest_hands_over_a_tea_set_on_play(patch, triggers):
    guest = _card(patch, "BG32_111")
    player = _player(patch, board=[guest])
    triggers.fire_on_place(player=player, placed=guest, shop_excluded_race=None)
    assert [c.card_id for c in player.hand if c is not None] == ["BG28_888"]


def test_nightmare_par_tea_guest_hands_over_another_on_death(patch):
    player = _player(patch)
    guest = _card(patch, "BG32_111")
    player.board = [guest]
    seat = PlayerCombatSeat(player, patch=patch)
    _fight(
        [guest], [_wall(hp=1, atk=40)], patch,
        seats=(seat, PlayerCombatSeat(_player(patch))),
    )
    assert seat.hand_adds == ["BG28_888"]


def test_gatekeeper_amalgam_casts_a_tea_set_when_spelled(patch):
    gatekeeper = _card(patch, "BG36_640")  # 6/6, and an Amalgam
    beast = _of("beast", Race.BEAST)
    player = _player(patch, board=[gatekeeper, beast])
    cast_tavern_spell(
        player,
        patch.tavern_spells["BG28_897"],  # give a minion +2/+2
        rng=np.random.default_rng(0),
        patch=patch,
        target=gatekeeper,
    )
    # 6/6, +2/+2 from the spell, then a Tea Set — and an Amalgam is every type,
    # so it is eligible for nearly every pick the Tea Set makes.
    assert gatekeeper.raw_attack > 8
    assert gatekeeper.max_health > 8


def test_gatekeeper_amalgam_does_not_cast_forever(patch):
    """Its own cast is not "a spell cast on this", or nothing would stop it."""
    gatekeeper = _card(patch, "BG36_640")
    player = _player(patch, board=[gatekeeper])
    cast_tavern_spell(
        player, patch.tavern_spells["BG28_897"], rng=np.random.default_rng(0),
        patch=patch, target=gatekeeper,
    )
    # 6/6, +2/+2 from the spell, and +2/+2 from the Tea Set it casts. Once:
    # "a friendly minion of each type" pays one body per type and each body
    # answers for one type, so being every type is not being paid nine times.
    assert gatekeeper.raw_attack == 6 + 2 + 2


def test_golden_gatekeeper_casts_twice(patch):
    (ability,) = patch.triple_merge_golden_abilities("BG36_640")
    assert ability.effect.repeats == 2
    assert ability.effect.untargeted is True


# --------------------------------------------------------------------------- #
# The guard that found all this
# --------------------------------------------------------------------------- #


def test_every_modern_deathrattle_has_a_handler(patch):
    """The guard covered the 2021 packages and not the one being written."""
    unhandled = [
        (cid, type(ab.effect).__name__)
        for cid, tpl in sorted(patch.templates.items())
        for ab in tpl.abilities
        if ab.trigger is Trigger.ON_DEATH
        and type(ab.effect) not in _DEATHRATTLE_HANDLERS
    ]
    assert not unhandled


def _die_with_a_seat(patch, card_id):
    """Play a card into a fight it loses, and hand back the seat it wrote to."""
    player = _player(patch)
    body = _card(patch, card_id)
    player.board = [body]
    seat = PlayerCombatSeat(player, patch=patch)
    _fight(
        [body], [_wall(hp=1, atk=40)], patch,
        seats=(seat, PlayerCombatSeat(_player(patch))),
    )
    return player, seat


def test_a_spell_handing_deathrattle_reaches_the_hand(patch):
    _player_out, seat = _die_with_a_seat(patch, "BG32_820")  # Firescale Hoarder
    assert seat.hand_adds == ["BG28_168"]  # a Shiny Ring


def test_a_lockbox_deathrattle_reaches_the_seat(patch):
    from src.bg_recruitment.lockbox import find_lockbox

    player, _seat = _die_with_a_seat(patch, "BG36_521")
    assert find_lockbox(player) is not None


def test_a_blood_gem_deathrattle_raises_what_a_gem_is_worth(patch):
    player, _seat = _die_with_a_seat(patch, "BG23_017")  # Sanguine Champion
    assert (player.blood_gem_bonus_attack, player.blood_gem_bonus_health) == (1, 1)
