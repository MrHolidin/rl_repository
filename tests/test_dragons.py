"""The Dragon family, whose new trick is casting a spell from inside a fight."""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pytest

from src.bg_catalog.cards import make_minion
from src.bg_catalog.patch_context import PatchContext
from src.bg_core.effects import Keyword
from src.bg_core.minion import Minion, Race
from src.bg_lobby.player import PlayerPhase, PlayerState
from src.bg_recruitment.combat_seat import PlayerCombatSeat
from src.bg_recruitment.shop_triggers import ShopTriggers
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


def _dragon(card_id="d", atk=1, hp=1) -> Minion:
    return Minion(card_id=card_id, base_attack=atk, base_health=hp, tier=1, race=Race.DRAGON)


def _wall(hp=40, atk=0):
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


# --------------------------------------------------------------------------- #
# Casting a spell mid-fight
# --------------------------------------------------------------------------- #


def test_runic_arcanist_casts_shiny_ring_at_start_of_combat(patch):
    arcanist = _card(patch, "BG36_245")
    mate = _dragon("mate", 1, 20)
    survivors, _ = _fight([arcanist, mate], [_wall()], patch)
    fought = next(m for m in survivors if m.card_id == "mate")
    assert (fought.raw_attack, fought.max_health) == (2, 21)


def test_crimson_vindicator_casts_dragonbreath_on_its_swing(patch):
    vindicator = _card(patch, "BG36_241")  # Divine Shield Dragon
    plain = Minion(card_id="p", base_attack=1, base_health=20, tier=1)
    dragon = _dragon("d", 1, 20)
    survivors, _ = _fight([vindicator, plain, dragon], [_wall(hp=1)], patch)
    # Everyone +1/+1; Dragons again; Divine Shields again.
    assert next(m for m in survivors if m.card_id == "p").raw_attack == 2
    assert next(m for m in survivors if m.card_id == "d").raw_attack == 3


def test_a_cast_in_combat_leaves_no_card_behind(patch):
    arcanist = _card(patch, "BG36_245")
    player = _player(patch, [arcanist])
    _fight(
        [arcanist], [_wall()], patch,
        seats=(PlayerCombatSeat(player, patch=patch), PlayerCombatSeat(_player(patch))),
    )
    assert all(c is None for c in player.hand)


# --------------------------------------------------------------------------- #
# Combat properties
# --------------------------------------------------------------------------- #


def test_warpwing_takes_nothing_back_from_its_target(patch):
    """The retaliation for its own swing, which would otherwise kill it."""
    warpwing = _card(patch, "BG24_004")  # 12/4
    survivors, _ = _fight([warpwing], [_wall(hp=1, atk=9)], patch)
    fought = next(m for m in survivors if m.card_id == "BG24_004")
    assert fought.damage_taken == 0


def test_warpwing_is_not_immune_when_it_is_the_one_attacked(patch):
    warpwing = _card(patch, "BG24_004")
    warpwing.bonus_health += 40  # outlive the fight; the damage is what is on trial
    survivors, _ = _fight([warpwing], [_wall(hp=200, atk=1)], patch)
    fought = next(m for m in survivors if m.card_id == "BG24_004")
    assert fought.damage_taken > 0


def test_obsidian_ravager_splashes_its_attack(patch):
    ravager = _card(patch, "BG27_017")  # 7/7
    enemies = [Minion(card_id=f"e{i}", base_attack=0, base_health=6, tier=1) for i in range(3)]
    _, deaths = _fight([ravager], enemies, patch)
    assert len([cid for side, cid in deaths if side == 1]) >= 2


def test_persistent_poet_gives_the_keep_to_its_neighbours(patch):
    poet = _card(patch, "BG29_813")
    neighbour = _dragon("keeper", 1, 40)
    player = _player(patch, [poet, neighbour])
    giver = Minion(
        card_id="giver", base_attack=1, base_health=40, tier=1, race=Race.DRAGON
    )
    from src.bg_core.effects import Ability, BuffMatching, BuffTarget, Trigger

    giver.abilities = (
        Ability(
            Trigger.ON_START_OF_COMBAT,
            BuffMatching(BuffTarget.ALL_FRIENDLY, attack=5, health=5),
        ),
    )
    player.board.append(giver)
    _fight(
        [poet, neighbour, giver], [_wall(hp=1)], patch,
        seats=(PlayerCombatSeat(player, patch=patch), PlayerCombatSeat(_player(patch))),
    )
    assert neighbour.raw_attack == 6  # kept its five
    assert poet.raw_attack == poet.base_attack  # the Poet keeps nothing itself


# --------------------------------------------------------------------------- #
# Shop
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("card_id,spell_id", [("BG32_820", "BG28_168")])
def test_a_card_hands_over_the_spell_it_names(patch, triggers, card_id, spell_id):
    source = _card(patch, card_id)
    player = _player(patch, [source])
    triggers.fire_on_place(source, player, None)
    assert any(c is not None and c.card_id == spell_id for c in player.hand)


def test_draconic_warden_hands_over_a_chromadrake(patch, triggers):
    warden = _card(patch, "BG34_633")
    player = _player(patch, [warden])
    triggers.fire_on_place(warden, player, None)
    got = next(c for c in player.hand if c is not None)
    assert got.card_id.startswith("BG34_63")


def test_ignition_specialist_hands_over_two_spells(patch, triggers):
    specialist = _card(patch, "BG28_595")
    player = _player(patch, [specialist])
    triggers.fire_on_turn_end(player)
    got = [c for c in player.hand if c is not None]
    assert len(got) == 2 and all(c.is_tavern_spell for c in got)


def test_kalecgos_wants_a_battlecry(patch, triggers):
    kalecgos = _card(patch, "BGS_041")
    dragon = _dragon()
    player = _player(patch, [kalecgos, dragon])
    plain = Minion(card_id="plain", base_attack=1, base_health=1, tier=1)
    triggers.fire_after_friendly_minion_placed(player, plain)
    assert dragon.raw_attack == 1

    from src.bg_core.effects import Ability, GainGoldThisTurnEffect, Trigger

    with_battlecry = Minion(
        card_id="bc", base_attack=1, base_health=1, tier=1,
        abilities=(Ability(Trigger.ON_PLACE, GainGoldThisTurnEffect(amount=1)),),
    )
    triggers.fire_after_friendly_minion_placed(player, with_battlecry)
    assert (dragon.raw_attack, dragon.max_health) == (3, 3)


def test_fire_forged_evoker_improves_with_tavern_spells(patch):
    from src.bg_recruitment.game_counts import TAVERN_SPELLS_CAST

    evoker = _card(patch, "BG32_822")
    mate = _dragon("mate", 1, 40)
    player = _player(patch, [evoker, mate])
    player.game_counts[TAVERN_SPELLS_CAST] = 1
    survivors, _ = _fight(
        [evoker, mate], [_wall()], patch,
        seats=(PlayerCombatSeat(player, patch=patch), PlayerCombatSeat(_player(patch))),
    )
    fought = next(m for m in survivors if m.card_id == "mate")
    assert (fought.raw_attack, fought.max_health) == (5, 42)  # twice over
