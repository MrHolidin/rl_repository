"""The last six Mechs — the Magnetize machinery reached from somewhere new."""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pytest

from src.bg_catalog.cards import make_minion
from src.bg_catalog.patch_context import PatchContext
from src.bg_core.effects import Keyword
from src.bg_core.minion import Minion, Race
from src.bg_lobby.player import PendingChoiceKind, PlayerPhase, PlayerState
from src.bg_recruitment.combat_seat import PlayerCombatSeat
from src.bg_recruitment.discover import resolve_discover_pick
from src.bg_recruitment.game_counts import (
    DEATHRATTLES_TRIGGERED,
    bump_seat_counter,
    refresh_count_bonuses,
)
from src.bg_recruitment.shop_triggers import ShopTriggers
from src.bg_recruitment.targeted_battlecry import apply_targeted_on_place_battlecries
from src.bg_recruitment.tavern_spells import cast_tavern_spell, tavern_spell_bonus
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


def _mech(card_id="m", atk=1, hp=1) -> Minion:
    return Minion(
        card_id=card_id, base_attack=atk, base_health=hp, tier=1, race=Race.MECHANICAL
    )


def _wall(hp=30, atk=0):
    return Minion(card_id="wall", base_attack=atk, base_health=hp, tier=1)


def _fight(board_0, board_1, patch, seats=None, seed=0):
    survivors: List[Minion] = []
    deaths: List[tuple] = []
    kwargs = {"seats": seats} if seats is not None else {}
    simulate_battle(
        board_0, board_1, p0_has_initiative=True, rng=np.random.default_rng(seed),
        patch=patch, p0_board_out=survivors, death_log=deaths, **kwargs,
    )
    return survivors, deaths


def _seat(patch, player):
    seat = PlayerCombatSeat(player, patch=patch)
    return seat, (seat, PlayerCombatSeat(_player(patch)))


# --------------------------------------------------------------------------- #
# Sentences other tribes already say
# --------------------------------------------------------------------------- #


def test_enchanted_sentinel_raises_tavern_spells_while_it_stands(patch):
    sentinel = _card(patch, "BG35_341")
    assert Keyword.MAGNETIC in sentinel.all_keywords
    assert tavern_spell_bonus(_player(patch)) == (0, 0)
    assert tavern_spell_bonus(_player(patch, board=[sentinel])) == (1, 1)


def test_charging_czarina_pays_only_the_divine_shields(patch):
    czarina = _card(patch, "BG28_741")  # 4/1, Divine Shield
    shielded = Minion(
        card_id="s", base_attack=1, base_health=1, tier=1,
        keywords=frozenset({Keyword.SHIELD}),
    )
    plain = Minion(card_id="p", base_attack=1, base_health=1, tier=1)
    player = _player(patch, board=[czarina, shielded, plain])
    cast_tavern_spell(
        player, patch.tavern_spells["BG28_810"], rng=np.random.default_rng(0), patch=patch
    )
    assert shielded.raw_attack == 5
    assert plain.raw_attack == 1
    assert czarina.raw_attack == 8  # it has one too


# --------------------------------------------------------------------------- #
# Welding from somewhere new
# --------------------------------------------------------------------------- #


def test_glambot_welds_a_satellite_onto_a_spelled_mech(patch):
    glambot = _card(patch, "BG36_853")
    mech = _mech()
    player = _player(patch, board=[glambot, mech])
    cast_tavern_spell(
        player,
        patch.tavern_spells["BG28_897"],  # give a minion +2/+2
        rng=np.random.default_rng(0),
        patch=patch,
        target=mech,
    )
    # 1/1, +2/+2 from the spell, then a 6/6 Satellite welded on.
    assert (mech.raw_attack, mech.max_health) == (9, 9)
    assert mech.magnetized_count == 1


def test_glambot_ignores_a_spell_cast_on_anything_else(patch):
    glambot = _card(patch, "BG36_853")
    beast = Minion(
        card_id="b", base_attack=1, base_health=1, tier=1, race=Race.BEAST
    )
    player = _player(patch, board=[glambot, beast])
    cast_tavern_spell(
        player, patch.tavern_spells["BG28_897"], rng=np.random.default_rng(0),
        patch=patch, target=beast,
    )
    assert (beast.raw_attack, beast.max_health) == (3, 3)
    assert beast.magnetized_count == 0


def test_golden_glambot_welds_twice(patch):
    (ability,) = patch.triple_merge_golden_abilities("BG36_853")
    assert ability.effect.effect.repeats == 2


def test_an_ordinary_tavern_spell_is_a_spell_cast_on_its_target(patch):
    """The event Blood Gems and Spellcraft already fired, and this one did not."""
    distractor = _card(patch, "BG36_762")  # a spell on this buffs the tavern for good
    player = _player(patch, board=[distractor])
    cast_tavern_spell(
        player, patch.tavern_spells["BG28_897"], rng=np.random.default_rng(0),
        patch=patch, target=distractor,
    )
    assert player.standing_bonuses  # the watcher heard it


# --------------------------------------------------------------------------- #
# Clunker Junker
# --------------------------------------------------------------------------- #


def test_clunker_junker_welds_its_discover_onto_the_chosen_mech(patch, triggers):
    junker = _card(patch, "BG29_503")
    target = _mech()
    player = _player(patch, board=[target, junker])
    apply_targeted_on_place_battlecries(
        triggers, player, junker, rng=np.random.default_rng(0), forced_buff_target=target
    )
    pc = player.pending_choice
    assert pc is not None and pc.kind is PendingChoiceKind.DISCOVER_TRIBE
    assert pc.magnetize_onto_board_idx == 0
    assert all(patch.templates[cid].race is Race.MECHANICAL for cid in pc.options)

    picked = patch.templates[pc.options[0]]
    resolve_discover_pick(
        player, 0, None, rng=np.random.default_rng(0),
        on_after_placed=lambda _p, _m: None, patch=patch,
    )
    assert target.magnetized_count == 1
    assert (target.raw_attack, target.max_health) == (
        1 + picked.raw_attack,
        1 + picked.max_health,
    )
    assert all(c is None for c in player.hand)  # it never was a card in hand


def test_clunker_junker_opens_even_with_a_full_hand(patch, triggers):
    """The pick is welded, not held, so a full hand is no reason to refuse it."""
    junker = _card(patch, "BG29_503")
    target = _mech()
    player = _player(patch, board=[target, junker])
    player.hand = [_card(patch, "BG25_008") for _ in range(10)]
    apply_targeted_on_place_battlecries(
        triggers, player, junker, rng=np.random.default_rng(0), forced_buff_target=target
    )
    assert player.pending_choice is not None


def test_clunker_junker_with_no_other_mech_does_nothing(patch, triggers):
    junker = _card(patch, "BG29_503")  # a Mech itself, and "a friendly Mech" is another
    player = _player(patch, board=[junker])
    apply_targeted_on_place_battlecries(
        triggers, player, junker, rng=np.random.default_rng(0)
    )
    assert player.pending_choice is None


def test_golden_clunker_junker_discovers_twice(patch):
    (ability,) = patch.triple_merge_golden_abilities("BG29_503")
    assert ability.effect.repeats == 2
    assert ability.effect.magnetize_onto_target is True


# --------------------------------------------------------------------------- #
# Scrap Scraper
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("seed", [0, 3, 7])
def test_scrap_scraper_fetches_a_magnetic_mech(patch, seed):
    player = _player(patch)
    scraper = _card(patch, "BG26_148")
    player.board = [scraper]
    seat, seats = _seat(patch, player)
    _fight([scraper], [_wall(hp=1, atk=40)], patch, seats=seats, seed=seed)
    assert seat.hand_adds
    for cid in seat.hand_adds:
        template = patch.templates[cid]
        assert template.race is Race.MECHANICAL
        assert Keyword.MAGNETIC in template.all_keywords


def test_golden_scrap_scraper_fetches_two(patch):
    (ability,) = patch.triple_merge_golden_abilities("BG26_148")
    assert ability.effect.count == 2
    assert ability.effect.keyword is Keyword.MAGNETIC


# --------------------------------------------------------------------------- #
# Falling Sky Golem
# --------------------------------------------------------------------------- #


def test_falling_sky_golem_grows_with_every_deathrattle(patch):
    player = _player(patch)
    golem = _card(patch, "BG35_342")  # 4/2, Divine Shield
    player.board = [golem]
    refresh_count_bonuses(player)
    assert (golem.raw_attack, golem.max_health) == (4, 2)
    for _ in range(3):
        bump_seat_counter(player, DEATHRATTLES_TRIGGERED)
    assert (golem.raw_attack, golem.max_health) == (4 + 12, 2 + 6)


def test_falling_sky_golem_counts_a_real_fights_deathrattles(patch):
    player = _player(patch)
    golem = _card(patch, "BG35_342")
    bonehead = _card(patch, "BG28_300")  # Deathrattle: two Skeletons
    player.board = [bonehead, golem]
    _, seats = _seat(patch, player)
    _fight([bonehead, golem], [_wall(hp=40, atk=3)], patch, seats=seats)
    assert player.game_counts[DEATHRATTLES_TRIGGERED] == 1
    assert (golem.raw_attack, golem.max_health) == (8, 4)


def test_falling_sky_golem_grows_in_hand_too(patch):
    """"wherever this is" — the tally is the seat's, not the board's."""
    player = _player(patch)
    golem = _card(patch, "BG35_342")
    player.hand[0] = golem
    bump_seat_counter(player, DEATHRATTLES_TRIGGERED)
    assert (golem.raw_attack, golem.max_health) == (8, 4)


def test_a_tavern_destroy_counts_toward_it(patch, triggers):
    """A body destroyed in the tavern fires its deathrattle, so it counts."""
    player = _player(patch)
    victim = _card(patch, "BG28_300")
    triggers.fire_tavern_deathrattle(victim, player)
    assert player.game_counts[DEATHRATTLES_TRIGGERED] == 1
