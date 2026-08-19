"""The Undead family: bodies traded away, and bodies that come back."""

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
from src.bg_recruitment.activate import activate_minion
from src.bg_recruitment.combat_seat import PlayerCombatSeat
from src.bg_recruitment.shop_triggers import ShopTriggers
from src.bg_recruitment.standing_bonuses import settle_standing_bonuses
from src.bg_recruitment.targeted_battlecry import apply_targeted_on_place_battlecries
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


def _undead(card_id="u", atk=1, hp=1) -> Minion:
    return Minion(card_id=card_id, base_attack=atk, base_health=hp, tier=1, race=Race.UNDEAD)


def _wall(hp=30, atk=0):
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


def _seat(patch, player):
    """The seat under test, with a throwaway one for the other side."""
    seat = PlayerCombatSeat(player, patch=patch)
    return seat, (seat, PlayerCombatSeat(_player(patch)))


def _reborn(card_id="r", atk=1, hp=1) -> Minion:
    return Minion(
        card_id=card_id, base_attack=atk, base_health=hp, tier=1,
        keywords=frozenset({Keyword.REBORN}),
    )


# --------------------------------------------------------------------------- #
# Trading a body away
# --------------------------------------------------------------------------- #


def test_maw_caster_eats_an_undead_and_opens_an_undead_discover(patch, triggers):
    food = _undead("f")
    caster = _card(patch, "BG32_340")
    player = _player(patch, board=[food, caster])
    apply_targeted_on_place_battlecries(
        triggers, player, caster, rng=np.random.default_rng(0), forced_buff_target=food
    )
    assert [m.card_id for m in player.board] == ["BG32_340"]
    pc = player.pending_choice
    assert pc is not None and pc.kind is PendingChoiceKind.DISCOVER_TRIBE
    assert pc.discover_tribe is Race.UNDEAD
    assert all(patch.templates[cid].race is Race.UNDEAD for cid in pc.options)


def test_maw_caster_with_nothing_to_eat_discovers_nothing(patch, triggers):
    caster = _card(patch, "BG32_340")
    player = _player(patch, board=[caster])
    apply_targeted_on_place_battlecries(
        triggers, player, caster, rng=np.random.default_rng(0)
    )
    assert player.pending_choice is None
    assert [m.card_id for m in player.board] == ["BG32_340"]


def test_disguised_graverobber_still_gets_its_plain_copy(patch, triggers):
    """The card the destroy effect was built for, after it grew three fields."""
    food = _undead("BG25_008")  # an Eternal Knight body, buffed
    food.bonus_attack += 10
    robber = _card(patch, "BG28_303")
    player = _player(patch, board=[food, robber])
    apply_targeted_on_place_battlecries(
        triggers, player, robber, rng=np.random.default_rng(0), forced_buff_target=food
    )
    held = [c for c in player.hand if c is not None]
    assert [c.card_id for c in held] == ["BG25_008"]
    assert held[0].bonus_attack == 0  # plain: what the body gained stays behind


def test_dead_bellringer_eats_an_undead_for_stats(patch):
    food = _undead("f")
    ringer = _card(patch, "BG36_511")  # 3/6, Activate (1)
    player = _player(patch, board=[ringer, food])
    activate_minion(
        player, 0, rng=np.random.default_rng(0), patch=patch, buff_target=food
    )
    assert [m.card_id for m in player.board] == ["BG36_511"]
    assert (ringer.raw_attack, ringer.max_health) == (7, 10)
    assert player.gold == 9


def test_dead_bellringer_will_not_eat_itself(patch):
    ringer = _card(patch, "BG36_511")  # an Undead, and the only one
    player = _player(patch, board=[ringer])
    activate_minion(player, 0, rng=np.random.default_rng(0), patch=patch)
    assert [m.card_id for m in player.board] == ["BG36_511"]
    assert ringer.raw_attack == ringer.base_attack


# --------------------------------------------------------------------------- #
# A tavern death is a death
# --------------------------------------------------------------------------- #


def test_plaguerunner_pays_double_outside_combat(patch, triggers):
    player = _player(patch)
    runner = _card(patch, "BG34_690")
    player.board = [runner]
    triggers.fire_tavern_deathrattle(runner, player)
    later = _undead("u")
    player.board = [later]
    settle_standing_bonuses(player)
    assert later.raw_attack == 1 + 4


def test_plaguerunner_pays_the_printed_price_in_combat(patch):
    player = _player(patch)
    runner = _card(patch, "BG34_690")
    player.board = [runner]
    _, seats = _seat(patch, player)
    _fight([runner], [_wall(hp=1, atk=40)], patch, seats=seats)
    later = _undead("u")
    player.board = [later]
    settle_standing_bonuses(player)
    assert later.raw_attack == 1 + 2


def test_a_destroyed_body_leaves_its_deathrattle_behind(patch, triggers):
    """Maw Caster eating a Plaguerunner is the combo the second price is for."""
    runner = _card(patch, "BG34_690")
    caster = _card(patch, "BG32_340")
    player = _player(patch, board=[runner, caster])
    apply_targeted_on_place_battlecries(
        triggers, player, caster, rng=np.random.default_rng(0), forced_buff_target=runner
    )
    later = _undead("u")
    player.board = [later]
    settle_standing_bonuses(player)
    assert later.raw_attack == 1 + 4


def test_forsaken_weaver_raises_the_undead_on_a_tavern_spell(patch):
    from src.bg_recruitment.tavern_spells import _fire_tavern_spell_cast

    weaver = _card(patch, "BG34_692")
    player = _player(patch, board=[weaver])
    _fire_tavern_spell_cast(
        player, rng=np.random.default_rng(0), patch=patch, shared_pool=None
    )
    later = _undead("u")
    player.board = [later]
    settle_standing_bonuses(player)
    assert later.raw_attack == 1 + 2


# --------------------------------------------------------------------------- #
# Coming back
# --------------------------------------------------------------------------- #


def test_barrier_banshee_answers_a_reborn_friendly(patch):
    banshee = _card(patch, "BG36_514")  # 7/7
    survivors, _ = _fight([_reborn(), banshee], [_wall(hp=30, atk=5)], patch)
    paid = next(m for m in survivors if m.card_id == "BG36_514")
    assert (paid.raw_attack, paid.max_health) == (14, 14)
    assert Keyword.SHIELD in paid.all_keywords


def test_barrier_banshee_is_quiet_without_reborn(patch):
    banshee = _card(patch, "BG36_514")
    plain = Minion(card_id="p", base_attack=1, base_health=1, tier=1)
    survivors, _ = _fight([plain, banshee], [_wall(hp=30, atk=1)], patch)
    quiet = next(m for m in survivors if m.card_id == "BG36_514")
    assert (quiet.raw_attack, quiet.max_health) == (7, 7)
    assert Keyword.SHIELD not in quiet.all_keywords


def test_snazzy_phantom_pays_the_right_most_undead(patch):
    phantom = _card(patch, "BG36_515")
    right = _undead("u", 2, 30)
    survivors, _ = _fight(
        [_reborn(atk=6), phantom, right], [_wall(hp=30, atk=5)], patch
    )
    grown = next(m for m in survivors if m.card_id == "u")
    assert (grown.raw_attack, grown.max_health) == (8, 36)  # +6/+6, the Attack it read


def test_golden_snazzy_phantom_doubles_the_attack_it_reads(patch):
    (ability,) = patch.triple_merge_golden_abilities("BG36_515")
    assert ability.effect.factor == 2


def test_eternal_summoner_leaves_an_eternal_knight(patch):
    survivors, deaths = _fight([_card(patch, "BG25_009")], [_wall(hp=1, atk=40)], patch)
    assert any(m.card_id == "BG25_008" for m in survivors)
    assert Keyword.REBORN not in {
        k for m in survivors if m.card_id == "BG25_009" for k in m.all_keywords
    }


def test_golden_eternal_summoner_leaves_a_golden_knight(patch):
    (ability,) = patch.triple_merge_golden_abilities("BG25_009")
    assert (ability.effect.token_id, ability.effect.count) == ("BG25_008_G", 1)


# --------------------------------------------------------------------------- #
# Avenge
# --------------------------------------------------------------------------- #


def test_drustfallen_butcher_pays_on_the_third_death(patch):
    player = _player(patch)
    butcher = _card(patch, "BG32_324")
    fodder = [Minion(card_id=f"f{i}", base_attack=0, base_health=1, tier=1) for i in range(3)]
    player.board = [butcher] + fodder
    seat, seats = _seat(patch, player)
    _fight([butcher] + fodder, [_wall(hp=40, atk=1)], patch, seats=seats)
    assert seat.hand_adds == ["BG28_604"]  # a Butchering


def test_drustfallen_butcher_is_quiet_on_two(patch):
    player = _player(patch)
    butcher = _card(patch, "BG32_324")
    fodder = [Minion(card_id=f"f{i}", base_attack=0, base_health=1, tier=1) for i in range(2)]
    player.board = [butcher] + fodder
    seat, seats = _seat(patch, player)
    _fight([butcher] + fodder, [_wall(hp=40, atk=1)], patch, seats=seats)
    assert seat.hand_adds == []


def test_golden_butcher_keeps_avenge_three_and_gets_two(patch):
    (ability,) = patch.triple_merge_golden_abilities("BG32_324")
    assert ability.effect.count == 3  # not six
    assert ability.effect.effect.count == 2


def test_deathly_striker_summons_an_undead_from_hand_on_death(patch):
    player = _player(patch)
    striker = _card(patch, "BG31_835")
    player.hand[0] = _card(patch, "BG25_008")  # an Eternal Knight, kept in hand
    player.board = [striker]
    _, seats = _seat(patch, player)
    _, deaths = _fight([striker], [_wall(hp=40, atk=20)], patch, seats=seats)
    assert any(cid == "BG25_008" for side, cid in deaths if side == 0)
    assert player.hand[0] is not None  # the card stays in hand


def test_golden_deathly_striker_fetches_and_summons_two(patch):
    avenge, deathrattle = patch.triple_merge_golden_abilities("BG31_835")
    assert avenge.effect.count == 4
    assert avenge.effect.effect.count == 2
    assert deathrattle.effect.count == 2


# --------------------------------------------------------------------------- #
# Stitched Salvager
# --------------------------------------------------------------------------- #


def test_stitched_salvager_eats_its_left_neighbour_and_gives_it_back(patch):
    left = Minion(card_id="L", base_attack=3, base_health=4, tier=1)
    salvager = _card(patch, "BG31_999")
    survivors, deaths = _fight([left, salvager], [_wall(hp=1, atk=40)], patch)
    # Eaten at the start, and back on the board once the Salvager fell.
    assert ("L", 3, 4) in [(m.card_id, m.raw_attack, m.max_health) for m in survivors]
    dead = [cid for side, cid in deaths if side == 0]
    assert dead.count("L") == 1 and "BG31_999" in dead


def test_stitched_salvager_keeps_what_the_body_had_gained(patch):
    left = Minion(card_id="L", base_attack=3, base_health=4, tier=1)
    left.bonus_attack += 7
    salvager = _card(patch, "BG31_999")
    survivors, _ = _fight([left, salvager], [_wall(hp=1, atk=40)], patch)
    back = next(m for m in survivors if m.card_id == "L")
    assert back.raw_attack == 10  # an *exact* copy, not the printed card


def test_stitched_salvager_will_not_eat_another_salvager(patch):
    twin = _card(patch, "BG31_999")
    salvager = _card(patch, "BG31_999")
    survivors, _ = _fight([twin, salvager], [_wall(hp=200, atk=0)], patch)
    assert len([m for m in survivors if m.card_id == "BG31_999"]) == 2


def test_stitched_salvager_alone_eats_nothing(patch):
    salvager = _card(patch, "BG31_999")
    survivors, _ = _fight([salvager], [_wall(hp=200, atk=0)], patch)
    assert [m.card_id for m in survivors] == ["BG31_999"]


def test_golden_stitched_salvager_eats_both_neighbours(patch):
    start, _dr = patch.triple_merge_golden_abilities("BG31_999")
    assert start.effect.adjacent is True
