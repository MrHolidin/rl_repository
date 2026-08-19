"""The Quilboar family: Blood Gems, Choose One, and three new countdowns."""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pytest

from src.bg_catalog.cards import make_minion
from src.bg_catalog.patch_context import PatchContext
from src.bg_core.minion import Minion, Race
from src.bg_lobby.player import PlayerPhase, PlayerState
from src.bg_recruitment.blood_gems import play_blood_gem_on
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


def _player(patch, board=(), shop=(), **kw) -> PlayerState:
    base = dict(
        health=30, gold=10, tavern_tier=6, board=list(board),
        shop=list(shop) + [None] * (7 - len(shop)), hand=[None] * 10,
        phase=PlayerPhase.SHOP, shop_actions_used=0, ruleset=patch.meta.ruleset,
    )
    base.update(kw)
    return PlayerState(**base)


def _boar(card_id="q", atk=1, hp=1) -> Minion:
    return Minion(card_id=card_id, base_attack=atk, base_health=hp, tier=1, race=Race.QUILBOAR)


def _wall(hp=40, atk=0):
    return Minion(card_id="wall", base_attack=atk, base_health=hp, tier=1)


def _fight(board_0, board_1, patch, seats=None):
    survivors: List[Minion] = []
    kwargs = {"seats": seats} if seats is not None else {}
    simulate_battle(
        board_0, board_1, p0_has_initiative=True, rng=np.random.default_rng(0),
        patch=patch, p0_board_out=survivors, **kwargs,
    )
    return survivors


# --------------------------------------------------------------------------- #
# Blood Gems
# --------------------------------------------------------------------------- #


def test_razorfen_vineweaver_keeps_the_gems_it_plays(patch):
    vineweaver = _card(patch, "BG33_883")
    player = _player(patch, [vineweaver])
    _fight(
        [vineweaver], [_wall(hp=1)], patch,
        seats=(PlayerCombatSeat(player, patch=patch), PlayerCombatSeat(_player(patch))),
    )
    # Three permanent Gems: the board minion keeps them after the fight.
    assert (vineweaver.raw_attack, vineweaver.max_health) == (
        vineweaver.base_attack + 3,
        vineweaver.base_health + 3,
    )


def test_vigilant_bristlemane_gems_its_neighbours(patch):
    bristlemane = _card(patch, "BG36_510")
    left, right = _boar("l"), _boar("r")
    player = _player(patch, [left, bristlemane, right])
    play_blood_gem_on(player, bristlemane, patch=patch)
    assert (left.raw_attack, left.max_health) == (2, 2)
    assert (right.raw_attack, right.max_health) == (2, 2)


def test_sanguine_champion_makes_every_gem_worth_more(patch, triggers):
    champion = _card(patch, "BG23_017")
    target = _boar()
    player = _player(patch, [champion, target])
    triggers.fire_on_place(champion, player, None)
    play_blood_gem_on(player, target, patch=patch)
    assert (target.raw_attack, target.max_health) == (3, 3)


def test_turbo_hogrider_answers_a_choose_one(patch, triggers):
    from src.bg_recruitment.choose_one import fire_choose_one_played

    hogrider = _card(patch, "BG31_323")
    boar = _boar()
    beast = Minion(card_id="b", base_attack=1, base_health=1, tier=1, race=Race.BEAST)
    player = _player(patch, [hogrider, boar, beast])
    fire_choose_one_played(
        player,
        lambda src, eff: triggers.apply_shop_effect(player, src, eff, None),
    )
    assert (boar.raw_attack, boar.max_health) == (2, 2)
    assert (beast.raw_attack, beast.max_health) == (1, 1)


def test_jailbird_juggernaut_summons_its_gems(patch):
    juggernaut = _card(patch, "BG36_333")
    player = _player(patch, [juggernaut])
    play_blood_gem_on(player, juggernaut, count=3, patch=patch)
    survivors = _fight(
        [juggernaut], [_wall(hp=1)], patch,
        seats=(PlayerCombatSeat(player, patch=patch), PlayerCombatSeat(_player(patch))),
    )
    golem = next(
        m for m in survivors if m.card_id == "BG36_333" and m is not juggernaut
        and m.base_attack == 3
    )
    assert (golem.base_attack, golem.base_health) == (3, 3)


# --------------------------------------------------------------------------- #
# Countdowns and charges
# --------------------------------------------------------------------------- #


def test_felboar_eats_every_third_spell(patch):
    felboar = _card(patch, "BG28_633")
    meals = [Minion(card_id=f"m{i}", base_attack=2, base_health=2, tier=1) for i in range(2)]
    player = _player(patch, [felboar], shop=meals)
    target = _boar("t")
    player.board.append(target)
    for _ in range(2):
        play_blood_gem_on(player, target, patch=patch)
    assert sum(1 for m in player.shop if m is not None) == 2
    play_blood_gem_on(player, target, patch=patch)  # the third
    assert sum(1 for m in player.shop if m is not None) == 1


def test_snare_trapper_can_buy_a_bigger_purse(patch, triggers):
    from src.bg_recruitment.choose_one import resolve_choose_one

    trapper = _card(patch, "BG36_332")
    player = _player(patch, [trapper])
    before = player.ruleset.gold_cap
    triggers.fire_on_place(trapper, player, None)
    resolve_choose_one(
        player, 1,
        apply_effect=lambda src, eff: triggers.apply_shop_effect(player, src, eff, None),
    )
    assert player.ruleset.gold_cap == before + 1


def test_snare_trappers_other_half_fetches_a_quilboar(patch, triggers):
    from src.bg_recruitment.choose_one import resolve_choose_one

    trapper = _card(patch, "BG36_332")
    player = _player(patch, [trapper])
    triggers.fire_on_place(trapper, player, None)
    resolve_choose_one(
        player, 0,
        apply_effect=lambda src, eff: triggers.apply_shop_effect(player, src, eff, None),
    )
    got = next(c for c in player.hand if c is not None)
    assert got.race == Race.QUILBOAR


def test_thorned_trailblazer_refills_its_charge_each_turn(patch, triggers):
    trailblazer = _card(patch, "BG31_327")
    player = _player(patch, [trailblazer])
    triggers.fire_on_turn_start(player)
    assert player.choose_one_combined_charges == 1
    triggers.fire_on_turn_start(player)
    assert player.choose_one_combined_charges == 2


def test_bramble_tunneler_fetches_a_choose_one_card(patch):
    """A Rally that fetches queues the card for after the fight, the way every
    combat hand-add does."""
    tunneler = _card(patch, "BG36_331")
    player = _player(patch, [tunneler])
    seat = PlayerCombatSeat(player, patch=patch)
    _fight(
        [tunneler], [_wall(hp=1)], patch,
        seats=(seat, PlayerCombatSeat(_player(patch))),
    )
    assert seat.hand_adds
    assert seat.hand_adds[0] in {
        "BG31_893", "BG31_880", "BG31_881", "BG31_890", "BG31_886", "BG31_884"
    }


def test_the_queued_choose_one_card_lands_as_a_spell(patch):
    """The queue is applied after the fight, and a spell arrives as one."""
    from src.bg_core.spell_card import SpellCard
    from src.bg_recruitment.hand_slots import apply_combat_hand_adds

    player = _player(patch)
    apply_combat_hand_adds(player, ["BG31_880"], patch)
    assert isinstance(player.hand[0], SpellCard)


def test_veteran_brigand_gems_everyone(patch, triggers):
    from src.bg_recruitment.choose_one import resolve_choose_one

    brigand = _card(patch, "BG36_341")
    mate = _boar("mate")
    player = _player(patch, [brigand, mate])
    triggers.fire_on_place(brigand, player, None)
    resolve_choose_one(
        player, 0,
        apply_effect=lambda src, eff: triggers.apply_shop_effect(player, src, eff, None),
    )
    assert (mate.raw_attack, mate.max_health) == (4, 4)
