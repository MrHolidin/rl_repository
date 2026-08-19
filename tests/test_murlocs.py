"""The Murloc family, whose subject is the hand.

They read it at Start of Combat, write into it from the board, and one of them
acts *while sitting in it* — the only listener in the game that is not on the
board.
"""

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


def _player(patch, board=(), hand=(), **kw) -> PlayerState:
    base = dict(
        health=30,
        gold=10,
        tavern_tier=6,
        board=list(board),
        shop=[None] * 7,
        hand=list(hand) + [None] * (10 - len(hand)),
        phase=PlayerPhase.SHOP,
        shop_actions_used=0,
        ruleset=patch.meta.ruleset,
    )
    base.update(kw)
    return PlayerState(**base)


def _murloc(card_id="m", atk=1, hp=1) -> Minion:
    return Minion(card_id=card_id, base_attack=atk, base_health=hp, tier=1, race=Race.MURLOC)


def _fight(board_0, board_1, patch, seats=None):
    survivors: List[Minion] = []
    kwargs = {"seats": seats} if seats is not None else {}
    simulate_battle(
        board_0,
        board_1,
        p0_has_initiative=True,
        rng=np.random.default_rng(0),
        patch=patch,
        p0_board_out=survivors,
        **kwargs,
    )
    return survivors


# --------------------------------------------------------------------------- #
# A card that acts while in hand
# --------------------------------------------------------------------------- #


def test_bream_counter_grows_while_it_waits_in_hand(patch, triggers):
    counter = _card(patch, "BG26_137")
    player = _player(patch, hand=[counter])
    triggers.fire_after_friendly_minion_placed(player, _murloc())
    assert (counter.raw_attack, counter.max_health) == (
        counter.base_attack + 6,
        counter.base_health + 6,
    )


def test_it_ignores_a_minion_of_another_tribe(patch, triggers):
    counter = _card(patch, "BG26_137")
    player = _player(patch, hand=[counter])
    beast = Minion(card_id="b", base_attack=1, base_health=1, tier=1, race=Race.BEAST)
    triggers.fire_after_friendly_minion_placed(player, beast)
    assert counter.raw_attack == counter.base_attack


def test_a_board_listener_is_not_a_hand_listener(patch, triggers):
    """The pass over hand cards is its own: a card on the board with the same
    trigger is not consulted, and vice versa."""
    counter = _card(patch, "BG26_137")
    player = _player(patch, board=[counter])
    triggers.fire_after_friendly_minion_placed(player, _murloc())
    assert counter.raw_attack == counter.base_attack


# --------------------------------------------------------------------------- #
# Reading the hand at Start of Combat
# --------------------------------------------------------------------------- #


def test_costume_enthusiast_takes_the_biggest_attack_in_hand(patch):
    enthusiast = _card(patch, "BG34_142")  # 4/5, Divine Shield
    player = _player(
        patch,
        board=[enthusiast],
        hand=[_murloc("small", 2, 2), _murloc("big", 9, 1)],
    )
    survivors = _fight(
        [enthusiast],
        [Minion(card_id="w", base_attack=0, base_health=40, tier=1)],
        patch,
        seats=(PlayerCombatSeat(player, patch=patch), PlayerCombatSeat(_player(patch))),
    )
    fought = next(m for m in survivors if m.card_id == "BG34_142")
    assert fought.raw_attack == enthusiast.base_attack + 9


def test_choral_mrrrglr_takes_everything_in_hand(patch):
    choral = _card(patch, "BG26_354")  # 6/6
    player = _player(
        patch,
        board=[choral],
        hand=[_murloc("a", 2, 3), _murloc("b", 4, 5)],
    )
    survivors = _fight(
        [choral],
        [Minion(card_id="w", base_attack=0, base_health=40, tier=1)],
        patch,
        seats=(PlayerCombatSeat(player, patch=patch), PlayerCombatSeat(_player(patch))),
    )
    fought = next(m for m in survivors if m.card_id == "BG26_354")
    assert (fought.raw_attack, fought.max_health) == (6 + 6, 6 + 8)


def test_an_empty_hand_gives_nothing(patch):
    choral = _card(patch, "BG26_354")
    player = _player(patch, board=[choral])
    survivors = _fight(
        [choral],
        [Minion(card_id="w", base_attack=0, base_health=40, tier=1)],
        patch,
        seats=(PlayerCombatSeat(player, patch=patch), PlayerCombatSeat(_player(patch))),
    )
    fought = next(m for m in survivors if m.card_id == "BG26_354")
    assert (fought.raw_attack, fought.max_health) == (6, 6)


# --------------------------------------------------------------------------- #
# Writing into the hand
# --------------------------------------------------------------------------- #


def test_twilight_tidehunter_pays_the_first_card_in_hand(patch):
    from src.bg_recruitment.blood_gems import play_blood_gem_on

    tidehunter = _card(patch, "BG36_703")
    first, second = _murloc("first"), _murloc("second")
    player = _player(patch, board=[tidehunter], hand=[first, second])
    play_blood_gem_on(player, tidehunter, patch=patch)
    assert (first.raw_attack, first.max_health) == (7, 7)
    assert (second.raw_attack, second.max_health) == (1, 1)


def test_shamanic_tidecaller_pays_murlocs_in_both_places(patch):
    from src.bg_recruitment.blood_gems import play_blood_gem_on

    tidecaller = _card(patch, "BG36_704")
    on_board = _murloc("board")
    in_hand = _murloc("hand")
    beast = Minion(card_id="b", base_attack=1, base_health=1, tier=1, race=Race.BEAST)
    player = _player(patch, board=[tidecaller, on_board, beast], hand=[in_hand])
    play_blood_gem_on(player, on_board, patch=patch)
    assert (in_hand.raw_attack, in_hand.max_health) == (4, 4)
    assert (on_board.raw_attack, on_board.max_health) == (5, 5)  # the Gem too
    assert (beast.raw_attack, beast.max_health) == (1, 1)


def test_futurefin_hands_its_stats_to_the_first_card(patch, triggers):
    futurefin = _card(patch, "BG34_145")  # 7/13
    waiting = _murloc("waiting")
    player = _player(patch, board=[futurefin], hand=[waiting])
    triggers.fire_on_turn_end(player)
    assert (waiting.raw_attack, waiting.max_health) == (8, 14)


# --------------------------------------------------------------------------- #
# The rest
# --------------------------------------------------------------------------- #


def test_primitive_painter_answers_only_small_cards(patch, triggers):
    painter = _card(patch, "BG33_893")
    murloc = _murloc()
    player = _player(patch, board=[painter, murloc])
    big = Minion(card_id="big", base_attack=1, base_health=1, tier=5, race=Race.BEAST)
    triggers.fire_after_friendly_minion_placed(player, big)
    assert murloc.raw_attack == 1

    small = Minion(card_id="small", base_attack=1, base_health=1, tier=3, race=Race.BEAST)
    triggers.fire_after_friendly_minion_placed(player, small)
    assert (murloc.raw_attack, murloc.max_health) == (3, 3)


def test_cousin_errgl_fetches_a_parent(patch, triggers):
    errgl = _card(patch, "BG35_142")
    player = _player(patch, board=[errgl])
    triggers.fire_on_turn_end(player)
    got = next(c for c in player.hand if c is not None)
    assert got.card_id in {"BG35_140", "BG35_141"}


def test_bile_spitter_shares_venomous_when_it_swings(patch):
    spitter = _card(patch, "BG33_318")  # 1/10 Venomous
    mate = _murloc("mate", 1, 30)
    survivors = _fight(
        [spitter, mate],
        [Minion(card_id="w", base_attack=0, base_health=40, tier=1)],
        patch,
    )
    fought = next(m for m in survivors if m.card_id == "mate")
    assert Keyword.VENOMOUS in fought.all_keywords


def test_primalfin_lookout_wants_another_murloc(patch, triggers):
    lookout = _card(patch, "BGS_020")
    player = _player(patch, board=[lookout])
    triggers.fire_on_place(lookout, player, None)
    assert player.pending_choice is None

    player.board.append(_murloc())
    triggers.fire_on_place(lookout, player, None)
    assert player.pending_choice is not None
