"""Tier-2 bindings of the 36.2.0 package, played rather than inspected.

Same rule as the tier-1 file: every card is built out of the real catalog, so a
binding that names the wrong tribe or hangs an effect off the wrong trigger
fails here rather than in a training run.
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
from src.bg_recruitment.blood_gems import is_blood_gem
from src.bg_recruitment.combat_seat import PlayerCombatSeat
from src.bg_recruitment.lockbox import find_lockbox, is_lockbox
from src.bg_recruitment.shop_triggers import ShopTriggers
from tests.minibg_helpers import simulate_battle

PATCH_DIR = Path("data/bgcore/36_2_0_248348")


@pytest.fixture(scope="module")
def patch() -> PatchContext:
    return PatchContext.load(PATCH_DIR)


@pytest.fixture()
def triggers(patch):
    return ShopTriggers(np.random.default_rng(0), patch=patch)


def _card(patch: PatchContext, card_id: str) -> Minion:
    return make_minion(card_id, patch=patch)


def _player(patch: PatchContext, board=(), **kw) -> PlayerState:
    base = dict(
        health=30,
        gold=10,
        tavern_tier=2,
        board=list(board),
        shop=[None] * 7,
        hand=[None] * 10,
        phase=PlayerPhase.SHOP,
        shop_actions_used=0,
        ruleset=patch.meta.ruleset,
    )
    base.update(kw)
    return PlayerState(**base)


def _wall(hp: int = 30, atk: int = 0) -> Minion:
    return Minion(card_id="wall", base_attack=atk, base_health=hp, tier=1)


def _fight(board_0, board_1, patch, seed: int = 0, seats=None):
    survivors: List[Minion] = []
    deaths: List[tuple] = []
    kwargs = {}
    if seats is not None:
        kwargs["seats"] = seats
    simulate_battle(
        board_0,
        board_1,
        p0_has_initiative=True,
        rng=np.random.default_rng(seed),
        patch=patch,
        p0_board_out=survivors,
        death_log=deaths,
        **kwargs,
    )
    return survivors, deaths


# --------------------------------------------------------------------------- #
# Shop
# --------------------------------------------------------------------------- #


def test_sellemental_pays_a_water_droplet_when_sold(patch, triggers):
    sellemental = _card(patch, "BGS_115")
    player = _player(patch, [sellemental])
    triggers.fire_on_sell(sellemental, player)
    got = [c for c in player.hand if c is not None]
    assert len(got) == 1 and got[0].card_id == "BGS_115t"


def test_bilgewater_breakout_hands_over_a_lockbox(patch, triggers):
    breakout = _card(patch, "BG36_520")
    player = _player(patch, [breakout])
    triggers.fire_on_place(breakout, player, None)
    assert find_lockbox(player) is not None


def test_a_second_breakout_hurries_the_lockbox_instead_of_adding_one(patch, triggers):
    """A seat only ever holds one, so the second copy pays in time, not cards."""
    breakout = _card(patch, "BG36_520")
    player = _player(patch, [breakout])
    triggers.fire_on_place(breakout, player, None)
    first = player.hand[find_lockbox(player)].turns_until_open
    triggers.fire_on_place(breakout, player, None)
    idx = find_lockbox(player)
    assert sum(1 for c in player.hand if c is not None and is_lockbox(c)) == 1
    assert player.hand[idx].turns_until_open == first - 1


def test_shell_collector_hands_over_a_tavern_coin(patch, triggers):
    collector = _card(patch, "BG23_002")
    player = _player(patch, [collector])
    triggers.fire_on_place(collector, player, None)
    got = [c for c in player.hand if c is not None]
    assert len(got) == 1
    assert got[0].card_id == "BG28_810" and got[0].is_tavern_spell


def test_electric_synthesizer_buffs_other_dragons_when_played(patch, triggers):
    synth = _card(patch, "BG26_963")
    dragon = Minion(card_id="d", base_attack=1, base_health=1, tier=1, race=Race.DRAGON)
    beast = Minion(card_id="b", base_attack=1, base_health=1, tier=1, race=Race.BEAST)
    player = _player(patch, [synth, dragon, beast])
    triggers.fire_on_place(synth, player, None)
    assert (dragon.raw_attack, dragon.max_health) == (2, 2)
    assert (beast.raw_attack, beast.max_health) == (1, 1)
    assert (synth.raw_attack, synth.max_health) == (synth.base_attack, synth.base_health)


# --------------------------------------------------------------------------- #
# Start of Combat
# --------------------------------------------------------------------------- #


def test_electric_synthesizer_buffs_again_at_start_of_combat(patch):
    synth = _card(patch, "BG26_963")
    dragon = Minion(card_id="d", base_attack=1, base_health=20, tier=1, race=Race.DRAGON)
    survivors, _ = _fight([synth, dragon], [_wall(hp=40)], patch)
    fought = next(m for m in survivors if m.card_id == "d")
    assert (fought.raw_attack, fought.max_health) == (2, 21)


def test_humming_bird_gives_your_beasts_attack_for_the_combat(patch):
    bird = _card(patch, "BG26_805")  # 1/4 Beast
    beast = Minion(card_id="b", base_attack=1, base_health=20, tier=1, race=Race.BEAST)
    murloc = Minion(card_id="m", base_attack=1, base_health=20, tier=1, race=Race.MURLOC)
    survivors, _ = _fight([bird, beast, murloc], [_wall(hp=60)], patch)
    assert next(m for m in survivors if m.card_id == "b").raw_attack == 2
    assert next(m for m in survivors if m.card_id == "m").raw_attack == 1


def test_the_start_of_combat_buff_does_not_follow_the_board_home(patch):
    """Combat runs on copies; the seat's own Beast is untouched afterwards."""
    bird = _card(patch, "BG26_805")
    beast = Minion(card_id="b", base_attack=1, base_health=20, tier=1, race=Race.BEAST)
    _fight([bird, beast], [_wall(hp=40)], patch)
    assert beast.raw_attack == 1


def test_paper_drake_buffs_only_the_left_most_dragon(patch):
    drake = _card(patch, "BG29_810")  # 2/3 Dragon, itself left-most here
    second = Minion(card_id="d2", base_attack=1, base_health=20, tier=1, race=Race.DRAGON)
    survivors, _ = _fight([drake, second], [_wall(hp=40)], patch)
    lead = next(m for m in survivors if m.card_id == "BG29_810")
    assert (lead.raw_attack, lead.max_health) == (drake.base_attack + 1, drake.base_health + 2)
    assert Keyword.WINDFURY in lead.all_keywords
    tail = next(m for m in survivors if m.card_id == "d2")
    assert Keyword.WINDFURY not in tail.all_keywords


def test_paper_drake_skips_past_a_non_dragon_to_find_one(patch):
    drake = _card(patch, "BG29_810")
    mech = Minion(card_id="mech", base_attack=1, base_health=20, tier=1, race=Race.MECHANICAL)
    survivors, _ = _fight([mech, drake], [_wall(hp=40)], patch)
    assert Keyword.WINDFURY not in next(m for m in survivors if m.card_id == "mech").all_keywords
    assert Keyword.WINDFURY in next(m for m in survivors if m.card_id == "BG29_810").all_keywords


# --------------------------------------------------------------------------- #
# Combat
# --------------------------------------------------------------------------- #


def test_scarlet_skull_leaves_a_friendly_undead_bigger(patch):
    skull = _card(patch, "BG25_022")  # 2/1 Undead, Reborn
    undead = Minion(card_id="u", base_attack=10, base_health=40, tier=1, race=Race.UNDEAD)
    survivors, _ = _fight([skull, undead], [_wall(hp=25, atk=2)], patch)
    grown = next(m for m in survivors if m.card_id == "u")
    assert (grown.raw_attack, grown.max_health) > (10, 40)


def test_scarlet_skull_will_not_buff_a_minion_of_another_tribe(patch):
    skull = _card(patch, "BG25_022")
    beast = Minion(card_id="b", base_attack=10, base_health=40, tier=1, race=Race.BEAST)
    survivors, _ = _fight([skull, beast], [_wall(hp=25, atk=2)], patch)
    untouched = next(m for m in survivors if m.card_id == "b")
    assert (untouched.raw_attack, untouched.max_health) == (10, 40)


def test_roadboar_hands_the_seat_a_gem_when_it_attacks(patch):
    roadboar = _card(patch, "BG20_101")  # 2/4 Quilboar, Rally: get a Blood Gem
    player = _player(patch, [roadboar])
    seat = PlayerCombatSeat(player)
    _fight([roadboar], [_wall(hp=30)], patch, seats=(seat, PlayerCombatSeat(_player(patch))))
    assert sum(1 for c in player.hand if c is not None and is_blood_gem(c)) >= 1


def test_a_seatless_combat_still_runs_a_rally_that_gives_gems(patch):
    """The recording seat collects them and applies nothing, as it always has."""
    survivors, _ = _fight([_card(patch, "BG20_101")], [_wall(hp=30)], patch)
    assert any(m.card_id == "BG20_101" for m in survivors)
