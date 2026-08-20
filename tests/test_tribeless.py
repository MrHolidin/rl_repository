"""The tribeless cards, held together by shape rather than by type.

Four say "your X happen more than once", two read the minion a Rally is
swinging at, and the rest are one of a kind.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pytest

from src.bg_catalog.cards import make_minion
from src.bg_catalog.patch_context import PatchContext
from src.bg_core.effects import Keyword, MultiplierKind
from src.bg_core.minion import Minion, Race
from src.bg_lobby.player import PendingChoiceKind, PlayerPhase, PlayerState
from src.bg_recruitment.activate import activate_minion
from src.bg_recruitment.combat_seat import PlayerCombatSeat
from src.bg_recruitment.economy import effective_sell_reward
from src.bg_recruitment.shop_triggers import ShopTriggers
from src.bg_recruitment.standing_bonuses import settle_standing_bonuses
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


def _wall(hp=30, atk=0, **kw):
    return Minion(card_id="wall", base_attack=atk, base_health=hp, tier=1, **kw)


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
    seat = PlayerCombatSeat(player, patch=patch)
    return seat, (seat, PlayerCombatSeat(_player(patch)))


# --------------------------------------------------------------------------- #
# "Your X happen more than once"
# --------------------------------------------------------------------------- #


def test_brann_doubles_battlecries(patch):
    player = _player(patch, board=[_card(patch, "BG_LOE_077")])
    assert ShopTriggers.battlecry_multiplier(player.board) == 2


def test_titus_rivendare_doubles_deathrattles(patch):
    (ability,) = patch.effects["BG25_354"]
    assert ability.effect.kind is MultiplierKind.DEATHRATTLE
    # Two Skeletons become four.
    bonehead = _card(patch, "BG28_300")
    titus = _card(patch, "BG25_354")
    _, deaths = _fight([bonehead, titus], [_wall(hp=2, atk=2)], patch)
    survivors, _ = _fight([bonehead, titus], [_wall(hp=2, atk=2)], patch)
    assert len([m for m in survivors if m.card_id == "BG_ICC_026t"]) == 4


def test_drakkari_enchanter_runs_the_end_of_turn_pass_twice(patch, triggers):
    lullabot = _card(patch, "BG26_146")  # end of turn: gain +1 Health
    triggers.fire_on_turn_end(_player(patch, board=[lullabot]))
    assert lullabot.max_health == lullabot.base_health + 1

    lullabot = _card(patch, "BG26_146")
    triggers.fire_on_turn_end(
        _player(patch, board=[lullabot, _card(patch, "BG26_ICC_901")])
    )
    assert lullabot.max_health == lullabot.base_health + 2


def test_golden_drakkari_runs_it_three_times(patch):
    (ability,) = patch.triple_merge_golden_abilities("BG26_ICC_901")
    assert ability.effect.factor == 3


def test_balinda_casts_a_targeted_spell_twice(patch):
    banana = patch.tavern_spells["BG28_897"]  # give a minion +2/+2
    target = Minion(card_id="t", base_attack=1, base_health=1, tier=1)
    player = _player(patch, board=[target, _card(patch, "BG35_883")])
    cast_tavern_spell(
        player, banana, rng=np.random.default_rng(0), patch=patch, target=target
    )
    assert (target.raw_attack, target.max_health) == (5, 5)


def test_balinda_leaves_an_untargeted_spell_alone(patch):
    ring = patch.tavern_spells["BG28_168"]  # give your minions +1/+1
    target = Minion(card_id="t", base_attack=1, base_health=1, tier=1)
    player = _player(patch, board=[target, _card(patch, "BG35_883")])
    cast_tavern_spell(player, ring, rng=np.random.default_rng(0), patch=patch)
    assert target.raw_attack == 2


# --------------------------------------------------------------------------- #
# Reading the Rally's target
# --------------------------------------------------------------------------- #


def test_heroic_underdog_takes_the_targets_attack(patch):
    underdog = _card(patch, "BG34_604")  # 1/10
    survivors, _ = _fight([underdog], [_wall(hp=8, atk=7)], patch)
    grown = next(m for m in survivors if m.card_id == "BG34_604")
    assert grown.raw_attack == 1 + 7


def test_golden_heroic_underdog_doubles_it(patch):
    (ability,) = patch.triple_merge_golden_abilities("BG34_604")
    assert ability.effect.factor == 2


def test_sindorei_strips_taunt_off_what_it_swings_at(patch):
    shot = _card(patch, "BG25_016")  # Divine Shield, Windfury
    taunt = _wall(hp=40, keywords=frozenset({Keyword.TAUNT, Keyword.REBORN}))
    behind = Minion(card_id="e", base_attack=0, base_health=2, tier=1)
    _, deaths = _fight([shot], [taunt, behind], patch)
    # The Taunt is gone, so the second swing reaches past it.
    assert any(cid == "e" for side, cid in deaths if side == 1)


def test_sindorei_leaves_the_targets_other_keywords(patch):
    shot = _card(patch, "BG25_016")
    shielded = _wall(hp=40, keywords=frozenset({Keyword.TAUNT, Keyword.SHIELD}))
    survivors, _ = _fight([shot], [shielded], patch)
    assert Keyword.SHIELD in shielded.all_keywords  # never printed on the card


# --------------------------------------------------------------------------- #
# One of a kind
# --------------------------------------------------------------------------- #


def test_leeroy_takes_his_killer_with_him(patch):
    leeroy = _card(patch, "BG23_318")  # 6/2
    _, deaths = _fight([leeroy], [_wall(hp=40, atk=9)], patch)
    assert any(cid == "wall" for side, cid in deaths if side == 1)


def test_leeroy_takes_nobody_when_nothing_killed_him(patch):
    """Eaten by his own Stitched Salvager: a death with no killer on the books."""
    leeroy = _card(patch, "BG23_318")
    salvager = _card(patch, "BG31_999")
    _, deaths = _fight([leeroy, salvager], [_wall(hp=5000, atk=0)], patch)
    assert any(cid == "BG23_318" for side, cid in deaths if side == 0)
    assert not any(side == 1 for side, _cid in deaths)


def test_kangors_apprentice_gives_back_two_mechs(patch):
    mech = Minion(card_id="m", base_attack=1, base_health=1, tier=1, race=Race.MECHANICAL)
    other = Minion(card_id="m2", base_attack=1, base_health=1, tier=1, race=Race.MECHANICAL)
    kangor = _card(patch, "BGS_012")
    _, deaths = _fight([mech, other, kangor], [_wall(hp=40, atk=4)], patch)
    # Each Mech dies twice: once for itself, once for the copy Kangor gave back.
    dead = [cid for side, cid in deaths if side == 0]
    assert dead.count("m") == 2 and dead.count("m2") == 2


def test_nomi_pays_the_tavern_after_an_elemental(patch, triggers):
    nomi = _card(patch, "BGS_104")
    elemental = Minion(
        card_id="e", base_attack=1, base_health=1, tier=1, race=Race.ELEMENTAL
    )
    player = _player(patch, board=[nomi, elemental])
    triggers.fire_after_friendly_minion_placed(player, elemental)
    offered = Minion(
        card_id="s", base_attack=1, base_health=1, tier=1, race=Race.ELEMENTAL
    )
    player.shop[0] = offered
    settle_standing_bonuses(player)
    assert (offered.raw_attack, offered.max_health) == (5, 5)


def test_nomi_is_quiet_after_anything_else(patch, triggers):
    nomi = _card(patch, "BGS_104")
    beast = Minion(card_id="b", base_attack=1, base_health=1, tier=1, race=Race.BEAST)
    player = _player(patch, board=[nomi, beast])
    triggers.fire_after_friendly_minion_placed(player, beast)
    offered = Minion(
        card_id="s", base_attack=1, base_health=1, tier=1, race=Race.ELEMENTAL
    )
    player.shop[0] = offered
    settle_standing_bonuses(player)
    assert offered.raw_attack == 1


def test_humongozz_raises_tavern_spells_while_it_stands(patch):
    assert tavern_spell_bonus(_player(patch)) == (0, 0)
    gozz = _card(patch, "BG32_341")
    assert tavern_spell_bonus(_player(patch, board=[gozz])) == (1, 2)


def test_humongozz_stops_paying_once_it_leaves(patch):
    """An aura, not a promise: the "this game" cards keep paying, this one does not."""
    gozz = _card(patch, "BG32_341")
    player = _player(patch, board=[gozz])
    player.board = []
    assert tavern_spell_bonus(player) == (0, 0)


def test_cataclysmic_harbinger_copies_the_last_spell_cast(patch, triggers):
    harbinger = _card(patch, "BG35_123")
    player = _player(patch, board=[harbinger])
    player.last_tavern_spell_cast = "BG28_810"  # Tavern Coin
    triggers.fire_on_turn_end(player)
    held = [c for c in player.hand if c is not None]
    assert [c.card_id for c in held] == ["BG28_810"]


def test_cataclysmic_harbinger_copies_nothing_before_a_cast(patch, triggers):
    harbinger = _card(patch, "BG35_123")
    player = _player(patch, board=[harbinger])
    triggers.fire_on_turn_end(player)
    assert all(c is None for c in player.hand)


def test_golden_harbinger_hands_over_two(patch):
    (ability,) = patch.triple_merge_golden_abilities("BG35_123")
    assert ability.effect.count == 2


def test_rodeo_performer_discovers_a_tavern_spell(patch, triggers):
    performer = _card(patch, "BG28_550")
    player = _player(patch, board=[performer])
    triggers.fire_on_place(player=player, placed=performer, shop_excluded_race=None)
    pc = player.pending_choice
    assert pc is not None and pc.kind is PendingChoiceKind.SPELL_DISCOVER
    assert all(cid in patch.tavern_spells for cid in pc.options)


def test_golden_rodeo_performer_discovers_twice(patch):
    (ability,) = patch.triple_merge_golden_abilities("BG28_550")
    assert ability.effect.repeats == 2


def test_highkeeper_ra_fetches_tier_six_on_its_swing(patch):
    player = _player(patch)
    ra = _card(patch, "BG34_319")
    player.board = [ra]
    seat, seats = _seat(patch, player)
    _fight([ra], [_wall(hp=30, atk=1)], patch, seats=seats)
    assert seat.hand_adds
    assert all(patch.templates[cid].tier == 6 for cid in seat.hand_adds)


def test_highkeeper_ra_fetches_on_the_battlecry_too(patch, triggers):
    ra = _card(patch, "BG34_319")
    player = _player(patch, board=[ra])
    triggers.fire_on_place(player=player, placed=ra, shop_excluded_race=None)
    held = [c for c in player.hand if c is not None]
    assert [c.tier for c in held] == [6]


def test_tyrael_sets_stats_rather_than_adding_to_them(patch):
    other = Minion(card_id="o", base_attack=60, base_health=60, tier=1)
    tyrael = _card(patch, "BG36_356")
    player = _player(patch, board=[tyrael, other])
    activate_minion(
        player, 0, rng=np.random.default_rng(0), patch=patch, buff_target=other
    )
    assert (other.raw_attack, other.max_health) == (40, 40)  # down, not up
    assert player.gold == 8


def test_tyrael_will_not_set_its_own_stats(patch):
    tyrael = _card(patch, "BG36_356")  # 8/8
    player = _player(patch, board=[tyrael])
    activate_minion(
        player, 0, rng=np.random.default_rng(0), patch=patch, buff_target=tyrael
    )
    assert (tyrael.raw_attack, tyrael.max_health) == (8, 8)


def test_tortollan_blue_shell_is_worth_more_after_a_loss(patch):
    shell = _card(patch, "BG24_018")
    player = _player(patch, board=[shell], last_combat_won=True)
    assert effective_sell_reward(shell, player) == 1
    player.last_combat_won = False
    assert effective_sell_reward(shell, player) == 5


def test_a_flat_printed_sell_price_still_works(patch):
    """Freedealing Gambler keeps its catalog-derived price, unconditionally."""
    classic = PatchContext.load(Path("data/bgcore/19_6_0_74257"))
    gambler = make_minion("BGS_049", patch=classic)
    assert effective_sell_reward(gambler) == 3
