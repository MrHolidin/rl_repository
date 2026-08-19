"""Tier-1 bindings of the 36.2.0 package, played rather than inspected.

Every test here builds the *real* card out of the 36.2.0 catalog — no synthetic
stand-in with a hand-written ability — so a binding that names the wrong token
or hangs an effect off the wrong trigger fails here and not in a training run.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pytest

from src.bg_catalog.cards import make_minion
from src.bg_catalog.patch_context import PatchContext
from src.bg_core.effects import Keyword, Trigger
from src.bg_core.minion import Minion, Race
from src.bg_lobby.player import PlayerPhase, PlayerState
from src.bg_recruitment import spellcraft
from src.bg_recruitment.blood_gems import is_blood_gem
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
        tavern_tier=1,
        board=list(board),
        shop=[None] * 7,
        hand=[None] * 10,
        phase=PlayerPhase.SHOP,
        shop_actions_used=0,
        ruleset=patch.meta.ruleset,
    )
    base.update(kw)
    return PlayerState(**base)


def _fight(board_0, board_1, patch, seed: int = 0):
    """Fight and report both what side 0 had left and everything that died.

    Two views because a summoned token usually does not survive the fight that
    summoned it: the death log is the only place a one-swing Beetle shows up.
    """
    survivors_0: List[Minion] = []
    deaths: List[tuple] = []
    simulate_battle(
        board_0,
        board_1,
        p0_has_initiative=True,
        rng=np.random.default_rng(seed),
        patch=patch,
        p0_board_out=survivors_0,
        death_log=deaths,
    )
    return survivors_0, deaths


def _wall(hp: int = 30, atk: int = 0) -> Minion:
    """A punching bag: big enough to outlive the fight, harmless if atk=0."""
    return Minion(card_id="wall", base_attack=atk, base_health=hp, tier=1)


# --------------------------------------------------------------------------- #
# Deathrattles that summon a token
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "card_id,token_id,count,stats",
    [
        ("BG31_803", "BG28_603t", 1, (2, 2)),  # Buzzing Vermin -> Beetle
        ("BG29_611", "BG_BOT_312t", 1, (1, 1)),  # Cord Puller -> Microbot
        ("BG28_300", "BG_ICC_026t", 2, (1, 1)),  # Harmless Bonehead -> 2 Skeletons
    ],
)
def test_deathrattle_summons_its_token(patch, card_id, token_id, count, stats):
    dies = _card(patch, card_id)
    killer = _wall(hp=30, atk=20)
    _, deaths = _fight([dies], [killer], patch)
    summoned = [cid for side, cid in deaths if side == 0 and cid == token_id]
    assert len(summoned) == count
    token = patch.templates[token_id]
    assert (token.base_attack, token.base_health) == stats


def test_the_summoned_tokens_are_real_catalog_cards(patch):
    """A token id that the catalog does not carry summons nothing at all."""
    for token_id in ("BG28_603t", "BG_BOT_312t", "BG_ICC_026t", "BG36_200t"):
        assert token_id in patch.templates


# --------------------------------------------------------------------------- #
# Rally — "whenever this attacks"
# --------------------------------------------------------------------------- #


def test_glim_guardian_gains_attack_on_its_swing(patch):
    guardian = _card(patch, "BG29_888")  # 1/4 Dragon, Rally: gain +2 Attack
    assert guardian.base_attack == 1
    survivors, _ = _fight([guardian], [_wall(hp=30)], patch)
    swung = next(m for m in survivors if m.card_id == "BG29_888")
    # It attacked at least once and each swing added +2.
    assert swung.raw_attack >= 3


def test_flittering_bat_summons_a_beast_when_it_attacks(patch):
    bat = _card(patch, "BG36_200")  # 1/3 Beast, Rally: summon a 1/1 Beast
    survivors, deaths = _fight([bat], [_wall(hp=30)], patch)
    on_board = {m.card_id for m in survivors} | {cid for side, cid in deaths if side == 0}
    assert "BG36_200t" in on_board


def test_tusked_camper_gems_itself_when_it_attacks(patch):
    camper = _card(patch, "BG33_886")  # 2/3 Quilboar, Rally: play a Blood Gem on itself
    base = (camper.base_attack, camper.base_health)
    survivors, _ = _fight([camper], [_wall(hp=30)], patch)
    swung = next(m for m in survivors if m.card_id == "BG33_886")
    assert (swung.raw_attack, swung.max_health) > base


# --------------------------------------------------------------------------- #
# Shop triggers
# --------------------------------------------------------------------------- #


def test_lullabot_gains_health_at_end_of_turn(patch, triggers):
    bot = _card(patch, "BG26_146")  # 2/2 Mech, Magnetic, end of turn: +1 Health
    player = _player(patch, [bot])
    before = bot.max_health
    triggers.fire_on_turn_end(player)
    assert bot.max_health == before + 1


def test_lullabot_keeps_its_printed_magnetic(patch):
    assert Keyword.MAGNETIC in _card(patch, "BG26_146").keywords


def test_razorfen_geomancer_gets_two_blood_gems(patch, triggers):
    geomancer = _card(patch, "BG20_100")  # Battlecry: get 2 Blood Gems
    player = _player(patch, [geomancer])
    triggers.fire_on_place(geomancer, player, None)
    assert sum(1 for c in player.hand if c is not None and is_blood_gem(c)) == 2


def test_wrath_weaver_pays_health_for_a_demon(patch, triggers):
    weaver = _card(patch, "BGS_004")  # 1/3 Demon
    demon = Minion(card_id="d", base_attack=1, base_health=1, tier=1, race=Race.DEMON)
    player = _player(patch, [weaver])
    triggers.fire_after_friendly_minion_placed(player, demon)
    assert (weaver.raw_attack, weaver.max_health) == (3, 5)
    assert player.health == 29


def test_molten_rock_grows_on_elementals_only(patch, triggers):
    rock = _card(patch, "BGS_127")  # 3/3 Elemental, +1 Health per Elemental played
    player = _player(patch, [rock])
    beast = Minion(card_id="b", base_attack=1, base_health=1, tier=1, race=Race.BEAST)
    triggers.fire_after_friendly_minion_placed(player, beast)
    assert rock.max_health == 3
    ele = Minion(card_id="e", base_attack=1, base_health=1, tier=1, race=Race.ELEMENTAL)
    triggers.fire_after_friendly_minion_placed(player, ele)
    assert rock.max_health == 4


def test_mini_myrmidon_hands_out_its_spellcraft_spell(patch, triggers):
    myrmidon = _card(patch, "BG23_000")  # Spellcraft: +2 Attack until next turn
    player = _player(patch, [myrmidon])
    triggers.fire_on_turn_start(player)
    spells = [c for c in player.hand if c is not None and spellcraft.is_spellcraft_spell(c)]
    assert len(spells) == 1


# --------------------------------------------------------------------------- #
# Cards that need no binding, and the queue that still does
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "card_id,keywords",
    [
        ("BGS_119", {Keyword.SHIELD, Keyword.WINDFURY}),  # Crackling Cyclone
        ("BG25_001", {Keyword.TAUNT, Keyword.REBORN}),  # Risen Rider
    ],
)
def test_keyword_only_cards_carry_their_keywords_unbound(patch, card_id, keywords):
    card = _card(patch, card_id)
    assert keywords <= card.keywords
    assert card.abilities == ()


def test_bindings_only_name_cards_the_catalog_has(patch):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "bindings_248348", PATCH_DIR / "bindings.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    for card_id in mod.EFFECTS:
        assert card_id in patch.templates, card_id
        # A token can carry an ability of its own (the Sewer Rat leaves a
        # Half-Shell), so a binding key is a pool card *or* a declared token.
        assert card_id in patch.pool_ids or card_id in mod.TOKEN_IDS, card_id
    for card_id in mod.KEYWORD_ONLY_POOL_IDS:
        assert card_id in patch.pool_ids, card_id
    for card_id in mod.UNBOUND_NEEDS_ENGINE:
        assert card_id in patch.templates, card_id


# --------------------------------------------------------------------------- #
# Mechanics added for tier 1
# --------------------------------------------------------------------------- #


def test_rot_hide_gnoll_grows_with_each_friendly_death(patch):
    gnoll = _card(patch, "BG25_013")  # 1/4 Undead, +1 Attack per friendly dead
    gnoll.bonus_health += 50  # outlive the fight; its Attack is what is on trial
    # 1/1s that die on their own swing to the wall's retaliation — two deaths,
    # no dependence on who the wall picks.
    fodder = [Minion(card_id=f"f{i}", base_attack=1, base_health=1, tier=1) for i in range(2)]
    survivors, _ = _fight([gnoll] + fodder, [_wall(hp=15, atk=1)], patch)
    grown = next(m for m in survivors if m.card_id == "BG25_013")
    assert grown.raw_attack == gnoll.base_attack + 2


def test_rot_hide_gnoll_starts_each_combat_at_its_printed_attack(patch):
    """The count is per combat: the buff lives on the copy, not on the board."""
    gnoll = _card(patch, "BG25_013")
    fodder = Minion(card_id="f", base_attack=0, base_health=1, tier=1)
    _fight([gnoll, fodder], [_wall(hp=40, atk=1)], patch)
    assert gnoll.raw_attack == gnoll.base_attack


def test_river_skipper_sells_into_a_tier_one_minion(patch, triggers):
    skipper = _card(patch, "BG33_140")
    player = _player(patch, [skipper], tavern_tier=4)
    triggers.fire_on_sell(skipper, player)
    got = [c for c in player.hand if c is not None]
    assert len(got) == 1
    # Tier 1 exactly, not "anything up to the seat's tier 4".
    assert got[0].tier == 1


def test_southsea_busker_pays_next_turn_not_this_one(patch, triggers):
    busker = _card(patch, "BG26_135")
    player = _player(patch, [busker], gold=3)
    triggers.fire_on_place(busker, player, None)
    assert player.gold == 3 and player.gold_next_turn == 1


def test_banked_gold_is_paid_once_and_cleared(patch):
    from src.bg_recruitment.economy import start_of_turn_gold

    player = _player(patch, gold=0, gold_next_turn=1)
    assert start_of_turn_gold(player, 4) == patch.meta.ruleset.gold_for_round(4) + 1
    assert player.gold_next_turn == 0
    assert start_of_turn_gold(player, 5) == patch.meta.ruleset.gold_for_round(5)


def test_prisonguard_activates_onto_another_minion(patch):
    from src.bg_recruitment.activate import activate_cost, activate_minion

    guard = _card(patch, "BG36_345")
    friend = Minion(card_id="f", base_attack=1, base_health=1, tier=1)
    player = _player(patch, [guard, friend], gold=5)
    assert activate_cost(guard) == 1
    activate_minion(
        player, 0, rng=np.random.default_rng(0), patch=patch, buff_target=friend
    )
    assert (friend.raw_attack, friend.max_health) == (4, 4)
    assert player.gold == 4
    assert (guard.raw_attack, guard.max_health) == (guard.base_attack, guard.base_health)


def test_prisonguard_activate_is_once_per_turn(patch):
    from src.bg_recruitment.activate import ActivateNotAllowed, activate_minion

    guard = _card(patch, "BG36_345")
    friend = Minion(card_id="f", base_attack=1, base_health=1, tier=1)
    player = _player(patch, [guard, friend], gold=5)
    activate_minion(player, 0, rng=np.random.default_rng(0), patch=patch, buff_target=friend)
    with pytest.raises(ActivateNotAllowed):
        activate_minion(player, 0, rng=np.random.default_rng(0), patch=patch, buff_target=friend)


def test_fleeing_fugitive_grows_when_a_spell_hits_it(patch):
    from src.bg_recruitment.blood_gems import play_blood_gem_on

    fugitive = _card(patch, "BG36_921")
    player = _player(patch, [fugitive])
    before = fugitive.max_health
    play_blood_gem_on(player, fugitive)
    # +1/+1 from the Gem itself, and +1 Health for the spell that carried it.
    assert fugitive.max_health == before + 2
    assert fugitive.raw_attack == fugitive.base_attack + 1


def test_a_spell_on_someone_else_leaves_the_fugitive_alone(patch):
    from src.bg_recruitment.blood_gems import play_blood_gem_on

    fugitive = _card(patch, "BG36_921")
    other = Minion(card_id="o", base_attack=1, base_health=1, tier=1)
    player = _player(patch, [fugitive, other])
    play_blood_gem_on(player, other)
    assert fugitive.max_health == fugitive.base_health


def test_scarlet_survivor_shields_itself_at_six_attack(patch):
    survivor = _card(patch, "BG35_814")  # 3/3 Dragon
    assert Keyword.SHIELD not in survivor.keywords
    survivor.bonus_attack += 3  # 3 -> 6
    from src.bg_recruitment.shop_auras import refresh_attack_thresholds

    refresh_attack_thresholds([survivor])
    assert Keyword.SHIELD in survivor.keywords and survivor.has_shield


def test_scarlet_survivor_below_the_threshold_stays_bare(patch):
    from src.bg_recruitment.shop_auras import refresh_attack_thresholds

    survivor = _card(patch, "BG35_814")
    survivor.bonus_attack += 2  # 3 -> 5
    refresh_attack_thresholds([survivor])
    assert Keyword.SHIELD not in survivor.keywords


def test_a_popped_shield_does_not_re_arm_on_the_next_recount(patch):
    """The latch fires once; without that, every recount would refresh it."""
    from src.bg_recruitment.shop_auras import refresh_attack_thresholds

    survivor = _card(patch, "BG35_814")
    survivor.bonus_attack += 3
    refresh_attack_thresholds([survivor])
    survivor.has_shield = False  # popped in the fight
    refresh_attack_thresholds([survivor])
    assert not survivor.has_shield


def test_scarlet_survivor_shields_itself_inside_the_combat(patch):
    """The latch is combat's too, not only the shop's — and it fires on the
    copy, leaving the seat's own board untouched the way combat always does."""
    survivor = _card(patch, "BG35_814")
    survivor.bonus_attack += 3  # 6 attack going in, never refreshed in the shop
    survivor.bonus_health += 50
    assert Keyword.SHIELD not in survivor.keywords
    survivors, _ = _fight([survivor], [_wall(hp=15, atk=1)], patch)
    fought = next(m for m in survivors if m.card_id == "BG35_814")
    assert Keyword.SHIELD in fought.keywords
    assert Keyword.SHIELD not in survivor.keywords


def test_flighty_scout_summons_itself_out_of_hand(patch):
    from src.bg_recruitment.combat_seat import PlayerCombatSeat

    scout = _card(patch, "BG32_330")  # 3/3 Murloc
    player = _player(patch, board=[_wall(hp=30)])
    player.hand[0] = scout
    seat = PlayerCombatSeat(player)
    survivors: List[Minion] = []
    simulate_battle(
        [player.board[0]],
        [_wall(hp=1)],
        p0_has_initiative=True,
        rng=np.random.default_rng(0),
        patch=patch,
        p0_board_out=survivors,
        seats=(seat, PlayerCombatSeat(_player(patch))),
    )
    assert any(m.card_id == "BG32_330" for m in survivors)


def test_a_scout_left_in_the_shop_summons_nothing(patch):
    """No seat, no hand — the seatless combat API keeps working unchanged."""
    survivors, _ = _fight([_wall(hp=30)], [_wall(hp=1)], patch)
    assert all(m.card_id != "BG32_330" for m in survivors)


def test_aureate_laureate_is_born_golden(patch):
    laureate = _card(patch, "BG32_236")
    assert laureate.is_golden


def test_three_laureates_do_not_make_a_triple(patch):
    """Golden copies never merge, which is the card's "no Triple Reward"."""
    from src.bg_recruitment.triples import resolve_triples_loop

    player = _player(patch, [_card(patch, "BG32_236") for _ in range(3)])
    resolve_triples_loop(player, patch=patch)
    assert len(player.board) == 3
