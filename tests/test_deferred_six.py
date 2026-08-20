"""The six cards deferred for a missing piece, now that the pieces exist.

Each was set aside for one thing the engine could not say. Every one of those
was later built for a different card — the tribe Discover for Maw Caster, the
tavern Rally for the Fishbait, the refilling countdown for Felboar — so these
are mostly bindings on top of work already done.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.bg_catalog.cards import make_minion
from src.bg_catalog.patch_context import PatchContext
from src.bg_core.effects import Keyword
from src.bg_core.minion import Minion, Race
from src.bg_lobby.player import PendingChoiceKind, PlayerPhase, PlayerState
from src.bg_recruitment.activate import activate_minion
from src.bg_recruitment.discover import resolve_discover_pick
from src.bg_recruitment.shop_triggers import ShopTriggers
from src.bg_recruitment.spellcraft import (
    give_spellcraft_spell,
    play_spellcraft_spell_from_hand,
)
from src.bg_recruitment.tavern_spells import play_tavern_spell_from_hand

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


def _plain(card_id="m", atk=1, hp=1, race=None) -> Minion:
    return Minion(card_id=card_id, base_attack=atk, base_health=hp, tier=1, race=race)


# --------------------------------------------------------------------------- #
# A targeted spell reaches the counter
# --------------------------------------------------------------------------- #


def test_a_tavern_spell_can_be_cast_at_a_minion_in_the_tavern(patch):
    player = _player(patch)
    offered = _plain("s")
    player.shop[2] = offered
    player.hand[0] = patch.tavern_spells["BG28_897"]  # give a minion +2/+2
    play_tavern_spell_from_hand(
        player, 0, rng=np.random.default_rng(0), patch=patch, target_shop_index=2
    )
    assert (offered.raw_attack, offered.max_health) == (3, 3)


def test_a_buff_bought_off_the_counter_comes_with_the_minion(patch):
    from src.bg_recruitment import economy

    player = _player(patch)
    offered = _plain("BG25_008", 4, 2, Race.UNDEAD)
    player.shop[0] = offered
    player.hand[0] = patch.tavern_spells["BG28_897"]
    play_tavern_spell_from_hand(
        player, 0, rng=np.random.default_rng(0), patch=patch, target_shop_index=0
    )
    economy.buy_from_shop(
        player, 0, patch=patch, on_bought=lambda _m, _p: None, on_triples=lambda _p: None
    )
    held = next(c for c in player.hand if c is not None)
    assert (held.raw_attack, held.max_health) == (6, 4)


def test_a_board_target_still_wins_when_both_are_named(patch):
    player = _player(patch, board=[_plain("b")])
    offered = _plain("s")
    player.shop[0] = offered
    player.hand[0] = patch.tavern_spells["BG28_897"]
    play_tavern_spell_from_hand(
        player, 0, rng=np.random.default_rng(0), patch=patch,
        target_board_index=0, target_shop_index=0,
    )
    assert player.board[0].raw_attack == 3
    assert offered.raw_attack == 1


# --------------------------------------------------------------------------- #
# Imposing Percussionist
# --------------------------------------------------------------------------- #


def test_imposing_percussionist_discovers_a_demon_and_charges_its_tier(patch, triggers):
    percussionist = _card(patch, "BG26_525")
    player = _player(patch, board=[percussionist])
    triggers.fire_on_place(
        player=player, placed=percussionist, shop_excluded_race=None
    )
    pc = player.pending_choice
    assert pc is not None and pc.kind is PendingChoiceKind.DISCOVER_TRIBE
    assert all(patch.templates[cid].race is Race.DEMON for cid in pc.options)

    tier = patch.templates[pc.options[0]].tier
    resolve_discover_pick(
        player, 0, None, rng=np.random.default_rng(0),
        on_after_placed=lambda _p, _m: None, patch=patch,
    )
    assert player.health == 30 - tier


def test_golden_percussionist_discovers_two_and_pays_for_both(patch):
    discover, damage = patch.triple_merge_golden_abilities("BG26_525")
    assert discover.effect.repeats == 2
    # One hit per pick, each for that pick's own Tier — "their Tiers".
    assert damage.effect.per_tier == 1


# --------------------------------------------------------------------------- #
# Deft Deserter
# --------------------------------------------------------------------------- #


def test_deft_deserter_pays_the_whole_tavern_and_hands_out_a_keyword(patch):
    deserter = _card(patch, "BG36_621")
    player = _player(patch, board=[deserter])
    for i in range(4):
        player.shop[i] = _plain(f"s{i}")
    activate_minion(player, 0, rng=np.random.default_rng(1), patch=patch)
    offered = [player.shop[i] for i in range(4)]
    assert all((m.raw_attack, m.max_health) == (9, 9) for m in offered)
    wanted = {Keyword.TAUNT, Keyword.SHIELD, Keyword.WINDFURY}
    assert all(m.all_keywords & wanted for m in offered)
    assert player.gold == 9


def test_deft_deserter_rolls_the_keyword_per_minion(patch):
    """"Taunt, Divine Shield, **or** Windfury" — a tavern comes out mixed."""
    deserter = _card(patch, "BG36_621")
    player = _player(patch, board=[deserter])
    for i in range(6):
        player.shop[i] = _plain(f"s{i}")
    activate_minion(player, 0, rng=np.random.default_rng(1), patch=patch)
    granted = {
        next(iter(m.all_keywords)) for m in player.shop[:6] if m.all_keywords
    }
    assert len(granted) > 1


def test_golden_deft_deserter_doubles_the_stats_only(patch):
    (ability,) = patch.triple_merge_golden_abilities("BG36_621")
    assert (ability.effect.attack, ability.effect.health) == (16, 16)
    assert len(ability.effect.keyword_choices) == 3


# --------------------------------------------------------------------------- #
# Trigger a friendly's ability
# --------------------------------------------------------------------------- #


def test_sky_hatch_runaway_fires_a_friendlys_rally(patch):
    runaway = _card(patch, "BG36_243")
    hyena = _card(patch, "BG36_210")  # Rally: summon a Tasty Lobster
    player = _player(patch, board=[runaway, hyena])
    activate_minion(
        player, 0, rng=np.random.default_rng(0), patch=patch, buff_target=hyena
    )
    assert "BG36_202" in [m.card_id for m in player.board]
    assert player.gold == 9


def test_sky_hatch_runaway_fires_a_rally_that_writes_to_the_seat(patch):
    runaway = _card(patch, "BG36_243")
    devastator = _card(patch, "BG33_323")  # Rally: your Undead +2 Attack this game
    player = _player(patch, board=[runaway, devastator])
    activate_minion(
        player, 0, rng=np.random.default_rng(0), patch=patch, buff_target=devastator
    )
    assert player.standing_bonuses


def test_kelp_keeper_fires_a_friendlys_battlecry_again(patch, triggers):
    keeper = _card(patch, "BG36_701")
    swarmer = _card(patch, "BG25_011")  # Battlecry: your Undead +1 Attack this game
    player = _player(patch, board=[keeper, swarmer])
    triggers.fire_on_place(player=player, placed=swarmer, shop_excluded_race=None)
    (banked,) = player.standing_bonuses.values()
    assert banked == (1, 0)
    activate_minion(
        player, 0, rng=np.random.default_rng(0), patch=patch, buff_target=swarmer
    )
    (banked,) = player.standing_bonuses.values()
    assert banked == (2, 0)


def test_a_retrigger_with_nobody_named_does_nothing(patch):
    keeper = _card(patch, "BG36_701")
    player = _player(patch, board=[keeper])
    activate_minion(player, 0, rng=np.random.default_rng(0), patch=patch)
    assert not player.standing_bonuses


@pytest.mark.parametrize("card_id", ["BG36_243", "BG36_701"])
def test_golden_retriggers_fire_twice(patch, card_id):
    (ability,) = patch.triple_merge_golden_abilities(card_id)
    assert ability.effect.repeats == 2


# --------------------------------------------------------------------------- #
# Unbound Tempest
# --------------------------------------------------------------------------- #


def _play_elemental(triggers, player, name):
    elemental = _plain(name, race=Race.ELEMENTAL)
    player.board.append(elemental)
    triggers.fire_on_place(player=player, placed=elemental, shop_excluded_race=None)


def test_unbound_tempest_answers_every_third_elemental(patch, triggers):
    tempest = _card(patch, "BG36_352")  # 3/12
    player = _player(patch, board=[tempest])
    player.shop[0] = _plain("big", 3, 9)
    player.shop[1] = _plain("small", 1, 2)
    for i in range(2):
        _play_elemental(triggers, player, f"e{i}")
    assert (tempest.raw_attack, tempest.max_health) == (3, 12)
    _play_elemental(triggers, player, "e2")
    assert (tempest.raw_attack, tempest.max_health) == (6, 21)  # the biggest, 3/9


def test_unbound_tempest_leaves_the_minion_on_the_counter(patch, triggers):
    tempest = _card(patch, "BG36_352")
    player = _player(patch, board=[tempest])
    player.shop[0] = _plain("big", 3, 9)
    for i in range(3):
        _play_elemental(triggers, player, f"e{i}")
    assert player.shop[0] is not None  # read, not eaten


def test_two_tempests_count_separately(patch, triggers):
    """"(3 left!)" is printed on the card, so the countdown is the card's."""
    first = _card(patch, "BG36_352")
    player = _player(patch, board=[first])
    player.shop[0] = _plain("big", 3, 9)
    for i in range(2):
        _play_elemental(triggers, player, f"e{i}")
    second = _card(patch, "BG36_352")
    player.board.append(second)
    _play_elemental(triggers, player, "e2")
    assert first.raw_attack == 6  # its third
    assert second.raw_attack == 3  # its first


def test_golden_tempest_takes_double(patch):
    (ability,) = patch.triple_merge_golden_abilities("BG36_352")
    assert ability.effect.threshold == 3  # a countdown, not a payout
    assert ability.effect.effect.factor == 2


# --------------------------------------------------------------------------- #
# Zesty Shaker
# --------------------------------------------------------------------------- #


def test_zesty_shaker_gets_a_copy_of_the_spell_cast_on_it(patch):
    shaker = _card(patch, "BG26_505")
    naga = _card(patch, "BG23_000")  # Spellcraft: Myrmidon's Might
    player = _player(patch, board=[shaker, naga])
    give_spellcraft_spell(player, naga.abilities[0].effect)
    slot = next(i for i, c in enumerate(player.hand) if c is not None)
    play_spellcraft_spell_from_hand(player, slot, 0, patch=patch)
    assert [c.card_id for c in player.hand if c is not None] == ["BG23_000t"]


def test_zesty_shaker_answers_once_a_turn(patch):
    shaker = _card(patch, "BG26_505")
    naga = _card(patch, "BG23_000")
    player = _player(patch, board=[shaker, naga])
    for _ in range(2):
        give_spellcraft_spell(player, naga.abilities[0].effect)
        slot = next(i for i, c in enumerate(player.hand) if c is not None)
        play_spellcraft_spell_from_hand(player, slot, 0, patch=patch)
    # One copy kept, not two: the second cast found the latch spent.
    assert [c.card_id for c in player.hand if c is not None] == ["BG23_000t"]


def test_zesty_shakers_answer_comes_back_next_turn(patch):
    from src.bg_recruitment.activate import reset_activations

    shaker = _card(patch, "BG26_505")
    naga = _card(patch, "BG23_000")
    player = _player(patch, board=[shaker, naga])
    give_spellcraft_spell(player, naga.abilities[0].effect)
    slot = next(i for i, c in enumerate(player.hand) if c is not None)
    play_spellcraft_spell_from_hand(player, slot, 0, patch=patch)
    assert shaker.spell_answered_this_turn
    reset_activations(player)
    assert not shaker.spell_answered_this_turn


def test_golden_zesty_shaker_gets_two(patch):
    (ability,) = patch.triple_merge_golden_abilities("BG26_505")
    assert ability.effect.count == 2
    assert ability.effect.once_per_turn is True
