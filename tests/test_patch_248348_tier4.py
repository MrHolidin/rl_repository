"""Tier-4 bindings of the 36.2.0 package, played rather than inspected."""

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
        health=30,
        gold=10,
        tavern_tier=4,
        board=list(board),
        shop=[None] * 7,
        hand=[None] * 10,
        phase=PlayerPhase.SHOP,
        shop_actions_used=0,
        ruleset=patch.meta.ruleset,
    )
    base.update(kw)
    return PlayerState(**base)


def _wall(hp=40, atk=0):
    return Minion(card_id="wall", base_attack=atk, base_health=hp, tier=1)


def _fight(board_0, board_1, patch, seed=0, seats=None):
    survivors: List[Minion] = []
    deaths: List[tuple] = []
    kwargs = {"seats": seats} if seats is not None else {}
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


def _summoned(survivors, deaths):
    return {cid for side, cid in deaths if side == 0} | {m.card_id for m in survivors}


# --------------------------------------------------------------------------- #
# Combat
# --------------------------------------------------------------------------- #


def test_auto_assembler_leaves_an_automaton(patch):
    _, deaths = _fight([_card(patch, "BG32_172")], [_wall(atk=20)], patch)
    assert any(cid == "BG_TTN_401" for side, cid in deaths if side == 0)


def test_boom_in_a_box_hits_everything_but_itself(patch):
    boom = _card(patch, "BG36_620")  # 5/10 Taunt
    friend = Minion(card_id="f", base_attack=1, base_health=2, tier=1)
    enemy = Minion(card_id="e", base_attack=1, base_health=2, tier=1)
    survivors, deaths = _fight([boom, friend], [enemy], patch)
    dead = {cid for _side, cid in deaths}
    assert "f" in dead and "e" in dead
    assert any(m.card_id == "BG36_620" for m in survivors)


def test_bonker_gems_its_others_when_it_swings(patch):
    bonker = _card(patch, "BG20_104")  # Windfury Quilboar
    friend = Minion(card_id="f", base_attack=1, base_health=40, tier=1)
    survivors, _ = _fight([bonker, friend], [_wall(hp=1)], patch)
    grown = next(m for m in survivors if m.card_id == "f")
    assert (grown.raw_attack, grown.max_health) == (2, 41)
    swung = next(m for m in survivors if m.card_id == "BG20_104")
    assert swung.raw_attack == bonker.base_attack  # "all your *other* minions"


def test_cage_gnawer_pays_the_beasts_when_one_attacks(patch):
    gnawer = _card(patch, "BG36_211")  # 2/7 Beast
    beast = Minion(card_id="b", base_attack=1, base_health=40, tier=1, race=Race.BEAST)
    murloc = Minion(card_id="m", base_attack=0, base_health=40, tier=1, race=Race.MURLOC)
    # Long enough that the Beast swings too: the Gnawer's own attack is not
    # "a friendly Beast attacks" as far as it is concerned.
    survivors, _ = _fight([gnawer, beast, murloc], [_wall(hp=30)], patch)
    assert next(m for m in survivors if m.card_id == "b").raw_attack > 1
    assert next(m for m in survivors if m.card_id == "m").raw_attack == 0


def test_blade_collector_cleaves(patch):
    """Its swing reaches the defender's neighbours as well."""
    collector = _card(patch, "BG26_817")
    collector.bonus_attack += 10
    collector.bonus_health += 40
    enemies = [
        Minion(card_id=f"e{i}", base_attack=0, base_health=4, tier=1) for i in range(3)
    ]
    _, deaths = _fight([collector], enemies, patch)
    # One swing, more than one body: the neighbours took it too.
    assert len([cid for side, cid in deaths if side == 1]) > 1


def test_headhunter_gryphon_fetches_a_beast_on_its_swing(patch):
    gryphon = _card(patch, "BG36_204")
    player = _player(patch, [gryphon])
    seat = PlayerCombatSeat(player, patch=patch)
    _fight(
        [gryphon],
        [_wall(hp=1)],
        patch,
        seats=(seat, PlayerCombatSeat(_player(patch))),
    )
    # A Rally that fetches a card queues it for after the fight, the way every
    # combat hand-add does.
    assert seat.hand_adds and patch.templates[seat.hand_adds[0]].race == Race.BEAST


# --------------------------------------------------------------------------- #
# Shop
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "card_id,trigger,spell_id",
    [
        ("BG36_760", "death", "BG28_518"),  # Captain Cookie -> Chef's Choice
        ("BG35_143", "place", "BG35_149"),  # Deepwater Chieftain -> Deepwater Clan
        ("BG34_682", "death", "BG34_689"),  # Razorfen Flapper -> Blood Gem Barrage
        ("BG34_684", "turn_end", "BG28_698"),  # Trench Fighter -> Gem Confiscation
    ],
)
def test_a_card_hands_over_the_spell_it_names(patch, triggers, card_id, trigger, spell_id):
    source = _card(patch, card_id)
    player = _player(patch, [source])
    if trigger == "place":
        triggers.fire_on_place(source, player, None)
    elif trigger == "turn_end":
        triggers.fire_on_turn_end(player)
    else:
        triggers.apply_shop_effect(player, source, source.abilities[-1].effect, None)
    assert any(c is not None and c.card_id == spell_id for c in player.hand)


def test_gearfin_hands_over_two_cheap_spells(patch, triggers):
    gearfin = _card(patch, "BG36_764")
    player = _player(patch, [gearfin])
    triggers.fire_on_turn_end(player)
    got = [c for c in player.hand if c is not None]
    assert len(got) == 2 and all(c.cost <= 1 for c in got)


def test_devilish_distractor_buffs_the_tavern_for_the_game(patch):
    from src.bg_recruitment.blood_gems import play_blood_gem_on
    from src.bg_recruitment.shop import refresh_shop

    distractor = _card(patch, "BG36_762")
    player = _player(patch, [distractor])
    play_blood_gem_on(player, distractor, patch=patch)  # a spell cast on it
    refresh_shop(player, None, rng=np.random.default_rng(2), patch=patch)
    assert all(m.bonus_attack >= 2 for m in player.shop if m is not None)


def test_en_djinn_blazer_buffs_one_minion_every_roll(patch, triggers):
    from src.bg_recruitment.shop import refresh_shop

    blazer = _card(patch, "BG34_865")
    player = _player(patch, [blazer])
    triggers.fire_on_place(blazer, player, None)
    for _ in range(2):
        refresh_shop(player, None, rng=np.random.default_rng(4), patch=patch)
        assert sum(1 for m in player.shop if m is not None and m.bonus_attack >= 7) == 1


def test_refreshing_anomaly_hands_over_two_free_rolls(patch, triggers):
    from src.bg_recruitment.economy import effective_roll_cost

    anomaly = _card(patch, "BGS_116")
    player = _player(patch, [anomaly])
    triggers.fire_on_place(anomaly, player, None)
    assert effective_roll_cost(player) == 0


def test_motley_phalanx_pays_one_of_each_tribe(patch, triggers):
    phalanx = _card(patch, "BG27_080")
    beast = Minion(card_id="b", base_attack=1, base_health=1, tier=1, race=Race.BEAST)
    murloc = Minion(card_id="m", base_attack=1, base_health=1, tier=1, race=Race.MURLOC)
    second_beast = Minion(card_id="b2", base_attack=1, base_health=1, tier=1, race=Race.BEAST)
    player = _player(patch, [phalanx, beast, murloc, second_beast])
    triggers.apply_shop_effect(player, phalanx, phalanx.abilities[0].effect, None)
    beasts_paid = sum(1 for m in (beast, second_beast) if m.raw_attack > 1)
    assert beasts_paid == 1 and murloc.raw_attack == 3


def test_maritime_extortionist_counts_golden_minions_played(patch, triggers):
    from src.bg_recruitment.game_counts import refresh_count_bonuses
    from src.bg_recruitment.place import place_from_hand

    extortionist = _card(patch, "BG36_524")
    player = _player(patch, [extortionist])
    assert extortionist.raw_attack == extortionist.base_attack

    golden = _card(patch, "BG25_001")
    golden.is_golden = True
    player.hand[0] = golden
    place_from_hand(
        player,
        0,
        None,
        board_size=7,
        triggers=ShopTriggers(np.random.default_rng(0), patch=patch),
        rng=np.random.default_rng(0),
    )
    refresh_count_bonuses(player)
    assert extortionist.raw_attack == extortionist.base_attack + 8


def test_a_plain_minion_played_does_not_count(patch):
    from src.bg_recruitment.game_counts import refresh_count_bonuses
    from src.bg_recruitment.place import place_from_hand

    extortionist = _card(patch, "BG36_524")
    player = _player(patch, [extortionist])
    player.hand[0] = _card(patch, "BG25_001")
    place_from_hand(
        player,
        0,
        None,
        board_size=7,
        triggers=ShopTriggers(np.random.default_rng(0), patch=patch),
        rng=np.random.default_rng(0),
    )
    refresh_count_bonuses(player)
    assert extortionist.raw_attack == extortionist.base_attack
