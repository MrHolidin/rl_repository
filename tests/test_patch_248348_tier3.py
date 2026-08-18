"""Tier-3 bindings of the 36.2.0 package, played rather than inspected.

Most of this tier lands on mechanics earlier tiers paid for — Tavern spells,
standing bonuses, the improve tally, the seat protocol — so these tests are
mostly about whether each card names the right one.
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


def _player(patch, board=(), **kw) -> PlayerState:
    base = dict(
        health=30,
        gold=10,
        tavern_tier=3,
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


# --------------------------------------------------------------------------- #
# Deathrattles and tokens
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "card_id,token,count",
    [
        ("BG30_125", "BG_ICC_026t", 3),  # Cadaver Caretaker -> three Skeletons
        ("BG25_010", "BG25_010t", 1),  # Handless Forsaken -> a Helping Hand
    ],
)
def test_a_deathrattle_summons_what_it_names(patch, card_id, token, count):
    _, deaths = _fight([_card(patch, card_id)], [_wall(atk=20)], patch)
    assert sum(1 for side, cid in deaths if side == 0 and cid == token) == count


def test_the_helping_hand_is_printed_with_reborn(patch):
    """What makes the token worth summoning; the keyword rides on the card."""
    assert Keyword.REBORN in patch.templates["BG25_010t"].keywords


def test_scourfin_feeds_a_card_in_hand(patch):
    scourfin = _card(patch, "BG26_360")
    player = _player(patch, [scourfin])
    held = Minion(card_id="held", base_attack=1, base_health=1, tier=1)
    player.hand[0] = held
    _fight(
        [scourfin],
        [_wall(atk=20)],
        patch,
        seats=(PlayerCombatSeat(player), PlayerCombatSeat(_player(patch))),
    )
    assert (held.raw_attack, held.max_health) == (8, 8)


def test_mummifier_gives_reborn_to_another_undead(patch):
    mummifier = _card(patch, "BG28_309")  # 5/2 Undead
    undead = Minion(
        card_id="u", base_attack=1, base_health=60, tier=1, race=Race.UNDEAD
    )
    # Enough attack to kill the Mummifier on its swing, too little to matter to
    # the Undead it pays.
    survivors, _ = _fight([mummifier, undead], [_wall(hp=8, atk=2)], patch)
    revived = next(m for m in survivors if m.card_id == "u")
    assert Keyword.REBORN in revived.all_keywords


# --------------------------------------------------------------------------- #
# Cards that land on the Tavern-spell system
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "card_id,spell_id",
    [
        ("BG34_683", "BG34_689"),  # Briarback Drummer -> Blood Gem Barrage
        ("BG31_326", "BG31_893"),  # Gem Rat -> Gem Day
    ],
)
def test_a_card_hands_over_the_spell_it_names(patch, triggers, card_id, spell_id):
    source = _card(patch, card_id)
    player = _player(patch, [source])
    triggers.fire_on_place(source, player, None)
    triggers.fire_on_turn_end(player)
    assert any(c is not None and c.card_id == spell_id for c in player.hand)


def test_fruit_vendor_hands_over_two_bananas(patch):
    from src.bg_recruitment.activate import activate_minion

    vendor = _card(patch, "BG36_346")
    player = _player(patch, [vendor], gold=5)
    activate_minion(player, 0, rng=np.random.default_rng(0), patch=patch)
    got = [c for c in player.hand if c is not None]
    assert len(got) == 2 and all(c.card_id == "BG28_897" for c in got)
    assert player.gold == 4


def test_timecapn_hooktail_answers_a_cast(patch):
    from src.bg_recruitment.tavern_spells import play_tavern_spell_from_hand

    hooktail = _card(patch, "BG27_005")
    friend = Minion(card_id="f", base_attack=1, base_health=1, tier=1)
    player = _player(patch, [hooktail, friend])
    player.hand[0] = patch.tavern_spells["BG28_810"]  # Tavern Coin
    play_tavern_spell_from_hand(player, 0, rng=np.random.default_rng(0), patch=patch)
    assert friend.raw_attack == 2
    assert hooktail.raw_attack == hooktail.base_attack + 1


def test_azsharan_cutlassier_raises_what_a_spell_gives(patch, triggers):
    from src.bg_recruitment.tavern_spells import play_tavern_spell_from_hand

    cutlassier = _card(patch, "BG33_830")
    target = Minion(card_id="t", base_attack=1, base_health=1, tier=1)
    player = _player(patch, [cutlassier, target])
    triggers.fire_on_place(cutlassier, player, None)
    player.hand[0] = patch.tavern_spells["BG28_897"]  # Banana: +2/+2
    play_tavern_spell_from_hand(
        player, 0, rng=np.random.default_rng(0), patch=patch, target_board_index=1
    )
    assert (target.raw_attack, target.max_health) == (4, 3)


# --------------------------------------------------------------------------- #
# Start of Combat and Rally
# --------------------------------------------------------------------------- #


def test_amber_guardian_shields_one_other_dragon(patch):
    guardian = _card(patch, "BG24_500")
    first = Minion(card_id="d1", base_attack=1, base_health=20, tier=1, race=Race.DRAGON)
    second = Minion(card_id="d2", base_attack=1, base_health=20, tier=1, race=Race.DRAGON)
    survivors, _ = _fight([guardian, first, second], [_wall()], patch)
    shielded = [m for m in survivors if m.card_id in ("d1", "d2") and m.has_shield]
    assert len(shielded) == 1
    assert (shielded[0].raw_attack, shielded[0].max_health) == (3, 22)


def test_dustbone_devastator_raises_undead_for_the_game(patch):
    devastator = _card(patch, "BG33_323")
    player = _player(patch, [devastator])
    in_hand = Minion(card_id="u", base_attack=1, base_health=1, tier=1, race=Race.UNDEAD)
    player.hand[0] = in_hand
    # One swing: a Rally fires per attack, so the enemy has to die to the first.
    _fight(
        [devastator],
        [_wall(hp=1)],
        patch,
        seats=(PlayerCombatSeat(player), PlayerCombatSeat(_player(patch))),
    )
    from src.bg_recruitment.standing_bonuses import settle_standing_bonuses

    settle_standing_bonuses(player)
    assert in_hand.raw_attack == 3


def test_wolf_pup_and_roaring_recruiter_pay_the_attacker(patch):
    recruiter = _card(patch, "BG29_816")  # another Dragon attacks: +3/+1
    dragon = Minion(card_id="d", base_attack=2, base_health=30, tier=1, race=Race.DRAGON)
    survivors, _ = _fight([recruiter, dragon], [_wall(hp=60)], patch)
    grown = next(m for m in survivors if m.card_id == "d")
    assert grown.raw_attack > 2


# --------------------------------------------------------------------------- #
# The improve tally, on two cards that share one
# --------------------------------------------------------------------------- #


def test_the_mrrgltons_improve_each_other(patch, triggers):
    murloc = Minion(card_id="m", base_attack=1, base_health=1, tier=1, race=Race.MURLOC)
    player = _player(patch, [murloc])
    mama = _card(patch, "BG35_140")
    player.board.append(mama)
    triggers.fire_on_place(mama, player, None)
    assert murloc.raw_attack == 3  # +2 at level one

    papa = _card(patch, "BG35_141")
    player.board.append(papa)
    triggers.fire_on_place(papa, player, None)
    assert murloc.max_health == 5  # +2 twice: level two


def test_a_lone_mrrglton_is_worth_what_it_prints(patch, triggers):
    murloc = Minion(card_id="m", base_attack=1, base_health=1, tier=1, race=Race.MURLOC)
    player = _player(patch, [murloc])
    mama = _card(patch, "BG35_140")
    player.board.append(mama)
    triggers.fire_on_place(mama, player, None)
    assert murloc.raw_attack == 3


# --------------------------------------------------------------------------- #
# Keyword-only bodies
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "card_id,keywords",
    [
        ("BG_BOT_911", {Keyword.MAGNETIC, Keyword.SHIELD, Keyword.TAUNT}),
        ("BGS_131", {Keyword.VENOMOUS}),
        ("BG_DEEP_015", {Keyword.MAGNETIC, Keyword.REBORN}),
    ],
)
def test_keyword_only_cards_need_no_binding(patch, card_id, keywords):
    card = _card(patch, card_id)
    assert keywords <= card.keywords and card.abilities == ()


# --------------------------------------------------------------------------- #
# Cards that needed the engine to learn something
# --------------------------------------------------------------------------- #


def test_wolf_pup_pays_everyone_but_itself(patch):
    pup = _card(patch, "BG36_207")  # 3/5 Beast
    friend = Minion(card_id="f", base_attack=1, base_health=30, tier=1)
    survivors, _ = _fight([pup, friend], [_wall(hp=1)], patch)
    grown = next(m for m in survivors if m.card_id == "f")
    assert (grown.raw_attack, grown.max_health) == (5, 32)
    swung = next(m for m in survivors if m.card_id == "BG36_207")
    assert (swung.raw_attack, swung.max_health) == (pup.base_attack, pup.base_health)


def test_graverobber_leaves_a_plain_copy(patch):
    """Plain is the whole cost: what the body had gained does not come along."""
    from src.bg_recruitment.targeted_battlecry import apply_targeted_on_place_battlecries

    robber = _card(patch, "BG28_303")
    victim = _card(patch, "BG25_022")  # Scarlet Skull, an Undead
    victim.bonus_attack += 10
    victim.bonus_health += 10
    player = _player(patch, [robber, victim])
    apply_targeted_on_place_battlecries(
        ShopTriggers(np.random.default_rng(0), patch=patch),
        player,
        robber,
        rng=np.random.default_rng(0),
        forced_buff_target=victim,
    )
    assert victim not in player.board
    copy = next(c for c in player.hand if c is not None)
    assert copy.card_id == "BG25_022"
    assert (copy.raw_attack, copy.max_health) == (copy.base_attack, copy.base_health)


def test_a_destroyed_undead_counts_as_a_death(patch):
    """Eternal Knight reads "died this game", and this is a death in the tavern
    — the one place the engine had none until this card."""
    from src.bg_recruitment.game_counts import refresh_count_bonuses
    from src.bg_recruitment.targeted_battlecry import apply_targeted_on_place_battlecries

    robber = _card(patch, "BG28_303")
    knight = _card(patch, "BG25_008")  # Eternal Knight
    other_knight = _card(patch, "BG25_008")
    player = _player(patch, [robber, knight])
    player.hand[0] = other_knight
    apply_targeted_on_place_battlecries(
        ShopTriggers(np.random.default_rng(0), patch=patch),
        player,
        robber,
        rng=np.random.default_rng(0),
        forced_buff_target=knight,
    )
    refresh_count_bonuses(player)
    assert (other_knight.raw_attack, other_knight.max_health) == (
        other_knight.base_attack + 4,
        other_knight.base_health + 2,
    )


def test_malchezaar_pays_for_refreshes_in_health(patch, triggers):
    from src.bg_recruitment.economy import roll_shop

    malchezaar = _card(patch, "BG26_524")
    player = _player(patch, [malchezaar], gold=10, health=30)
    triggers.fire_on_turn_start(player)
    roll_shop(player, None, rng=np.random.default_rng(0), patch=patch)
    assert (player.gold, player.health) == (10, 29)


def test_only_two_refreshes_a_turn_cost_health(patch, triggers):
    from src.bg_recruitment.economy import roll_shop

    malchezaar = _card(patch, "BG26_524")
    player = _player(patch, [malchezaar], gold=10, health=30)
    triggers.fire_on_turn_start(player)
    for _ in range(3):
        roll_shop(player, None, rng=np.random.default_rng(0), patch=patch)
    assert (player.gold, player.health) == (9, 28)


def test_a_card_that_undoes_hero_damage_undoes_the_refresh_too(patch, triggers):
    """The payment is hero damage, so everything that reads hero damage sees
    it — including Soul Rewinder, which hands the refresh back for free."""
    from src.bg_recruitment.economy import roll_shop

    malchezaar = _card(patch, "BG26_524")
    rewinder = _card(patch, "BG26_174")  # Soul Rewinder
    player = _player(patch, [malchezaar, rewinder], gold=10, health=30)
    triggers.fire_on_turn_start(player)
    roll_shop(player, None, rng=np.random.default_rng(0), patch=patch)
    assert (player.gold, player.health) == (10, 30)
    assert rewinder.max_health == rewinder.base_health + 1


def test_armor_absorbs_the_refresh_payment(patch, triggers):
    from src.bg_recruitment.economy import roll_shop

    malchezaar = _card(patch, "BG26_524")
    player = _player(patch, [malchezaar], gold=10, health=30, armor=5)
    triggers.fire_on_turn_start(player)
    roll_shop(player, None, rng=np.random.default_rng(0), patch=patch)
    assert (player.health, player.armor) == (30, 4)


def test_treasure_parrot_pays_out_once_it_has_dealt_forty(patch):
    from src.bg_recruitment.game_counts import DAMAGE_DEALT, counter_key

    parrot = _card(patch, "BG36_763")  # 5/5
    player = _player(patch, [parrot])
    seat = PlayerCombatSeat(player, patch=patch)
    _fight(
        [parrot],
        [_wall(hp=200)],
        patch,
        seats=(seat, PlayerCombatSeat(_player(patch))),
    )
    dealt = player.game_counts[counter_key(DAMAGE_DEALT, "BG36_763")]
    assert dealt >= 40
    assert any(c is not None and c.card_id == "BG28_830" for c in player.hand)


def test_the_parrot_carries_its_tally_between_fights(patch):
    """"(40 left!)" counts down over the game, not over one battle."""
    from src.bg_recruitment.game_counts import DAMAGE_DEALT, counter_key

    parrot = _card(patch, "BG36_763")
    player = _player(patch, [parrot])
    key = counter_key(DAMAGE_DEALT, "BG36_763")
    for _ in range(2):
        _fight(
            [parrot],
            [_wall(hp=11)],
            patch,
            seats=(PlayerCombatSeat(player, patch=patch), PlayerCombatSeat(_player(patch))),
        )
    assert player.game_counts[key] > 11  # both fights counted


def test_the_parrot_pays_out_only_once(patch):
    parrot = _card(patch, "BG36_763")
    player = _player(patch, [parrot])
    for _ in range(3):
        _fight(
            [parrot],
            [_wall(hp=200)],
            patch,
            seats=(PlayerCombatSeat(player, patch=patch), PlayerCombatSeat(_player(patch))),
        )
    assert sum(1 for c in player.hand if c is not None and c.card_id == "BG28_830") == 1
