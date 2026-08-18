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
    parrot = _card(patch, "BG36_763")  # 5/5
    player = _player(patch, [parrot])
    seat = PlayerCombatSeat(player, patch=patch)
    _fight(
        [parrot],
        [_wall(hp=200)],
        patch,
        seats=(seat, PlayerCombatSeat(_player(patch))),
    )
    assert parrot.damage_dealt_total >= 40
    assert any(c is not None and c.card_id == "BG28_830" for c in player.hand)


def test_the_parrot_carries_its_tally_between_fights(patch):
    """"(40 left!)" counts down over the game, not over one battle."""
    parrot = _card(patch, "BG36_763")
    player = _player(patch, [parrot])
    for _ in range(2):
        _fight(
            [parrot],
            [_wall(hp=11)],
            patch,
            seats=(PlayerCombatSeat(player, patch=patch), PlayerCombatSeat(_player(patch))),
        )
    assert parrot.damage_dealt_total > 11  # both fights counted


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


def test_two_parrots_count_separately(patch):
    """The tally is the body's, not the printing's: one swinging does not bring
    the other one closer to its reward."""
    swinging = _card(patch, "BG36_763")
    idle = _card(patch, "BG36_763")
    player = _player(patch, [swinging, idle])
    _fight(
        [swinging],
        [_wall(hp=200)],
        patch,
        seats=(PlayerCombatSeat(player, patch=patch), PlayerCombatSeat(_player(patch))),
    )
    assert swinging.damage_dealt_total >= 40
    assert idle.damage_dealt_total == 0
    assert not idle.damage_reward_paid


def test_a_golden_parrot_starts_its_count_over(patch):
    """Three merge into one new card, and the new card has dealt nothing —
    even when the copies that made it had already been paid."""
    from src.bg_recruitment.triples import resolve_triples_loop

    parrots = [_card(patch, "BG36_763") for _ in range(3)]
    player = _player(patch, parrots)
    _fight(
        [parrots[0]],
        [_wall(hp=200)],
        patch,
        seats=(PlayerCombatSeat(player, patch=patch), PlayerCombatSeat(_player(patch))),
    )
    assert parrots[0].damage_reward_paid

    resolve_triples_loop(player, patch=patch)
    golden = next(
        c for c in player.hand if c is not None and getattr(c, "is_golden", False)
    )
    assert golden.card_id == "BG36_763"
    assert (golden.damage_dealt_total, golden.damage_reward_paid) == (0, False)


# --------------------------------------------------------------------------- #
# The last six
# --------------------------------------------------------------------------- #


def test_devout_hellcaller_grows_when_a_demon_deals_damage(patch):
    hellcaller = _card(patch, "BG33_155")  # 2/2 Demon
    hellcaller.bonus_health += 40
    demon = Minion(card_id="d", base_attack=3, base_health=40, tier=1, race=Race.DEMON)
    player = _player(patch, [hellcaller, demon])
    survivors, _ = _fight(
        [hellcaller, demon],
        [_wall(hp=6)],
        patch,
        seats=(PlayerCombatSeat(player, patch=patch), PlayerCombatSeat(_player(patch))),
    )
    fought = next(m for m in survivors if m.card_id == "BG33_155")
    assert fought.raw_attack > hellcaller.base_attack
    # "Permanently": the owner's own body kept it too.
    assert hellcaller.raw_attack > hellcaller.base_attack


def test_a_beast_dealing_damage_does_not_feed_the_hellcaller(patch):
    hellcaller = _card(patch, "BG33_155")
    hellcaller.bonus_health += 40
    beast = Minion(card_id="b", base_attack=3, base_health=40, tier=1, race=Race.BEAST)
    player = _player(patch, [hellcaller, beast])
    _fight(
        [hellcaller, beast],
        [_wall(hp=6)],
        patch,
        seats=(PlayerCombatSeat(player, patch=patch), PlayerCombatSeat(_player(patch))),
    )
    assert hellcaller.raw_attack == hellcaller.base_attack


def test_diremuck_forager_summons_the_best_murloc_in_hand(patch):
    forager = _card(patch, "BG27_556")
    player = _player(patch, [forager])
    player.hand[0] = _card(patch, "BG23_002")  # Shell Collector, a Naga
    player.hand[1] = _card(patch, "BG36_507")  # Breakout Mastermind, a Murloc
    survivors, deaths = _fight(
        [forager],
        [_wall(hp=1)],
        patch,
        seats=(PlayerCombatSeat(player, patch=patch), PlayerCombatSeat(_player(patch))),
    )
    on_board = {m.card_id for m in survivors} | {cid for side, cid in deaths if side == 0}
    assert "BG36_507" in on_board and "BG23_002" not in on_board


def test_hired_mount_hands_over_one_of_the_five_chromadrakes(patch):
    from src.bg_recruitment.activate import activate_minion

    mount = _card(patch, "BG36_240")
    player = _player(patch, [mount], gold=5)
    activate_minion(player, 0, rng=np.random.default_rng(3), patch=patch)
    got = next(c for c in player.hand if c is not None)
    assert got.card_id in {"BG34_634t", "BG34_635t", "BG34_636t", "BG34_637t", "BG34_638t"}
    assert player.gold == 3


def test_meteorite_crasher_grows_when_an_elemental_is_sold(patch, triggers):
    crasher = _card(patch, "BG31_843")
    elemental = Minion(
        card_id="e", base_attack=1, base_health=1, tier=1, race=Race.ELEMENTAL
    )
    player = _player(patch, [crasher, elemental])
    triggers.fire_on_sell(elemental, player)
    assert (crasher.raw_attack, crasher.max_health) == (
        crasher.base_attack + 2,
        crasher.base_health + 2,
    )


def test_selling_something_else_leaves_the_crasher_alone(patch, triggers):
    crasher = _card(patch, "BG31_843")
    beast = Minion(card_id="b", base_attack=1, base_health=1, tier=1, race=Race.BEAST)
    player = _player(patch, [crasher, beast])
    triggers.fire_on_sell(beast, player)
    assert crasher.raw_attack == crasher.base_attack


def test_sly_raptor_summons_a_six_six_beast(patch):
    survivors, deaths = _fight([_card(patch, "BG25_806")], [_wall(atk=20)], patch)
    summoned = [cid for side, cid in deaths if side == 0 and cid != "BG25_806"]
    assert summoned, "the deathrattle summoned nothing"
    beast = patch.templates[summoned[0]]
    assert beast.race == Race.BEAST
    # Set, not added: whatever it rolled, it fought as a 6/6.


def test_the_raptors_beast_fights_at_six_six(patch):
    """Set, not added: whatever it rolled, it lands as a 6/6."""
    # Enough attack to kill the 1/3 Raptor on its swing, little enough health
    # that the 6/6 it leaves behind finishes the fight and can be inspected.
    survivors, _ = _fight([_card(patch, "BG25_806")], [_wall(hp=6, atk=3)], patch)
    summoned = next(m for m in survivors if m.card_id != "BG25_806")
    assert (summoned.base_attack, summoned.base_health) == (6, 6)


def test_waveling_buffs_the_tavern_on_every_roll_from_now_on(patch):
    from src.bg_recruitment.shop import refresh_shop

    waveling = _card(patch, "BG34_856")
    player = _player(patch, [waveling])
    _fight(
        [waveling],
        [_wall(atk=20)],
        patch,
        seats=(PlayerCombatSeat(player, patch=patch), PlayerCombatSeat(_player(patch))),
    )
    assert player.refresh_buffs == ((3, 3),)

    for _ in range(2):
        refresh_shop(player, None, rng=np.random.default_rng(1), patch=patch)
        buffed = [m for m in player.shop if m is not None and m.bonus_attack >= 3]
        assert len(buffed) == 1


def test_a_seat_with_no_waveling_rolls_a_plain_tavern(patch):
    from src.bg_recruitment.shop import refresh_shop

    player = _player(patch)
    refresh_shop(player, None, rng=np.random.default_rng(1), patch=patch)
    assert all(m.bonus_attack == 0 for m in player.shop if m is not None)
