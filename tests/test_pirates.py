"""The Pirate family: gold spent, Bounties, and making a minion Golden."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.bg_catalog.cards import make_minion
from src.bg_catalog.patch_context import PatchContext
from src.bg_core.minion import Minion, Race
from src.bg_lobby.player import PlayerPhase, PlayerState
from src.bg_recruitment.shop_triggers import ShopTriggers
from src.bg_recruitment.targeted_battlecry import apply_targeted_on_place_battlecries

PATCH_DIR = Path("data/bgcore/36_2_0_248348")
BOUNTIES = {"BG33_811", "BG33_812", "BG33_813", "BG33_814", "BG33_815"}


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
        tavern_tier=6,
        board=list(board),
        shop=[None] * 7,
        hand=[None] * 10,
        phase=PlayerPhase.SHOP,
        shop_actions_used=0,
        ruleset=patch.meta.ruleset,
    )
    base.update(kw)
    return PlayerState(**base)


def _pirate(card_id="p", atk=1, hp=1) -> Minion:
    return Minion(card_id=card_id, base_attack=atk, base_health=hp, tier=1, race=Race.PIRATE)


def _place(triggers, player, source, forced=None):
    triggers.fire_on_place(source, player, None)
    apply_targeted_on_place_battlecries(
        triggers, player, source, rng=np.random.default_rng(0), forced_buff_target=forced
    )


# --------------------------------------------------------------------------- #
# Gold spent
# --------------------------------------------------------------------------- #


def test_gunpowder_courier_pays_the_pirates_every_five_gold(patch, triggers):
    courier = _card(patch, "BG26_810")
    pirate = _pirate()
    beast = Minion(card_id="b", base_attack=1, base_health=1, tier=1, race=Race.BEAST)
    player = _player(patch, [courier, pirate, beast])
    triggers.fire_gold_spent(player, 3)
    assert pirate.raw_attack == 1
    triggers.fire_gold_spent(player, 2)
    assert pirate.raw_attack == 3
    assert beast.raw_attack == 1


def test_the_countdown_carries_the_remainder(patch, triggers):
    courier = _card(patch, "BG26_810")
    pirate = _pirate()
    player = _player(patch, [courier, pirate])
    triggers.fire_gold_spent(player, 12)  # two full fives, two left over
    assert pirate.raw_attack == 5
    assert courier.gold_spent_seen == 2


def test_dual_wield_corsair_pays_exactly_two(patch, triggers):
    corsair = _card(patch, "BG31_824")
    crew = [_pirate(f"p{i}") for i in range(3)]
    player = _player(patch, [corsair] + crew)
    triggers.fire_gold_spent(player, 5)
    paid = [m for m in crew if m.raw_attack > 1]
    assert len(paid) + (1 if corsair.raw_attack > corsair.base_attack else 0) == 2


def test_enterprising_escapee_hands_over_a_lockbox(patch, triggers):
    from src.bg_recruitment.lockbox import find_lockbox

    escapee = _card(patch, "BG36_523")
    player = _player(patch, [escapee])
    triggers.fire_gold_spent(player, 5)
    assert find_lockbox(player) is not None


def test_spending_is_counted_wherever_the_gold_goes(patch):
    """Buying, rolling and levelling all spend, and the cards say only "spend"."""
    from src.bg_recruitment.economy import roll_shop

    courier = _card(patch, "BG26_810")
    pirate = _pirate()
    player = _player(patch, [courier, pirate], gold=10)
    for _ in range(5):
        roll_shop(player, None, rng=np.random.default_rng(0), patch=patch)
    assert pirate.raw_attack == 3


# --------------------------------------------------------------------------- #
# This turn's gold, which is a different tally
# --------------------------------------------------------------------------- #


def test_lovesick_balladist_scales_with_this_turns_spending(patch, triggers):
    balladist = _card(patch, "BG26_814")
    pirate = _pirate()
    player = _player(patch, [balladist, pirate])
    triggers.fire_gold_spent(player, 3)
    _place(triggers, player, balladist, forced=pirate)
    # +1 Health, once plus once per Gold spent this turn.
    assert pirate.max_health == 1 + 4


def test_that_tally_resets_and_the_other_does_not(patch, triggers):
    courier = _card(patch, "BG26_810")
    player = _player(patch, [courier])
    triggers.fire_gold_spent(player, 3)
    assert (player.gold_spent_this_turn, courier.gold_spent_seen) == (3, 3)
    triggers.fire_on_turn_start(player)
    assert (player.gold_spent_this_turn, courier.gold_spent_seen) == (0, 3)


# --------------------------------------------------------------------------- #
# Bounties
# --------------------------------------------------------------------------- #


def test_shipwrecked_rascal_hands_over_a_bounty(patch, triggers):
    rascal = _card(patch, "BG33_821")
    player = _player(patch, [rascal])
    triggers.fire_on_place(rascal, player, None)
    got = [c for c in player.hand if c is not None]
    assert len(got) == 1 and got[0].card_id in BOUNTIES


def test_a_bounty_pays_the_number_of_minions_it_names(patch):
    from src.bg_recruitment.tavern_spells import cast_tavern_spell

    crew = [_pirate(f"p{i}") for i in range(6)]
    player = _player(patch, crew)
    cast_tavern_spell(
        player,
        patch.tavern_spells["BG33_812"],  # Hostile Bounty: four minions +4 Attack
        rng=np.random.default_rng(0),
        patch=patch,
    )
    assert sum(1 for m in crew if m.raw_attack > 1) == 4


def test_the_selfish_bounty_pays_the_left_most(patch):
    from src.bg_recruitment.tavern_spells import cast_tavern_spell

    crew = [_pirate(f"p{i}") for i in range(3)]
    player = _player(patch, crew)
    cast_tavern_spell(
        player,
        patch.tavern_spells["BG33_813"],
        rng=np.random.default_rng(0),
        patch=patch,
    )
    assert (crew[0].raw_attack, crew[0].max_health) == (7, 7)
    assert (crew[1].raw_attack, crew[1].max_health) == (1, 1)


# --------------------------------------------------------------------------- #
# Making a minion Golden
# --------------------------------------------------------------------------- #


def test_captain_sanders_makes_a_friendly_golden(patch, triggers):
    sanders = _card(patch, "BG25_034")
    target = _card(patch, "BG25_001")  # Risen Rider 2/1, tier 1
    player = _player(patch, [sanders, target])
    _place(triggers, player, sanders, forced=target)
    assert target.is_golden
    assert (target.base_attack, target.base_health) == (4, 2)


def test_a_made_golden_owes_no_triple_reward(patch, triggers):
    """Nothing merged, so nothing is owed — unlike a forged Golden."""
    sanders = _card(patch, "BG25_034")
    target = _card(patch, "BG25_001")
    player = _player(patch, [sanders, target])
    _place(triggers, player, sanders, forced=target)
    assert player.pending_choice is None
    assert all(c is None for c in player.hand)


def test_a_minion_above_the_cap_is_left_alone(patch, triggers):
    sanders = _card(patch, "BG25_034")
    big = _card(patch, "BG27_016")  # Champion of Sargeras, tier 7
    player = _player(patch, [sanders, big])
    _place(triggers, player, sanders, forced=big)
    assert not big.is_golden


def test_a_golden_minion_is_not_made_golden_twice(patch, triggers):
    sanders = _card(patch, "BG25_034")
    already = _card(patch, "BG32_236")  # Aureate Laureate, born golden
    before = (already.base_attack, already.base_health)
    player = _player(patch, [sanders, already])
    _place(triggers, player, sanders, forced=already)
    assert (already.base_attack, already.base_health) == before


# --------------------------------------------------------------------------- #
# The last four
# --------------------------------------------------------------------------- #


def test_proud_privateer_casts_a_bounty_twice(patch, triggers):
    from src.bg_recruitment.tavern_spells import cast_tavern_spell

    privateer = _card(patch, "BG33_825")
    crew = [_pirate(f"p{i}") for i in range(3)]
    player = _player(patch, [privateer] + crew)
    triggers.fire_on_place(privateer, player, None)
    assert player.bounties_cast_twice

    cast_tavern_spell(
        player,
        patch.tavern_spells["BG33_813"],  # Selfish Bounty: left-most +6/+6
        rng=np.random.default_rng(0),
        patch=patch,
    )
    assert (privateer.raw_attack, privateer.max_health) == (
        privateer.base_attack + 12,
        privateer.base_health + 12,
    )


def test_without_the_privateer_a_bounty_casts_once(patch):
    from src.bg_recruitment.tavern_spells import cast_tavern_spell

    crew = [_pirate(f"p{i}") for i in range(2)]
    player = _player(patch, crew)
    cast_tavern_spell(
        player,
        patch.tavern_spells["BG33_813"],
        rng=np.random.default_rng(0),
        patch=patch,
    )
    assert (crew[0].raw_attack, crew[0].max_health) == (7, 7)


def test_an_ordinary_spell_is_not_doubled(patch, triggers):
    """The promise is about Bounties, and the package says which spells are."""
    from src.bg_recruitment.tavern_spells import cast_tavern_spell

    privateer = _card(patch, "BG33_825")
    target = _pirate()
    player = _player(patch, [privateer, target], gold=0)
    triggers.fire_on_place(privateer, player, None)
    cast_tavern_spell(
        player,
        patch.tavern_spells["BG28_810"],  # Tavern Coin: +1 Gold, not a Bounty
        rng=np.random.default_rng(0),
        patch=patch,
    )
    assert player.gold == 1


def test_silent_deliverer_hands_over_a_golden_tier_four(patch, triggers):
    deliverer = _card(patch, "BG36_343")
    player = _player(patch, [deliverer])
    triggers.fire_on_place(deliverer, player, None)
    got = next(c for c in player.hand if c is not None)
    assert got.is_golden and got.tier == 4
    assert player.pending_choice is None  # nothing merged, nothing owed


def test_friendly_bounty_reads_the_board_for_a_tribe(patch):
    from src.bg_recruitment.tavern_spells import cast_tavern_spell

    board = [_pirate("p1"), _pirate("p2")]
    board.append(Minion(card_id="b", base_attack=1, base_health=1, tier=1, race=Race.BEAST))
    player = _player(patch, board)
    cast_tavern_spell(
        player,
        patch.tavern_spells["BG33_814"],
        rng=np.random.default_rng(0),
        patch=patch,
    )
    got = next(c for c in player.hand if c is not None)
    assert got.race == Race.PIRATE


def test_a_tribeless_board_names_no_type(patch):
    from src.bg_recruitment.tavern_spells import cast_tavern_spell

    player = _player(patch, [Minion(card_id="x", base_attack=1, base_health=1, tier=1)])
    cast_tavern_spell(
        player,
        patch.tavern_spells["BG33_814"],
        rng=np.random.default_rng(0),
        patch=patch,
    )
    assert all(c is None for c in player.hand)


def test_hooktusk_answers_a_discover(patch, triggers):
    from src.bg_recruitment.discover import resolve_discover_pick
    from src.bg_recruitment.tavern_spells import open_tavern_spell_discover

    hooktusk = _card(patch, "BG36_344")
    mate = _pirate("mate")
    player = _player(patch, [hooktusk, mate], tavern_tier=3)
    open_tavern_spell_discover(player, rng=np.random.default_rng(0), patch=patch)
    resolve_discover_pick(
        player, 0, None, rng=np.random.default_rng(0),
        on_after_placed=lambda p, m: None, patch=patch,
    )
    assert (mate.raw_attack, mate.max_health) == (2, 2)
    # "your *other* Pirates"
    assert (hooktusk.raw_attack, hooktusk.max_health) == (
        hooktusk.base_attack,
        hooktusk.base_health,
    )


def test_hooktusk_improves_with_golden_minions_played(patch, triggers):
    from src.bg_recruitment.discover import resolve_discover_pick
    from src.bg_recruitment.game_counts import GOLDEN_PLAYED
    from src.bg_recruitment.tavern_spells import open_tavern_spell_discover

    hooktusk = _card(patch, "BG36_344")
    mate = _pirate("mate")
    player = _player(patch, [hooktusk, mate], tavern_tier=3)
    player.game_counts[GOLDEN_PLAYED] = 2
    open_tavern_spell_discover(player, rng=np.random.default_rng(0), patch=patch)
    resolve_discover_pick(
        player, 0, None, rng=np.random.default_rng(0),
        on_after_placed=lambda p, m: None, patch=patch,
    )
    assert (mate.raw_attack, mate.max_health) == (4, 4)  # three times over
