"""Magnetize, and the cards that watch one land.

Every Magnetization goes through one entry point, which is what lets four cards
across four tiers all see it however it arrived: welded out of hand, made on the
spot by Spark Snapper, doubled by Drone Duplicator, or echoed by Polarizing
Beatboxer onto a second body.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.bg_catalog.cards import make_minion
from src.bg_catalog.patch_context import PatchContext
from src.bg_core.effects import Keyword
from src.bg_core.minion import Minion, Race
from src.bg_lobby.player import PlayerPhase, PlayerState
from src.bg_recruitment.place import (
    can_magnetize_onto,
    magnet_from_hand,
    magnetize,
)
from src.bg_recruitment.shop_triggers import ShopTriggers

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
        tavern_tier=5,
        board=list(board),
        shop=[None] * 7,
        hand=[None] * 10,
        phase=PlayerPhase.SHOP,
        shop_actions_used=0,
        ruleset=patch.meta.ruleset,
    )
    base.update(kw)
    return PlayerState(**base)


def _mech(card_id="mech", atk=1, hp=1) -> Minion:
    return Minion(
        card_id=card_id, base_attack=atk, base_health=hp, tier=1, race=Race.MECHANICAL
    )


# --------------------------------------------------------------------------- #
# The weld itself
# --------------------------------------------------------------------------- #


def test_a_weld_adds_stats_and_counts(patch, triggers):
    target = _mech(atk=2, hp=3)
    player = _player(patch, [target])
    magnetize(player, target, _card(patch, "BG26_146"), triggers=triggers)  # Lullabot 2/2
    assert (target.raw_attack, target.max_health) == (4, 5)
    assert target.magnetized_count == 1


def test_prosthetic_hand_welds_to_an_undead(patch):
    """Its text says where it may land, and the binding is what says it."""
    hand = _card(patch, "BG_DEEP_015")
    undead = Minion(card_id="u", base_attack=1, base_health=1, tier=1, race=Race.UNDEAD)
    beast = Minion(card_id="b", base_attack=1, base_health=1, tier=1, race=Race.BEAST)
    assert can_magnetize_onto(hand, undead)
    assert can_magnetize_onto(hand, _mech())
    assert not can_magnetize_onto(hand, beast)


def test_an_ordinary_magnet_still_wants_a_mech(patch):
    lullabot = _card(patch, "BG26_146")
    undead = Minion(card_id="u", base_attack=1, base_health=1, tier=1, race=Race.UNDEAD)
    assert can_magnetize_onto(lullabot, _mech())
    assert not can_magnetize_onto(lullabot, undead)


# --------------------------------------------------------------------------- #
# The cards that watch one
# --------------------------------------------------------------------------- #


def test_mechagnome_interpreter_pays_a_weld_as_well_as_a_play(patch, triggers):
    """"Whenever you play *or Magnetize* a Mech" — one card, two sites."""
    interpreter = _card(patch, "BG31_177")
    target = _mech(atk=2, hp=3)
    player = _player(patch, [interpreter, target])
    magnetize(player, target, _card(patch, "BG26_146"), triggers=triggers)
    # 2/3 plus the Lullabot's 2/2, plus the Interpreter's +3/+1.
    assert (target.raw_attack, target.max_health) == (7, 6)


def test_drone_duplicator_doubles_the_next_weld(patch, triggers):
    from src.bg_recruitment.activate import activate_minion

    duplicator = _card(patch, "BG36_506")  # 4/4 Mech
    player = _player(patch, [duplicator], gold=5)
    activate_minion(player, 0, rng=np.random.default_rng(0), patch=patch)
    assert duplicator.magnet_doubles_next

    magnetize(player, duplicator, _card(patch, "BG26_146"), triggers=triggers)
    assert (duplicator.raw_attack, duplicator.max_health) == (8, 8)  # 4/4 + 2/2 twice
    assert duplicator.magnetized_count == 2
    assert not duplicator.magnet_doubles_next


def test_the_doubling_is_spent_by_one_weld(patch, triggers):
    duplicator = _card(patch, "BG36_506")
    player = _player(patch, [duplicator])
    duplicator.magnet_doubles_next = True
    magnetize(player, duplicator, _card(patch, "BG26_146"), triggers=triggers)
    magnetize(player, duplicator, _card(patch, "BG26_146"), triggers=triggers)
    assert duplicator.magnetized_count == 3  # two, then one


def test_polarizing_beatboxer_takes_a_copy_of_every_weld(patch, triggers):
    beatboxer = _card(patch, "BG26_149")  # 5/10 Mech
    other = _mech(atk=2, hp=2)
    player = _player(patch, [beatboxer, other])
    magnetize(player, other, _card(patch, "BG26_146"), triggers=triggers)
    assert other.magnetized_count == 1
    assert beatboxer.magnetized_count == 1
    assert (beatboxer.raw_attack, beatboxer.max_health) == (7, 12)


def test_two_beatboxers_do_not_answer_each_other_forever(patch, triggers):
    first = _card(patch, "BG26_149")
    second = _card(patch, "BG26_149")
    target = _mech(atk=1, hp=1)
    player = _player(patch, [first, second, target])
    magnetize(player, target, _card(patch, "BG26_146"), triggers=triggers)
    assert (first.magnetized_count, second.magnetized_count) == (1, 1)


def test_utility_drone_pays_per_weld_carried(patch, triggers):
    drone = _card(patch, "BG26_152")
    welded = _mech(atk=1, hp=1)
    welded.magnetized_count = 2
    plain = _mech(card_id="plain", atk=1, hp=1)
    player = _player(patch, [drone, welded, plain])
    triggers.fire_on_turn_end(player)
    assert (welded.raw_attack, welded.max_health) == (9, 9)
    assert (plain.raw_attack, plain.max_health) == (1, 1)


def test_spark_snapper_welds_a_satellite_onto_a_played_mech(patch, triggers):
    snapper = _card(patch, "BG36_851")
    newcomer = _mech(atk=1, hp=1)
    player = _player(patch, [snapper])
    player.board.append(newcomer)
    triggers.fire_after_friendly_minion_placed(player, newcomer)
    assert newcomer.magnetized_count == 1
    assert (newcomer.raw_attack, newcomer.max_health) == (4, 4)


def test_the_snapper_improves_with_each_mech(patch, triggers):
    """"and improve this" — the second Mech gets two Satellites."""
    snapper = _card(patch, "BG36_851")
    player = _player(patch, [snapper])
    first, second = _mech(card_id="m1"), _mech(card_id="m2")
    for m in (first, second):
        player.board.append(m)
        triggers.fire_after_friendly_minion_placed(player, m)
    assert first.magnetized_count == 1
    assert second.magnetized_count == 2


def test_a_non_mech_played_gets_no_satellite(patch, triggers):
    snapper = _card(patch, "BG36_851")
    beast = Minion(card_id="b", base_attack=1, base_health=1, tier=1, race=Race.BEAST)
    player = _player(patch, [snapper, beast])
    triggers.fire_after_friendly_minion_placed(player, beast)
    assert beast.magnetized_count == 0


def test_welding_from_hand_goes_through_the_same_door(patch, triggers):
    interpreter = _card(patch, "BG31_177")
    target = _mech(atk=2, hp=3)
    player = _player(patch, [interpreter, target])
    player.hand[0] = _card(patch, "BG26_146")
    magnet_from_hand(player, 0, 1, patch=patch, triggers=triggers)
    assert target.magnetized_count == 1
    assert target.raw_attack == 7  # the Interpreter saw it


def test_the_doubling_promise_lifts_at_turn_start(patch, triggers):
    duplicator = _card(patch, "BG36_506")
    player = _player(patch, [duplicator])
    duplicator.magnet_doubles_next = True
    triggers.fire_on_turn_start(player)
    assert not duplicator.magnet_doubles_next


# --------------------------------------------------------------------------- #
# What a Magnetization owes the rest of the game
# --------------------------------------------------------------------------- #


def test_the_part_goes_back_to_the_lobby_pool(patch, triggers):
    """Patch 27.0.0.181554: "Magnetic minions now return to the minion pool
    immediately after being Magnetized." The body is gone into the host, so
    nothing later releases it."""
    from src.bg_lobby.shared_pool import build_initial_shared_pool

    pool = build_initial_shared_pool(patch=patch)
    player = _player(patch, [_card(patch, "BG28_741")])
    player.hand[0] = _card(patch, "BG26_146")
    before = pool.remaining_copies("BG26_146")
    magnet_from_hand(player, 0, 0, patch=patch, triggers=triggers, shared_pool=pool)
    assert pool.remaining_copies("BG26_146") == before + 1


def test_magnetizing_counts_as_playing_a_minion(patch, triggers):
    """Magnetizing counts as playing a minion, but not as summoning one."""
    from src.bg_recruitment.game_counts import GOLDEN_PLAYED

    player = _player(patch, [_card(patch, "BG28_741")])
    part = _card(patch, "BG26_146")
    part.is_golden = True
    player.hand[0] = part
    magnet_from_hand(player, 0, 0, patch=patch, triggers=triggers)
    assert player.game_counts.get(GOLDEN_PLAYED) == 1


def test_a_magnetized_host_keeps_its_parts_through_a_triple(patch, triggers):
    """Patch 29.2.2.198608 fixed the same thing: the merge rebuilds the golden
    from the printed card, so the parts had to be carried over explicitly."""
    from src.bg_recruitment.triples import merge_three_non_golden_into_golden

    host_id, part_id = "BG28_741", "BG_BOT_911"
    part = _card(patch, part_id)
    worn = _card(patch, host_id)
    magnetize(_player(patch, [worn]), worn, part, triggers=triggers)
    plain_a, plain_b = _card(patch, host_id), _card(patch, host_id)

    golden = merge_three_non_golden_into_golden(
        host_id, worn, plain_a, plain_b, patch=patch
    )
    printed = patch.templates[host_id]
    assert golden.raw_attack == printed.base_attack * 2 + part.base_attack
    assert golden.max_health == printed.base_health * 2 + part.base_health
    assert golden.magnetized_count == 1
    # ...and the part's text, behind the golden's own.
    assert len(golden.abilities) >= len(part.abilities)
    for ability in part.abilities:
        assert ability in golden.abilities


def test_an_unworn_triple_is_unchanged(patch):
    from src.bg_recruitment.triples import merge_three_non_golden_into_golden

    host_id = "BG28_741"
    bodies = [_card(patch, host_id) for _ in range(3)]
    golden = merge_three_non_golden_into_golden(host_id, *bodies, patch=patch)
    printed = patch.templates[host_id]
    assert golden.raw_attack == printed.base_attack * 2
    assert golden.max_health == printed.base_health * 2
    assert golden.magnetized_count == 0
