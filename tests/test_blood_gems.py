"""Blood Gems — the Quilboar currency.

A Gem is worth +1/+1 *plus* whatever the seat has banked ("Your Blood Gems give
an extra +1/+1 this game"), and it arrives by two routes: off your hand onto a
minion you pick, or played by a card onto minions the card names. Both routes
run through one helper, and these tests hold it to that: the same value, the
same bookkeeping, whichever way the Gem was played.

The bookkeeping is not decoration. Jailbird Juggernaut summons a Golem "with
stats equal to this minion's Blood Gems" and Gem Confiscation "steals all Blood
Gems from its neighbors" — both read back what a Gem gave, which a plain +1/+1
buff cannot answer.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.bg_catalog.patch_context import PatchContext
from src.bg_core.effects import (
    Ability,
    BloodGemTarget,
    GainBloodGemsEffect,
    IncreaseBloodGemBonusEffect,
    Keyword,
    PlayBloodGemsEffect,
    Trigger,
)
from src.bg_core.minion import Minion, Race
from src.bg_lobby.player import PlayerPhase, PlayerState
from src.bg_recruitment import blood_gems
from src.bg_recruitment.shop_triggers import ShopTriggers

PATCH_DIR = "data/bgcore/19_6_0_74257"


@pytest.fixture(scope="module")
def patch():
    from pathlib import Path

    return PatchContext.load(Path(PATCH_DIR))


def _minion(card_id: str, atk: int = 1, hp: int = 1, race=None, **kw) -> Minion:
    return Minion(card_id=card_id, base_attack=atk, base_health=hp, tier=1, race=race, **kw)


def _player(board=None, hand_slots: int = 10, **kw) -> PlayerState:
    base = dict(
        health=40,
        gold=10,
        tavern_tier=1,
        board=list(board or []),
        shop=[None] * 6,
        hand=[None] * hand_slots,
        phase=PlayerPhase.SHOP,
        shop_actions_used=0,
    )
    base.update(kw)
    return PlayerState(**base)


def _fire(patch, player, source, effect):
    """Run one shop effect through the real dispatcher."""
    ShopTriggers(np.random.default_rng(0), patch=patch).apply_shop_effect(
        player, source, effect, placed=None
    )


# --------------------------------------------------------------------------- #
# What a Gem is worth
# --------------------------------------------------------------------------- #


def test_a_gem_is_one_and_one_by_default():
    player = _player()
    assert blood_gems.blood_gem_value(player) == (1, 1)


def test_the_seats_bonus_raises_every_later_gem(patch):
    player = _player([_minion("boar", race=Race.QUILBOAR)])
    _fire(patch, player, player.board[0], IncreaseBloodGemBonusEffect(attack=1, health=1))
    assert blood_gems.blood_gem_value(player) == (2, 2)


def test_the_bonus_can_move_attack_alone(patch):
    """Gem Day grants "+1 Attack this game; or +1 Health" — not a pair."""
    player = _player([_minion("boar")])
    _fire(patch, player, player.board[0], IncreaseBloodGemBonusEffect(attack=1))
    assert blood_gems.blood_gem_value(player) == (2, 1)


def test_a_later_bonus_does_not_regrow_gems_already_played():
    player = _player([_minion("boar", 3, 3)])
    target = player.board[0]
    blood_gems.play_blood_gem_on(player, target)
    player.blood_gem_bonus_attack = 5
    assert target.raw_attack == 4, "the Gem already played keeps the stats it gave"


# --------------------------------------------------------------------------- #
# Playing one
# --------------------------------------------------------------------------- #


def test_playing_a_gem_grows_the_minion_and_is_recorded():
    player = _player([_minion("boar", 3, 4)])
    target = player.board[0]
    blood_gems.play_blood_gem_on(player, target)
    assert (target.raw_attack, target.max_health) == (4, 5)
    assert (target.blood_gem_attack, target.blood_gem_health) == (1, 1)


def test_the_record_tracks_the_stats_given_not_the_gems_counted():
    """Two Gems at +2/+2 read as 4/4 of Gems, which is what the cards ask for."""
    player = _player([_minion("boar", 1, 1)], blood_gem_bonus_attack=1, blood_gem_bonus_health=1)
    target = player.board[0]
    blood_gems.play_blood_gem_on(player, target, count=2)
    assert (target.blood_gem_attack, target.blood_gem_health) == (4, 4)


def test_a_gem_from_hand_leaves_the_hand():
    player = _player([_minion("boar", 1, 1)])
    blood_gems.give_blood_gems(player, 1)
    assert blood_gems.is_blood_gem(player.hand[0])
    blood_gems.play_blood_gem_from_hand(player, 0, 0)
    assert player.hand[0] is None
    assert player.board[0].raw_attack == 2


def test_a_gem_from_hand_needs_a_minion_to_land_on():
    player = _player([])
    blood_gems.give_blood_gems(player, 1)
    with pytest.raises(ValueError, match="no minion at board index"):
        blood_gems.play_blood_gem_from_hand(player, 0, 0)


def test_only_a_gem_can_be_played_as_one():
    player = _player([_minion("boar")])
    player.hand[0] = _minion("not_a_gem")
    with pytest.raises(ValueError, match="does not hold a Blood Gem"):
        blood_gems.play_blood_gem_from_hand(player, 0, 0)


# --------------------------------------------------------------------------- #
# The keyword-granting printings
# --------------------------------------------------------------------------- #


def test_a_keyword_gem_arms_a_quilboar():
    player = _player([_minion("boar", race=Race.QUILBOAR)])
    blood_gems.play_blood_gem_on(player, player.board[0], quilboar_keyword=Keyword.TAUNT)
    assert Keyword.TAUNT in player.board[0].all_keywords


def test_a_keyword_gem_on_a_non_quilboar_is_only_stats():
    """"If it's a Quilboar, also give it Taunt" — the card means it."""
    player = _player([_minion("murloc", race=Race.MURLOC)])
    blood_gems.play_blood_gem_on(player, player.board[0], quilboar_keyword=Keyword.TAUNT)
    assert Keyword.TAUNT not in player.board[0].all_keywords
    assert player.board[0].raw_attack == 2


# --------------------------------------------------------------------------- #
# Cards handing out and playing Gems
# --------------------------------------------------------------------------- #


def test_a_card_hands_gems_to_the_hand(patch):
    """Razorfen Geomancer's shape: "Battlecry: Get 2 Blood Gems"."""
    player = _player([_minion("geomancer")])
    _fire(patch, player, player.board[0], GainBloodGemsEffect(count=2))
    assert sum(1 for c in player.hand if blood_gems.is_blood_gem(c)) == 2


def test_gems_that_do_not_fit_the_hand_are_lost(patch):
    """What a Gem with no free slot does is UNVERIFIED against the client.

    Every source found says only that Gems occupy hand slots; none says what
    happens to one that cannot fit. This pins the engine's current choice so a
    future correction is a deliberate edit rather than a silent drift.
    """
    player = _player([_minion("geomancer")], hand_slots=1)
    made = blood_gems.give_blood_gems(player, 3)
    assert made == 1
    assert sum(1 for c in player.hand if blood_gems.is_blood_gem(c)) == 1


def test_a_gem_with_no_minion_to_land_on_waits_in_hand(patch):
    """The documented Battlegrounds state: Gems in hand, empty board, stuck.

    The Gem is neither discarded nor spent — it is simply unplayable, which is
    what makes the soft-lock possible in the first place.
    """
    player = _player([])
    blood_gems.give_blood_gems(player, 2)
    assert blood_gems.can_play_blood_gem(player) is False
    assert sum(1 for c in player.hand if blood_gems.is_blood_gem(c)) == 2

    player.board.append(_minion("boar"))
    assert blood_gems.can_play_blood_gem(player) is True


@pytest.mark.parametrize(
    "target,expected",
    [
        (BloodGemTarget.SELF, {"source"}),
        (BloodGemTarget.ALL_FRIENDLY, {"source", "left", "right", "boar"}),
        (BloodGemTarget.ALL_OTHER_FRIENDLY, {"left", "right", "boar"}),
        (BloodGemTarget.ADJACENT, {"left", "right"}),
        (BloodGemTarget.ALL_FRIENDLY_QUILBOAR, {"boar"}),
    ],
)
def test_a_card_plays_gems_on_whom_it_names(patch, target, expected):
    board = [
        _minion("left"),
        _minion("source", race=Race.QUILBOAR),
        _minion("right"),
        _minion("boar", race=Race.QUILBOAR),
    ]
    player = _player(board)
    source = board[1]
    # The source is a Quilboar here on purpose: ALL_FRIENDLY_QUILBOAR is
    # printed "all your **other** Quilboar", so it must leave the source out
    # while SELF is nothing but the source.
    expected = set(expected)
    _fire(patch, player, source, PlayBloodGemsEffect(target=target, count=1))
    grown = {m.card_id for m in board if m.blood_gem_attack > 0}
    assert grown == expected


def test_a_card_can_play_several_gems_at_once(patch):
    """"This plays 3 permanent Blood Gems on itself" — count, not repetition."""
    player = _player([_minion("vineweaver", 2, 2)])
    source = player.board[0]
    _fire(patch, player, source, PlayBloodGemsEffect(target=BloodGemTarget.SELF, count=3))
    assert (source.raw_attack, source.max_health) == (5, 5)


# --------------------------------------------------------------------------- #
# Reading the record back
# --------------------------------------------------------------------------- #


def test_gems_can_be_stolen_from_neighbours():
    """Gem Confiscation: the stats move, and so does the record with them."""
    thief = _minion("confiscator", 1, 1)
    victim = _minion("victim", 1, 1)
    player = _player([victim, thief])
    blood_gems.play_blood_gem_on(player, victim, count=3)
    assert (victim.raw_attack, victim.max_health) == (4, 4)

    blood_gems.steal_blood_gems(thief, [victim])
    assert (victim.raw_attack, victim.max_health) == (1, 1), "the Gems left the victim"
    assert (victim.blood_gem_attack, victim.blood_gem_health) == (0, 0)
    assert (thief.raw_attack, thief.max_health) == (4, 4)
    assert (thief.blood_gem_attack, thief.blood_gem_health) == (3, 3)


def test_stealing_takes_only_gem_stats_not_other_buffs():
    thief = _minion("confiscator", 1, 1)
    victim = _minion("victim", 1, 1)
    victim.bonus_attack += 10  # a buff from somewhere else entirely
    player = _player([victim, thief])
    blood_gems.play_blood_gem_on(player, victim)

    blood_gems.steal_blood_gems(thief, [victim])
    assert victim.raw_attack == 11, "the unrelated +10 stays put"
    assert thief.raw_attack == 2


def test_a_blood_gem_is_not_a_tavern_spell():
    """The distinction the card pool depends on — see SpellCard."""
    gem = blood_gems.make_blood_gem()
    assert gem.is_blood_gem
    assert not gem.is_tavern_spell
