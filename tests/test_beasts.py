"""The Beast family, whose shared trick is paying a minion as it is summoned."""

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
from src.bg_recruitment.standing_bonuses import settle_standing_bonuses
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


def _beast(card_id="b", atk=1, hp=1) -> Minion:
    return Minion(card_id=card_id, base_attack=atk, base_health=hp, tier=1, race=Race.BEAST)


def _wall(hp=30, atk=0):
    return Minion(card_id="wall", base_attack=atk, base_health=hp, tier=1)


def _fight(board_0, board_1, patch, seats=None):
    survivors: List[Minion] = []
    deaths: List[tuple] = []
    kwargs = {"seats": seats} if seats is not None else {}
    simulate_battle(
        board_0, board_1, p0_has_initiative=True, rng=np.random.default_rng(0),
        patch=patch, p0_board_out=survivors, death_log=deaths, **kwargs,
    )
    return survivors, deaths


def _seats(patch, player):
    return (PlayerCombatSeat(player, patch=patch), PlayerCombatSeat(_player(patch)))


def _lobsters(survivors):
    """Every Tasty Lobster the Hoarding Hyena put on the board, left to right."""
    return [m for m in survivors if m.card_id == "BG36_202"]


# --------------------------------------------------------------------------- #
# Paying the minion that arrives
# --------------------------------------------------------------------------- #


def test_banana_slamma_doubles_a_summoned_beasts_attack(patch):
    hyena = _card(patch, "BG36_210")  # Rally: summon a 1/1 Tasty Lobster
    slamma = _card(patch, "BG26_802")
    survivors, _ = _fight([hyena, slamma], [_wall(hp=30)], patch)
    lobsters = _lobsters(survivors)
    assert lobsters and all(m.raw_attack == 2 for m in lobsters)


def test_banana_slamma_leaves_a_summoned_non_beast_alone(patch):
    # Harmless Bonehead dies and leaves 1/1 Undead Skeletons.
    bonehead = _card(patch, "BG28_300")
    slamma = _card(patch, "BG26_802")
    survivors, _ = _fight([bonehead, slamma], [_wall(hp=2, atk=2)], patch)
    bones = [m for m in survivors if m.card_id == "BG_ICC_026t"]
    assert bones and all(b.raw_attack == 1 for b in bones)


def test_golden_banana_slamma_triples_rather_than_quadruples(patch):
    (ability,) = patch.triple_merge_golden_abilities("BG26_802")
    assert ability.effect.factor == 3


def test_stalwart_kodo_hands_over_its_own_stats(patch):
    hyena = _card(patch, "BG36_210")
    kodo = _card(patch, "BG34_322")  # 16/32
    survivors, _ = _fight([hyena, kodo], [_wall(hp=30)], patch)
    lobsters = _lobsters(survivors)
    assert lobsters and all((m.raw_attack, m.max_health) == (17, 33) for m in lobsters)


def test_stalwart_kodo_stops_after_three_summons(patch):
    hyena = _card(patch, "BG36_210")
    kodo = _card(patch, "BG34_322")
    # A long fight, so the Hyena's Rally fires more times than the Kodo can pay.
    survivors, _ = _fight([hyena, kodo], [_wall(hp=300)], patch)
    lobsters = _lobsters(survivors)
    assert len(lobsters) > 3  # the fight really did run past the charges
    assert sum(1 for m in lobsters if m.raw_attack == 17) == 3
    assert all(m.raw_attack in (1, 17) for m in lobsters)


def test_stalwart_kodo_charges_refill_next_combat(patch):
    player = _player(patch)
    hyena, kodo = _card(patch, "BG36_210"), _card(patch, "BG34_322")
    player.board = [hyena, kodo]
    _fight([hyena, kodo], [_wall(hp=30)], patch, seats=_seats(patch, player))
    # The charge is spent by the combat copy; the seat's own body never counts.
    assert kodo.combat_uses_left == -1


def test_golden_kodo_doubles_the_stats_and_keeps_three_charges(patch):
    (ability,) = patch.triple_merge_golden_abilities("BG34_322")
    assert (ability.effect.charges, ability.effect.factor) == (3, 2)


def test_lurking_leviathan_improves_as_it_pays(patch):
    hyena = _card(patch, "BG36_210")
    leviathan = _card(patch, "BG35_602")
    survivors, _ = _fight([hyena, leviathan], [_wall(hp=30)], patch)
    # +2, then +4, then +6 — newest Lobster first, since each is summoned
    # beside the Hyena.
    assert [m.raw_attack for m in _lobsters(survivors)] == [7, 5, 3]


def test_lurking_leviathan_keeps_its_improve_after_the_fight(patch):
    player = _player(patch)
    hyena, leviathan = _card(patch, "BG36_210"), _card(patch, "BG35_602")
    player.board = [hyena, leviathan]
    _fight([hyena, leviathan], [_wall(hp=30)], patch, seats=_seats(patch, player))
    assert leviathan.self_improves == 3


def test_leviathan_ignores_a_summoned_non_beast(patch):
    bonehead = _card(patch, "BG28_300")
    leviathan = _card(patch, "BG35_602")
    survivors, _ = _fight([bonehead, leviathan], [_wall(hp=2, atk=2)], patch)
    bones = [m for m in survivors if m.card_id == "BG_ICC_026t"]
    assert bones and all(b.raw_attack == 1 for b in bones)


# --------------------------------------------------------------------------- #
# Deathrattles
# --------------------------------------------------------------------------- #


def test_sewer_lord_leaves_rats_that_leave_half_shells(patch):
    lord = _card(patch, "BG35_604")
    survivors, deaths = _fight([lord], [_wall(hp=1, atk=40)], patch)
    assert [m.card_id for m in survivors] == ["BG19_010", "BG19_010"]
    assert ("0", "BG35_604") != deaths[0]  # the Lord itself died
    # The Rats survive here, so their own deathrattle is checked on its own.
    rat = _card(patch, "BG19_010")
    survivors, _ = _fight([rat], [_wall(hp=1, atk=40)], patch)
    assert [m.card_id for m in survivors] == ["BG19_010t"]


def test_half_shell_has_taunt(patch):
    shell = _card(patch, "BG19_010t")
    assert Keyword.TAUNT in shell.all_keywords
    assert (shell.raw_attack, shell.max_health) == (2, 3)


def test_golden_sewer_lord_leaves_two_golden_rats_not_four(patch):
    (ability,) = patch.triple_merge_golden_abilities("BG35_604")
    assert (ability.effect.token_id, ability.effect.count) == ("BG19_010_G", 2)
    golden_rat = _card(patch, "BG19_010_G")
    assert golden_rat.is_golden and (golden_rat.raw_attack, golden_rat.max_health) == (6, 4)


def test_turquoise_skitterer_raises_the_beetles_this_game(patch):
    player = _player(patch)
    skitterer = _card(patch, "BG31_809")
    player.board = [skitterer]
    _fight([skitterer], [_wall(hp=1, atk=40)], patch, seats=_seats(patch, player))
    beetle = _card(patch, "BG28_603t")  # bought after the fight
    player.board = [beetle]
    settle_standing_bonuses(player)
    assert (beetle.raw_attack, beetle.max_health) == (7, 7)


def test_goldrinn_pays_only_the_beasts(patch):
    goldrinn = _card(patch, "BGS_018")  # 8/8
    beast = _beast("b", 1, 40)
    plain = Minion(card_id="p", base_attack=1, base_health=40, tier=1)
    survivors, _ = _fight([goldrinn, beast, plain], [_wall(hp=8, atk=8)], patch)
    grown = next(m for m in survivors if m.card_id == "b")
    assert (grown.raw_attack, grown.max_health) == (9, 48)
    assert next(m for m in survivors if m.card_id == "p").raw_attack == 1


# --------------------------------------------------------------------------- #
# Watching an attack
# --------------------------------------------------------------------------- #


def test_ravaging_scorpid_raises_the_beetles_on_every_swing(patch):
    player = _player(patch)
    scorpid = _card(patch, "BG36_209")
    player.board = [scorpid]
    _fight(
        [scorpid, _beast("b", 1, 40)], [_wall(hp=30)], patch, seats=_seats(patch, player)
    )
    beetle = _card(patch, "BG28_603t")
    player.board = [beetle]
    settle_standing_bonuses(player)
    # Three swings landed before the wall fell, at +3/+3 apiece.
    assert (beetle.raw_attack, beetle.max_health) == (11, 11)


def test_deathstrider_fires_the_left_most_deathrattle(patch):
    lord = _card(patch, "BG35_604")  # the left-most Deathrattle
    hyena = _card(patch, "BG36_210")  # a Rally minion
    strider = _card(patch, "BG36_208")
    survivors, _ = _fight([lord, hyena, strider], [_wall(hp=30)], patch)
    # The Lord is still standing and its Rats are on the board anyway.
    assert any(m.card_id == "BG35_604" for m in survivors)
    assert sum(1 for m in survivors if m.card_id == "BG19_010") == 2


def test_deathstrider_ignores_a_plain_attacker(patch):
    lord = _card(patch, "BG35_604")
    plain = Minion(card_id="p", base_attack=5, base_health=40, tier=1)
    strider = _card(patch, "BG36_208")
    survivors, _ = _fight([lord, plain, strider], [_wall(hp=30)], patch)
    assert not any(m.card_id == "BG19_010" for m in survivors)


def test_golden_deathstrider_fires_twice(patch):
    (ability,) = patch.triple_merge_golden_abilities("BG36_208")
    assert ability.effect.repeats == 2


# --------------------------------------------------------------------------- #
# The tavern
# --------------------------------------------------------------------------- #


def test_hoarding_hyena_summons_a_lobster_on_its_swing(patch):
    hyena = _card(patch, "BG36_210")
    survivors, _ = _fight([hyena], [_wall(hp=30)], patch)
    assert _lobsters(survivors)


def test_golden_hoarding_hyena_summons_the_golden_lobster(patch):
    (ability,) = patch.triple_merge_golden_abilities("BG36_210")
    assert (ability.effect.token_id, ability.effect.count) == ("BG36_202_G", 1)
    assert _card(patch, "BG36_202_G").is_golden


def test_snarky_shark_refreshes_with_a_bait_its_beast_eats(patch, triggers):
    from src.bg_recruitment.fishbait import FISHBAIT_CARD_ID

    shark = _card(patch, "BG36_206")
    beast = _beast("b", 2, 3)
    player = _player(patch, board=[beast, shark])
    triggers.fire_on_sell(shark, player)
    assert (beast.raw_attack, beast.max_health) == (7, 8)  # the bait's +5/+5
    assert not any(c is not None and c.card_id == FISHBAIT_CARD_ID for c in player.shop)
    assert any(c is not None for c in player.shop)  # and the tavern was refreshed


def test_snarky_shark_does_not_feed_itself(patch, triggers):
    shark = _card(patch, "BG36_206")  # a Beast, and the only one
    player = _player(patch, board=[shark])
    triggers.fire_on_sell(shark, player)
    assert shark.raw_attack == shark.base_attack


def test_snarky_shark_bait_stays_when_no_beast_can_eat_it(patch, triggers):
    from src.bg_recruitment.fishbait import FISHBAIT_CARD_ID

    shark = _card(patch, "BG36_206")
    plain = Minion(card_id="p", base_attack=1, base_health=1, tier=1)
    player = _player(patch, board=[plain, shark])
    triggers.fire_on_sell(shark, player)
    assert any(c is not None and c.card_id == FISHBAIT_CARD_ID for c in player.shop)
