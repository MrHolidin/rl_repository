"""Rules the audit found wrong, pinned so they cannot drift back.

Four findings from a five-way audit against the real game. Each is a rule the
engine had backwards, not a card binding — so each is pinned here rather than
in a per-tribe file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.bg_catalog.patch_context import PatchContext
from src.bg_core.effects import Keyword
from src.bg_core.minion import Minion
from src.bg_lobby.player import PendingChoiceKind, PlayerPhase, PlayerState
from src.bg_player_turn.engine import PlayerTurnContext, PlayerTurnEngine
from src.bg_recruitment.shop_triggers import ShopTriggers
from tests.minibg_helpers import simulate_battle
import src.envs.minibg.actions as A

PATCH_DIR = Path("data/bgcore/36_2_0_248348")


@pytest.fixture(scope="module")
def patch() -> PatchContext:
    return PatchContext.load(PATCH_DIR)


def _player(patch, board=(), **kw) -> PlayerState:
    base = dict(
        health=30, gold=10, tavern_tier=6, board=list(board), shop=[None] * 7,
        hand=[None] * 10, phase=PlayerPhase.SHOP, shop_actions_used=0,
        ruleset=patch.meta.ruleset,
    )
    base.update(kw)
    return PlayerState(**base)


def _plain(card_id="m", atk=1, hp=1) -> Minion:
    return Minion(card_id=card_id, base_attack=atk, base_health=hp, tier=1)


# --------------------------------------------------------------------------- #
# Reborn returns a fresh copy of the printed card
# --------------------------------------------------------------------------- #


def test_reborn_drops_everything_the_body_had_gained(patch):
    body = patch.make_minion("BG25_010t")  # Helping Hand, printed 2/1 Reborn
    body.bonus_attack += 7
    body.bonus_health += 19
    body.granted_keywords = frozenset({Keyword.TAUNT})
    survivors: list = []
    simulate_battle(
        [body],
        [_plain("k", 25, 1)],
        p0_has_initiative=False,
        rng=np.random.default_rng(0),
        patch=patch,
        p0_board_out=survivors,
    )
    (back,) = survivors
    assert (back.raw_attack, back.max_health) == (2, 1)
    assert back.max_health - back.damage_taken == 1
    assert Keyword.TAUNT not in back.all_keywords


def test_reborn_brings_a_printed_divine_shield_back_up(patch):
    """A fresh copy has never been hit, so its printed shield is armed."""
    import src.bg_combat.battle.effects as fx

    seen: list = []
    original = fx._reborn_copy
    fx._reborn_copy = lambda rt, dead: (
        lambda fresh: (seen.append(fresh.has_shield), fresh)[1]
    )(original(rt, dead))
    try:
        body = patch.make_minion("BG_BOT_911")  # printed Divine Shield + Taunt
        body.granted_keywords = frozenset(body.granted_keywords | {Keyword.REBORN})
        simulate_battle(
            [body],
            [_plain(f"k{i}", 40, 1) for i in range(3)],
            p0_has_initiative=False,
            rng=np.random.default_rng(0),
            patch=patch,
            p0_board_out=[],
        )
    finally:
        fx._reborn_copy = original
    assert seen == [True]


def test_a_reborn_minion_dies_twice(patch):
    """Two deaths, so the cards that watch a friendly die see two."""
    deaths: list = []
    simulate_battle(
        [patch.make_minion("BG25_010t")],
        [_plain("w", 20, 200)],
        p0_has_initiative=True,
        rng=np.random.default_rng(0),
        patch=patch,
        p0_board_out=[],
        death_log=deaths,
    )
    assert [cid for side, cid in deaths if side == 0] == ["BG25_010t", "BG25_010t"]


# --------------------------------------------------------------------------- #
# Overkill is printed on a swing
# --------------------------------------------------------------------------- #


def test_a_counter_swing_cannot_overkill(patch):
    """Wildfire Elemental is "After **this attacks** and kills a minion"."""
    import src.bg_combat.battle.engine as engine
    from src.bg_combat.battle.events import Overkill

    sides: list = []
    original = engine._dispatch

    def traced(rt, ev):
        if isinstance(ev, Overkill):
            sides.append(ev.attacker_side_idx)
        return original(rt, ev)

    engine._dispatch = traced
    try:
        wildfire = patch.make_minion("BGS_126")  # 6/3
        wildfire.bonus_health += 500
        # One minion a side, so the fight is over the moment the 1/1 dies to
        # the counter — the only swing that could overkill is the answer.
        simulate_battle(
            [_plain("atk", 1, 1)],
            [wildfire],
            p0_has_initiative=True,
            rng=np.random.default_rng(0),
            patch=patch,
            p0_board_out=[],
        )
    finally:
        engine._dispatch = original
    assert sides == []


# --------------------------------------------------------------------------- #
# A Divine Shield prevents the damage, so nothing "takes damage"
# --------------------------------------------------------------------------- #


def test_a_popped_shield_is_not_damage_taken(patch):
    """Very Hungry Winterfinner answers damage; a shield stops there being any."""
    from src.bg_combat.battle.seat import RecordingSeat

    def hand_buffs(with_shield: bool):
        finner = patch.make_minion("BG29_300")
        finner.bonus_health += 40
        if with_shield:
            finner.keywords = frozenset(finner.keywords | {Keyword.SHIELD})
            finner.has_shield = True
        seat = RecordingSeat()
        simulate_battle(
            [finner],
            [_plain("k", 3, 40)],
            p0_has_initiative=False,
            rng=np.random.default_rng(0),
            patch=patch,
            p0_board_out=[],
            seats=(seat, RecordingSeat()),
            max_attacks=1,
        )
        return seat.hand_buffs

    # Exactly one hit lands either way: bare, it is damage; shielded, it is not.
    assert hand_buffs(with_shield=False) == [(2, 1)]
    assert hand_buffs(with_shield=True) == []


# --------------------------------------------------------------------------- #
# Choose One reaches its own resolver
# --------------------------------------------------------------------------- #


def _open_choose_one(patch):
    triggers = ShopTriggers(np.random.default_rng(0), patch=patch)
    miner = patch.make_minion("BG31_320")  # Choose One: 2 Blood Gems; or Gem Day
    player = _player(patch, board=[miner])
    triggers.fire_on_place(player=player, placed=miner, shop_excluded_race=None)
    return player, triggers


def test_a_choose_one_offers_only_its_two_options(patch):
    player, _ = _open_choose_one(patch)
    assert player.pending_choice.kind is PendingChoiceKind.CHOOSE_ONE
    legal = PlayerTurnEngine().legal_actions(player, None)
    assert legal == [int(A.Action.DISCOVER_PICK_0), int(A.Action.DISCOVER_PICK_1)]


@pytest.mark.parametrize(
    "pick,expected",
    [(0, ["BG20_GEM", "BG20_GEM"]), (1, ["BG31_893"])],
)
def test_a_choose_one_pick_resolves_the_half_it_named(patch, pick, expected):
    """Nothing routed a pick to ``resolve_choose_one``, so every Choose One
    card fell through to the Discover path and dropped the chosen half."""
    player, triggers = _open_choose_one(patch)
    ctx = PlayerTurnContext(
        rng=np.random.default_rng(0), triggers=triggers, patch=patch
    )
    PlayerTurnEngine().apply(player, int(A.Action.DISCOVER_PICK_0) + pick, ctx)
    assert [c.card_id for c in player.hand if c is not None] == expected
    assert player.pending_choice is None


# --------------------------------------------------------------------------- #
# "A friendly minion" includes the one it is talking about
# --------------------------------------------------------------------------- #


def test_a_watcher_hears_its_own_attack(patch):
    """Cage Gnawer is "whenever **a** friendly Beast attacks", and is one."""
    gnawer = patch.make_minion("BG36_211")  # printed 2/7 Beast
    gnawer.bonus_health += 50
    survivors: list = []
    simulate_battle(
        [gnawer],
        [_plain("wall", 0, 200)],
        p0_has_initiative=True,
        rng=np.random.default_rng(0),
        patch=patch,
        p0_board_out=survivors,
    )
    assert survivors[0].raw_attack > 2


def test_a_watcher_that_says_another_does_not(patch):
    """Roaring Recruiter is "whenever **another** friendly Dragon attacks"."""
    recruiter = patch.make_minion("BG29_816")  # printed 2/8 Dragon
    recruiter.bonus_health += 50
    survivors: list = []
    simulate_battle(
        [recruiter],
        [_plain("wall", 0, 200)],
        p0_has_initiative=True,
        rng=np.random.default_rng(0),
        patch=patch,
        p0_board_out=survivors,
    )
    assert survivors[0].raw_attack == 2


# --------------------------------------------------------------------------- #
# Goldens whose upgrade is not a bigger number
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("card_id", ["BG24_500", "BG29_810"])
def test_a_golden_that_reaches_one_more_body_keeps_its_numbers(patch, card_id):
    """"Give **two** other friendly Dragons +2/+2" — the count moves, not the buff."""
    (plain,) = patch.effects[card_id]
    (golden,) = patch.triple_merge_golden_abilities(card_id)
    assert golden.effect.limit == plain.effect.limit + 1
    assert (golden.effect.attack, golden.effect.health) == (
        plain.effect.attack,
        plain.effect.health,
    )
    assert golden.effect.grant_keyword is plain.effect.grant_keyword


def test_a_limit_does_not_move_when_the_golden_says_nothing_about_it(patch):
    """Tasty Lobster's Golden pays double; it still reaches two Beasts."""
    (plain, _bump) = patch.effects["BG36_202"]
    (golden, _g_bump) = patch.triple_merge_golden_abilities("BG36_202")
    assert golden.effect.effect.limit == plain.effect.effect.limit


@pytest.mark.parametrize(
    "patch_dir,card_id",
    [
        ("data/bgcore/36_2_0_248348", "BG36_620"),  # Boom-in-a-Box
        ("data/bgcore/19_6_0_74257", "FP1_024"),  # Unstable Ghoul
    ],
)
def test_a_golden_that_says_twice_deals_two_instances(patch_dir, card_id):
    """One Divine Shield eats one instance, not the whole doubled hit."""
    ctx = PatchContext.load(Path(patch_dir))
    (plain,) = [a for a in ctx.effects[card_id] if hasattr(a.effect, "repeats")]
    (golden,) = [
        a for a in ctx.triple_merge_golden_abilities(card_id)
        if hasattr(a.effect, "repeats")
    ]
    assert golden.effect.amount == plain.effect.amount
    assert golden.effect.repeats == 2


def test_obsidian_ravager_hits_one_neighbour_and_its_golden_hits_both(patch):
    from src.bg_core.effects import Keyword as _Kw

    def splashed(golden: bool, seed: int):
        ravager = patch.make_minion("BG27_017")
        if golden:
            ravager.abilities = patch.triple_merge_golden_abilities("BG27_017")
        ravager.bonus_attack += 5
        ravager.bonus_health += 500
        enemies = [_plain(f"e{i}", 0, 100000) for i in range(3)]
        enemies[1].keywords = frozenset({_Kw.TAUNT})  # force the middle target
        out: list = []
        simulate_battle(
            [ravager], enemies,
            p0_has_initiative=True,
            rng=np.random.default_rng(seed),
            patch=patch,
            p1_board_out=out,
            max_attacks=1,
        )
        return sum(1 for m in out if m.damage_taken)

    assert {splashed(False, s) for s in range(6)} == {2}
    assert {splashed(True, s) for s in range(6)} == {3}


# --------------------------------------------------------------------------- #
# Four more cards the derivation could not say
# --------------------------------------------------------------------------- #


def test_golden_sly_raptor_summons_one_beast_at_double_stats(patch):
    """"Summon **a** random Beast. Set its stats to 12/12" — one, not two."""
    (plain,) = patch.effects["BG25_806"]
    (golden,) = patch.triple_merge_golden_abilities("BG25_806")
    assert golden.effect.count == plain.effect.count == 1
    assert (golden.effect.set_attack, golden.effect.set_health) == (12, 12)


def test_golden_stone_age_slab_triples(patch):
    """"+10/+10 and **triple** its stats" — the multiple moves, the flat stays."""
    (plain,) = patch.effects["BG34_950"]
    (golden,) = patch.triple_merge_golden_abilities("BG34_950")
    assert (golden.effect.attack, golden.effect.health) == (10, 10)
    assert (plain.effect.stat_multiplier, golden.effect.stat_multiplier) == (2, 3)


def test_stone_age_slab_pays_what_it_prints(patch, monkeypatch):
    """A 1/1 bought becomes 22/22, and 33/33 off the Golden."""
    from src.bg_recruitment.shop_triggers import ShopTriggers

    def bought_stats(ability):
        slab = patch.make_minion("BG34_950")
        slab.abilities = (ability,)
        player = _player(patch, board=[slab])
        bought = _plain("b", 1, 1)
        ShopTriggers(np.random.default_rng(0), patch=patch).fire_on_bought(
            player, bought
        )
        return (bought.raw_attack, bought.max_health)

    assert bought_stats(patch.effects["BG34_950"][0]) == (22, 22)
    assert bought_stats(patch.triple_merge_golden_abilities("BG34_950")[0]) == (33, 33)


def test_the_jailbird_golem_is_a_golem(patch):
    """Pointing at its own card id put a second tier-5 Quilboar on the board."""
    from src.bg_recruitment.blood_gems import play_blood_gem_on
    from src.bg_recruitment.combat_seat import PlayerCombatSeat

    juggernaut = patch.make_minion("BG36_333")
    player = _player(patch, board=[juggernaut])
    play_blood_gem_on(player, juggernaut, count=3, patch=patch)
    juggernaut.bonus_health += 200
    survivors: list = []
    simulate_battle(
        [juggernaut],
        [_plain("w", 0, 400)],
        p0_has_initiative=True,
        rng=np.random.default_rng(0),
        patch=patch,
        p0_board_out=survivors,
        seats=(PlayerCombatSeat(player, patch=patch), PlayerCombatSeat(_player(patch))),
    )
    golems = [m for m in survivors if m.card_id != "BG36_333"]
    assert golems
    assert all(m.race is None for m in golems)
    assert all((m.raw_attack, m.max_health) == (3, 3) for m in golems)


def test_humming_bird_pays_a_beast_summoned_later(patch):
    """"For the rest of this combat" — the buff stays open."""
    import src.bg_combat.battle.engine as engine
    from src.bg_combat.battle.events import MinionSummoned

    def summoned_stats(with_bird: bool):
        seen: list = []
        original = engine._dispatch

        def traced(rt, ev):
            out = original(rt, ev)
            if isinstance(ev, MinionSummoned):
                m = rt.find_minion(ev.side_idx, ev.instance_id)
                if m is not None:
                    seen.append((m.raw_attack, m.max_health))
            return out

        engine._dispatch = traced
        try:
            board = [patch.make_minion("BG31_803")]  # Deathrattle: a 2/2 Beetle
            if with_bird:
                bird = patch.make_minion("BG26_805")
                bird.bonus_health += 400
                board.insert(0, bird)
            simulate_battle(
                board,
                [_plain("w", 2, 400)],
                p0_has_initiative=True,
                rng=np.random.default_rng(0),
                patch=patch,
                p0_board_out=[],
            )
        finally:
            engine._dispatch = original
        return seen

    assert summoned_stats(with_bird=False) == [(2, 2)]
    assert summoned_stats(with_bird=True) == [(3, 2)]
