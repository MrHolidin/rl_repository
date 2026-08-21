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
    fx._reborn_copy = lambda rt, side_idx, dead: (
        lambda fresh: (seen.append(fresh.has_shield), fresh)[1]
    )(original(rt, side_idx, dead))
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


# --------------------------------------------------------------------------- #
# The 36.2.0 numbers
# --------------------------------------------------------------------------- #


def test_the_tavern_stops_at_tier_six(patch):
    from src.bg_player_turn.engine import PlayerTurnEngine

    rs = patch.meta.ruleset
    assert rs.max_tier == 6
    engine = PlayerTurnEngine()
    at_six = _player(patch, tavern_tier=6, gold=99)
    at_five = _player(patch, tavern_tier=5, gold=99)
    assert int(A.Action.LEVEL_UP) in engine.legal_actions(at_five, rs)
    assert int(A.Action.LEVEL_UP) not in engine.legal_actions(at_six, rs)


def test_the_upgrade_ladder_is_the_patchs_own(patch):
    """28.2 cut 4->5 from 11 to 10; 34.2 put a Gold back on 4->5 and on 5->6."""
    rs = patch.meta.ruleset
    assert {t: rs.level_up_cost(t) for t in range(1, 6)} == {
        1: 5, 2: 7, 3: 8, 4: 11, 5: 12
    }


def test_a_triple_at_the_top_discovers_at_the_top(patch):
    from src.bg_recruitment.discover_pool import triple_reward_discover_tier

    assert triple_reward_discover_tier(5, patch=patch) == 6
    assert triple_reward_discover_tier(6, patch=patch) == 6


# --------------------------------------------------------------------------- #
# "Give N friendly minions" picks N of them, not the first N
# --------------------------------------------------------------------------- #


def _bounty_hits(patch, spell_id, seed, n_board=6):
    board = [_plain(f"m{i}") for i in range(n_board)]
    player = _player(patch, board=board)
    triggers = ShopTriggers(np.random.default_rng(seed), patch=patch)
    for ability in patch.tavern_spells[spell_id].abilities:
        triggers.apply_shop_effect(player, None, ability.effect, placed=None)
    return tuple(m.max_health > 1 or m.raw_attack > 1 for m in board)


def test_four_friendly_minions_is_four_at_random(patch):
    seen = {_bounty_hits(patch, "BG33_811", seed) for seed in range(8)}
    assert len(seen) > 1  # not the same four every time
    assert all(sum(hits) == 4 for hits in seen)


def test_your_left_most_minion_is_always_the_left_most(patch):
    seen = {_bounty_hits(patch, "BG33_813", seed, n_board=4) for seed in range(6)}
    assert seen == {(True, False, False, False)}


def test_a_positional_buff_in_combat_stays_positional(patch):
    """Thousandth Paper Drake says "your **left-most** Dragon"."""
    (ability,) = patch.effects["BG29_810"]
    assert ability.effect.leftmost is True
    assert patch.effects["BG24_500"][0].effect.leftmost is False


# --------------------------------------------------------------------------- #
# Three kinds of Discover, three tier rules
# --------------------------------------------------------------------------- #


def _tribe_discover_tiers(patch, tavern_tier, tribe, rolls=200):
    from src.bg_recruitment.discover_pool import roll_discover_tribe_triple

    rng = np.random.default_rng(0)
    seen = set()
    for _ in range(rolls):
        for cid in roll_discover_tribe_triple(
            rng, tavern_tier, None, tribe=tribe, patch=patch
        ):
            seen.add(patch.templates[cid].tier)
    return seen


@pytest.mark.parametrize("tavern_tier", [1, 3, 4, 5, 6])
def test_a_tribe_discover_never_reaches_above_the_seat(patch, tavern_tier):
    """A bare "Discover a Beast" prints no tier, so it takes the default."""
    from src.bg_core.minion import Race as _R

    assert max(_tribe_discover_tiers(patch, tavern_tier, _R.BEAST)) <= tavern_tier


def test_a_tribe_discover_offers_what_there_is(patch):
    """Capped at the seat, Tier 1 holds only two Beasts — two options, not an error."""
    from src.bg_recruitment.discover_pool import roll_discover_tribe_triple
    from src.bg_core.minion import Race as _R

    opts = roll_discover_tribe_triple(
        np.random.default_rng(0), 1, None, tribe=_R.BEAST, patch=patch
    )
    assert 0 < len(opts) < 3


def test_the_triple_reward_is_one_tier_up_and_fixed_when_placed(patch):
    """The other rule: +1, and snapshotted so a later upgrade cannot move it."""
    from src.bg_recruitment.discover_pool import triple_reward_discover_tier
    from src.bg_recruitment.triples import resolve_triples_loop

    assert {t: triple_reward_discover_tier(t, patch=patch) for t in range(1, 7)} == {
        1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 6
    }
    player = _player(patch, tavern_tier=3)
    player.board = [patch.make_minion("BGS_115") for _ in range(3)]
    resolve_triples_loop(player, patch=patch)
    reward = next(
        c for c in player.hand if c is not None and getattr(c, "triple_discover_tier", 0)
    )
    assert reward.triple_discover_tier == 4
    player.tavern_tier = 6  # upgrading afterwards does not move it
    assert reward.triple_discover_tier == 4


def test_a_printed_tier_discover_ignores_the_seat(patch):
    """"Discover a Tier 1 minion" is a Tier 1 minion at any tavern tier."""
    (ability,) = patch.tavern_spells["BG33_101"].abilities
    assert ability.effect.tier == 1


def test_a_discover_is_weighted_by_what_is_left_in_the_pool(patch):
    """A Discover draws from the shared pool, so a card's chance is its
    remaining copies — which skews the spread *low*, because the pool holds
    15 copies of each Tier 1 minion against 7 of each Tier 6."""
    from collections import Counter

    from src.bg_core.minion import Race as _R
    from src.bg_lobby.shared_pool import build_initial_shared_pool
    from src.bg_recruitment.discover_pool import (
        roll_discover_tribe_triple,
        tribe_discover_card_ids,
    )

    pool = build_initial_shared_pool(patch=patch)
    rng = np.random.default_rng(0)
    seen: Counter = Counter()
    for _ in range(1500):
        for cid in roll_discover_tribe_triple(
            rng, 6, None, tribe=_R.BEAST, shared_pool=pool, patch=patch
        ):
            seen[patch.templates[cid].tier] += 1

    copies = patch.meta.pool_copies_by_tier
    expected: Counter = Counter()
    for cid in tribe_discover_card_ids(_R.BEAST, patch=patch):
        tier = patch.templates[cid].tier
        if tier <= 6:
            expected[tier] += copies[tier]

    drawn = sum(seen.values())
    total = sum(expected.values())
    for tier in expected:
        assert abs(seen[tier] / drawn - expected[tier] / total) < 0.03, tier
    # And the low tiers really are the common ones, which is the whole point.
    assert seen[1] > seen[6]


def test_a_discover_without_a_pool_is_flat(patch):
    """Nothing to weigh by, so every eligible card is equally likely — and in
    particular not weighted by how close its tier is to the seat's."""
    from collections import Counter

    from src.bg_core.minion import Race as _R
    from src.bg_recruitment.discover_pool import roll_discover_tribe_triple

    rng = np.random.default_rng(0)
    seen: Counter = Counter()
    for _ in range(1500):
        for cid in roll_discover_tribe_triple(rng, 6, None, tribe=_R.BEAST, patch=patch):
            seen[patch.templates[cid].tier] += 1
    # Tier 6 holds 3 Beasts and Tier 3 holds 4; flat means the 4 win.
    assert seen[3] > seen[6]


def test_every_pool_draw_uses_the_same_weighting(patch):
    """The tavern refresh has drawn by remaining copies all along; the Discover
    rolls and the forced-tribe slot were picking uniformly among distinct card
    ids beside it."""
    from collections import Counter

    from src.bg_lobby.shared_pool import build_initial_shared_pool
    from src.bg_recruitment.discover_pool import draw_from_pool

    pool = build_initial_shared_pool(patch=patch)
    # Two cards of different tiers: the Tier 1 has 15 copies, the Tier 6 has 7.
    # Picked from a sorted list rather than the id *set*, and each draw gets its
    # own generator, so the ratio measures the weighting and nothing else.
    cheap = next(c for c in sorted(patch.pool_ids) if patch.templates[c].tier == 1)
    dear = next(c for c in sorted(patch.pool_ids) if patch.templates[c].tier == 6)
    expected = patch.meta.pool_copies_by_tier[1] / patch.meta.pool_copies_by_tier[6]

    seen = Counter(
        draw_from_pool(
            np.random.default_rng(seed), [cheap, dear], 1, shared_pool=pool
        )[0]
        for seed in range(4000)
    )
    assert abs(seen[cheap] / seen[dear] - expected) < 0.25

    flat = Counter(
        draw_from_pool(np.random.default_rng(seed), [cheap, dear], 1)[0]
        for seed in range(4000)
    )
    assert abs(flat[cheap] / flat[dear] - 1.0) < 0.15  # no pool, no weighting


def test_a_forced_tribe_slot_draws_like_an_ordinary_one(patch):
    """Ysera's extra Dragon is a tavern offer; being forced to a tribe does not
    change how the pool is read."""
    import inspect

    from src.bg_recruitment import shop

    source = inspect.getsource(shop._fill_forced_tribe_slot)
    assert "draw_from_pool" in source
    assert "rng.integers" not in source


# --------------------------------------------------------------------------- #
# Six numbers and knobs the package carried but nothing read
# --------------------------------------------------------------------------- #


def test_the_pool_holds_fifteen_of_each_tier_one(patch):
    assert patch.meta.pool_copies_by_tier[1] == 15


def test_a_seat_starts_at_thirty_and_takes_armor_from_its_hero(patch):
    assert patch.meta.ruleset.starting_health == 30
    # The classic packages predate armor and keep their flat 40.
    classic = PatchContext.load(Path("data/bgcore/19_6_0_74257"))
    assert classic.meta.ruleset.starting_health == 40


def test_the_damage_cap_ramps_and_lifts_in_the_top_four(patch):
    rs = patch.meta.ruleset
    assert [rs.damage_cap_for_round(n) for n in (1, 3, 4, 7, 8, 12)] == [
        5, 5, 10, 10, 15, 15
    ]
    assert rs.effective_damage_cap(12, alive_count=5) == 15
    assert rs.effective_damage_cap(12, alive_count=4) > 100  # lifted


@pytest.mark.parametrize(
    "field,call",
    [
        ("buy_cost", lambda eco, p, m: eco.effective_buy_cost(p)),
        ("roll_cost", lambda eco, p, m: eco.effective_roll_cost(p)),
        ("sell_reward", lambda eco, p, m: eco.effective_sell_reward(m, p)),
    ],
)
def test_the_economy_knobs_are_read_not_ignored(patch, field, call):
    """They were fields on Ruleset that nothing consulted — the module
    constants were used directly, so a package setting them did nothing."""
    from dataclasses import replace

    from src.bg_recruitment import economy

    player = _player(patch, ruleset=replace(patch.meta.ruleset, **{field: 37}))
    assert call(economy, player, _plain("m")) == 37


def test_gold_on_a_tavern_spell_is_gold_spent(patch):
    from src.bg_recruitment.tavern_spells import buy_tavern_spell, offer_tavern_spells

    player = _player(patch)
    offer_tavern_spells(player, rng=np.random.default_rng(0), patch=patch)
    cost = player.tavern_spell_offers[0].cost
    buy_tavern_spell(player, 0, patch=patch)
    assert player.gold_spent_this_turn == cost


def test_gold_on_an_activate_is_gold_spent(patch):
    from src.bg_recruitment.activate import activate_minion

    castaway = patch.make_minion("BG36_342")  # Activate (2)
    player = _player(patch, board=[castaway])
    activate_minion(player, 0, rng=np.random.default_rng(0), patch=patch)
    assert player.gold_spent_this_turn == 2


def test_the_offers_per_tier_come_from_the_patch(patch):
    """The count per tier is a patch number; MAX_SHOP_SLOTS is the action
    space and deliberately is not."""
    import inspect

    from src.bg_recruitment import shop

    assert "shop_offers_by_tier" in inspect.getsource(shop.shop_offers_for_tier)
    for tier, expected in patch.meta.layout.shop_offers_by_tier.items():
        if tier > patch.meta.ruleset.max_tier:
            continue
        player = _player(patch, tavern_tier=tier)
        assert shop.shop_offers_for_tier(player) == expected


# --------------------------------------------------------------------------- #
# "This game" bonuses reach a body summoned mid-combat
# --------------------------------------------------------------------------- #


def _seat_for(player):
    from src.bg_recruitment.combat_seat import PlayerCombatSeat

    return PlayerCombatSeat(player)


def test_a_token_summoned_in_combat_carries_the_seats_this_game_bonus(patch):
    """Forest Rover prints "wherever they are" — a Beetle summoned by a
    deathrattle is somewhere, and Beetles exist only as combat summons."""
    from src.bg_combat.battle.seat import RecordingSeat
    from src.bg_core.effects import ScopeKind
    from src.bg_recruitment.standing_bonuses import (
        BonusScope,
        raise_standing_bonus,
        settle_standing_bonuses,
    )

    player = _player(patch, [patch.make_minion("BG31_801")])
    raise_standing_bonus(player, BonusScope(ScopeKind.CARD, "BG28_603t"), 2, 1)
    settle_standing_bonuses(player)

    out: list = []
    simulate_battle(
        [patch.make_minion("BG31_801")],
        [_plain("killer", 40, 1)],
        p0_has_initiative=False,
        rng=np.random.default_rng(0),
        patch=patch,
        p0_board_out=out,
        seats=(_seat_for(player), RecordingSeat()),
    )
    beetle = next(m for m in out if m.card_id == "BG28_603t")
    assert (beetle.raw_attack, beetle.max_health) == (4, 3)


def test_a_summoned_copy_of_a_paid_minion_is_not_paid_twice(patch):
    """Idempotent by the body's own absorbed record, which the copy carries."""
    from src.bg_combat.battle.seat import RecordingSeat
    from src.bg_combat.battle.state import BattleSide, _CombatRuntime, battle_copy
    from src.bg_combat.battle.summon import _summon_append
    from src.bg_core.effects import ScopeKind
    from src.bg_recruitment.standing_bonuses import (
        BonusScope,
        raise_standing_bonus,
        settle_standing_bonuses,
    )

    body = patch.make_minion("BG28_603t")
    player = _player(patch, [body])
    raise_standing_bonus(player, BonusScope(ScopeKind.CARD, "BG28_603t"), 2, 1)
    settle_standing_bonuses(player)
    assert (body.raw_attack, body.max_health) == (4, 3)

    rt = _CombatRuntime(
        sides=(BattleSide([]), BattleSide([])),
        rng=np.random.default_rng(0),
        combat_board_max=7,
        damage_cap=15,
        patch=patch,
        seats=(_seat_for(player), RecordingSeat()),
    )
    clone = _summon_append(rt, 0, body)
    assert (clone.raw_attack, clone.max_health) == (4, 3)


# --------------------------------------------------------------------------- #
# Death order: deathrattle, then Avenge, then Reborn
# --------------------------------------------------------------------------- #


def test_avenge_fires_after_the_dead_minions_deathrattle(patch):
    from src.bg_combat.battle import effects as fx

    order: list = []
    fire_dr, fire_av = fx._fire_deathrattle, fx._fire_avenge
    fx._fire_deathrattle = lambda *a, **k: (order.append("dr"), fire_dr(*a, **k))[1]
    fx._fire_avenge = lambda *a, **k: (order.append("avenge"), fire_av(*a, **k))[1]
    try:
        simulate_battle(
            [_plain("victim", 0, 1)],
            [_plain("killer", 40, 40)],
            p0_has_initiative=False,
            rng=np.random.default_rng(0),
            patch=patch,
        )
    finally:
        fx._fire_deathrattle, fx._fire_avenge = fire_dr, fire_av
    assert order[:2] == ["dr", "avenge"]


# --------------------------------------------------------------------------- #
# Spellcraft: on play, a shield that is up, a wait, and an expiry that reaches
# --------------------------------------------------------------------------- #


def _triggers(patch):
    return ShopTriggers(np.random.default_rng(0), patch=patch)


def _spellcraft_in_hand(player):
    from src.bg_recruitment.spellcraft import is_spellcraft_spell

    return [c for c in player.hand if is_spellcraft_spell(c)]


def test_playing_a_spellcraft_naga_hands_the_spell_over_at_once(patch):
    """Blizzard: the spell is made when the minion is played *and* at the start
    of each Recruit phase. Eight of nine cards only ever did the second."""
    naga = patch.make_minion("BG23_000")  # Mini-Myrmidon
    player = _player(patch, [naga])
    _triggers(patch).fire_on_place(naga, player, None)
    assert len(_spellcraft_in_hand(player)) == 1
    _triggers(patch).fire_on_turn_start(player)
    assert len(_spellcraft_in_hand(player)) == 2


def test_zarjira_keeps_making_its_spell_every_turn(patch):
    """The mirror of the same bug: bound ON_PLACE only, so one spell ever."""
    naga = patch.make_minion("BG27_514")
    player = _player(patch, [naga])
    _triggers(patch).fire_on_place(naga, player, None)
    _triggers(patch).fire_on_turn_start(player)
    assert len(_spellcraft_in_hand(player)) == 2


def test_brann_does_not_double_a_spellcraft_spell(patch):
    """Spellcraft is a keyword, not a Battlecry."""
    naga = patch.make_minion("BG23_000")
    brann = patch.make_minion("BG_LOE_077")  # Brann: battlecry multiplier
    player = _player(patch, [brann, naga])
    _triggers(patch).fire_on_place(naga, player, None)
    assert len(_spellcraft_in_hand(player)) == 1


def test_a_spellcraft_spell_waits_for_a_hand_slot(patch):
    """The keyword's own exception to the full-hand rule."""
    from src.bg_recruitment.spellcraft import flush_pending_spellcraft

    naga = patch.make_minion("BG23_000")
    player = _player(patch, [naga])
    for i in range(len(player.hand)):
        player.hand[i] = _plain(f"filler{i}")
    _triggers(patch).fire_on_place(naga, player, None)
    assert len(player.pending_spellcraft) == 1
    assert _spellcraft_in_hand(player) == []

    player.hand[0] = None
    flush_pending_spellcraft(player)
    assert player.pending_spellcraft == ()
    assert len(_spellcraft_in_hand(player)) == 1


def test_a_waiting_spellcraft_spell_still_dies_at_end_of_turn(patch):
    """Waiting is within the turn it was made for."""
    from src.bg_recruitment.spellcraft import discard_spellcraft_spells

    naga = patch.make_minion("BG23_000")
    player = _player(patch, [naga])
    for i in range(len(player.hand)):
        player.hand[i] = _plain(f"filler{i}")
    _triggers(patch).fire_on_place(naga, player, None)
    assert discard_spellcraft_spells(player) == 1
    assert player.pending_spellcraft == ()


def test_a_spellcraft_divine_shield_is_actually_up(patch):
    """Combat asks for the keyword *and* the flag; Glowscale set only one."""
    from src.bg_recruitment.spellcraft import play_spellcraft_spell_from_hand

    glowscale = patch.make_minion("BG23_008")
    target = _plain("target", 1, 1)
    player = _player(patch, [glowscale, target])
    _triggers(patch).fire_on_place(glowscale, player, None)
    slot = next(i for i, c in enumerate(player.hand) if c is not None)
    play_spellcraft_spell_from_hand(player, slot, 1, patch=patch)
    assert Keyword.SHIELD in target.all_keywords and target.has_shield


def test_the_expiry_takes_down_only_the_shield_it_put_up(patch):
    from src.bg_recruitment.spellcraft import expire_temporary_buffs

    printed = patch.make_minion("BG_BOT_911")  # printed Divine Shield
    printed.temp_keywords = frozenset({Keyword.SHIELD})
    expire_temporary_buffs(_player(patch, [printed]))
    assert printed.has_shield

    borrowed = _plain("borrowed")
    borrowed.temp_keywords = frozenset({Keyword.SHIELD})
    borrowed.has_shield = True
    expire_temporary_buffs(_player(patch, [borrowed]))
    assert not borrowed.has_shield


def test_an_until_next_turn_buff_expires_on_a_minion_in_the_tavern(patch):
    """A Spellcraft spell reaches the counter, so the expiry has to as well —
    otherwise buying the minion carries the stats past the boundary."""
    from src.bg_recruitment.spellcraft import (
        expire_temporary_buffs,
        play_spellcraft_spell_from_hand,
    )

    naga = patch.make_minion("BG23_000")  # +2 Attack until next turn
    player = _player(patch, [naga])
    offered = _plain("on-the-counter", 1, 1)
    player.shop[0] = offered
    _triggers(patch).fire_on_place(naga, player, None)
    slot = next(i for i, c in enumerate(player.hand) if c is not None)
    play_spellcraft_spell_from_hand(player, slot, shop_index=0, patch=patch)
    assert offered.raw_attack == 3
    expire_temporary_buffs(player)
    assert offered.raw_attack == 1


# --------------------------------------------------------------------------- #
# Whose multiplier it is, and where a spell can reach
# --------------------------------------------------------------------------- #


def _cast_spell(patch, player, spell_id, target=None):
    from src.bg_recruitment.tavern_spells import cast_tavern_spell

    cast_tavern_spell(
        player,
        patch.tavern_spells[spell_id],
        rng=np.random.default_rng(0),
        patch=patch,
        target=target,
    )


def _bounty_gold(patch, board):
    player = _player(patch, board)
    before = player.gold
    _cast_spell(patch, player, "BG33_815")  # Wealthy Bounty: gain 2 Gold
    return player.gold - before


def test_proud_privateer_only_doubles_while_it_stands(patch):
    """"Your Bounties cast twice" is an ongoing effect of the body in play, not
    a promise the seat keeps after it is sold."""
    assert _bounty_gold(patch, []) == 2
    assert _bounty_gold(patch, [patch.make_minion("BG33_825")]) == 4


def test_a_golden_privateer_casts_a_bounty_three_times(patch):
    from src.bg_recruitment.targeted_battlecry import make_golden

    privateer = patch.make_minion("BG33_825")
    make_golden(privateer, patch=patch)
    assert _bounty_gold(patch, [privateer]) == 6


def test_balinda_does_not_double_a_spell_aimed_at_the_tavern(patch):
    """"Your spells that target **friendly** minions cast twice" — a minion on
    the counter is not friendly."""
    on_board = _plain("mine", 1, 1)
    player = _player(patch, [patch.make_minion("BG35_883"), on_board])
    _cast_spell(patch, player, "BG28_897", target=on_board)  # +2/+2
    assert (on_board.raw_attack, on_board.max_health) == (5, 5)

    offered = _plain("theirs", 1, 1)
    player = _player(patch, [patch.make_minion("BG35_883")])
    player.shop[0] = offered
    _cast_spell(patch, player, "BG28_897", target=offered)
    assert (offered.raw_attack, offered.max_health) == (3, 3)


def test_a_blood_gem_can_land_on_a_minion_in_the_tavern(patch):
    """Patch 27.4.0.185749 — the fix for the empty-board soft-lock."""
    from src.bg_recruitment.blood_gems import (
        can_play_blood_gem,
        give_blood_gems,
        play_blood_gem_from_hand,
    )

    player = _player(patch)
    give_blood_gems(player, 1)
    offered = _plain("on-the-counter", 1, 1)
    player.shop[0] = offered
    assert can_play_blood_gem(player)
    slot = next(i for i, c in enumerate(player.hand) if c is not None)
    play_blood_gem_from_hand(player, slot, shop_index=0)
    assert (offered.raw_attack, offered.max_health) == (2, 2)


def test_a_gem_with_nowhere_to_land_still_waits_in_hand(patch):
    from src.bg_recruitment.blood_gems import can_play_blood_gem, give_blood_gems

    player = _player(patch)
    give_blood_gems(player, 1)
    assert not can_play_blood_gem(player)
    assert any(c is not None for c in player.hand)


def test_a_count_scaled_minion_is_already_grown_on_the_counter(patch):
    """"+4/+2 for each friendly Eternal Knight that died this game" — a Knight
    rolled onto the counter after five have died is a 24/12 there, not the 4/2
    it prints and corrects itself to once bought."""
    from src.bg_recruitment.game_counts import DIED, bump_game_count
    from src.bg_recruitment.shop import refresh_shop

    counts: dict = {}
    seed_player = _player(patch)
    for _ in range(5):
        bump_game_count(seed_player, DIED, "BG25_008")
    counts = dict(seed_player.game_counts)

    for seed in range(200):
        player = _player(patch)
        player.game_counts = dict(counts)
        refresh_shop(player, None, rng=np.random.default_rng(seed), patch=patch)
        rolled = next(
            (m for m in player.shop if m is not None and m.card_id == "BG25_008"), None
        )
        if rolled is not None:
            assert (rolled.raw_attack, rolled.max_health) == (24, 12)
            return
    pytest.skip("no Eternal Knight rolled in 200 shops")


# --------------------------------------------------------------------------- #
# Crashes and dead cards the second audit turned up
# --------------------------------------------------------------------------- #


def test_a_promise_finds_its_body_after_the_state_is_copied(patch):
    """``instance_id`` cannot name a body across turns: ``__copy__`` re-issues
    it and the seat's state is copied once per action, so Winner's Bread's
    second half could never find what it had promised."""
    from src.bg_lobby.player import copy_player_state
    from src.bg_recruitment.tavern_spells import cast_tavern_spell

    body = _plain("mine", 1, 1)
    player = _player(patch, [body])
    cast_tavern_spell(
        player,
        patch.tavern_spells["BG36_883"],
        rng=np.random.default_rng(0),
        patch=patch,
        target=body,
    )
    assert (player.board[0].raw_attack, player.board[0].max_health) == (3, 4)
    for _ in range(3):  # three shop actions' worth of copying
        player = copy_player_state(player)
    player.last_combat_won = True
    ShopTriggers(np.random.default_rng(0), patch=patch).fire_on_turn_start(player)
    assert (player.board[0].raw_attack, player.board[0].max_health) == (7, 10)


def test_a_gold_spent_watcher_cannot_steal_the_buyers_hand_slot(patch):
    """The mask ruled BUY legal on the hand as it was; paying must not change
    that out from under it."""
    import inspect

    from src.bg_recruitment import economy

    body = inspect.getsource(economy.buy_from_shop)
    assert body.index("player.hand[h] = minion") < body.index("note_gold_spent")


def test_a_golden_token_with_no_template_is_forged(patch):
    """Golden rows are bindings, not cards, so the templates hold the plain
    printings only — and a summon pointed at one raised mid-fight."""
    (ability,) = patch.triple_merge_golden_abilities("BG25_009")
    assert ability.effect.token_id == "BG25_008_G"
    token = patch.make_minion("BG25_008_G")
    printed = patch.templates["BG25_008"]
    assert token.is_golden
    assert (token.base_attack, token.base_health) == (
        printed.base_attack * 2,
        printed.base_health * 2,
    )


def test_destroying_a_friendly_in_the_tavern_skips_a_combat_deathrattle(patch):
    """A body summoned from hand or stashed has nowhere to go outside a fight,
    and there is no killer to punish — these used to reach the dispatcher and
    raise."""
    from src.bg_recruitment.shop_triggers import _COMBAT_ONLY_ON_DEATH
    from src.bg_core.effects import (
        BuffRandomHandMinionEffect,
        DestroyKillerEffect,
        SummonBestFromHandEffect,
        SummonStashedEffect,
    )

    for effect in (
        SummonBestFromHandEffect,
        SummonStashedEffect,
        DestroyKillerEffect,
        BuffRandomHandMinionEffect,
    ):
        assert issubclass(effect, object) and effect in _COMBAT_ONLY_ON_DEATH


def test_a_named_eater_still_has_to_be_the_tribe_the_card_asks_for(patch):
    """"Choose a friendly **Demon**" — the filter was read only on the random
    branch, so a named body of any tribe ate."""
    from src.bg_catalog.cards import make_minion
    from src.bg_core.minion import Race
    from src.bg_recruitment.tavern_spells import cast_tavern_spell

    beast = next(
        c for c in sorted(patch.pool_ids) if patch.templates[c].race is Race.BEAST
    )
    undead = next(
        c for c in sorted(patch.pool_ids) if patch.templates[c].race is Race.UNDEAD
    )
    eater = make_minion(beast, patch=patch)
    player = _player(patch, [eater])
    for i in range(3):
        player.shop[i] = make_minion(undead, patch=patch)
    before = (eater.raw_attack, eater.max_health)

    cast_tavern_spell(
        player,
        patch.tavern_spells["BG28_607"],  # "Choose a friendly Demon"
        rng=np.random.default_rng(0),
        patch=patch,
        target=eater,
    )
    assert (eater.raw_attack, eater.max_health) == before
    assert sum(1 for m in player.shop if m is not None) == 3


# --------------------------------------------------------------------------- #
# The second audit's open list, closed
# --------------------------------------------------------------------------- #


def test_a_minion_eaten_off_the_counter_goes_back_to_the_lobby(patch):
    from src.bg_lobby.shared_pool import build_initial_shared_pool
    from src.bg_recruitment.targeted_battlecry import consume_tavern_minion

    pool = build_initial_shared_pool(patch=patch)
    player = _player(patch, [patch.make_minion("BGS_119")])
    player.shop[0] = patch.make_minion("BG25_008")
    pool.try_reserve_offer("BG25_008")
    before = pool.remaining_copies("BG25_008")
    consume_tavern_minion(
        player, player.board[0], rng=np.random.default_rng(0), shared_pool=pool
    )
    assert pool.remaining_copies("BG25_008") == before + 1


def test_a_token_is_worth_no_copies_at_all(patch):
    """It was never lent, so releasing it invented a key the lobby never had."""
    from src.bg_lobby.shared_pool import build_initial_shared_pool, copies_for_minion

    token_id = next(c for c in sorted(patch.templates) if patch.templates[c].is_token)
    token = patch.make_minion(token_id)
    assert copies_for_minion(token) == 0
    pool = build_initial_shared_pool(patch=patch)
    pool.release_minion(token)
    assert pool.remaining_copies(token_id) == 0


def test_a_ghost_fight_answers_the_win_and_tie_questions(patch):
    import inspect

    from src.bg_lobby import eight_player

    body = inspect.getsource(eight_player.resolve_combat_round)
    ghost = body[: body.index("assert match.b is not None")]
    assert "live.last_combat_won" in ghost
    assert "live.last_combat_tied" in ghost


def test_a_card_firing_without_an_rng_still_draws_a_stream(patch):
    """Every one of these built ``default_rng(0)`` afresh, so a "random" pick
    was the same pick every time."""
    from src.bg_core.board_helpers import seat_rng

    player = _player(patch)
    player.side_rng_seed = 3
    drawn = {int(seat_rng(player).integers(0, 1000)) for _ in range(20)}
    assert len(drawn) > 1

    # ...and two seats do not march in lockstep.
    def _first_ten(seed):
        seat = _player(patch)
        seat.side_rng_seed = seed
        return [int(seat_rng(seat).integers(0, 1000)) for _ in range(10)]

    assert _first_ten(3) != _first_ten(4)
    assert _first_ten(3) == _first_ten(3)  # and are still fixed by the seed


def test_the_stat_giving_predicate_sees_more_than_two_shapes(patch):
    from src.bg_recruitment.tavern_spells import spell_gives_stats

    bound = [s for s in patch.tavern_spells.values() if s.in_pool and s.abilities]
    giving = {s.name for s in bound if spell_gives_stats(s)}
    for name in ("Shiny Ring", "Sanctify", "Queen's Command", "Wave of Gold"):
        assert name in giving


def test_methodical_madness_takes_the_bonus_keywords_too(patch):
    from src.bg_core.minion import Race
    from src.bg_recruitment.tavern_spells import cast_tavern_spell

    demon = next(
        c for c in sorted(patch.pool_ids) if patch.templates[c].race is Race.DEMON
    )
    eater = patch.make_minion(demon)
    player = _player(patch, [eater])
    player.shop[0] = _plain("taunted", 5, 6)
    player.shop[0].keywords = frozenset({Keyword.TAUNT})
    player.shop[1] = _plain("windy", 7, 8)
    player.shop[1].keywords = frozenset({Keyword.WINDFURY})

    cast_tavern_spell(
        player,
        patch.tavern_spells["BG36_880"],
        rng=np.random.default_rng(0),
        patch=patch,
        target=eater,
    )
    assert Keyword.TAUNT in eater.all_keywords
    assert Keyword.WINDFURY in eater.all_keywords


def test_one_of_each_type_pays_one_body_per_type(patch):
    """An All-type minion answers for one type, not for all nine."""
    from src.bg_core.minion import Race
    from src.bg_recruitment.tavern_spells import cast_tavern_spell

    amalgam, beast = _plain("amalgam", 1, 1), _plain("beast", 1, 1)
    amalgam.race, beast.race = Race.ALL, Race.BEAST
    player = _player(patch, [amalgam, beast])
    cast_tavern_spell(
        player,
        patch.tavern_spells["BG28_888"],
        rng=np.random.default_rng(0),
        patch=patch,
    )
    assert (amalgam.raw_attack, beast.raw_attack) == (3, 3)


def test_a_shared_type_buff_includes_the_body_that_named_the_type(patch):
    from src.bg_core.minion import Race
    from src.bg_recruitment.tavern_spells import cast_tavern_spell

    mine, other = _plain("mine", 1, 1), _plain("other", 1, 1)
    mine.race, other.race = Race.DEMON, Race.BEAST
    offered = _plain("on-the-counter", 1, 3)
    offered.race = Race.DEMON
    player = _player(patch, [mine, other])
    player.shop[0] = offered
    cast_tavern_spell(
        player,
        patch.tavern_spells["BG28_845"],
        rng=np.random.default_rng(0),
        patch=patch,
        target=offered,
    )
    assert mine.raw_attack == 4
    assert offered.raw_attack == 4  # it named the type; it is one of the type
    assert other.raw_attack == 1


def test_tomb_turnings_pick_dies_if_played_the_same_turn(patch):
    from src.bg_recruitment.discover import resolve_discover_pick
    from src.bg_recruitment.place import place_from_hand
    from src.bg_recruitment.tavern_spells import cast_tavern_spell

    def _discovered(player):
        cast_tavern_spell(
            player,
            patch.tavern_spells["BG34_888"],
            rng=np.random.default_rng(0),
            patch=patch,
        )
        resolve_discover_pick(
            player,
            0,
            None,
            rng=np.random.default_rng(0),
            on_after_placed=lambda *_: None,
            patch=patch,
        )
        return next(c for c in player.hand if c is not None)

    def _play(player, card):
        place_from_hand(
            player,
            player.hand.index(card),
            None,
            board_size=7,
            triggers=ShopTriggers(np.random.default_rng(0), patch=patch),
            rng=np.random.default_rng(0),
        )

    player = _player(patch)
    card = _discovered(player)
    assert card.dies_if_played_this_turn
    _play(player, card)
    assert player.board == []

    player = _player(patch)
    card = _discovered(player)
    ShopTriggers(np.random.default_rng(0), patch=patch).fire_on_turn_start(player)
    assert not card.dies_if_played_this_turn
    _play(player, card)
    assert [m.card_id for m in player.board] == [card.card_id]
