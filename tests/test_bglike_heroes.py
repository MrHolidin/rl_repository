"""Behavioral tests for bglike hero passives (patch 19.6) + backward compat.

Each test force-assigns a specific hero and exercises the event hook that drives
its power, asserting the verified patch-19.6 numbers. The backward-compat tests
pin that ``with_heroes=False`` leaves heroes off and obs/action dims unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.bg_catalog.cards import make_minion
from src.bg_catalog.patch_context import load_patch_context
from src.bg_core.effects import Keyword
from src.bg_core.minion import Race
from src.bg_lobby.player import PlayerPhase, PlayerState
from src.bg_recruitment import economy, hero_passives
from src.bg_recruitment import shop as recruitment_shop
from src.envs.minibg import actions as A

PATCH_DIR = "data/bgcore/19_6_0_74257"


@pytest.fixture(scope="module")
def patch():
    return load_patch_context(PATCH_DIR)


def _hero(patch, hid):
    return patch.heroes[hid]


def _card_of(patch, race, tier=1):
    for cid, t in sorted(patch.templates.items()):
        if (
            t.race == race
            and t.tier == tier
            and not t.is_token
            and not t.is_golden
            and not t.is_triple_reward_spell
        ):
            return cid
    raise AssertionError(f"no tier-{tier} {race} in patch")


def _player(patch, hero=None, *, tier=1, gold=10, board=None, shop=None):
    return PlayerState(
        health=40,
        gold=gold,
        tavern_tier=tier,
        next_tier_up_cost=A.LEVEL_UP_COSTS.get(tier, 0),
        board=board if board is not None else [],
        shop=shop if shop is not None else [None] * A.MAX_SHOP_SLOTS,
        hand=[None] * A.HAND_SIZE,
        phase=PlayerPhase.SHOP,
        shop_actions_used=0,
        hero=hero,
    )


def _rng(seed=0):
    return np.random.default_rng(seed)


# --------------------------------------------------------------------------- #
# Economy heroes
# --------------------------------------------------------------------------- #


def test_patchwerk_starts_with_55_health(patch):
    p = _player(patch, _hero(patch, "patchwerk"))
    hero_passives.apply_hero_on_game_start(p, 1, patch=patch, rng=_rng())
    assert p.health == 55


def test_nozdormu_first_refresh_free_then_normal(patch):
    p = _player(patch, _hero(patch, "nozdormu"))
    hero_passives.apply_hero_on_turn_start(p, 1, patch=patch, rng=_rng())
    assert economy.effective_roll_cost(p) == 0
    economy.roll_shop(p, None, rng=_rng(), patch=patch)
    assert p.hero_free_roll_pending is False
    assert economy.effective_roll_cost(p) == A.ROLL_COST


def test_millhouse_flat_costs(patch):
    p = _player(patch, _hero(patch, "millhouse"))
    assert economy.effective_buy_cost(p) == 2
    assert economy.effective_roll_cost(p) == 2
    # Tavern upgrades cost (1) more than the base.
    assert economy.effective_level_up_cost(p) == p.next_tier_up_cost + 1


def test_omu_plus_two_gold_on_upgrade(patch):
    p = _player(patch, _hero(patch, "omu"), gold=0)
    hero_passives.apply_hero_on_level_up(p)
    assert p.gold == 2


def test_chenvaala_discount_after_three_elementals(patch):
    p = _player(patch, _hero(patch, "chenvaala"))
    base = economy.effective_level_up_cost(p)
    for _ in range(2):
        hero_passives.apply_hero_on_elemental_played(p)
    assert p.hero_upgrade_discount == 0  # not yet 3
    hero_passives.apply_hero_on_elemental_played(p)
    assert p.hero_upgrade_discount == 3
    assert economy.effective_level_up_cost(p) == max(0, base - 3)


# --------------------------------------------------------------------------- #
# Shop heroes
# --------------------------------------------------------------------------- #


def test_millificent_buffs_shop_mechs(patch):
    p = _player(patch, _hero(patch, "millificent"), shop=[None] * A.MAX_SHOP_SLOTS)
    # Force a mech and a non-mech into the tavern; only the mech is buffed.
    recruitment_shop.add_random_minion_to_shop(
        p, Race.MECHANICAL, None, rng=_rng(0), patch=patch
    )
    recruitment_shop.add_random_minion_to_shop(
        p, Race.BEAST, None, rng=_rng(0), patch=patch
    )
    mechs = [m for m in p.shop if m is not None and m.race == Race.MECHANICAL]
    beasts = [m for m in p.shop if m is not None and m.race == Race.BEAST]
    assert mechs and beasts
    for m in mechs:
        assert m.bonus_attack == 1 and m.bonus_health == 1
    for m in beasts:
        assert m.bonus_attack == 0 and m.bonus_health == 0


def test_ysera_extra_dragon_slot(patch):
    p = _player(patch, _hero(patch, "ysera"))
    base = A.shop_offers_count(p.tavern_tier)
    assert recruitment_shop.effective_shop_offers_count(p) == base + 1
    recruitment_shop.refresh_shop(p, None, rng=_rng(2), patch=patch)
    assert p.shop[base] is not None and p.shop[base].race == Race.DRAGON


# --------------------------------------------------------------------------- #
# Buy / sell heroes
# --------------------------------------------------------------------------- #


def test_kaelthas_every_third_buy(patch):
    p = _player(patch, _hero(patch, "kaelthas"))
    mons = [make_minion(_card_of(patch, Race.BEAST), patch=patch) for _ in range(3)]
    for m in mons:
        hero_passives.apply_hero_on_bought(m, p)
    assert (mons[0].bonus_attack, mons[0].bonus_health) == (0, 0)
    assert (mons[1].bonus_attack, mons[1].bonus_health) == (0, 0)
    assert (mons[2].bonus_attack, mons[2].bonus_health) == (2, 2)


def test_rat_king_rotating_tribe_buff(patch):
    p = _player(patch, _hero(patch, "rat_king"))
    p.hero_rotating_tribe = Race.BEAST
    beast = make_minion(_card_of(patch, Race.BEAST), patch=patch)
    murloc = make_minion(_card_of(patch, Race.MURLOC), patch=patch)
    hero_passives.apply_hero_on_bought(beast, p)
    hero_passives.apply_hero_on_bought(murloc, p)
    assert (beast.bonus_attack, beast.bonus_health) == (2, 2)
    assert (murloc.bonus_attack, murloc.bonus_health) == (0, 0)


def test_deryl_buffs_two_shop_minions_on_sell(patch):
    shop = [make_minion(_card_of(patch, Race.BEAST), patch=patch) for _ in range(4)]
    p = _player(patch, _hero(patch, "deryl"), shop=shop + [None, None])
    sold = make_minion(_card_of(patch, Race.BEAST), patch=patch)
    hero_passives.apply_hero_on_sell(sold, p, rng=_rng(1), patch=patch)
    total_atk = sum(m.bonus_attack for m in p.shop if m is not None)
    total_hp = sum(m.bonus_health for m in p.shop if m is not None)
    assert total_atk == 2 and total_hp == 2  # 2 minions × +1/+1


def test_flurgl_adds_murloc_on_murloc_sell(patch):
    # Shop with one empty active slot for the added murloc to land in.
    shop = [make_minion(_card_of(patch, Race.BEAST), patch=patch), None, None]
    p = _player(patch, _hero(patch, "flurgl"), shop=shop + [None] * 3)
    murloc = make_minion(_card_of(patch, Race.MURLOC), patch=patch)
    hero_passives.apply_hero_on_sell(murloc, p, rng=_rng(1), patch=patch)
    added = [m for m in p.shop if m is not None and m.race == Race.MURLOC]
    assert added, "Flurgl should add a murloc to the tavern"


def test_flurgl_ignores_non_murloc_sell(patch):
    shop = [make_minion(_card_of(patch, Race.BEAST), patch=patch), None, None]
    p = _player(patch, _hero(patch, "flurgl"), shop=shop + [None] * 3)
    beast = make_minion(_card_of(patch, Race.BEAST), patch=patch)
    hero_passives.apply_hero_on_sell(beast, p, rng=_rng(1), patch=patch)
    assert [m for m in p.shop if m is not None and m.race == Race.MURLOC] == []


# --------------------------------------------------------------------------- #
# Start-of-game heroes
# --------------------------------------------------------------------------- #


def test_curator_starts_with_amalgam(patch):
    p = _player(patch, _hero(patch, "curator"))
    hero_passives.apply_hero_on_game_start(p, 1, patch=patch, rng=_rng())
    amalgams = [m for m in p.hand if m is not None and m.race == Race.ALL]
    assert len(amalgams) == 1


def test_afkay_zero_gold_and_two_tier3_minions(patch):
    p = _player(patch, _hero(patch, "afkay"), gold=3)
    hero_passives.apply_hero_on_game_start(p, 1, patch=patch, rng=_rng())
    assert p.gold == 0  # round 1 skipped
    in_hand = [m for m in p.hand if m is not None]
    assert len(in_hand) == 2
    assert all(m.tier == 3 for m in in_hand)
    # Round 2 also skipped; round 3 is a normal turn.
    p.gold = 5
    hero_passives.apply_hero_on_turn_start(p, 2, patch=patch, rng=_rng())
    assert p.gold == 0
    p.gold = 5
    hero_passives.apply_hero_on_turn_start(p, 3, patch=patch, rng=_rng())
    assert p.gold == 5


# --------------------------------------------------------------------------- #
# Combat heroes
# --------------------------------------------------------------------------- #


def test_deathwing_attack_aura(patch):
    from src.bg_combat.battle import attack_value, build_battle_side

    assert _hero(patch, "deathwing").combat_attack_aura() == 3
    m = make_minion(_card_of(patch, Race.BEAST), patch=patch)
    side = build_battle_side([m], patch=patch)
    bm = side.minions[0]
    base = attack_value(bm, side, death_resolution=False)
    side.attack_aura_all = 3
    assert attack_value(bm, side, death_resolution=False) == base + 3


def test_alakir_grants_leftmost_keywords(patch):
    from src.bg_combat.battle import (
        BattleSide,
        _CombatRuntime,
        _build_side,
        _fire_start_of_combat,
    )

    kws = _hero(patch, "alakir").start_combat_leftmost_keywords()
    assert kws == frozenset({Keyword.WINDFURY, Keyword.SHIELD, Keyword.TAUNT})

    left = make_minion(_card_of(patch, Race.BEAST), patch=patch)
    other = make_minion(_card_of(patch, Race.MURLOC), patch=patch)
    rt = _CombatRuntime(
        sides=(BattleSide(), BattleSide()),
        rng=_rng(),
        combat_board_max=10,
        damage_cap=99,
        patch=patch,
    )
    rt.sides = (_build_side([left, other], rt), _build_side([other], rt))
    rt.sides[0].start_combat_keywords = kws
    _fire_start_of_combat(rt)
    bm = rt.sides[0].minions[0]
    assert bm.shield_armed
    assert Keyword.WINDFURY in bm.template.all_keywords
    assert Keyword.TAUNT in bm.template.all_keywords
    # Non-leftmost is untouched.
    assert not rt.sides[0].minions[1].shield_armed


# --------------------------------------------------------------------------- #
# Backward compatibility
# --------------------------------------------------------------------------- #


def test_no_heroes_leaves_seats_heroless(patch):
    from src.envs.bglike.game import BGLikeGame

    g = BGLikeGame(seed=7, with_heroes=False, patch_dir=PATCH_DIR)
    s = g.initial_state()
    assert all(pl.hero is None for pl in s.players)


def test_with_heroes_assigns_every_seat(patch):
    from src.envs.bglike.game import BGLikeGame

    g = BGLikeGame(seed=7, with_heroes=True, patch_dir=PATCH_DIR)
    s = g.initial_state()
    assert all(pl.hero is not None for pl in s.players)


def test_action_and_obs_dims_unchanged():
    # Heroes must not change the action space or observation layout.
    assert A.NUM_ACTIONS == 73
    from src.envs.bglike.obs_v5 import OBS_DIM_V5

    assert OBS_DIM_V5 == 2536


def test_no_hero_rollout_is_deterministic(patch):
    from src.envs.bglike.game import BGLikeGame

    def rollout(seed):
        g = BGLikeGame(seed=seed, with_heroes=False, patch_dir=PATCH_DIR)
        s = g.initial_state()
        pick = np.random.default_rng(seed)
        sig = []
        steps = 0
        while not g.is_terminal(s) and steps < 1500:
            legal = g.legal_actions(s)
            if not legal:
                break
            a = int(pick.choice(legal))
            s = g.apply_action(s, a)
            sig.append(tuple(pl.health for pl in s.players))
            steps += 1
        return sig

    assert rollout(11) == rollout(11)
