"""Shared tavern pool variant B: reserve on shop offer, release on clear."""

from __future__ import annotations

import numpy as np

from src.bg_catalog.cards import make_minion
from src.bg_lobby.shared_pool import (
    POOL_SIZE_BY_TIER,
    SharedCardPool,
    build_initial_shared_pool,
)
from src.bg_recruitment.economy import buy_from_shop, roll_shop, sell_from_board
from src.bg_recruitment.shop import clear_shop_slot, fill_shop_slot
from src.bg_lobby.player import PlayerPhase, PlayerState
from src.envs.minibg.actions import HAND_SIZE, MAX_SHOP_SLOTS


def _empty_player(*, tier: int = 1) -> PlayerState:
    return PlayerState(
        health=30,
        gold=10,
        tavern_tier=tier,
        next_tier_up_cost=5,
        board=[],
        shop=[None] * MAX_SHOP_SLOTS,
        hand=[None] * HAND_SIZE,
        phase=PlayerPhase.SHOP,
        shop_actions_used=0,
    )


def test_initial_pool_tier_counts():
    pool = build_initial_shared_pool(None)
    cid = "EX1_162"  # recruit tier 1
    assert pool.remaining_copies(cid) == POOL_SIZE_BY_TIER[1]


def test_fill_slot_reserves_release_on_clear():
    pool = build_initial_shared_pool(None)
    p = _empty_player()
    rng = np.random.default_rng(0)
    cid = "EX1_162"
    before = pool.remaining_copies(cid)
    pool.try_reserve_offer(cid)
    p.shop[0] = make_minion(cid)
    assert pool.remaining_copies(cid) == before - 1
    clear_shop_slot(p, 0, pool, release_to_pool=True)
    assert pool.remaining_copies(cid) == before
    assert p.shop[0] is None


def test_buy_does_not_release_shop_reservation():
    pool = build_initial_shared_pool(None)
    p = _empty_player()
    cid = "EX1_162"
    pool.try_reserve_offer(cid)
    p.shop[0] = make_minion(cid)
    before = pool.remaining_copies(cid)
    buy_from_shop(
        p,
        0,
        on_bought=lambda _m, _p: None,
        on_triples=lambda _p: None,
        shared_pool=pool,
    )
    assert pool.remaining_copies(cid) == before
    assert p.hand[0] is not None


def test_roll_releases_unbought_offers():
    pool = build_initial_shared_pool(None)
    p = _empty_player()
    rng = np.random.default_rng(1)
    fill_shop_slot(p, 0, None, rng=rng, shared_pool=pool)
    m = p.shop[0]
    assert m is not None
    cid = m.card_id
    after_fill = pool.remaining_copies(cid)
    roll_shop(p, None, rng=np.random.default_rng(2), shared_pool=pool)
    # at least one reroll happened; reserved copy returned when slot cleared
    assert pool.remaining_copies(cid) >= after_fill


def test_sell_returns_copy_to_pool():
    pool = build_initial_shared_pool(None)
    p = _empty_player()
    cid = "EX1_162"
    assert pool.acquire_new(cid, 1)
    before = pool.remaining_copies(cid)
    p.board = [make_minion(cid)]
    sell_from_board(p, 0, on_triples=lambda _p: None, shared_pool=pool)
    assert pool.remaining_copies(cid) == before + 1


def test_two_players_share_pool_via_game():
    from src.envs.minibg.game import MiniBGGame

    fresh_cap = sum(build_initial_shared_pool(None).remaining.values())
    g = MiniBGGame(seed=0, use_shared_pool=True)
    s = g.initial_state()
    assert s.shared_pool is not None
    after_setup = sum(s.shared_pool.remaining.values())
    assert 0 < after_setup < fresh_cap
