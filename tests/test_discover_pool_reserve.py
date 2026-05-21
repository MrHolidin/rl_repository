"""Discover options reserve lobby copies at modal open."""

from __future__ import annotations

import numpy as np

from src.bg_lobby.player import PendingChoice, PendingChoiceKind, PlayerPhase, PlayerState
from src.bg_lobby.shared_pool import build_initial_shared_pool
from src.bg_recruitment.discover import resolve_discover_pick, try_open_hand_discover_modal
from src.bg_recruitment.pool_ledger import reserve_discover_options
from src.envs.minibg.actions import HAND_SIZE


def _player() -> PlayerState:
    return PlayerState(
        health=40,
        gold=10,
        tavern_tier=3,
        next_tier_up_cost=5,
        board=[],
        shop=[None] * 6,
        hand=[None] * HAND_SIZE,
        phase=PlayerPhase.SHOP,
        shop_actions_used=0,
    )


def test_discover_pick_succeeds_when_only_reserved_copies_remain():
    pool = build_initial_shared_pool(None)
    cid = "BGS_020"
    while pool.remaining_copies(cid) > 3:
        assert pool.acquire_new(cid, 1)
    assert pool.remaining_copies(cid) == 3

    opts = ("BGS_020", "BGS_022", "EX1_507")
    for card in opts:
        assert pool.remaining_copies(card) >= 1

    player = _player()
    assert try_open_hand_discover_modal(
        player,
        PendingChoiceKind.DISCOVER_MURLOC,
        opts,
        0,
        shared_pool=pool,
    )
    assert player.pending_choice is not None
    assert player.pending_choice.options_pool_reserved
    assert pool.remaining_copies(cid) == 2

    pick = 0
    resolve_discover_pick(
        player,
        pick,
        None,
        rng=np.random.default_rng(1),
        on_after_placed=lambda _p, _m: None,
        shared_pool=pool,
    )
    assert player.pending_choice is None
    assert player.hand[0] is not None and player.hand[0].card_id == cid
    assert pool.remaining_copies(cid) == 2
    assert pool.remaining_copies("BGS_022") == pool._initial["BGS_022"]
    assert pool.remaining_copies("EX1_507") == pool._initial["EX1_507"]


def test_discover_release_unpicked_copies():
    pool = build_initial_shared_pool(None)
    opts = ("EX1_162", "BGS_020", "BGS_022")
    before = pool.remaining_copies("EX1_162")
    assert reserve_discover_options(pool, opts)
    assert pool.remaining_copies("EX1_162") == before - 1
    player = _player()
    player.pending_choice = PendingChoice(
        PendingChoiceKind.DISCOVER_MURLOC,
        opts,
        0,
        options_pool_reserved=True,
    )
    resolve_discover_pick(
        player,
        0,
        None,
        rng=np.random.default_rng(2),
        on_after_placed=lambda _p, _m: None,
        shared_pool=pool,
    )
    assert pool.remaining_copies("EX1_162") == before - 1
    assert pool.remaining_copies("BGS_020") == pool._initial["BGS_020"]
    assert pool.remaining_copies("BGS_022") == pool._initial["BGS_022"]
