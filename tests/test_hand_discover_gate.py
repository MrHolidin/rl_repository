"""Hand-discover modals require a free hand slot at open time."""

from unittest.mock import MagicMock

import numpy as np

from src.bg_catalog.cards import make_minion
from src.bg_recruitment.discover import (
    is_hand_discover_kind,
    resolve_discover_pick,
    try_open_hand_discover_modal,
)
from src.bg_recruitment.place import place_from_hand
from src.bg_recruitment.shop_triggers import ShopTriggers
from src.envs.minibg.actions import BOARD_SIZE, HAND_SIZE
from src.envs.minibg.game import MiniBGGame
from src.envs.minibg.state import (
    MiniBGState,
    PendingChoice,
    PendingChoiceKind,
    PlayerPhase,
    PlayerState,
)


def test_try_open_rejects_full_hand():
    p = PlayerState(
        health=40,
        gold=10,
        tavern_tier=3,
        next_tier_up_cost=5,
        board=[],
        shop=[None] * 6,
        hand=[make_minion("recruit") for _ in range(HAND_SIZE)],
        phase=PlayerPhase.SHOP,
        shop_actions_used=0,
    )
    assert not try_open_hand_discover_modal(
        p,
        PendingChoiceKind.DISCOVER_MURLOC,
        ("a", "b", "c"),
        0,
    )
    assert p.pending_choice is None


def test_brann_chain_truncated_to_free_slots():
    g = MiniBGGame(seed=5)
    p = PlayerState(
        health=40,
        gold=10,
        tavern_tier=3,
        next_tier_up_cost=5,
        board=[make_minion("recruit")],
        shop=[None] * 6,
        hand=[None, make_minion("BGS_020"), None, None, None],
        phase=PlayerPhase.SHOP,
        shop_actions_used=0,
    )
    triggers = ShopTriggers(g._rng, on_triples=lambda _pl: None)
    placed = p.hand[1]
    assert placed is not None
    p.hand[1] = None
    p.board.append(placed)
    triggers.fire_on_place(placed, p, None)
    assert p.pending_choice is not None
    assert p.pending_choice.kind == PendingChoiceKind.DISCOVER_MURLOC
    assert p.pending_choice.extra_modals_after == 0


def test_discover_pick_triples_after_chain_closed():
    g = MiniBGGame(seed=7)
    rec = make_minion("recruit")
    p = PlayerState(
        health=40,
        gold=10,
        tavern_tier=3,
        next_tier_up_cost=5,
        board=[],
        shop=[None] * 6,
        hand=[rec, rec, None, None, None],
        phase=PlayerPhase.SHOP,
        shop_actions_used=0,
    )
    p.pending_choice = PendingChoice(
        PendingChoiceKind.DISCOVER_MURLOC,
        ("recruit", "recruit", "recruit"),
        0,
    )
    on_after = MagicMock()
    resolve_discover_pick(
        p, 0, None, rng=g._rng, on_after_placed=on_after
    )
    assert p.pending_choice is None
    assert any(h is not None and h.is_golden for h in p.hand)


def test_is_hand_discover_kind():
    assert is_hand_discover_kind(PendingChoiceKind.DISCOVER_MURLOC)
    assert is_hand_discover_kind(PendingChoiceKind.TRIPLE_REWARD_DISCOVER)
    assert not is_hand_discover_kind(PendingChoiceKind.ADAPT)
