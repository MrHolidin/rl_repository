"""Triple reward: discover spell in hand, PLACE opens modal (not on golden play)."""

from copy import copy

import numpy as np

from tests.conftest import PATCH_CTX

from tests.minibg_helpers import make_minion
from src.bg_recruitment.discover_pool import triple_reward_discover_tier
from src.bg_recruitment.triples import (
    is_triple_reward_discover_spell,
    resolve_one_triple,
)
from src.bg_recruitment.place import place_from_hand
from src.bg_recruitment.shop_triggers import ShopTriggers
from src.envs.minibg.actions import BOARD_SIZE, HAND_SIZE
from src.envs.minibg.game import MiniBGGame
from src.envs.minibg.state import PendingChoiceKind, PlayerState


def _player_with_triple_in_hand(card_id: str = "recruit") -> PlayerState:
    m = make_minion(card_id)
    return PlayerState(
        health=40,
        gold=10,
        tavern_tier=2,
        next_tier_up_cost=5,
        board=[],
        shop=[None] * 6,
        hand=[copy(m), copy(m), copy(m), None, None],
        phase=0,
        shop_actions_used=0,
    )


def test_triple_merge_grants_spell_not_immediate_discover():
    p = _player_with_triple_in_hand()
    assert resolve_one_triple(p, patch=PATCH_CTX)
    spells = [h for h in p.hand if h is not None and is_triple_reward_discover_spell(h)]
    goldens = [h for h in p.hand if h is not None and h.is_golden]
    assert len(goldens) == 1
    assert len(spells) == 1
    assert spells[0].triple_discover_tier == triple_reward_discover_tier(2)
    assert p.pending_choice is None


def test_place_spell_on_full_board_opens_discover():
    g = MiniBGGame(seed=0)
    rng = np.random.default_rng(0)
    triggers = g._shop_triggers
    p = PlayerState(
        health=40,
        gold=10,
        tavern_tier=3,
        next_tier_up_cost=5,
        board=[make_minion("recruit") for _ in range(BOARD_SIZE)],
        shop=[None] * 6,
        hand=[None] * HAND_SIZE,
        phase=0,
        shop_actions_used=0,
    )
    tier = triple_reward_discover_tier(3)
    from src.bg_recruitment.triples import make_triple_reward_discover_spell

    p.hand[0] = make_triple_reward_discover_spell(discover_tier=tier, patch=PATCH_CTX)
    place_from_hand(
        p,
        0,
        None,
        board_size=BOARD_SIZE,
        triggers=triggers,
        rng=rng,
    )
    assert len(p.board) == BOARD_SIZE
    assert p.pending_choice is not None
    assert p.pending_choice.kind == PendingChoiceKind.TRIPLE_REWARD_DISCOVER
    assert p.hand[0] is None


def test_place_golden_does_not_open_discover():
    g = MiniBGGame(seed=1)
    rng = np.random.default_rng(1)
    p = _player_with_triple_in_hand()
    resolve_one_triple(p, patch=PATCH_CTX)
    golden_slot = next(
        i for i, h in enumerate(p.hand) if h is not None and h.is_golden
    )
    p.board = [make_minion("recruit") for _ in range(BOARD_SIZE - 1)]
    place_from_hand(
        p,
        golden_slot,
        None,
        board_size=BOARD_SIZE,
        triggers=g._shop_triggers,
        rng=rng,
    )
    assert p.pending_choice is None
