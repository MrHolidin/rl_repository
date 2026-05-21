"""Staged PLACE + TARGET_BOARD for Argus / targeted battlecries in BGLike."""

from __future__ import annotations

from src.bg_recruitment.place import place_from_hand
from src.envs.minibg.rl_place import (
    RlPlacePlan,
    commit_rl_place_plan,
    commit_simple_place_from_hand,
    open_rl_place_plan,
)

from . import actions as bglike_actions
from .game import BGLikeGame
from .state import BGLikeState

__all__ = [
    "RlPlacePlan",
    "commit_bglike_rl_place",
    "commit_bglike_simple_place",
    "open_rl_place_plan",
]


def commit_bglike_simple_place(
    state: BGLikeState,
    seat: int,
    hand_slot: int,
    game: BGLikeGame,
) -> BGLikeState:
    new_state = game._copy_state(state)
    player = new_state.players[seat]
    commit_simple_place_from_hand(
        player,
        hand_slot,
        new_state.shop_excluded_race,
        board_size=bglike_actions.BOARD_SIZE,
        triggers=game._shop_triggers,
        rng=game._rng,
        shared_pool=new_state.shared_pool,
    )
    return new_state


def commit_bglike_rl_place(
    state: BGLikeState,
    seat: int,
    plan: RlPlacePlan,
    game: BGLikeGame,
) -> BGLikeState:
    return commit_rl_place_plan(
        state,
        seat,
        plan,
        board_size=bglike_actions.BOARD_SIZE,
        shop_excluded_race=state.shop_excluded_race,
        triggers=game._shop_triggers,
        rng=game._rng,
        reorder_board=game.reorder_board,
        shared_pool=state.shared_pool,
    )
