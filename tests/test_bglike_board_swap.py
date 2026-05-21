"""Env-level adjacent board swaps in BGLike shop (no ORDER game phase)."""

from __future__ import annotations

import numpy as np

from src.bg_catalog.cards import make_minion
from src.envs.bglike.action_map import (
    A_FINISH,
    A_SWAP_BOARD_0,
    NUM_ENV_ACTIONS,
    NUM_SWAP_ADJ,
    is_swap_board,
)
from src.envs.bglike.heuristic_bots.common import choose_final_order
from src.envs.bglike.lobby_env import BGLobbyEnv, lobby_from_learned_seats
from src.envs.bglike.state import PlayerPhase
from src.agents.random_agent import RandomAgent


def test_legal_mask_includes_swap_when_board_has_two_minions():
    agent = RandomAgent(seed=1)
    configs = lobby_from_learned_seats((0,), agent_by_seat={0: agent}, seed=3)
    lobby = BGLobbyEnv(configs, learned_seats=(0,), seed=3)
    lobby.reset(seed=10)
    seat = 0
    while lobby.current_seat() != seat:
        lobby.step_auto(deterministic=True)
    p = lobby.state.players[seat]
    p.board = [make_minion("recruit"), make_minion("guard")]
    mask = lobby.legal_mask_for_seat(seat)
    assert mask[A_SWAP_BOARD_0]
    assert mask[A_FINISH]
    assert NUM_SWAP_ADJ == 6
    assert NUM_ENV_ACTIONS == int(mask.shape[0])


def test_swap_adjacent_does_not_end_shop_turn():
    agent = RandomAgent(seed=2)
    configs = lobby_from_learned_seats((0,), agent_by_seat={0: agent}, seed=4)
    lobby = BGLobbyEnv(configs, learned_seats=(0,), seed=4)
    lobby.reset(seed=11)
    seat = 0
    while lobby.current_seat() != seat:
        lobby.step_auto(deterministic=True)
    p = lobby.state.players[seat]
    p.board = [make_minion("recruit"), make_minion("guard")]
    before = [m.card_id for m in p.board]
    lobby.step_action(seat, A_SWAP_BOARD_0)
    after = [m.card_id for m in lobby.state.players[seat].board]
    assert before == after[::-1]
    assert lobby.state.players[seat].phase == PlayerPhase.SHOP
    assert lobby.state.current_player_index == seat


def test_heuristic_choose_final_order_picks_swap():
    mask = np.zeros(NUM_ENV_ACTIONS, dtype=bool)
    mask[A_SWAP_BOARD_0] = True
    mask[A_FINISH] = True
    # High attack left, low attack right — default order wants taunt/right-heavy left
    board = [make_minion("guard"), make_minion("recruit")]
    action = choose_final_order(board, mask, None)
    assert is_swap_board(action)
