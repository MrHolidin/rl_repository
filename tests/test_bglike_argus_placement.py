"""Argus-style staged PLACE + TARGET_BOARD in BGLike lobby."""

from __future__ import annotations

from src.agents.random_agent import RandomAgent
from src.bg_catalog.cards import make_minion
from src.bg_core.effects import Keyword
from src.envs.bglike.action_map import (
    A_APPLY_EFFECT_SKIP,
    A_PLACE_BASE,
    A_TARGET_BOARD_BASE,
    is_apply_effect_skip,
    is_target_board,
)
from src.envs.bglike.lobby_env import BGLobbyMultiCurrentEnv
from src.envs.bglike.obs import encode_pending_choice
from src.envs.minibg.obs import PENDING_IS_APPLY_OFFSET
from src.envs.bglike.seat_config import build_training_lobby_configs


def _lobby_for_seat(seat: int = 0) -> BGLobbyMultiCurrentEnv:
    cur = RandomAgent(seed=1)
    opp = RandomAgent(seed=2)
    env = BGLobbyMultiCurrentEnv([seat], seed=10)
    env.set_agents(cur, {s: opp for s in range(1, 8)})
    env.reset()
    return env


def test_argus_two_targets_then_skip_not_needed():
    env = _lobby_for_seat(0)
    lobby = env.lobby
    lobby.reset(seed=10)
    seat = 0
    while lobby.current_seat() != seat:
        lobby.step_auto(deterministic=True)

    p = lobby.state.players[seat]
    p.board = [make_minion("recruit"), make_minion("guard")]
    p.hand[0] = make_minion("defender_argus")

    lobby.step_action(seat, A_PLACE_BASE)
    assert seat in lobby._rl_pending
    mask = lobby.legal_mask_for_seat(seat)
    assert mask[A_TARGET_BOARD_BASE]
    assert mask[A_TARGET_BOARD_BASE + 1]
    assert not mask[A_APPLY_EFFECT_SKIP]

    lobby.step_action(seat, A_TARGET_BOARD_BASE)
    mask2 = lobby.legal_mask_for_seat(seat)
    assert mask2[A_TARGET_BOARD_BASE + 1]
    assert mask2[A_APPLY_EFFECT_SKIP]

    lobby.step_action(seat, A_TARGET_BOARD_BASE + 1)
    assert seat not in lobby._rl_pending

    b = lobby.state.players[seat].board
    assert len(b) == 3
    assert b[1].card_id == "EX1_093"
    assert b[0].bonus_attack == 1 and Keyword.TAUNT in b[0].keywords
    assert b[2].bonus_attack == 1 and Keyword.TAUNT in b[2].keywords


def test_argus_single_neighbor_skip_second():
    env = _lobby_for_seat(0)
    lobby = env.lobby
    lobby.reset(seed=11)
    seat = 0
    while lobby.current_seat() != seat:
        lobby.step_auto(deterministic=True)

    p = lobby.state.players[seat]
    p.board = [make_minion("recruit"), make_minion("guard")]
    p.hand[0] = make_minion("defender_argus")

    lobby.step_action(seat, A_PLACE_BASE)
    lobby.step_action(seat, A_TARGET_BOARD_BASE + 1)
    assert lobby._rl_pending[seat].can_skip_second_adjacent()
    lobby.step_action(seat, A_APPLY_EFFECT_SKIP)

    b = lobby.state.players[seat].board
    assert len(b) == 3
    assert b[1].bonus_attack == 1 and Keyword.TAUNT in b[1].keywords


def test_obs_encodes_rl_pending_apply():
    env = _lobby_for_seat(0)
    lobby = env.lobby
    lobby.reset(seed=12)
    seat = 0
    while lobby.current_seat() != seat:
        lobby.step_auto(deterministic=True)
    p = lobby.state.players[seat]
    p.board = [make_minion("recruit"), make_minion("guard")]
    p.hand[0] = make_minion("defender_argus")
    lobby.step_action(seat, A_PLACE_BASE)
    pend = encode_pending_choice(p, rl_pending=lobby.rl_pending_for_seat(seat))
    assert pend[PENDING_IS_APPLY_OFFSET] == 1.0
