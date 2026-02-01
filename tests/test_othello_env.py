"""Tests for Othello environment."""

import numpy as np
import pytest

from src.envs.othello import OthelloEnv, OthelloGame, OthelloState


def test_othello_initial_state():
    env = OthelloEnv()
    obs = env.reset()
    
    assert env.board.shape == (8, 8)
    assert env.board[3, 3] == -1
    assert env.board[3, 4] == 1
    assert env.board[4, 3] == 1
    assert env.board[4, 4] == -1
    assert not env.done
    assert env.current_player_token == 1


def test_othello_legal_moves():
    env = OthelloEnv()
    env.reset()
    
    legal_actions = env.get_legal_actions()
    assert len(legal_actions) > 0
    
    legal_mask = env.legal_actions_mask
    assert legal_mask.shape == (64,)
    assert np.sum(legal_mask) == len(legal_actions)


def test_othello_valid_move():
    env = OthelloEnv()
    env.reset()
    
    legal_actions = env.get_legal_actions()
    assert len(legal_actions) > 0
    
    action = legal_actions[0]
    result = env.step(action)
    
    assert not result.info["invalid_action"]
    assert result.reward >= 0 or not result.terminated


def test_othello_invalid_move():
    env = OthelloEnv()
    env.reset()
    
    result = env.step(0)
    
    if 0 not in env.get_legal_actions():
        assert result.info["invalid_action"]
        assert result.reward < 0


def test_othello_game_flow():
    env = OthelloEnv()
    env.reset()
    
    max_moves = 100
    moves = 0
    
    while not env.done and moves < max_moves:
        legal_actions = env.get_legal_actions()
        if not legal_actions:
            break
        
        action = legal_actions[0]
        result = env.step(action)
        moves += 1
    
    assert moves > 0


def test_othello_game_rules():
    game = OthelloGame()
    state = game.initial_state()
    
    assert state.board[3, 3] == -1
    assert state.board[3, 4] == 1
    assert state.board[4, 3] == 1
    assert state.board[4, 4] == -1
    
    legal_actions = game.legal_actions(state)
    assert len(legal_actions) == 4


def test_othello_flipping():
    env = OthelloEnv()
    env.reset()
    
    initial_count = np.sum(env.board != 0)
    
    legal_actions = env.get_legal_actions()
    action = legal_actions[0]
    env.step(action)
    
    final_count = np.sum(env.board != 0)
    assert final_count > initial_count


def test_othello_state_copy():
    env = OthelloEnv()
    env.reset()
    
    state1 = env.get_state()
    legal_actions = env.get_legal_actions()
    env.step(legal_actions[0])
    state2 = env.get_state()
    
    assert not np.array_equal(state1.board, state2.board)
    
    env.set_state(state1)
    assert np.array_equal(env.board, state1.board)


def test_othello_render():
    env = OthelloEnv()
    env.reset()
    
    env.render()
